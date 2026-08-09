# protocols/ppm/sequential.py
#
# Copied from https://github.com/John-YuehanZhang/CircLS @ 8802a5b — the
# author's own repository (John Yuehan Zhang).  Source: circls/
# sequential_ppm_ls.py (``SequentialPPMExperiment``), distilled to the
# minimal explicit-route driver.
#
# Intentionally NOT copied: liveness / first-use init, auto-rotation
# planning, snake steps, Y initial states (Gidney births), joint-parity
# record capture and program-bit observable assembly.  Every ``PPMStep``
# must carry an explicit ``route`` (``[]`` for cell-adjacent targets);
# ``colour_swapped`` is kept (rule-table rows 2/3 depend on patch types).

"""Sequential PPM driver — run a sequence of joint Pauli-product measurements
on rotated routed surface-code patches; between PPMs measure out ONLY the bus
(corridor), target patches persist with state.

Cell-adjacent two-patch steps are classified live by the four-row rule table
(:mod:`.seam_rules`): rows 2/3 take the stretched-stabilizer wall coupler
(diagonal schedule mandatory), rows 1/4 take the plain/recoloured zero-cell
merge.  Distant targets go through the explicit-route corridor.
"""
from dataclasses import replace
from typing import Dict, List

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.bent_joint_se import se_round_chunk
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)

from .spec import PPMStep, cell_index, conjugate_patch_records
from .coupler import route_and_build, RotatedRoutedMultiPatchCoupler
from .seam_rules import (
    RotatedSeamWallCoupler, SeamRuleError, classify_seam, patch_view,
    wall_spec)

__all__ = ["SequentialPPMExperiment"]

_CONJ = {'X': 'Z', 'Z': 'X'}
_FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}


class SequentialPPMExperiment:
    """Run ``ppm_sequence`` (a list of :class:`~.spec.PPMStep`) on the
    patches declared by ``patches`` (a list of :class:`~.spec.PatchSpec`
    on the seam-column coarse grid, pitch ``2d+2``).

    All patches are allocated and initialised up front; each PPM activates
    its pre-registered coupler, runs ``rounds`` merged SE rounds, splits, and
    measures out the corridor only; the final data readout measures every
    patch in its ``final_measure_states`` letter (the builder emits the
    standing logical observables there).
    """

    def __init__(self, patches, ppm_sequence, *, initial_states,
                 final_measure_states, rounds=3, rounds_init=1, idle_rounds=0,
                 noise_params=None, noise_model='circuit_level',
                 schedule: str = 'auto', colour_swapped=frozenset()):
        self.patches = list(patches)
        self.ppm_sequence = list(ppm_sequence)
        self.initial_states = {k: v.upper() for k, v in initial_states.items()}
        self.final_measure_states = {k: v.upper()
                                     for k, v in final_measure_states.items()}
        bad = {k: v for k, v in self.initial_states.items()
               if v not in ('X', 'Z')}
        if bad:
            raise ValueError(
                f"initial_states letters must be 'X' or 'Z'; got {bad} "
                f"(Y initial states are not part of this minimal variant)")
        bad = {k: v for k, v in self.final_measure_states.items()
               if v not in ('X', 'Z')}
        if bad:
            raise ValueError(
                f"final_measure_states letters must be 'X' or 'Z'; got {bad}")
        assert rounds_init >= 1, (
            f"rounds_init must be >= 1 (a freshly initialised patch needs at "
            f"least one standalone SE round to establish its stabilizers "
            f"before the merge); got {rounds_init}")
        self.rounds = rounds
        self.rounds_init = rounds_init
        self.idle_rounds = idle_rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        for i, step in enumerate(self.ppm_sequence):
            if step.route is None:
                raise ValueError(
                    f"PPM {i}: route is required in this variant — pass the "
                    f"explicit corridor cells, or [] for cell-adjacent targets")
            for name, P in step.interaction_type:
                if P not in ('X', 'Z'):
                    raise ValueError(
                        f"PPM Pauli must be 'X' or 'Z', got {P!r} on patch {name}")
        patch_names = {s.name for s in self.patches}
        missing_init = patch_names - self.initial_states.keys()
        if missing_init:
            raise ValueError(
                f"patch(es) {sorted(missing_init)} missing from initial_states")
        missing_final = patch_names - self.final_measure_states.keys()
        if missing_final:
            raise ValueError(
                f"patch(es) {sorted(missing_final)} missing from "
                f"final_measure_states")
        # schedule policy for MERGED rounds (decide right after routing — any
        # bend in the corridor, or a wall in the step, forces the WHOLE merged
        # block onto the diagonal schedule; straight corridors stay bent):
        # 'auto' (default) | 'bent' (wall steps raise) | 'diagonal'
        if schedule not in ('auto', 'bent', 'diagonal'):
            raise ValueError(
                f"schedule must be 'auto', 'bent' or 'diagonal', got {schedule!r}")
        self.schedule = schedule
        # colour_swapped: patches whose red/blue COLOURS are swapped at
        # allocation (conjugate_patch_records on top of the standard build).
        # This is a COLOUR knob — weight-2 positions (textbook/conjugate)
        # are untouched; the live logical orientation flips with the labels.
        self.colour_swapped = frozenset(colour_swapped or ())
        unknown_cs = self.colour_swapped - patch_names
        if unknown_cs:
            raise ValueError(
                f"colour_swapped names {sorted(unknown_cs)} are not patches")
        self._orient = {s.name: (_FLIP_O[s.orientation]
                                 if s.name in self.colour_swapped
                                 else s.orientation)
                        for s in self.patches}

    @staticmethod
    def _bus_of(interaction_type):
        n_x = sum(1 for _, P in interaction_type if P == 'X')
        return 'X' if n_x >= len(interaction_type) - n_x else 'Z'

    def _conj(self, step):
        """The conj-registered (colour-swapped) subset of the step's targets —
        reaches route_and_build as ``conj_names`` so the seam table is
        classified against the REAL registrations."""
        return frozenset(nm for nm, _ in step.interaction_type
                         if nm in self.colour_swapped)

    def _specs(self):
        """The patch specs the router sees: every patch, stamped with its
        live orientation (a colour-swapped patch's labels flip at allocation,
        so its live orientation starts flipped too)."""
        return [replace(s, orientation=self._orient[s.name])
                for s in self.patches]

    def _adjacent_pair(self, step):
        """True iff the step's targets are exactly two patches on
        edge-adjacent coarse cells — the direct-seam territory of the rule
        table (rows 1-4); distant targets go through the routed corridor."""
        names = [nm for nm, _ in step.interaction_type]
        if len(names) != 2:
            return False
        cells = [cell_index(self._by_name[nm].origin,
                            self._by_name[nm].distance, seam=True)
                 for nm in names]
        (a1, b1), (a2, b2) = cells
        return abs(a1 - a2) + abs(b1 - b2) == 1

    def _register_adjacent(self, i, step):
        """Classify the direct seam by the rule table (live probe = ground
        truth) and register the chosen construction: rows 2/3 -> the
        stretched-stabilizer wall coupler; rows 1/4 -> the plain/recoloured
        merge through the zero-cell corridor (route=[])."""
        names = [nm for nm, _ in step.interaction_type]
        views = {nm: patch_view(self.system, nm) for nm in names}
        tgt = dict(step.interaction_type)
        rule = classify_seam(views[names[0]], tgt[names[0]],
                             views[names[1]], tgt[names[1]])
        cons = step.construction if step.construction != 'auto' \
            else rule.construction
        if cons == 'wall':
            spec = wall_spec(views[names[0]], views[names[1]], tgt)
            self.system.register_coupler(
                RotatedSeamWallCoupler(), patch_names=names, name=f'ppm_{i}',
                spec=spec, occupied=frozenset(self.system.index_map))
            self._walls[i] = spec
            self._sched[i] = self._pick_schedule(step, is_wall=True,
                                                 collinear=True)
        else:
            self._register_ppm(i, step)
            self._sched[i] = self._pick_schedule(step, is_wall=False,
                                                 collinear=True)
        self._rules[i] = rule

    def _pick_schedule(self, step, *, is_wall, collinear):
        """The merged-round schedule, decided right after routing: a wall or
        any corridor bend forces the whole merged block onto the diagonal
        schedule; straight corridors stay bent."""
        if is_wall:
            if step.schedule == 'bent' or self.schedule == 'bent':
                raise SeamRuleError(
                    "a wall step cannot run the bent schedule: the bent chunk "
                    "cannot express stretched (kf) checks — use 'diagonal'")
            return 'diagonal'
        if step.schedule is not None:
            return step.schedule
        if self.schedule in ('bent', 'diagonal'):
            return self.schedule
        return 'bent' if collinear else 'diagonal'

    def _register_ppm(self, i, step):
        """Route (explicit route only) and register PPM ``i``'s coupler."""
        specs = self._specs()
        r = route_and_build(specs, step.interaction_type, seam=True,
                            route=step.route, conj_names=self._conj(step))
        if r.status != 'ok':
            raise ValueError(
                f'route_and_build failed at PPM {i}: {r.status} — {r.message}')
        # a dispatch-raised stretched wall in the layout makes this a WALL
        # step: kf records demand the diagonal schedule
        has_kf = any(ch.get('kf')
                     for ch in (r.layout.checks if r.layout is not None else ()))
        if i not in self._sched:
            # schedule decision straight off the route geometry: any bend
            # (cells not collinear) forces the diagonal schedule
            names = [nm for nm, _ in step.interaction_type]
            cells = [cell_index(self._by_name[nm].origin,
                                self._by_name[nm].distance, seam=True)
                     for nm in names] + [tuple(c) for c in (r.tree or ())]
            collinear = (len({a for a, _ in cells}) == 1
                         or len({b for _, b in cells}) == 1)
            self._sched[i] = self._pick_schedule(step, is_wall=has_kf,
                                                 collinear=collinear)
        self.system.register_coupler(
            RotatedRoutedMultiPatchCoupler(),
            patch_names=[nm for nm, _ in step.interaction_type],
            name=f'ppm_{i}', specs=specs,
            target=step.interaction_type, subset_route=r, seam=True,
            minority_names=self._conj(step))
        self._routes[i] = r

    def _alloc_patch(self, name):
        """Create one patch's physical qubits in the system: the DECLARED
        orientation, colours from ``colour_swapped``."""
        s = self._by_name[name]
        p = RotatedSurfaceCode(distance=s.distance)
        if s.orientation == 'X_horizontal':
            p.transpose_coords()
        if name in self.colour_swapped:
            conjugate_patch_records(p)      # colour knob: swap red/blue only
        self.system.add_patch(p, name=name,
                              offset=(s.origin[0] - 1, s.origin[1] - 1))

    def _setup(self):
        self.tracker = SyndromeTracker(
            num_qubits=self.system.num_qubits,
            expected_num_logicals=self.system.num_logicals)
        self.builder = CircuitBuilder(tracker=self.tracker,
                                      system_config=self.system,
                                      if_detector=True)
        self.system.register_tracker(self.tracker)
        self.system.register_builder(self.builder)

    def _standalone_se(self, n_rounds):
        if n_rounds < 1:
            return
        owner = self.system.index_to_owner_map
        domains = {tuple(self.system.qubit_coords[q]): self._orient[owner[q]]
                   for q in self.system.data_indices
                   if owner.get(q) in self._orient}
        chunk = se_round_chunk(self.system, domains=domains)
        self.builder.apply_syndrome_extraction(circuit_chunk=chunk,
                                               rounds=n_rounds)

    def _merged_chunk(self, i):
        """The one-round chunk of PPM ``i``'s merged system."""
        if self._sched.get(i, 'bent') == 'diagonal':
            # any bend in the corridor (or a forced override) puts the whole
            # merged block on the diagonal schedule — one stretched/bent
            # check's tick budget is the global tick budget
            return DiagonalSurfaceCodeExtractionBlock(self.system).circuit
        lay = self._routes[i].layout
        if any(c.get('kf') for c in lay.checks):
            # a dispatch-built stretched wall in the layout: the bent
            # chunk cannot express kf relay checks
            return DiagonalSurfaceCodeExtractionBlock(self.system).circuit
        domains_merged = dict(getattr(lay, 'domains', {}) or {})
        # SPECTATOR hook-safety: the routed layout's domains cover only the
        # step's patches + corridor.  Every OTHER live patch must still run
        # its orientation-correct schedule — a defaulted hook direction
        # aligned with the idle logical halves its distance to (d+1)/2
        owner = self.system.index_to_owner_map
        for q in self.system.data_indices:
            nm = owner.get(q)
            if nm in self._orient:
                domains_merged.setdefault(
                    tuple(self.system.qubit_coords[q]), self._orient[nm])
        return se_round_chunk(self.system, domains=domains_merged)

    def _apply_ppm_step(self, i, step):
        bus = self._bus_of(step.interaction_type)
        cname = f'ppm_{i}'
        # idle standalone SE between PPMs (debug knob; default 0)
        if self.idle_rounds:
            self._standalone_se(self.idle_rounds)
        # merge (coupler already registered)
        self.builder.activate_coupler(cname)
        coupler_patch = self.system.coupler_patches[cname]
        l2g = self.system.local_to_global_map[cname]
        coupler_init = {l2g[q]: _CONJ[bus] for q in coupler_patch.data_indices}
        if coupler_init:
            self.builder.initialize(init_dict=coupler_init,
                                    n=self.system.num_qubits)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=self._merged_chunk(i), rounds=self.rounds)
        # split: the bus/corridor readout measures out the corridor ONLY; the
        # joint patch-patch relation it created persists, so it must not
        # resolve absorbed_ops.
        self.builder.deactivate_coupler(cname)
        self.builder.apply_data_readout(final_measurements=dict(coupler_init),
                                        resolve_absorbed=False)

    def _apply_wall_step(self, i, step):
        """A stretched-stabilizer wall step: activate the wall coupler (its
        apparatus adds no data qubits, so there is no corridor init/readout),
        run the merged rounds on the DIAGONAL schedule (mandatory for kf
        checks), then split by deactivating — the paused facing lobes come
        back automatically."""
        cname = f'ppm_{i}'
        if self.idle_rounds:
            self._standalone_se(self.idle_rounds)
        self.builder.activate_coupler(cname)
        self.builder.apply_syndrome_extraction(
            DiagonalSurfaceCodeExtractionBlock(self.system).circuit,
            rounds=self.rounds)
        self.builder.deactivate_coupler(cname)

    def _init_and_baseline(self, names):
        """Reset the data qubits owned by ``names`` to their initial states,
        then run the baseline SE (``rounds_init``) that establishes their
        stabilizers."""
        owner = self.system.index_to_owner_map
        init_dict = {q: self.initial_states[owner[q]]
                     for q in self.system.data_indices if owner.get(q) in names}
        if init_dict:
            self.builder.initialize(init_dict=init_dict,
                                    n=self.system.num_qubits)
        self._standalone_se(self.rounds_init)

    def build(self):
        # build() is re-runnable: everything below re-derives from the specs.
        self._orient = {s.name: (_FLIP_O[s.orientation]
                                 if s.name in self.colour_swapped
                                 else s.orientation)
                        for s in self.patches}
        self.system = QECSystem()
        self._by_name = {s.name: s for s in self.patches}
        # direct-seam steps (two cell-adjacent targets): classified by the
        # rule table against the LIVE system, so always registered lazily
        self._adj_steps = {i for i, st in enumerate(self.ppm_sequence)
                           if self._adjacent_pair(st)}
        self._walls = {}
        self._rules = {}
        self._sched = {}

        # allocate ALL patches up front (no liveness / first-use init here)
        for s in self.patches:
            self._alloc_patch(s.name)

        # register the routed couplers up front (BEFORE the tracker exists);
        # adjacent-pair steps register lazily in the loop, off the live probe
        self._routes: List = [None] * len(self.ppm_sequence)
        for i, step in enumerate(self.ppm_sequence):
            if i in self._adj_steps:
                continue
            self._register_ppm(i, step)

        self._setup()
        self.builder.write_coordinates()
        self._init_and_baseline({s.name for s in self.patches})

        for i, step in enumerate(self.ppm_sequence):
            if i in self._adj_steps and i not in self._walls \
                    and self._routes[i] is None:
                self._register_adjacent(i, step)
            if i in self._walls:
                self._apply_wall_step(i, step)
                continue
            self._apply_ppm_step(i, step)

        owner = self.system.index_to_owner_map
        meas_dict = {q: self.final_measure_states[owner[q]]
                     for q in self.system.data_indices
                     if owner.get(q) in self.final_measure_states}
        if meas_dict:
            self.builder.apply_data_readout(final_measurements=meas_dict)

        if self.noise_params is not None:
            return self.builder.build_noisy_circuit(
                noise_params=self.noise_params, noise_model=self.noise_model)
        return self.builder.circuit
