"""Runnable sequential PPM experiments for rotated surface-code patches.

Adapted from John Yuehan Zhang's CircLS repository at commit ``8802a5b``.
Every step carries an explicit coarse-grid route; this driver does not perform
logical placement or routing.

Cell-adjacent two-patch steps are classified live by the four-row rule table.
The lowering layer can describe all four rows, but this initial experiment
driver executes only rows 1/4 (plain/recoloured merge). Rows 2/3 require a
mixed-measurement wall lifecycle that is not yet part of the tracker contract
and are rejected explicitly. Distant targets use the explicit-route corridor.
"""
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.bent_joint_se import se_round_chunk
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.ppm.lowering import (
    PPMOutcome,
    RotatedSurfacePPMRequest,
    apply_ppm_plan,
    joint_pauli_vector,
    lower_ppm,
    record_parity,
)
from lightstim.qec_code.surface_code.rotated.ppm.placement import (
    RotatedSurfacePatchPlacement,
    conjugate_patch_records,
)

__all__ = [
    "PPMOutcome",
    "RotatedSurfacePPMExperiment",
    "RotatedSurfacePPMStep",
    "UnsupportedPPMExperimentError",
]

_FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}


class UnsupportedPPMExperimentError(NotImplementedError):
    """A valid lowering plan that this experiment driver cannot execute."""


@dataclass(frozen=True)
class RotatedSurfacePPMStep:
    """One explicit-route PPM in a rotated-surface experiment.

    ``targets`` contains ``(patch_name, Pauli)`` pairs with Pauli ``X`` or
    ``Z``. ``route`` contains the exact coarse-grid corridor cells, or is
    empty for a cell-adjacent pair.
    """

    targets: Tuple[Tuple[str, str], ...]
    route: Tuple[Tuple[int, int], ...]
    construction: str = 'auto'
    schedule: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            'targets',
            tuple((name, pauli) for name, pauli in self.targets),
        )
        if self.route is not None:
            object.__setattr__(
                self,
                'route',
                tuple(tuple(cell) for cell in self.route),
            )


def _steps_commute(a, b):
    """Whether two logical Pauli products commute on their shared patches."""
    pa = dict(a.targets)
    pb = dict(b.targets)
    anti = sum(pa[name] != pb[name] for name in pa.keys() & pb.keys())
    return anti % 2 == 0


class RotatedSurfacePPMExperiment:
    """Run PPM steps on rotated patches placed on the seam-column grid.

    ``patches`` is a sequence of :class:`RotatedSurfacePatchPlacement` and
    ``ppm_sequence`` is a sequence of :class:`RotatedSurfacePPMStep`.

    All patches are allocated and initialised up front; each PPM lowers and
    registers its coupler immediately before use, runs ``rounds`` merged SE
    rounds, splits, and
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
        patch_name_list = [s.name for s in self.patches]
        if len(set(patch_name_list)) != len(patch_name_list):
            raise ValueError(
                f"patch names must be unique; got {patch_name_list}")
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
        if rounds_init < 1:
            raise ValueError(
                "rounds_init must be >= 1 (a freshly initialised patch needs "
                "at least one standalone SE round to establish its "
                f"stabilizers before the merge); got {rounds_init}"
            )
        self.rounds = rounds
        self.rounds_init = rounds_init
        self.idle_rounds = idle_rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        for i, step in enumerate(self.ppm_sequence):
            try:
                RotatedSurfacePPMRequest(
                    targets=step.targets,
                    route=(None if step.route is None else
                           tuple(tuple(c) for c in step.route)),
                    construction=step.construction,
                    schedule=step.schedule,
                )
            except ValueError as exc:
                raise type(exc)(f"PPM {i}: {exc}") from exc
        for i, step in enumerate(self.ppm_sequence):
            for j, earlier in enumerate(self.ppm_sequence[:i]):
                if not _steps_commute(earlier, step):
                    raise UnsupportedPPMExperimentError(
                        f"PPM {i} anti-commutes with PPM {j}; the current "
                        "experiment driver supports commuting sequences only")
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

    def _conj(self, step):
        """The conj-registered (colour-swapped) subset of the step's targets —
        reaches the layout builder as ``conj_names`` so the seam table is
        classified against the REAL registrations."""
        return frozenset(nm for nm, _ in step.targets
                         if nm in self.colour_swapped)

    def _specs(self):
        """The patch specs the router sees: every patch, stamped with its
        live orientation (a colour-swapped patch's labels flip at allocation,
        so its live orientation starts flipped too)."""
        return [replace(s, orientation=self._orient[s.name])
                for s in self.patches]

    def _request(self, step):
        """Bridge a RotatedSurfacePPMStep to the kernel's declarative request."""
        return RotatedSurfacePPMRequest(
            targets=step.targets,
            route=step.route,
            construction=step.construction,
            schedule=step.schedule,
        )

    @property
    def plans(self):
        """Lowering plans from the most recent build, in sequence order."""
        plans = getattr(self, '_plans', {})
        return tuple(plans[i] for i in range(len(plans)))

    def _register_step(self, i, step):
        """Lower PPM ``i`` through the kernel and register its plan.  All
        construction decisions (rule row, wall vs merge, schedule, corridor
        routing, certificate) are the kernel's; the driver only stores the
        plan and applies it."""
        try:
            plan = lower_ppm(self._specs(), self._request(step),
                             system=self.system, conj_names=self._conj(step),
                             schedule_policy=self.schedule)
        except ValueError as exc:
            raise type(exc)(f'PPM {i}: {exc}') from exc
        if plan.kind == 'wall':
            raise UnsupportedPPMExperimentError(
                f"PPM {i} lowers to a stretched-stabilizer wall. The lowering "
                "plan is available, but RotatedSurfacePPMExperiment does not yet "
                "execute wall rounds because they mix disposable syndrome "
                "measurements with retained data measurements.")
        apply_ppm_plan(self.system, plan, f'ppm_{i}')
        self._plans[i] = plan
        self._sched[i] = plan.schedule
        self._routes[i] = plan.route_result
        if plan.rule is not None:
            self._rules[i] = plan.rule

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

    def _capture_outcome(self, i, step, pre):
        """Bank PPM ``i``'s protocol output: the joint's record parity
        before the merge (None = free coin) and after the split."""
        post = record_parity(
            self.tracker, joint_pauli_vector(self.system,
                                             step.targets))
        self.ppm_outcomes[i] = PPMOutcome(
            step=i, targets=step.targets,
            records_pre_merge=tuple(pre) if pre is not None else None,
            records_post_split=tuple(post) if post is not None else None)

    def _apply_ppm_step(self, i, step):
        init_basis = self._plans[i].corridor_init_basis
        cname = f'ppm_{i}'
        pre = record_parity(
            self.tracker, joint_pauli_vector(self.system,
                                             step.targets))
        # idle standalone SE between PPMs (debug knob; default 0)
        if self.idle_rounds:
            self._standalone_se(self.idle_rounds)
        # merge (coupler already registered)
        self.builder.activate_coupler(cname)
        coupler_patch = self.system.coupler_patches[cname]
        l2g = self.system.local_to_global_map[cname]
        coupler_init = {l2g[q]: init_basis for q in coupler_patch.data_indices}
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
        self._capture_outcome(i, step, pre)

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
        self._rules = {}
        self._sched = {}
        self._plans = {}
        # protocol outputs, one per applied PPM (review §3: distinct from
        # the evaluation observables the final readout emits)
        self.ppm_outcomes: Dict[int, PPMOutcome] = {}

        # allocate ALL patches up front (no liveness / first-use init here)
        for s in self.patches:
            self._alloc_patch(s.name)

        self._routes: List = [None] * len(self.ppm_sequence)

        self._setup()
        self.builder.write_coordinates()
        self._init_and_baseline({s.name for s in self.patches})

        for i, step in enumerate(self.ppm_sequence):
            self._register_step(i, step)
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
