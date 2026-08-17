# protocols/ppm/coupler.py
#
# Copied from https://github.com/John-YuehanZhang/CircLS @ 8802a5b — the
# author's own repository (John Yuehan Zhang).  Source: circls/
# multi_patch_coupler.py, the EXPLICIT-ROUTE branch of ``route_and_build``
# and its transitive closure only (deterministic rule constructor, physics
# oracle ``MultiPatchLayout.verify``, seam-table wall dispatch,
# ``RotatedRoutedMultiPatchCoupler``).
#
# Intentionally NOT copied: the automatic router (corridor graph, Steiner /
# EMV candidate search, geometry caches) — in this LightStim variant
# ``route`` is REQUIRED (pass ``[]`` for cell-adjacent targets) and
# ``route=None`` raises ``BentLayoutError``.

from dataclasses import dataclass, field
import itertools
from types import SimpleNamespace

import numpy as np

from lightstim.ir.coupler import LogicalCouplerProtocol as _LCP
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode

from .spec import (BentLayoutError, PatchSpec, _FLIP, _pitch, cell,
                       cell_index, origin_of, place_patch)

__all__ = ["route_and_build", "SubsetRoute", "MultiPatchLayout",
           "RotatedRoutedMultiPatchCoupler", "rule_based_joint_checks",
           "path_to_corridor"]








#: Post-routing seam-table dispatch (route first, then classify every attach
#: seam on the path by rule-table v3).  Same-letter type-split joints:
#: the minority-type target is REGISTERED in the recolour convention and its
#: seam carries the verified mixed wall family — the rule-table row algebra
#: row2 = row4 o row3 (user-confirmed, measured full-distance end-to-end on
#: straight, L-bend and T-junction geometries, 2026-07-29).
TABLE_WALL_DISPATCH = True


def _bent_plaquettes(data, retype, phase):
    """Every (even,even) plaquette center with >=2 data corners, typed by the re-typing model.

    Base type at center ``(cx, cy)`` is 'X' if ``((cx+cy)//2 + phase)`` is odd else 'Z'.
    Each corner's Pauli equals the base type, flipped (X<->Z) if the corner lies in the
    re-typed set ``retype`` (the X-side arm).  All-equal corner Paulis -> a pure check of
    that type; mixed -> a domain-wall check of type 'M'.  ``phase`` (0/1) absorbs the
    position-dependent parity of ``(cx+cy)//2`` so the construction is translation-correct.
    """
    xs = [c for c, _ in data]
    ys = [r for _, r in data]
    out = []
    for cx in range(min(xs) - 1, max(xs) + 2):
        for cy in range(min(ys) - 1, max(ys) + 2):
            if cx % 2 or cy % 2:
                continue
            present = [(cx + a, cy + b) for a in (-1, 1) for b in (-1, 1)
                       if (cx + a, cy + b) in data]
            if len(present) < 2:
                continue
            base = "X" if ((cx + cy) // 2 + phase) % 2 == 1 else "Z"
            pauli = {q: (_FLIP[base] if q in retype else base) for q in sorted(present)}
            types = set(pauli.values())
            t = next(iter(types)) if len(types) == 1 else "M"
            out.append({"syn": (cx, cy), "type": t, "pauli": pauli,
                        "corners": sorted(pauli)})
    return out


def _commute(a, b, n):
    return int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0






def _symplectic(data):
    idx = {c: i for i, c in enumerate(sorted(data))}
    n = len(idx)

    def sv(pauli):
        v = np.zeros(2 * n, np.uint8)
        for c, P in pauli.items():
            if P in ("X", "Y"):
                v[idx[c]] ^= 1
            if P in ("Z", "Y"):
                v[n + idx[c]] ^= 1
        return v
    return sv, n


def _no_tick_collision(circuit):
    """True iff no qubit is touched by two operations within the same TICK window."""
    skip = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK",
            "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR"}
    seen = set()
    for op in circuit.flattened():
        if op.name == "TICK":
            seen = set()
            continue
        if op.name in skip:
            continue
        for t in op.targets_copy():
            if t.is_qubit_target:
                if t.value in seen:
                    return False
                seen.add(t.value)
    return True


# -----------------------------------------------------------------------------
# bit-packed GF(2) machinery — Pauli vectors as Python ints (X in bits 0..n-1, Z in n..2n-1).
# Int XOR / int.bit_count() are C-level, so the boundary search scales to N=5/6 and d=5.
# -----------------------------------------------------------------------------

def _int_symplectic(data):
    """Return ``(isv, n)`` where ``isv({coord: pauli})`` packs a Pauli into a 2n-bit int."""
    idx = {c: i for i, c in enumerate(sorted(data))}
    n = len(idx)

    def isv(pauli):
        v = 0
        for c, P in pauli.items():
            i = idx[c]
            if P in ("X", "Y"):
                v |= 1 << i
            if P in ("Z", "Y"):
                v |= 1 << (n + i)
        return v
    return isv, n


def _icommute(a, b, n):
    mask = (1 << n) - 1
    return (((a & mask) & (b >> n)).bit_count() + ((a >> n) & (b & mask)).bit_count()) & 1 == 0


class _IntBasis:
    """An incremental GF(2) row basis over bit-packed ints (reduction by highest set bit)."""
    __slots__ = ("piv",)

    def __init__(self):
        self.piv = {}                      # highest-set-bit -> reduced row int

    def copy(self):
        b = _IntBasis()
        b.piv = dict(self.piv)
        return b

    def reduce(self, v):
        while v:
            r = self.piv.get(v.bit_length() - 1)
            if r is None:
                return v
            v ^= r
        return v

    def add(self, v):                      # returns True iff v is independent (and adds it)
        r = self.reduce(v)
        if r:
            self.piv[r.bit_length() - 1] = r
            return True
        return False

    def contains(self, v):
        return self.reduce(v) == 0

    @property
    def rank(self):
        return len(self.piv)


def _cols(d):
    return sorted({c for c, _ in d})


def _rows(d):
    return sorted({r for _, r in d})


def _connected(region):
    """True iff ``region`` (a set of (odd,odd) coords) is connected via the plaquette graph."""
    region = set(region)
    if not region:
        return True
    start = next(iter(region))
    seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2),
                       (x + 2, y + 2), (x + 2, y - 2), (x - 2, y + 2), (x - 2, y - 2)):
            if (nx, ny) in region and (nx, ny) not in seen:
                seen.add((nx, ny)); stack.append((nx, ny))
    return len(seen) == len(region)




def _logical_direction(pauli, orientation):
    """The direction the measured ``pauli`` logical string must run, per the patch ``orientation``.

    ``orientation`` names the X̄ direction (``"X_horizontal"`` ⇒ X̄ runs along a row, ``"X_vertical"``
    ⇒ X̄ runs up a column); Z̄ is perpendicular to X̄.  So a measured ``X`` on an ``X_horizontal``
    patch runs ``"horizontal"``, a measured ``Z`` on an ``X_horizontal`` patch runs ``"vertical"``,
    and vice-versa for ``X_vertical``.
    """
    x_is_horizontal = orientation == "X_horizontal"
    if pauli == "X":
        return "horizontal" if x_is_horizontal else "vertical"
    return "vertical" if x_is_horizontal else "horizontal"


def _patch_rep(patch, pauli, direction, F, sv, n):
    """A patch-spanning ``pauli`` string **in the requested ``direction``** commuting with bulk ``F``.

    The orientation declared on the ``PatchSpec`` is honoured: an ``X_horizontal`` X-logical is
    represented by a horizontal string, an ``X_vertical`` X-logical by a vertical one (and Z̄
    perpendicular).  Returns the support, or ``None`` if no string in that direction commutes — the
    caller tries the other parity phase, and if both fail the placement is rejected rather than the
    logical being silently re-oriented.
    """
    c, r = _cols(patch), _rows(patch)
    cands = ([[(x, y) for y in r] for x in c] if direction == "vertical"
             else [[(x, y) for x in c] for y in r])
    for sup in cands:
        v = sv({q: pauli for q in sup})
        if all(_commute(v, f, n) for f in F):
            return sup
    return None
def _readout_chain(data, checks, logicals):
    """Check syndromes whose GF(2) product equals the joint ``∏ᵢ P̄ᵢ`` (for highlighting/readout)."""
    from lightstim.utils.linear_algebra import solve_linear_decomposition
    sv, n = _symplectic(data)
    basis = np.array([sv(ch["pauli"]) for ch in checks], np.uint8)
    target = np.zeros(2 * n, np.uint8)
    for P, sup in logicals:
        target ^= sv({c: P for c in sup})
    co, dep, _ = solve_linear_decomposition(basis=basis, targets=target.reshape(1, -1),
                                             reduce_weight=True)
    if not dep[0]:
        return set()
    return {checks[i]["syn"] for i in np.where(co[0])[0]}


# -----------------------------------------------------------------------------
# layout + builder
# -----------------------------------------------------------------------------

@dataclass
class MultiPatchLayout:
    """A rotated N-patch joint-measurement layout (``M(∏ᵢ P̄ᵢ)``).

    ``logicals`` is a list of ``(name, measured_pauli, support)`` — one per patch.
    ``x_observable`` is the X-type logical the X-memory circuit reads.
    """
    distance: int
    data: list
    checks: list
    logicals: list
    x_observable: list
    readout_chain: set = field(default_factory=set)
    specs: list = field(default_factory=list)
    target: list = field(default_factory=list)
    #: orientation-domain map (data coord -> 'X_horizontal' | 'X_vertical') used
    #: by the hook-benign scheduler; empty means "all default (X_horizontal)".
    domains: dict = field(default_factory=dict)
    #: the native (corridor) Pauli basis this layout was built for.  ``None`` means
    #: "majority target basis (tie -> X)"; an explicit 'X'/'Z' is a PLANNED bus that
    #: forces which patches attach through mixed walls / are conjugated.
    bus: str = None
    #: bus data qubits on the RECOLOURED side of a corridor-internal colour wall
    #: (segment recolouring, flip_cells): they are initialized/read out in the
    #: OPPOSITE basis of the rest of the bus.  Empty for uniform corridors.
    retyped: frozenset = frozenset()

    @property
    def N(self):
        return len(self.logicals)

    def build_circuit(self, rounds=None, p=0.0):
        """No-MPP gate-level syndrome-extraction circuit.

        The memory experiment runs in the **bus basis** — the majority target basis
        (tie -> X), matching the corridor's native type.  ``x_observable`` holds a
        bus-basis logical representative (see ``_construct_phase``), so basis and
        tracked observable always agree; running the opposite-basis experiment on a
        bus layout is ill-posed (a single measurement flip would flip the observable
        undetected).

        WARNING: standalone builder (wraps ``RotatedBentJointMeasurement.circuit``);
        its observable is a single bus-basis logical, NOT the ``SyndromeTracker``
        joint m.  Good for single-patch memory + determinism/collision checks, but
        its MULTI-PATCH graphlike distance is NOT the real joint distance -- use
        :class:`RoutedMultiPatchLSExperiment` for joint distance / LER."""
        from lightstim.qec_code.surface_code.rotated.bent_joint_se import RotatedBentJointMeasurement
        if rounds is None:
            rounds = self.distance
        if self.bus is not None:
            basis = self.bus
        else:
            n_x = sum(1 for _, P, _ in self.logicals if P == "X")
            basis = "X" if n_x >= len(self.logicals) - n_x else "Z"
        return RotatedBentJointMeasurement(
            list(self.data), self.checks, list(self.x_observable),
            domains=self.domains,
        ).circuit(rounds=rounds, p=p, basis=basis)

    def verify(self, rounds=None):
        """The eleven N-patch acceptance checks; returns a dict of booleans (all True == valid)."""
        isv, n = _int_symplectic(self.data)
        S = [isv(ch["pauli"]) for ch in self.checks]
        singles = [isv({c: P for c in sup}) for _, P, sup in self.logicals]
        joint = 0
        for v in singles:
            joint ^= v
        N = len(self.logicals)
        subs = []                                       # all proper non-empty subset products
        for r in range(1, N):
            for comb in itertools.combinations(range(N), r):
                v = 0
                for k in comb:
                    v ^= singles[k]
                subs.append(v)
        B = _IntBasis()
        for v in S:
            B.add(v)
        commute = all(_icommute(S[i], S[j], n) for i in range(len(S)) for j in range(i + 1, len(S)))
        twist = any(P == "Y" for ch in self.checks for P in ch["pauli"].values())

        def w1():
            for q in self.data:
                for P in "XZ":
                    v = isv({q: P})
                    if B.reduce(v) != 0 and all(_icommute(v, b, n) for b in B.piv.values()):
                        return True
            return False

        out = dict(
            commute=commute,
            joint=B.contains(joint),
            no_single=not any(B.contains(v) for v in singles),
            no_subjoint=not any(B.contains(v) for v in subs),
            no_twist=not twist,
            logical_count=len(self.data) - B.rank == N - 1,
            no_weight1_logical=not w1(),
        )
        if any(ch.get('kf') for ch in self.checks):
            # a layout hosting stretched (kf) checks cannot run the bent
            # standalone builder — the circuit-level items are covered by the
            # end-to-end diagonal pipeline instead; the algebraic oracle above
            # is the acceptance decision here
            return out
        circuit = self.build_circuit(rounds=rounds)
        out["no_mpp"] = "MPP" not in str(circuit)
        out["no_tick_collision"] = _no_tick_collision(circuit)
        try:
            dem = circuit.detector_error_model(decompose_errors=True)
            dem_ok = (dem.num_detectors == circuit.num_detectors
                      and dem.num_observables == circuit.num_observables)
        except Exception:
            dem_ok = False
        det, obs = circuit.compile_detector_sampler(seed=0).sample(200, separate_observables=True)
        out["dem_valid"] = dem_ok and not det.any() and not obs.any()
        return out


_ORTH = ((2, 0), (-2, 0), (0, 2), (0, -2))


def _corner_qubits(dset):
    """Exposed corner data qubits: ≤2 orthogonal data neighbours."""
    return {q for q in dset
            if sum(((q[0] + dx, q[1] + dy) in dset) for dx, dy in _ORTH) <= 2}


class _Builder:
    """One deterministic greedy construction on a fixed region and phase.

    ``forbidden`` is a set of syndrome sites the boundary may NOT use — the ancilla
    qubits already owned by neighbouring idle patches; vetoing them makes the
    alternating pattern interleave with the neighbour on a shared ancilla line, so
    edge-adjacent placement is physically collision-free whenever the parity allows.
    """

    def __init__(self, placed, target, orient, data, retype, phase, forbidden=frozenset(),
                 native_lock=False, bus=None, conj_names=None, extra_forced=(),
                 retire_lobes=frozenset()):
        self.data = sorted(data)
        self.forbidden = frozenset(forbidden)
        self.target = target
        self.patchq = set().union(*[placed[nm] for nm, _ in target])
        self._placed = placed
        self._orient = orient
        self._retype = set(retype)
        if bus is None:
            n_x = sum(1 for _, P in target if P == "X")
            bus = "X" if n_x >= len(target) - n_x else "Z"
        self._bus = bus
        self.plaqs = _bent_plaquettes(self.data, retype, phase)
        self.forced = [p for p in self.plaqs if len(p["pauli"]) >= 4]
        # injected forced checks (e.g. a corridor-internal stretched wall's kf
        # records): they join the forced set verbatim, so the rank/joint
        # acceptance metric and the logical representatives account for them;
        # their feet are protected from the convex-corner cut rule like patch
        # qubits (a cut foot would orphan the wall record)
        self.forced = self.forced + [dict(e) for e in extra_forced]
        self.patchq |= {q for e in extra_forced for q in e["pauli"]}
        self.pool = sorted((p for p in self.plaqs if len(p["pauli"]) < 4),
                           key=lambda p: p["syn"])
        if retire_lobes:
            # the wall band's two ancilla lines belong to the wall (same
            # retire semantics as the patch|patch wall coupler): no rule tile
            # may sit there
            self.pool = [p for p in self.pool if p["syn"] not in retire_lobes]
        if extra_forced:
            # the injected records' apparatus sites (syn + kf flag/shared) are
            # occupied — expel pool tiles there; anything else (e.g. the corner
            # lobe at the wall band's free end) stays available to the rules
            occ = set()
            for e in extra_forced:
                occ.add(tuple(e["syn"]))
                kf = e.get("kf") or {}
                for k in ("flag", "shared"):
                    if k in kf:
                        occ.add(tuple(kf[k]))
            self.pool = [p for p in self.pool if p["syn"] not in occ]
        # rule 0 -- native lock: a target patch whose interior matches the global
        # checkerboard (not retyped, phase-compatible) keeps its TEXTBOOK standalone
        # construction verbatim: its native boundary semicircles are seeded as forced
        # picks, and non-native tiles fully inside the patch are expelled from the
        # pool.  The one exception is a facing semicircle whose forced-bulk extension
        # (superset whose extra corners lie OUTSIDE the patch, i.e. seam/corridor
        # qubits) exists -- the extension replaces it, per joint-measurement rank.
        self.native_seed, self.locked_cells = [], set()
        if native_lock:
            forced_by_sup = {frozenset(p["pauli"]): p for p in self.forced}
            pool_by_key = {(frozenset(p["pauli"]), p["type"]): p for p in self.pool}
            if bus is None:
                n_x = sum(1 for _, P in target if P == "X")
                bus = "X" if n_x >= len(target) - n_x else "Z"
            _FLIP_O = {"X_horizontal": "X_vertical", "X_vertical": "X_horizontal"}
            _FLIP_P = {"X": "Z", "Z": "X"}
            for nm, _P in target:
                cells = placed[nm]
                # conjugate-registered patch: its lock target is the CONJUGATE-
                # CONVENTION construction (transposed native, types swapped).
                # conj_names is the explicit registration set; None infers it
                # from the measured Pauli (minority basis), the historical rule.
                flipped = (nm in conj_names) if conj_names is not None \
                    else (_P != bus)
                dd = len({x for x, _ in cells})
                o = (min(x for x, _ in cells), min(y for _, y in cells))
                onm = _FLIP_O[orient[nm]] if flipped else orient[nm]
                try:
                    nat = place_patch(SimpleNamespace(
                        origin=o, distance=dd, orientation=onm))["checks"]
                except Exception:
                    continue
                if flipped:
                    nat = [dict(c, type=_FLIP_P[c["type"]],
                                pauli={q: _FLIP_P[P] for q, P in c["pauli"].items()})
                           for c in nat]
                seeds, ok = [], True
                for c in nat:
                    sup = frozenset(c["pauli"])
                    if len(sup) >= 4:              # interior: must equal a forced tile
                        f = forced_by_sup.get(sup)
                        if f is None or f["type"] != c["type"]:
                            ok = False
                            break
                    else:                          # boundary semicircle / corner tile
                        if tuple(c["syn"]) in retire_lobes:
                            continue               # facing an ARM WALL: the lobe is
                                                   # retired, the wall records replace it
                        if any(sup < frozenset(f["pauli"])
                               and not ((frozenset(f["pauli"]) - sup) & cells)
                               for f in self.forced):
                            continue               # facing: replaced by its extension
                        cand = pool_by_key.get((sup, c["type"]))
                        if cand is None:
                            ok = False
                            break
                        seeds.append(cand)
                if not ok:                         # rule 0 is HARD: a phase that cannot
                    self.fail = (f"native lock: patch {nm} textbook construction is "
                                 f"incompatible with checkerboard phase {phase}")
                    return                         # host a native patch is rejected
                self.native_seed.extend(seeds)
                self.locked_cells |= cells
            if self.locked_cells:
                nset = {(frozenset(p["pauli"]), p["type"]) for p in self.native_seed}
                self.pool = [p for p in self.pool
                             if not (set(p["pauli"]) <= self.locked_cells
                                     and (frozenset(p["pauli"]), p["type"]) not in nset)]
        self.seed_syns = {p["syn"] for p in self.native_seed}
        self.isv, self.n = _int_symplectic(self.data)
        sv, _ = _symplectic(self.data)
        F = [sv(p["pauli"]) for p in self.forced]
        self.reps = {}
        for nm, P in target:
            sup = _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, self.n)
            if sup is None:
                self.fail = (f"no {P}-representative for patch {nm} commutes with the "
                             f"forced bulk (checkerboard phase {phase})")
                return
            self.reps[nm] = (P, sup)
        self.fail = None
        self.repv = [self.isv({q: P for q in sup}) for P, sup in self.reps.values()]
        self.fvecs = [self.isv(p["pauli"]) for p in self.forced]

    def _seg_letter(self, p):
        """Boundary LETTER for candidate ``p``, or None to fall back to
        parity.  Measured ground truth (engine dump, docs §2.6): the merged
        complex's free boundary carries the BUS letter everywhere outside
        the patches — bus-letter lobes forbid dual-letter chains from
        terminating across the corridor, which is what protects the joint
        measurement.  The patches keep their native lobes verbatim (rule-0
        anchors); any other all-inside-patch slot belongs to the native
        construction, not to this rule.  Mixed (domain-wall) tiles fall back
        to the parity scan."""
        if p["type"] not in "XZ":
            return None                       # mixed/domain-wall tile
        for nm, _P in self.target:
            if all(q in self._placed[nm] for q in p["pauli"]):
                if self.native_seed:
                    return None               # locked: seeds own the patches
                # UNLOCKED ladder fallback (native constructions cannot share
                # the attempt's phase): the patch boundary is re-derived like
                # an ordinary patch at this phase — each side keeps its
                # orientation-determined letter (X̄-horizontal: E/W carry X,
                # N/S carry Z; X̄-vertical swaps), measured from the engine's
                # unlocked successes (docs §2.6)
                cells = self._placed[nm]
                xs = [x for x, _ in cells]
                ys = [y for _, y in cells]
                fx = {x for x, _ in p["pauli"]}
                fy = {y for _, y in p["pauli"]}
                if fx == {min(xs)} or fx == {max(xs)}:
                    side_ew = True
                elif fy == {min(ys)} or fy == {max(ys)}:
                    side_ew = False
                else:
                    return None               # corner-spanning tile: undecided
                o = self._orient[nm]
                # OPEN (docs §2.6): in the unlocked fallback ONE of the
                # patches (the parity law's recoloured minority) follows the
                # FLIPPED side map — the selection criterion is not yet
                # transcribed; until it is, the orientation map is used for
                # all patches and the residual per-patch lobe deltas are the
                # known gap (cross-check harness tracks them).
                return ("X" if side_ew else "Z") if o == "X_horizontal" \
                    else ("Z" if side_ew else "X")
        if self._bus not in ("X", "Z"):
            return self._bus                  # legacy abutting: no bus letter
        feet = p["pauli"]
        inside = sum(1 for q in feet if q in self._retype)
        if inside == len(feet):
            # recoloured corridor: the M column flips letter AND colour
            # together (component ledger, docs §2.6), so the region's own
            # colour dictates the conjugate bus letter here
            return "X" if self._bus == "Z" else "Z"
        if inside:
            return None                       # straddles the recolour boundary
        return self._bus

    def direct_build(self):
        """Handbook §10.4 DIRECT boundary construction — zero search, zero
        per-tile commutation tests.

        The patches' native weight-2 stabilizers (rule-0 seeds) anchor the
        boundary pattern; wall records' feet count as occupied.  Every other
        boundary candidate is visited outward from the anchors (multi-source
        BFS over the foot-sharing chain graph, lexicographic within each
        layer) and placed iff it shares NO data foot with an already-placed
        boundary stabilizer.  Sharing a foot IS the paper's "one lattice
        spacing" (skip); disjoint is "two spacings" (place): straight-edge
        alternation and the concave weight-3 rule are local instances of this
        single test, and the convex-corner criterion stays with rule 6 in the
        caller.  The engine's sub-product exclusion is deliberately absent —
        the rules are supposed to make it unreachable (cross-checked against
        :meth:`build` by the regression harness).

        Returns the same ``(selected, corner_picks, metric, w1)`` tuple as
        :meth:`build`; the metric is computed ONCE at the end (it gates the
        caller's phase/cut ladders, it steers nothing here)."""
        n, isv = self.n, self.isv
        selected = []
        placed_feet = set()
        placed_pauli = {}                          # qubit -> letter already laid
        for p in self.native_seed:                 # rule 0: anchors, verbatim
            selected.append(p)
            placed_feet.update(p["pauli"])
            placed_pauli.update(p["pauli"])
        for e in self.forced:                      # wall/kf records own their feet
            if len(e["pauli"]) < 4:
                placed_feet.update(e["pauli"])
                placed_pauli.update(e["pauli"])

        # parity-propagating BFS over the whole boundary graph.  Nodes: every
        # pool position (anchors included; forbidden positions participate as
        # TRANSIT nodes — a neighbouring patch's own lobe sits there, so the
        # alternation parity flows through but nothing of ours is placed).
        # Edges: sharing a data foot (one lattice spacing), with a Chebyshev-2
        # fallback so the wave crosses seam endpoints where no common foot
        # survives.  A node at even chain-distance from an anchor is a "place"
        # slot, odd is a "skip" slot; a foot conflict with anything already
        # placed vetoes regardless (waves meeting out of phase — the parity
        # law's edge case — resolve to skip, like the paper's spacing rule).
        # ---- transient chain graph (the user's anchored-walk counting) ----
        # nodes: integer slots (pool candidates, untouched records) plus ONE
        # node per wall band (the single-slot rule).  Edges: sharing a data
        # foot — physical one-lattice-spacing adjacency ONLY (unstitched gaps
        # break chains; no geometric shortcuts).  The count walks outward
        # from the patch-native anchors: even = place slot, odd = skip slot;
        # a wall band advances the count by exactly one.
        nodes = {p["syn"]: p for p in self.pool}
        for p in self.native_seed:
            nodes.setdefault(p["syn"], p)
        # wall band -> one node PER FOOT-LINE (the pair of feet at one
        # along-wall coordinate).  A crossing chain pays exactly one slot
        # (enter 1, leave 0); consecutive lines of the same wall are one
        # slot apart along the wall.  A single blob node is wrong twice
        # over: symmetric costs make a crossing cost two slots, and any
        # shared node lets same-side tiles far apart along the wall borrow
        # a one-slot shortcut through it.
        band_feet = {}                       # line id -> foot pair
        for e in self.forced:
            if e.get('kf'):
                feet = set(map(tuple, e["pauli"]))
                xs = frozenset(x for x, _ in feet)
                ys = frozenset(y for _, y in feet)
                if len(xs) == 2:             # vertical wall: lines by y
                    for yv in sorted(ys):
                        band_feet.setdefault(('bl', 'v', xs, yv), set()).update(
                            (x, yv) for x in xs)
                else:                        # horizontal wall: lines by x
                    for xv in sorted(xs):
                        band_feet.setdefault(('bl', 'h', ys, xv), set()).update(
                            (xv, y) for y in ys)
        by_foot = {}
        for s, p in nodes.items():
            for q in p["pauli"]:
                by_foot.setdefault(q, []).append(s)
        for bid, bf in band_feet.items():
            for q in bf:
                by_foot.setdefault(q, []).append(bid)

        def _feet(nid):
            return band_feet[nid] if nid in band_feet else set(nodes[nid]["pauli"])

        def _line_neighbours(nid):
            # consecutive foot-lines of the SAME wall: one slot apart
            _tag, ax, span, v = nid
            for dv in (-2, 2):
                nb = ('bl', ax, span, v + dv)
                if nb in band_feet:
                    yield nb

        # 0-1 BFS: a STRAIGHT neighbour step advances the count by one slot;
        # a DIAGONAL corner step keeps it ((cx+cy)/2 unchanged — corner slots
        # share the corner foot at the SAME parity, the corner-w3 lesson);
        # entering a wall band costs one slot (the single-slot rule)
        from collections import deque
        def _w(s, t):
            if t in band_feet:
                return 1                  # entering the line: the single slot
            if s in band_feet:
                return 0                  # leaving is free - one slot in total
            return 0 if (abs(s[0] - t[0]) == 2 and abs(s[1] - t[1]) == 2) else 1
        depth = {}
        dq = deque()
        for s in sorted(s for s in self.seed_syns if s in nodes):
            depth[s] = 0
            dq.append(s)
        while dq:
            s = dq.popleft()
            nbrs = [t for q in _feet(s) for t in by_foot.get(q, ()) if t != s]
            if s in band_feet:
                nbrs.extend(_line_neighbours(s))
            for t in nbrs:
                w = 1 if (s in band_feet and t in band_feet) else _w(s, t)
                nd = depth[s] + w
                if t not in depth or nd < depth[t]:
                    depth[t] = nd
                    (dq.appendleft if w == 0 else dq.append)(t)
        order = sorted((s for s in depth if s not in band_feet),
                       key=lambda s: (depth[s], s))
        order += sorted(set(nodes) - set(depth))        # anchorless: fallback

        walked = set(depth)
        for s in order:
            if s in self.seed_syns:
                continue                            # anchors already placed
            if s in self.forbidden:
                continue                            # neighbour-owned site
            p = nodes[s]
            if s in walked:
                # letter-driven placement (slice-3 bus-boundary rule): the
                # slot's letter is the REGION's letter (bus letter, conjugate
                # inside a recoloured region — the M column flips letter and
                # colour together, so the colour shortcut tracks it), XOR'd
                # by the kf-crossing parity (a kf wall flips the letter only,
                # invisible to colour).  The walk count sequences the scan
                # and decides only the letter-undecided slots (recolour-
                # boundary straddlers, mixed tiles).
                if p["type"] in "XZ":
                    letter = self._seg_letter(p)
                    if letter is None:
                        if depth[s] % 2 == 1:
                            continue            # letter-undecided: count rules
                    else:
                        if p["type"] != letter:
                            continue
                        if any(placed_pauli.get(q, P) != P
                               for q, P in p["pauli"].items()):
                            continue            # region-junction guard: a
                                                # shared foot with a DIFFERENT
                                                # letter anticommutes; corner
                                                # slots legitimately share a
                                                # same-letter foot
                else:
                    if depth[s] % 2 == 1:
                        continue
                    if not placed_feet.isdisjoint(p["pauli"]):
                        continue                # mixed tile foot veto
            else:
                # anchorless chain: wall-free by construction — the
                # checkerboard-letter shortcut is exact there
                if p["type"] in "XZ":
                    letter = self._seg_letter(p)
                    if letter is None or p["type"] != letter:
                        continue
                else:
                    if not placed_feet.isdisjoint(p["pauli"]):
                        continue
            selected.append(p)
            placed_feet.update(p["pauli"])
            placed_pauli.update(p["pauli"])

        # acceptance metric, computed once (same contract as build())
        B = _IntBasis()
        svecs = list(self.fvecs)
        for v in self.fvecs:
            B.add(v)
        for p in selected:
            v = isv(p["pauli"])
            svecs.append(v)
            B.add(v)
        w1 = []
        for q in self.data:
            for P in "XZ":
                v = isv({q: P})
                if B.reduce(v) != 0 and all(_icommute(v, s, n) for s in svecs):
                    w1.append(q)
        N = len(self.target)
        deficit = (len(self.data) - B.rank) - (N - 1)
        joint = 0
        for v in self.repv:
            joint ^= v
        metric = (max(deficit, 0), len(w1), 0 if B.contains(joint) else 1)
        return selected, [], metric, w1


def _direct_metric(bl, selected):
    """Acceptance metric over the CURRENT forced+selected sets — vectors
    only, no re-scan (the local corner-cut semantics modifies a forced
    plaquette in place and re-checks acceptance without rebuilding)."""
    B = _IntBasis()
    svecs = []
    for p in bl.forced:
        v = bl.isv(p["pauli"])
        svecs.append(v)
        B.add(v)
    for p in selected:
        v = bl.isv(p["pauli"])
        svecs.append(v)
        B.add(v)
    w1 = []
    for q in bl.data:
        for P in "XZ":
            v = bl.isv({q: P})
            if B.reduce(v) != 0 and all(_icommute(v, s, bl.n) for s in svecs):
                w1.append(q)
    N = len(bl.target)
    deficit = (len(bl.data) - B.rank) - (N - 1)
    joint = 0
    for v in bl.repv:
        joint ^= v
    return (max(deficit, 0), len(w1), 0 if B.contains(joint) else 1), w1


def _convex_corner_cuts_local(bl, selected, w1):
    """Fig 34 label 2, the user's EXACT scope (2026-07-31): a convex bus
    corner is cut iff a NEIGHBOURING boundary stabilizer — one sharing a
    data foot with the corner plaquette, i.e. at ONE lattice spacing —
    exists; two spacings (no shared foot) means no cut.  A weight-1
    leftover on a bus corner also cuts.  (The retired Chebyshev-2 test
    was wider: in dense geometries it counted unrelated pieces at
    diagonal distance as neighbours and over-cut.)"""
    corner_q = _corner_qubits(set(bl.data)) - bl.patchq
    cuts = {q for q in w1 if q in corner_q}
    for q in corner_q:
        w4 = next((p for p in bl.forced if q in p["pauli"]), None)
        if w4 is None:
            continue
        w4_feet = set(w4["pauli"])
        for p in selected:
            feet = set(p["pauli"])
            if q in feet:
                continue
            if w4_feet & feet:
                cuts.add(q)      # one lattice spacing: shares a foot
                break
    return cuts


def _construct_phase(placed, target, orient, dset, retype, phase, forbidden=frozenset(),
                     native_lock=False, bus=None, conj_names=None, extra_forced=(),
                     retire_lobes=frozenset()):
    """One deterministic construction attempt (direct scan + LOCAL corner
    cuts).  Returns ``("ok", checks, logicals, x_obs, applied_cuts)`` or
    ``("fail", reason)``."""
    bl = _Builder(placed, target, orient, dset, retype, phase, forbidden,
                  native_lock=native_lock, bus=bus, conj_names=conj_names,
                  extra_forced=extra_forced, retire_lobes=retire_lobes)
    if bl.fail is not None:
        return ("fail", bl.fail)

    applied_cuts = set()
    # 10.4 direct construction, slice-3 letter rule: a slot's letter is
    # its REGION's letter (bus letter, conjugate inside a recoloured
    # region), so the placement is a lookup, not a propagated count; the
    # walk count sequences the scan and decides only letter-undecided
    # slots.  Wall-carrying builds count each kf band foot-line as ONE
    # slot (enter 1 / leave 0); retired-lobe reuse builds go through the
    # same rules.  (The legacy greedy engine and its rule-5 re-anchoring
    # are deleted: census showed zero reachable customers.)
    selected, picks, metric, w1 = bl.direct_build()
    if metric != (0, 0, 0):
        # rule 6, LOCAL semantics (user ruling 2026-07-31): a convex
        # corner that fails the neighbour-spacing test is cut IN PLACE
        # - the corner plaquette drops that foot (w4 -> w3), nothing
        # else moves, no region rebuild.  The old rebuild ladder
        # re-clipped the outline after every cut, manufactured phantom
        # corners at the notch and cascaded (the 'corner-cut
        # avalanche' was an artifact of that loop, not geometry).
        cuts = _convex_corner_cuts_local(bl, selected, w1)
        for q in sorted(cuts):
            if any(q in p["pauli"] for p in selected):
                continue          # a laid boundary piece owns this foot
            w4 = next((p for p in bl.forced
                       if q in p["pauli"] and len(p["pauli"]) >= 4), None)
            if w4 is None:
                continue
            del w4["pauli"][q]    # w4 -> w3, in place
            applied_cuts.add(q)
        if applied_cuts:
            bl.data = [q for q in bl.data if q not in applied_cuts]
            metric, w1 = _direct_metric(bl, selected)
        if metric != (0, 0, 0) and applied_cuts:
            # variant 2 - cut-then-RELAY: the region is FIRST reduced by
            # the round-one cut demands, then the boundary is laid ONCE
            # on the final region (fresh scan; no further cuts, so no
            # iteration and no cascade).  Some wall-adjacent layouts
            # only form the JOINT product in the relaid pattern
            # (measured: the role-switch south wall), while the flanked
            # corner family needs the kept pattern (variant 1).  Both
            # variants are bounded rule evaluations.
            bl2 = _Builder(placed, target, orient,
                           set(bl.data), {q for q in retype
                                          if q in set(bl.data)},
                           phase, forbidden, native_lock=native_lock,
                           bus=bus, conj_names=conj_names,
                           extra_forced=extra_forced,
                           retire_lobes=retire_lobes)
            if bl2.fail is None:
                sel2, picks2, met2, w12 = bl2.direct_build()
                if met2 == (0, 0, 0):
                    bl, selected, metric, w1 = bl2, sel2, met2, w12
        if metric != (0, 0, 0):
            import os as _os
            if _os.environ.get('LIGHTSTIM_DEBUG_JOINT'):
                import sys as _sys
                _B = _IntBasis()
                for _p2 in bl.forced:
                    _B.add(bl.isv(_p2["pauli"]))
                for _p2 in selected:
                    _B.add(bl.isv(_p2["pauli"]))
                _joint = 0
                for _v in bl.repv:
                    _joint ^= _v
                _res = _B.reduce(_joint)
                _n2 = bl.n
                _terms = []
                for _k, _q in enumerate(bl.data):
                    _x = (_res >> _k) & 1
                    _z = (_res >> (_n2 + _k)) & 1
                    if _x or _z:
                        _terms.append(
                            f"{'Y' if _x and _z else ('X' if _x else 'Z')}"
                            f"@{_q}")
                print(f"[joint] phase={phase} residual({len(_terms)}): "
                      f"{' '.join(_terms[:14])}", file=_sys.stderr)
            return ("fail", f"direct-scan acceptance shortfall {metric} "
                            f"(rank deficit, weight-1 count, joint "
                            f"missing) after {len(applied_cuts)} local "
                            f"corner cuts (phase {phase})")


    checks = []
    for p in bl.forced + selected:
        c = dict(p)
        c["corners"] = sorted(c["pauli"])
        checks.append(c)
    logicals = [(nm, bl.reps[nm][0], bl.reps[nm][1]) for nm, _ in target]
    # tracked observable: a BUS-basis representative — the memory experiment runs in the
    # bus basis, so the tracked logical must be of that type.  The bus is the PLANNED
    # ``bus`` when given, else the majority target basis (tie -> X).
    if bus is None:
        n_x = sum(1 for _, P, _ in logicals if P == "X")
        bus = "X" if n_x >= len(logicals) - n_x else "Z"
    x_obs = next((s for nm, P, s in logicals if P == bus), logicals[0][2])
    return ("ok", checks, logicals, x_obs, frozenset(applied_cuts))


def rule_based_joint_checks(placed, target, orient, data, retype, d, max_cut=4,
                            forbidden=frozenset(), native_lock=False, bus=None,
                            conj_names=None, extra_forced_fn=None,
                            retire_lobes=frozenset()):
    """Deterministic construction on the routed region (phases 0/1; corner
    cuts are applied LOCALLY inside each attempt per the user's 2026-07-31
    ruling — no rebuild ladder).
    Returns ``dict(checks, data, cut, phase, logicals, x_observable, reason="")`` on
    success, else ``dict(checks=None, reason=<why per phase>)``.  Pure function of the
    geometry — no randomness.

    ``native_lock`` (rule 0, default OFF in the abutting pipeline — in that
    geometry the greedy's alternation is sometimes load-bearing for mixed-joint
    distance laws; the seam-column pipeline passes True): every target patch whose interior is
    compatible with the global checkerboard (not retyped, phase-matched) keeps its
    TEXTBOOK standalone construction verbatim — its native boundary semicircles are
    forced picks and non-native tiles inside the patch are expelled, so a boundary-
    alternation tie can never relocate an observable's guard.  The facing semicircle
    is the sole exception: when its forced-bulk extension into seam/corridor qubits
    exists, the extension replaces it (joint-measurement rank demands this).  If the
    lock is unsatisfiable in every phase/strategy, the ladder is retried without it
    (noted in ``reason``).
    """
    reasons = []

    def attempt(phase, lock=True):
        rt = {q for q in retype if q in set(data)}
        ef = extra_forced_fn(phase) if extra_forced_fn is not None else ()
        res = _construct_phase(placed, target, orient, set(data), rt, phase,
                               forbidden, native_lock=lock, bus=bus,
                               conj_names=conj_names, extra_forced=ef,
                               retire_lobes=retire_lobes)
        if res[0] == "ok":
            lcut = set(res[4]) if len(res) > 4 else set()
            return dict(checks=res[1],
                        data=sorted(set(data) - lcut),
                        cut=tuple(sorted(lcut)),
                        phase=phase, logicals=res[2], x_observable=res[3],
                        reason="")
        reasons.append(f"phase {phase}: {res[1]}")
        return None

    # phases 0/1 with the native lock HARD (an unsatisfiable lock is a
    # failed build); corner cuts happen LOCALLY inside the attempt, so no
    # cut ladder exists any more.  ``max_cut`` is kept in the signature
    # for API compatibility only.
    lock = bool(native_lock)
    for phase in (0, 1):
        out = attempt(phase, lock=lock)
        if out is not None:
            return out
    return dict(checks=None, data=sorted(data), cut=(), phase=None,
                logicals=None, x_observable=None, reason="; ".join(reasons))


NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _seam_qubits(occupied, d):
    """The seam data qubits joining every pair of edge-adjacent occupied coarse cells
    (seam-column grid, pitch ``2d+2``): a vertical column of ``d`` qubits between
    horizontal neighbours, a horizontal row between vertical neighbours."""
    P = 2 * d + 2
    out = set()
    for (a, b) in occupied:
        if (a + 1, b) in occupied:
            out |= {(P * (a + 1) - 1, P * b + 1 + 2 * j) for j in range(d)}
        if (a, b + 1) in occupied:
            out |= {(P * a + 1 + 2 * i, P * (b + 1) - 1) for i in range(d)}
    return out


def _specs_to_cells(patches, target=None, seam=False):
    """Map the user-facing :class:`PatchSpec` list onto the internal coarse-cell model.

    Returns ``(patch_at, orient, d)`` where ``patch_at[name] = (a, b)`` is the coarse cell whose
    ``d×d`` footprint equals the patch's placed data qubits, ``orient[name]`` is the declared
    orientation, and ``d`` is the (shared) code distance.  This is the single adapter every public
    entry point runs first, so the whole subset API speaks ``PatchSpec`` while the routing internals
    keep working in coarse cells.

    Raises ``TypeError`` / ``ValueError`` with a concrete reason when the specs can't be placed on the
    coarse grid — a non-``PatchSpec`` element (e.g. the old ``{name:(a,b)}`` cell dict), mixed
    distances, duplicate names, an origin off the pitch-``2d`` grid, or a bad orientation — or when
    ``target`` names an unknown patch, repeats one, or uses a non-``X``/``Z`` Pauli.
    """
    patches = list(patches)
    if not patches:
        raise ValueError("need at least one PatchSpec")
    if not all(isinstance(s, PatchSpec) for s in patches):
        raise TypeError("patches must be a list of PatchSpec(name, origin, distance, "
                        "orientation); the old {name: (a, b)} cell-dict form is no "
                        "longer accepted — build a PatchSpec per patch (origin_of(a, b, d) places "
                        "one on coarse cell (a, b)).")
    d = patches[0].distance
    if any(s.distance != d for s in patches):
        raise ValueError(f"all patches must share one distance, got {sorted({s.distance for s in patches})}")
    names = [s.name for s in patches]
    if len(set(names)) != len(names):
        raise ValueError(f"patch names must be unique, got {names}")
    patch_at, orient = {}, {}
    for s in patches:
        if s.orientation not in ("X_horizontal", "X_vertical"):
            raise ValueError(f"patch {s.name!r}: orientation must be 'X_horizontal'|'X_vertical', "
                             f"got {s.orientation!r}")
        try:
            patch_at[s.name] = cell_index(s.origin, d, seam)
        except ValueError:
            raise ValueError(
                f"patch {s.name!r}: origin {s.origin} is not on the coarse routing grid (pitch "
                f"{_pitch(d, seam)} for d={d}, seam={seam}).  Subset routing places patches on cells "
                f"whose bus-facing corner is origin_of(a, b, d, seam={seam}).")
        orient[s.name] = s.orientation
    if target is not None:
        tnames = [nm for nm, _ in target]
        unknown = [nm for nm in tnames if nm not in patch_at]
        if unknown:
            raise ValueError(f"target names {unknown} are not among the patches {names}")
        if len(set(tnames)) != len(tnames):
            raise ValueError(f"target lists a patch more than once: {tnames}")
        if any(P not in ("X", "Z") for _, P in target):
            raise ValueError(f"target paulis must be 'X' or 'Z', got {[P for _, P in target]}")
    return patch_at, orient, d


def _cheb(ab, cd):
    """King-move (Chebyshev) distance between two coarse cells."""
    return max(abs(ab[0] - cd[0]), abs(ab[1] - cd[1]))


def _legal_attach_groups(patch_at, target, orient, corridor):
    """Per-target group of corridor cells adjacent through a PARALLEL-LAW
    legal seam (the same rule the stitching policy applies).  Returns
    (groups, missing) — ``missing`` lists targets with no legal cell."""
    groups, missing = [], []
    for nm, P in target:
        pa = tuple(patch_at[nm])
        cells = []
        for da, db in NEIGH:
            c = (pa[0] + da, pa[1] + db)
            if c not in corridor:
                continue
            seam_ew = (da != 0)
            want = 'X_horizontal' if (P == 'Z') == seam_ew else 'X_vertical'
            if orient[nm] == want:
                cells.append(c)
        groups.append(cells)
        if not cells:
            missing.append(nm)
    return groups, missing


def path_to_corridor(tree_cells, placed, target, d, seam=False, patch_at=None, bus=None,
                     conj_names=None, flip_cells=None, skip_seam_pairs=None):
    """Convert a set of corridor **cells** into the joint code's ``(data, retype)``.

    ``data`` = the target patch cells ∪ the corridor cells' footprints.

    **Bus-basis rule**: the corridor's native basis is ``bus`` when given, else the
    *majority* target basis (a tie keeps the historical X-bus).  Only the
    non-bus-basis patches attach through mixed (XZ) domain walls — so a pure-X or
    pure-Z joint uses NO mixed stabilizer at all (the pure-Z corridor is the exact CSS
    dual of the pure-X one).  Passing an explicit ``bus`` forces the native basis to a
    PLANNED choice (used to keep a shared patch's conjugation status consistent across
    a PPM sequence); the physics is valid for either basis (conjugating the non-bus
    patches reduces the joint to a pure-bus merge), the minority basis merely raises
    more mixed 'M' walls.

    ``conj_names``: the target patches whose LIVE construction is the colour-conjugate
    registration.  A patch's region is retyped iff it is conjugate-registered — its
    physical checks really are colour-flipped, and the wall the retype boundary raises
    is what reconciles them with the corridor.  ``None`` (the historical default)
    infers conjugation from the measured Pauli: minority-basis patches are the
    conjugate-registered ones (first-use minority allocation), which reproduces the
    original letter rule byte for byte.  Passing the set explicitly decouples the two:
    a conjugate patch measuring the BUS basis (its recoloured seam still needs the
    wall) routes without any rotation.

    ``retype`` encodes this for :func:`._bent_plaquettes`: for EITHER bus letter,
    exactly the conjugate-registered cells (plus ``flip_cells``) are flipped.  The
    complement is an exact STATIC dual — flipping *all* of ``data`` equals flipping
    none at the opposite checkerboard ``phase`` — but patches carry measurement
    history, so the dual is not free to choose; the direct convention is the
    only valid one.  Both checkerboard phases are tried downstream.
    """
    tnames = [nm for nm, _ in target]
    data = set()
    for nm in tnames:
        data |= placed[nm]
    for c in tree_cells:
        data |= cell(*c, d, seam)
    if seam:                    # seam-column design: join every adjacent occupied pair
        if patch_at is None:
            raise ValueError("path_to_corridor(seam=True) needs patch_at (coarse cells)")
        occupied = set(map(tuple, tree_cells)) | {tuple(patch_at[nm]) for nm in tnames}
        data |= _seam_qubits(occupied, d)
        # a walled junction gets NO seam qubits (the stretched-stabilizer band
        # lives in the gap; the wall records bridge the two segments)
        for pair in (skip_seam_pairs or ()):
            data -= _seam_qubits({tuple(pair[0]), tuple(pair[1])}, d)
    xnames = [nm for nm, P in target if P == "X"]
    znames = [nm for nm, P in target if P == "Z"]
    if bus is None:
        bus = "X" if len(xnames) >= len(znames) else "Z"
    if conj_names is None:      # historical inference: conjugate ⟺ minority basis
        conj_names = frozenset(znames if bus == "X" else xnames)
    flip = set().union(*[placed[nm] for nm in tnames if nm in conj_names]) \
        if conj_names else set()
    if flip_cells:
        # segment recolouring: these corridor cells join the flipped-gauge side,
        # WITH the seam qubits interior to the flipped region (flipped cell to
        # flipped cell / flipped cell to conjugate patch) — a plain stitch inside
        # one colour region flips as a whole
        fcs = {tuple(c) for c in flip_cells}
        for c in fcs:
            flip |= cell(*c, d, seam)
        if seam:
            fents = fcs | {tuple(patch_at[nm]) for nm in tnames
                           if nm in conj_names}
            flip |= _seam_qubits(fents, d) & data
    # DIRECT convention for both bus letters (stage-A hardening): retype is
    # exactly the flipped side - conj-registered footprints plus flip_cells.
    # The old X-bus complement shortcut realized the colour-conjugate DUAL of
    # the intended layout; statically equivalent, but a patch with measurement
    # history must keep its own letters, so the dual is not free to choose.
    retype = {q for q in data if q in flip}
    return data, retype


# -----------------------------------------------------------------------------
# physics-layer reuse: a routed (data, retype) region -> a verified MultiPatchLayout
# -----------------------------------------------------------------------------

def _assemble_region(placed, target, orient, data, retype, d, seed=0, max_trials=5000, max_cut=4,
                     forbidden=frozenset(), native_lock=False, bus=None, conj_names=None,
                     extra_forced_fn=None, extra_connect=frozenset(),
                     retyped_bus=frozenset(), retire_lobes=frozenset()):
    """Hand a routed region to the **deterministic rule-based** physics layer.

    The stabilizers are constructed by :func:`.deterministic_checks.rule_based_joint_checks`
    (the documented Handbook §10.4 / Fig 33-34 rules: forced bulk, alternating-spacing
    boundary, concave/convex corner rules with rule-driven corner cuts) — no randomized
    search.  ``seed`` / ``max_trials`` are accepted for backward compatibility and ignored.
    Returns a :class:`.MultiPatchLayout` (whose ``data`` may be smaller than the input when
    the convex-corner rule cut bus-corner qubits), or ``None`` if the rules cannot host this
    geometry.  The oracle decision itself is ``MultiPatchLayout.verify()``.
    """
    data = sorted(data)
    # a walled junction leaves a data gap; its would-be seam qubits are handed
    # in as extra_connect so the two segments count as one region
    if not _connected(set(data) | set(extra_connect)):
        return None
    rb = rule_based_joint_checks(placed, target, orient, data, set(retype), d, max_cut=max_cut,
                                 forbidden=forbidden, native_lock=native_lock, bus=bus,
                                 conj_names=conj_names, extra_forced_fn=extra_forced_fn,
                                 retire_lobes=retire_lobes)
    if rb["checks"] is None:
        return None
    log_pairs = [(P, sup) for _, P, sup in rb["logicals"]]
    # orientation-domain map for the hook-benign scheduler: patch cells carry
    # their declared orientation; bus/corridor cells follow the majority
    tally = {}
    for nm, _ in target:
        o = orient[nm]
        tally[o] = tally.get(o, 0) + 1
    bus_orient = max(sorted(tally), key=lambda o: tally[o])
    domains = {}
    for nm, _ in target:
        for q in placed[nm]:
            domains[q] = orient[nm]
    for q in rb["data"]:
        domains.setdefault(q, bus_orient)
    return MultiPatchLayout(distance=d, data=rb["data"], checks=rb["checks"],
                            logicals=rb["logicals"], x_observable=rb["x_observable"],
                            readout_chain=_readout_chain(rb["data"], rb["checks"], log_pairs),
                            target=list(target), domains=domains, bus=bus,
                            retyped=frozenset(retyped_bus))


# -----------------------------------------------------------------------------
# the router
# -----------------------------------------------------------------------------

@dataclass
class SubsetRoute:
    """Result of :func:`route_and_build`.  ``status == "ok"`` iff ``layout`` is a verified joint.

    On a **failure** an obstacle-free corridor may still exist but be un-hostable by the physics
    layer — ``attempted`` / ``attempted_arms`` carry the shortest such corridor (empty only when
    ``status == "no_path"``) so a caller can *draw the route that was found* and label it correctly
    (a rejected corridor is **not** "no route").
    """
    status: str    # "ok" | "no_path" (unreachable OR only disconnected corridors) | "no_verified_route"
    message: str = ""
    layout: object = None                             # MultiPatchLayout when status == "ok"
    root: str = None
    arms: dict = field(default_factory=dict)          # z-name -> corridor-cell path (VERIFIED bus)
    tree: set = field(default_factory=set)            # corridor cells of the VERIFIED bus
    attempted: set = field(default_factory=set)       # corridor cells of the shortest candidate (any status)
    attempted_arms: dict = field(default_factory=dict)  # z-name -> shortest candidate path
    data: set = field(default_factory=set)            # data qubits of the routed region
    placed: dict = field(default_factory=dict)        # name -> patch cells (ALL patches)
    target: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)     # non-target patch names
    obstacle_fp: set = field(default_factory=set)     # non-target patch data qubits
    corridor: set = field(default_factory=set)        # all corridor-eligible cells
    tried: int = 0
    how: str = "standard"                             # "standard" | "corner-cut" (route_and_build)
    cut: tuple = ()                                   # convex-corner qubits removed (corner-cut only)
    n_walls: int = 0                                  # stretched (kf) seam walls in the layout - band
                                                      # apparatus is real cost, chargeable like cells
    certificate: dict = None                          # MultiPatchLayout.verify() items for the
                                                      # accepted layout (None on failures/probes)

    @property
    def ok(self):
        return self.status == "ok"


def _obstacle_ancillas(patches, tnames):
    """The ancilla sites actually USED by the non-target (idle obstacle) patches.

    Each idle patch keeps its standalone construction; its selected syndrome sites are
    physical qubits the routed joint code may not re-use.  These are handed to the rule
    constructor as ``forbidden`` boundary positions, so a corridor sharing an ancilla
    line with an idle neighbour interleaves with it instead of colliding — edge-adjacent
    placement (keepout=0) is then physically sound whenever the parity allows.
    """
    out = set()
    for s in patches:
        if s.name in tnames:
            continue
        for c in place_patch(s)["checks"]:
            out.add(tuple(int(v) for v in c["syn"]))
    return frozenset(out)


def _table_wall_dispatch(patch_at, target, orient, tree, conj_names, bus,
                         ns_pairs=frozenset()):
    """Post-routing seam-table dispatch — the user's TWO-BIT table (letter,
    colour), 2026-07-30.  TWO wall rows raise a stretched kf wall; TYPE
    never triggers one (same-letter same-colour type-split seams are plain
    row #1).  Step 2 never re-registers a patch: ``conj_eff`` is passed
    through, not grown.

      #4 native (letter same, colour diff): a conj-registered target
         measuring the BUS letter — the snake family.  On a vertical seam
         the corridor painting flips (g = 1: #4->#6 and #1->#7 wholesale).
      #6 native (letter diff, colour same): a STANDARD-colour target
         measuring the OTHER letter (user ruling 2026-07-30: the mixed
         kf wall is the row's construct — the spatial-Hadamard interface
         turns the letter at the seam).  The wall hardware is the SAME
         local-rule family; the corridor never flips for it.  Fires only
         under an EXPLICIT ``conj_names`` — with ``conj_names=None`` the
         historical minority-conjugate inference keeps the legacy plain
         path (C4 mixed cells).

    Returns ``(arm_walls, conj_eff, flip_corridor)`` or ``None`` when no
    wall is due; ``(None, conj_eff, False)`` when a due wall cannot be
    hosted on this tree."""
    bus_eff = bus
    if bus_eff is None:
        n_x = sum(1 for _, P in target if P == "X")
        bus_eff = "X" if n_x >= len(target) - n_x else "Z"
    conj_eff = conj_names if conj_names is not None else \
        frozenset(nm for nm, P in target if P != bus_eff)
    # #4-native walls: colour-diff (conj) targets measuring the bus letter
    snake_nms = sorted(nm for nm, P in target
                       if P == bus_eff and nm in conj_eff)
    # #6-native walls: standard-colour targets measuring the other letter
    direct6_nms = sorted(nm for nm, P in target
                         if P != bus_eff and nm not in conj_eff) \
        if conj_names is not None else []
    wall_nms = snake_nms + direct6_nms
    if not wall_nms:
        return None
    # completeness: every NON-wall target must sit on row #1 (letter same,
    # colour same — plain stitch).  A leftover (T,T)/#7 seam in the same
    # joint has no verified construction alongside these walls (measured:
    # the layout passes the group oracle but the OBSERVABLE goes
    # non-deterministic) — decline to the legacy path instead.
    for nm, P in target:
        if nm not in wall_nms and not (P == bus_eff and nm not in conj_eff):
            return None
    tset = {tuple(c) for c in tree}
    # wall seams live on the LIVE-frame parallel-law-legal faces — the same
    # frame _auto_stitch judges plain seams in.  A wall reconciles the
    # CORRIDOR side of the seam (#4: the colour bit, #6: the bus letter),
    # never the target's own seam letter: a seam that is perpendicular in
    # the live frame is an ODD row — illegal, no wall due (measured: the
    # registered-frame faces admitted a S-seam #4 on a rotate_90'd conj
    # target and both construction phases refused it — 20-patch q17)
    groups_all, _miss = _legal_attach_groups(patch_at, target, orient, tset)
    legal_of = {nm: set(g) for (nm, _), g in zip(target, groups_all)}
    aw, snake_vertical = [], False

    def _hosts_direct6(nm, pc, wc):
        """Tree-dependent #6 hosting law (24/24 measured matrix, 2026-08-02:
        both orientations × all four sides × all exits, each combo built with
        a stubbed dispatch and checked verify + p0 + graphlike distance).
        The walled cell must be TERMINAL (degree 1 — the joint product only
        splices through an end wall; the interior middle-wall case measured
        (0,0,1) acceptance), and the bus may leave it straight ahead or with
        the scan-friendly turn only: the lexicographic NW-first scan lays the
        letter-turned continuation on ONE handedness.  With s = wc − pc and
        e = exit − wc, decline iff cross(s, e) == −1 for an X̄-vertical
        target and +1 for X̄-horizontal — the transpose is a REFLECTION, so
        the bad handedness flips with the family (ptype transpose symmetry
        makes everything else covariant, same argument as litinski ±x)."""
        bad = -1 if orient.get(nm) == 'X_vertical' else 1
        sx, sy = wc[0] - pc[0], wc[1] - pc[1]
        deg = 0
        for c in tset:
            ex_, ey_ = c[0] - wc[0], c[1] - wc[1]
            if abs(ex_) + abs(ey_) != 1:
                continue
            deg += 1
            if deg > 1:
                return False        # interior bus cell: splice unverified
            if sx * ey_ - sy * ex_ == bad:
                return False        # scan-hostile turn out of the walled cell
        return True

    for nm in wall_nms:
        pc0 = tuple(patch_at[nm])
        # a user-unstitched seam never hosts a wall (no_stitch wins over
        # every table row — measured: the fig5-six PPM4 wall on q4's
        # unstitched (2,1) seam admits, then the walled build fails and
        # the plain fallback strands q1)
        cells = sorted(c for c in legal_of.get(nm, ())
                       if frozenset((pc0, tuple(c))) not in ns_pairs)
        if nm in direct6_nms:
            # first HOSTABLE legal cell wins (a target with two legal faces
            # may host the wall on either; the hosting law filters per cell)
            cells = [c for c in cells if _hosts_direct6(nm, pc0, tuple(c))]
        if not cells:
            return None, conj_eff, False   # this path cannot host the wall
        wc = cells[0]
        if nm in snake_nms and wc[0] == pc0[0]:
            # snake wall on a HORIZONTAL (N/S) seam — the #4-native
            # horizontal band has no verified construction (measured on
            # the minimal std|conj pair: phase 0 acceptance (0,0,1)
            # joint missing, phase 1 native-lock refusal).  The verified
            # dispatch-snake family is the VERTICAL seam with the
            # corridor flip, mirroring the gateway snake.  Decline so
            # probe = builder.
            return None, conj_eff, False
        aw.append((nm, wc))
        if nm in snake_nms and wc[0] != tuple(patch_at[nm])[0]:
            snake_vertical = True
    if len(direct6_nms) > 1:
        # multiple native-#6 walls in one joint: the joint product must
        # splice through EVERY wall and the verified splice algebra covers
        # ONE wall (measured: two clean vertical walls, acceptance (0,0,1)
        # joint missing).  Decline; the planner reaches the same measurement
        # through a rotation (the conj target's #7 column is in production).
        return None, conj_eff, False
    if len({wc for _, wc in aw}) < len(aw):
        # two walls on the SAME corridor cell (both seams of one cell
        # walled) — no verified construction (the two-wall family puts
        # each wall on its own cell); measured: the build falls through
        # and dies on the plain path.  Decline so the probe agrees with
        # the builder and the planner takes another convention (the
        # all-swapped pair goes through the global-conjugate shortcut).
        return None, conj_eff, False
    if snake_vertical and direct6_nms:
        # the snake flip would repaint the corridor under the native-#6
        # walls, moving their seams to row #7 — no verified construction
        return None, conj_eff, False
    if snake_vertical:
        # g = 1 (flipped painting): a STANDARD-colour non-wall target's
        # plain seam becomes the recoloured column (#1 -> #7), verified
        # ONLY as the snake's horizontal-seam family — decline otherwise
        for nm, _P in target:
            if nm in wall_nms or nm in conj_eff:
                continue
            pc = tuple(patch_at[nm])
            adj = [c for c in tset
                   if abs(c[0] - pc[0]) + abs(c[1] - pc[1]) == 1]
            if not all(c[0] == pc[0] for c in adj):
                return None, conj_eff, False
    return aw, conj_eff, snake_vertical


def route_and_build(patches, target, pad=1, per_z=6, max_std=48, cut_budget=4, max_cut=4,
                    keepout=0, seed=0, max_trials=5000, route=None, seam=False, bus=None,
                    conj_names=None, flip_cells=None, no_stitch=None, wall_junction=None,
                    arm_walls=None, probe=False, geom_cache=None):
    """Fully-automatic route **and** build: no hand-written corridor needed.

    ``route`` (optional): an **explicit corridor** — a list of coarse cells ``[(a, b), …]``.
    When given, NO automatic routing happens: the joint code is built on exactly these
    cells by the deterministic rule constructor and gated by the full oracle.  The cells
    must not overlap any patch; the obstacle **keep-out margin is NOT enforced** for an
    explicit route (you are overriding the router), so check ``collision_report`` if
    obstacles sit next to your corridor.  Omit ``route`` (default) for auto-routing.

    ``conj_names`` (optional): the target patches whose LIVE construction is the
    colour-conjugate registration (see :func:`path_to_corridor`).  ``None`` keeps the
    historical inference (conjugate ⟺ minority basis).  Passing the true set lets a
    conjugate patch attach in the BUS basis — its recoloured seam wall is raised by
    the retype boundary, no rotation needed.

    ``flip_cells`` + ``wall_junction`` (optional, explicit route only): a SNAKE bus.
    ``flip_cells`` recolours those corridor cells into the conjugate gauge (they
    plain-stitch the conjugate targets); ``wall_junction = (cell_a, cell_b)`` names
    the adjacent corridor pair whose seam is REPLACED by a stretched-stabilizer
    colour wall (the K&F spatial-Hadamard interface: uniform pure-letter dominoes,
    flips the colours, keeps the letters).  No seam qubits are added there; the
    wall's kf records are injected as forced checks, so a same-letter joint between
    a colour-swapped and a standard patch closes with zero rotations and NO mixed
    checks anywhere.

    Propose-and-verify with retry: (1) candidate arms leave the X-anchor from **any** face (not just
    its X-faces), so clean below-/side-attach corridors are found; (2) arm-product unions are
    augmented with greedy **Steiner** candidates (:func:`_steiner_trees`) whose arms share a common
    trunk; (3) a candidate whose corridor is **disconnected** (its arms meet only through a target
    patch — two separate buses) is illegal and never tried; if *every* candidate is disconnected the
    result is an honest ``no_path``; (4) each surviving candidate is built by the **deterministic
    rule constructor** (:mod:`.deterministic_checks` — cut-free first, then rule-driven convex-corner
    cuts up to ``max_cut``) and gated by the full oracle.  Candidates are tried
    **fewest-corridor-cells first**, so the smallest *valid* bus wins — a shared straight trunk is
    preferred over fat multi-arm unions.  ``per_z`` / ``max_std`` / ``keepout`` shape the candidate
    pool; ``cut_budget`` / ``seed`` / ``max_trials`` are accepted for backward compatibility and
    ignored (there is no randomized search any more).  This is the routine the demo notebook calls
    instead of pasting cells.

    Returns a :class:`SubsetRoute` with ``status == "ok"`` and ``.layout`` / ``.tree`` (route cells) /
    ``.how`` (``"standard"`` | ``"corner-cut"``) / ``.cut`` set, or a failure ``SubsetRoute`` with
    status ``target_obstacle_conflict`` / ``no_path`` / ``no_verified_route``.
    """
    if route is None:
        raise BentLayoutError(
            "this LightStim variant has no auto-router: pass an explicit "
            "route (the coarse corridor cells, in order; [] for "
            "cell-adjacent targets)")
    patch_at, orient, d = _specs_to_cells(patches, target, seam=seam)
    tnames = [nm for nm, _ in target]
    # no_stitch: adjacencies that stay UNSTITCHED — the two boundaries sit next
    # to each other without merging (no seam qubits, no seam-orientation rule;
    # each side keeps its own free boundary).  Entries are (patch, cell) or
    # (patch, patch); normalize to cell pairs here.
    ns_pairs = frozenset(
        frozenset((tuple(patch_at[a]) if isinstance(a, str) else tuple(a),
                   tuple(patch_at[b]) if isinstance(b, str) else tuple(b)))
        for a, b in (no_stitch or ()))
    onames = [nm for nm in patch_at if nm not in tnames]

    def _auto_stitch(tree_cells, wall_pairs=frozenset()):
        """Parallel-law stitching policy: a target|cell (or target|target)
        adjacency is STITCHED iff the seam runs parallel to that target's
        measured logical; an illegal adjacency stays UNSTITCHED — the two
        boundaries simply sit next to each other, separated by the empty
        seam line (a corridor cell may pass BY one target to reach
        another).  User ``no_stitch`` pairs always stay unstitched; a
        walled junction counts as an attachment.  Returns
        ``(skip_pairs, violation)`` — ``violation`` is set when a target
        ends up with NO attachment at all."""
        skip = set(ns_pairs)
        attached = {nm: 0 for nm in tnames}
        # two-bit TABLE legality of a PLAIN (non-wall) stitch: row #1
        # (letter == bus, standard colours) or row #7 (letter != bus,
        # conj-registered — the retype-boundary M-column family).  The
        # wall rows #4/#6 cannot plain-stitch: without a dispatched wall
        # the seam stays open.  Mirrors the rule constructor exactly, so
        # the step-1 probe never green-lights a seam step 2 cannot build.
        n_x_s = sum(1 for _, P in target if P == "X")
        bus_s = bus if bus is not None else (
            "X" if n_x_s >= len(target) - n_x_s else "Z")
        conj_s = conj_names if conj_names is not None else \
            frozenset(nm for nm, P in target if P != bus_s)

        def _row_plain_ok(nm, P):
            return (P != bus_s) == (nm in conj_s)

        for nm, P in target:
            pa = tuple(patch_at[nm])
            legal_cells = []
            for tc in map(tuple, tree_cells):
                da, db = tc[0] - pa[0], tc[1] - pa[1]
                if abs(da) + abs(db) != 1:
                    continue
                pair = frozenset((pa, tc))
                if pair in wall_pairs:
                    attached[nm] += 1          # the wall IS the attachment
                    continue
                if pair in skip:
                    continue                   # user unstitch wins
                seam_ew = (da != 0)
                want = ('X_horizontal' if (P == 'Z') == seam_ew
                        else 'X_vertical')
                if orient[nm] != want or not _row_plain_ok(nm, P):
                    skip.add(pair)
                else:
                    legal_cells.append(tc)
            # SINGLE ATTACHMENT (user rule 2026-07-31): when the bus wraps a
            # target on several sides, the patch attaches through EXACTLY
            # ONE seam — any legal side works (measured: the flanked-patch
            # double stitch is what wrecked the corner geometry), the choice
            # is free; deterministic first-by-order here.  Extra legal
            # adjacencies stay unstitched (the corridor passes by).
            if attached[nm] == 0 and legal_cells:
                keep = sorted(legal_cells)[0]
                attached[nm] += 1
                for tc in legal_cells:
                    if tc != keep:
                        skip.add(frozenset((pa, tc)))
            else:
                for tc in legal_cells:
                    skip.add(frozenset((pa, tc)))
        tl = list(target)
        for i1, (n1, P1) in enumerate(tl):
            c1 = tuple(patch_at[n1])
            for n2, P2 in tl[i1 + 1:]:
                c2 = tuple(patch_at[n2])
                da, db = c2[0] - c1[0], c2[1] - c1[1]
                if abs(da) + abs(db) != 1:
                    continue
                pair = frozenset((c1, c2))
                if pair in wall_pairs:
                    attached[n1] += 1
                    attached[n2] += 1
                    continue
                if pair in skip:
                    continue
                seam_ew = (da != 0)
                legal = all(orient[nm] == ('X_horizontal'
                                           if (P == 'Z') == seam_ew
                                           else 'X_vertical')
                            and _row_plain_ok(nm, P)
                            for nm, P in ((n1, P1), (n2, P2)))
                if legal:
                    attached[n1] += 1
                    attached[n2] += 1
                else:
                    skip.add(pair)
        for nm, P in target:
            if attached[nm] == 0:
                return skip, (
                    f"{nm}: no parallel-law-legal attach seam to this "
                    f"corridor — measuring {P} needs a seam PARALLEL to the "
                    f"measured logical ({'vertical (E/W)' if (P == 'X') == (orient[nm] == 'X_vertical') else 'horizontal (N/S)'} "
                    f"for orientation {orient[nm]!r}); every adjacency is "
                    f"perpendicular, user-unstitched, or absent")
        # CONNECTIVITY: per-target degree >= 1 is not enough — two adjacent
        # targets can legally stitch to EACH OTHER while their corridor
        # seams are all skipped, forming an island the bus never reaches
        # (fig5 q6|q4 under all-vertical declarations).  The stitched merge
        # graph (corridor cells + target cells; edges = corridor internal
        # adjacency + every stitched/walled pair) must put ALL targets in
        # the component containing the corridor.
        cells_t = {tuple(patch_at[nm]): nm for nm in tnames}
        nodes = set(map(tuple, tree_cells)) | set(cells_t)
        adj = {n: set() for n in nodes}
        tset_c = set(map(tuple, tree_cells))
        for a in tset_c:
            for b in tset_c:
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                    adj[a].add(b)
                    adj[b].add(a)
        def _link(a, b):
            pair = frozenset((a, b))
            if pair in wall_pairs or pair not in skip:
                adj[a].add(b)
                adj[b].add(a)
        for nm in tnames:
            pa = tuple(patch_at[nm])
            for tc in tset_c:
                if abs(tc[0] - pa[0]) + abs(tc[1] - pa[1]) == 1:
                    _link(pa, tc)
            for nm2 in tnames:
                pb = tuple(patch_at[nm2])
                if nm2 != nm and abs(pb[0] - pa[0]) + abs(pb[1] - pa[1]) == 1:
                    _link(pa, pb)
        seed = next(iter(tset_c), tuple(patch_at[tnames[0]]))
        seen_c, frontier = {seed}, [seed]
        while frontier:
            n = frontier.pop()
            for m in adj[n]:
                if m not in seen_c:
                    seen_c.add(m)
                    frontier.append(m)
        stranded = [nm for nm in tnames if tuple(patch_at[nm]) not in seen_c]
        if stranded:
            return skip, (
                f"{stranded}: stitched to a neighbouring target but the "
                f"island never reaches the corridor — every corridor seam "
                f"of the island is perpendicular, user-unstitched, or "
                f"absent (the joint product cannot flow onto the bus)")
        return skip, None

    placed_all = {nm: cell(*ab, d, seam) for nm, ab in patch_at.items()}
    obstacle_fp0 = set().union(*[placed_all[nm] for nm in onames]) if onames else set()
    base0 = dict(placed=placed_all, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp0,
                 corridor=set())
    for tn in tnames:
        bad = [on for on in onames if _cheb(patch_at[tn], patch_at[on]) <= keepout]
        if bad:
            return SubsetRoute(status="target_obstacle_conflict", root=tnames[0],
                               message=(f"target {tn} is within keepout={keepout} of obstacle(s) "
                                        f"{bad}: their boundary ancillas would collide."), **base0)
    # route from a bus-basis anchor; ``bus`` forces a PLANNED basis, else the majority
    # target basis (tie keeps the historical X anchor)
    n_x = sum(1 for _, P in target if P == "X")
    n_z = sum(1 for _, P in target if P == "Z")
    bus_basis = bus if bus is not None else ("X" if n_x >= n_z else "Z")
    root0 = next((nm for nm, P in target if P == bus_basis), tnames[0])
    if wall_junction is not None and route is None:
        raise ValueError("wall_junction needs an explicit route")

    if route is not None:                  # explicit corridor: build on EXACTLY these cells
        tree = {tuple(c) for c in route}
        occupied_cells = set(patch_at.values())
        bad = sorted(tree & occupied_cells)
        if bad:
            raise ValueError(f"explicit route cells {bad} overlap patch cells; the corridor "
                             f"may only use empty coarse cells")
        # stitching decided by the parallel-law policy AFTER the wall
        # junctions are known (a walled seam counts as an attachment)
        # ---- corridor walls: mid-bus junctions (experimental) and ARM walls
        # (a stretched wall ON a target's attach seam — the production device
        # for rows #4/#6 of the seam table; the closed form is the VERIFIED
        # patch|patch spec, the corridor cell simply takes the far-patch role)
        # post-path seam-table dispatch (same rule as the auto branch): a
        # GIVEN route is still a routed path — when the caller assigned no
        # walls, classify the attach seams and assign the minority-type
        # targets' walls here
        if probe:
            # STEP-1 PROBE (stage B): geometric feasibility + table
            # classification only - route legality, parallel-law stitching
            # and the per-seam wall count, NO construction, NO oracle.
            wall_pairs2, aw_n = frozenset(), 0
            if seam and TABLE_WALL_DISPATCH and not arm_walls \
                    and wall_junction is None:
                dp = _table_wall_dispatch(patch_at, target, orient, tree,
                                          conj_names, bus,
                                          ns_pairs=ns_pairs)
                if dp is not None and dp[0] is not None:
                    aw_n = len(dp[0])
                    wall_pairs2 = frozenset(
                        frozenset((tuple(patch_at[nm]), tuple(wc)))
                        for nm, wc in dp[0])
            if arm_walls:
                aw_n += len(arm_walls)
                wall_pairs2 |= frozenset(
                    frozenset((tuple(patch_at[nm]), tuple(wc)))
                    for nm, wc in arm_walls)
            if seam:
                _, viol = _auto_stitch(tree, wall_pairs=wall_pairs2)
                if viol:
                    return SubsetRoute(status="no_verified_route", root=root0,
                                       tried=0, attempted=tree,
                                       message=f"probe: {viol}", **base0)
            return SubsetRoute(status="ok", message="probe", tree=tree,
                               attempted=tree, tried=0,
                               n_walls=aw_n, **base0)

        if seam and TABLE_WALL_DISPATCH and not arm_walls \
                and wall_junction is None:
            disp = _table_wall_dispatch(patch_at, target, orient, tree,
                                        conj_names, bus,
                                        ns_pairs=ns_pairs)
            if disp is not None and disp[0] is not None:
                # the dispatched wall IS this route's construction — its
                # outcome is FINAL (user ruling 2026-07-31: a failed step-2
                # construction is a hard error, never a fallback; the old
                # swallow-and-fall-to-plain masked the real error as a
                # misleading 'no parallel-law-legal seam')
                aw2, conj_eff2, vertical2 = disp
                return route_and_build(
                    patches, target, pad=pad, per_z=per_z,
                    max_std=max_std, cut_budget=cut_budget,
                    max_cut=max_cut, keepout=keepout, seed=seed,
                    max_trials=max_trials, route=sorted(tree),
                    seam=True, bus=bus, conj_names=conj_eff2,
                    flip_cells=(sorted(tree) if vertical2
                                else flip_cells),
                    no_stitch=no_stitch, arm_walls=aw2)

        wall_descr = []          # (c1, c2, kind, patch_name)
        if wall_junction is not None:
            c1, c2 = tuple(wall_junction[0]), tuple(wall_junction[1])
            if c1 not in tree or c2 not in tree:
                raise ValueError(f"wall_junction {(c1, c2)} must be route cells")
            wall_descr.append((c1, c2, 'mid', None))
        for nm, wcell in (arm_walls or ()):
            pc, wc2 = tuple(patch_at[nm]), tuple(wcell)
            if wc2 not in tree:
                raise ValueError(f"arm wall cell {wc2} is not a route cell")
            if conj_names is None:
                raise ValueError("arm_walls needs an explicit conj_names")
            wall_descr.append((pc, wc2, 'arm', nm))
        walls = []
        retire_lobes = set()
        extra_connect = set()
        for c1, c2, kind, wnm in wall_descr:
            da, db = c2[0] - c1[0], c2[1] - c1[1]
            if abs(da) + abs(db) != 1:
                raise ValueError(f"wall cells {(c1, c2)} are not adjacent")
            if da == 0:
                near_c = c1 if db > 0 else c2
                axis = 'horizontal'
            else:
                near_c = c1 if da > 0 else c2
                axis = 'vertical'
            ox, oy = origin_of(*near_c, d, seam=seam)
            a0, n0 = (ox, oy) if axis == 'horizontal' else (oy, ox)
            near_ln = n0 + 2 * (d - 1)
            lo, seam_ln, hi, far_ln = (near_ln + 1, near_ln + 2, near_ln + 3,
                                       near_ln + 4)
            P = ((lambda a, m: (a, m)) if axis == 'horizontal'
                 else (lambda a, m: (m, a)))
            bus_eff2 = None
            if kind == 'arm':
                bus_eff2 = bus
                if bus_eff2 is None:
                    n_x2 = sum(1 for _, PP in target if PP == "X")
                    bus_eff2 = "X" if n_x2 >= len(target) - n_x2 else "Z"
                if (dict(target)[wnm] == bus_eff2
                        and wnm not in (conj_names or frozenset())):
                    raise BentLayoutError(
                        f"arm wall {wnm}|{c2}: the target is same-type with "
                        f"the corridor (rule-table row 1 = plain merge, no "
                        f"wall is due); register it in the recolour "
                        f"convention (conj_names) to raise a wall")
            end_row = None
            if kind == 'arm':
                # the stretched end lobe goes at the end where it is NOT next
                # to one of the attaching patch's own weight-2 lobes (the
                # boundary alternation rule); probe the patch's REGISTERED
                # construction for a lobe foot on the facing corner data
                s_w = next(s for s in patches if s.name == wnm)
                _fo = {"X_horizontal": "X_vertical",
                       "X_vertical": "X_horizontal"}
                reg_o = (_fo[s_w.orientation] if wnm in conj_names
                         else s_w.orientation)
                nat = place_patch(SimpleNamespace(
                    origin=s_w.origin, distance=d, orientation=reg_o))["checks"]
                pside = near_ln if tuple(patch_at[wnm]) == near_c else far_ln
                band_lines = {lo, hi}
                ax_i = 1 if axis == 'horizontal' else 0
                blocked = set()
                for c in nat:
                    if len(c["pauli"]) != 2:
                        continue
                    if tuple(c["syn"])[ax_i] in band_lines:
                        continue        # the patch's own FACING lobes — they
                                        # are retired by the wall, not blockers
                    for q in c["pauli"]:
                        if q == P(a0, pside):
                            blocked.add('low')
                        if q == P(a0 + 2 * (d - 1), pside):
                            blocked.add('high')
                if 'low' not in blocked:
                    end_row = a0
                elif 'high' not in blocked:
                    end_row = a0 + 2 * (d - 1)
                else:
                    raise BentLayoutError(
                        f"arm wall {wnm}|{c2}: both band ends sit next to the "
                        f"patch's own weight-2 lobes — no legal end for the "
                        f"stretched lobe")
            walls.append(dict(axis=axis, ox=ox, oy=oy, a0=a0,
                              near_ln=near_ln, lo=lo, seam_ln=seam_ln, hi=hi,
                              far_ln=far_ln, P=P, kind=kind,
                              end_row=end_row, bus_letter=bus_eff2))
            # the true facing-lobe sites (a tile there would couple two
            # band feet on the same facing line) are the wall's territory
            for a in range(a0 + 1, a0 + 2 * (d - 1), 2):
                retire_lobes.add(P(a, lo))
                retire_lobes.add(P(a, hi))
            extra_connect |= _seam_qubits({c1, c2}, d)
        wall_pairs = frozenset(frozenset((tuple(c1), tuple(c2)))
                               for c1, c2, _, _n in wall_descr)
        if seam:
            stitch_skip, viol = _auto_stitch(tree, wall_pairs=wall_pairs)
            if viol:
                raise BentLayoutError(viol)
        else:
            stitch_skip = ns_pairs
        data, retype = path_to_corridor(tree, placed_all, target, d, seam=seam,
                                        patch_at=patch_at, bus=bus,
                                        conj_names=conj_names,
                                        flip_cells=flip_cells,
                                        skip_seam_pairs=([(c1, c2)
                                                          for c1, c2, _, _n
                                                          in wall_descr]
                                                         + [tuple(p) for p
                                                            in stitch_skip])
                                        or None)
        wall_fn = None
        retyped_bus = frozenset()
        if walls:
            from .seam_rules import (_plaquette_type as _pt,
                                     _wall_checks as _wc)
            _retype = frozenset(retype)

            def wall_fn(phase, _wc=_wc, d=d, walls=walls, _retype=_retype):
                out = []
                for w in walls:
                    P = w['P']
                    if w['kind'] == 'arm':
                        # the user's arrangement rules (K&F Fig 4 a/b):
                        # each stretched weight-4's side pair carries the
                        # OPPOSITE letter of the cell stabilizer right next
                        # to it on that side; the stretched weight-2 end
                        # lobe carries the flip of its neighbouring stretched
                        # record, at the end away from the patch's own lobes
                        nl, fl = w['near_ln'], w['far_ln']
                        nf = 1 if P(w['a0'], nl) in _retype else 0
                        ff = 1 if P(w['a0'], fl) in _retype else 0

                        def _cl(pos_a, line_m, flip):
                            c = P(pos_a, line_m)
                            L = _pt(c[0], c[1], phase)
                            return _FLIP[L] if flip else L

                        recs = []
                        for k in range(d - 1):
                            y = w['a0'] + 2 * k
                            Ll = _FLIP[_cl(y + 1, nl - 1, nf)]
                            Lr = _FLIP[_cl(y + 1, fl + 1, ff)]
                            feet = {P(y, nl): Ll, P(y + 2, nl): Ll,
                                    P(y, fl): Lr, P(y + 2, fl): Lr}
                            A, B = ((P(y + 1, w['hi']), P(y + 1, w['lo']))
                                    if Lr == 'Z'
                                    else (P(y + 1, w['lo']), P(y + 1, w['hi'])))
                            recs.append(
                                {'syn': B, 'type': 'M', 'pauli': feet,
                                 'corners': sorted(feet),
                                 'kf': {'flag': A,
                                        'shared': P(y + 1, w['seam_ln']),
                                        'orient': ('+' if A == P(y + 1, w['hi'])
                                                   else '-')}})
                        e = w['end_row']
                        adj = recs[0] if e == w['a0'] else recs[-1]
                        Ll_e = _FLIP[adj['pauli'][P(e, nl)]]
                        Lr_e = _FLIP[adj['pauli'][P(e, fl)]]
                        erow = e - 1 if e == w['a0'] else e + 1
                        feet = {P(e, nl): Ll_e, P(e, fl): Lr_e}
                        A, B = ((P(erow, w['hi']), P(erow, w['lo']))
                                if Lr_e == 'Z'
                                else (P(erow, w['lo']), P(erow, w['hi'])))
                        recs.append(
                            {'syn': B, 'type': 'M', 'pauli': feet,
                             'corners': sorted(feet),
                             'kf': {'flag': A, 'shared': P(erow, w['seam_ln']),
                                    'orient': ('+' if A == P(erow, w['hi'])
                                               else '-')}})
                        out.extend(recs)
                        continue
                    rep_near = P(w['a0'], w['near_ln'])
                    rep_far = P(w['a0'], w['far_ln'])
                    phi_n = phase ^ (1 if rep_near in _retype else 0)
                    phi_f = phase ^ (1 if rep_far in _retype else 0)
                    recs = _wc(d, phi_n, phi_f, w['ox'], w['oy'], w['axis'])
                    part = [{'syn': tuple(B), 'type': 'M', 'pauli': dict(p),
                             'corners': sorted(p), 'kf': dict(kf)}
                            for B, p, kf in recs]
                    if True:
                        # mid-bus junction (experimental): both band ends are
                        # free edges — mirror the closed form's end lobe
                        ends = {w['a0'] - 1, w['a0'] - 1 + 2 * d}
                        ax_i = 0 if w['axis'] == 'horizontal' else 1
                        used = next(a for a in ends
                                    if any(r['syn'][ax_i] == a for r in part))
                        m_e = next(a for a in ends if a != used)
                        sgn = 1 if m_e == w['a0'] - 1 else -1
                        adj = next(r for r in part
                                   if abs(r['syn'][ax_i] - (m_e + 2 * sgn)) <= 1)
                        letter = _FLIP[next(iter(set(adj['pauli'].values())))]
                        feet = {P(m_e + sgn, w['near_ln']): letter,
                                P(m_e + sgn, w['far_ln']): letter}
                        A, B = ((P(m_e, w['hi']), P(m_e, w['lo']))
                                if letter == 'Z'
                                else (P(m_e, w['lo']), P(m_e, w['hi'])))
                        part.append({'syn': B, 'type': 'M', 'pauli': feet,
                                     'corners': sorted(feet),
                                     'kf': {'flag': A,
                                            'shared': P(m_e - sgn, w['seam_ln']),
                                            'orient': '+' if A == P(m_e, w['hi'])
                                            else '-'}})
                    out.extend(part)
                return out
        if flip_cells:
            fq = set()
            for c in flip_cells:
                fq |= cell(*tuple(c), d, seam)
            if seam:
                fents = {tuple(c) for c in flip_cells} | \
                    {tuple(patch_at[nm]) for nm in tnames
                     if conj_names and nm in conj_names}
                fq |= _seam_qubits(fents, d)
            patchq = set().union(*[placed_all[nm] for nm in tnames])
            retyped_bus = frozenset((fq & set(data)) - patchq)
        # idle neighbours' USED ancilla sites — computed HERE, at the actual
        # construction, so the step-1 probe never touches place_patch (iron
        # rule: path finding is construction-free; this was also pure waste
        # on every probe — the probe branch cannot reach this point)
        forbidden = _obstacle_ancillas(patches, tnames)
        layout = _assemble_region(placed_all, target, orient, data, retype, d, seed,
                                  max_trials, max_cut=max_cut, forbidden=forbidden,
                                  native_lock=True, bus=bus, conj_names=conj_names,
                                  extra_forced_fn=wall_fn,
                                  extra_connect=frozenset(extra_connect),
                                  retyped_bus=retyped_bus,
                                  retire_lobes=frozenset(retire_lobes))
        cert = layout.verify() if layout is not None else None
        if cert is not None and all(cert.values()):
            cutq = tuple(sorted(set(data) - set(layout.data)))
            how = "corner-cut" if cutq else "standard"
            msg = f"verified subset joint ({how}, rule-based, EXPLICIT route)"
            return SubsetRoute(status="ok", message=msg,
                               layout=layout, root=root0, tree=tree, attempted=tree,
                               data=sorted(layout.data), tried=1, how=how, cut=cutq,
                               n_walls=len(wall_descr), certificate=cert,
                               **base0)
        return SubsetRoute(status="no_verified_route", root=root0, tried=1, attempted=tree,
                           message=("the EXPLICIT route does not pass the rule-based "
                                    "construction; the physics layer cannot host this "
                                    "corridor"), **base0)


class RotatedRoutedMultiPatchCoupler(_LCP):
    """Routed multi-patch coupler for ROTATED patches (seam-column design)."""

    EXPECTED_PATCH_COUNT = None
    _FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}

    @staticmethod
    def _key(ch):
        return (tuple(ch['syn']),
                frozenset((tuple(q), P) for q, P in ch['pauli'].items()))

    def _build_coupler_geometry(self, coupler_patch, patches, *, specs, target,
                                subset_route=None, seam=True, route=None,
                                minority_names=frozenset()):
        from types import SimpleNamespace
        r = subset_route
        if r is None:
            r = route_and_build(specs, target, seam=seam, route=route)
        if r.status != 'ok':
            raise ValueError(f'route_and_build failed: {r.status} — {r.message}')
        lay = r.layout
        tnames = {nm for nm, _ in target}
        coupler_patch.conflicting_stabilizer_coords = set()

        merged_keys = {self._key(ch) for ch in lay.checks}
        kept, native_syns, patch_cells = set(), set(), set()
        for s in specs:
            if s.name not in tnames:
                continue
            patch_cells |= {(s.origin[0] + 2 * i, s.origin[1] + 2 * j)
                            for i in range(s.distance) for j in range(s.distance)}
            # the construction the patch REGISTERED in the system; minority
            # patches register the conjugate-convention construction
            # (transposed geometry, records typeswapped)
            reg_orient = self._FLIP_O[s.orientation] if s.name in minority_names \
                else s.orientation
            reg = place_patch(SimpleNamespace(origin=s.origin, distance=s.distance,
                                              orientation=reg_orient))['checks']
            if s.name in minority_names:
                reg = [dict(ch, type=_FLIP[ch['type']],
                            pauli={q: _FLIP[P] for q, P in ch['pauli'].items()})
                       for ch in reg]
            for ch in reg:
                native_syns.add(tuple(ch['syn']))
                key = self._key(ch)
                if key not in merged_keys:
                    coupler_patch.conflicting_stabilizer_coords.add(tuple(ch['syn']))
                else:
                    kept.add(key)

        # new data qubits: corridor + seam columns
        for q in sorted(set(map(tuple, lay.data)) - patch_cells):
            coupler_patch.add_qubit(q[0], q[1], role='data')
        # coupler-owned checks + any genuinely new ancilla positions
        added = set()
        for ch in lay.checks:
            if self._key(ch) in kept:
                continue
            syn = tuple(ch['syn'])
            typ = ch.get('type') or next(iter(set(ch['pauli'].values())))
            kf = ch.get('kf')
            if kf is not None:
                # stretched (kf) record: B reads out in X, flag A and the
                # shared relay S in Z (same apparatus convention as
                # RotatedSeamWallCoupler)
                if syn not in native_syns and syn not in added:
                    coupler_patch.add_qubit(syn[0], syn[1], role='syndrome_x')
                    added.add(syn)
                for coord, role in ((tuple(kf['flag']), 'syndrome_z'),
                                    (tuple(kf['shared']), 'syndrome_z')):
                    if coord not in native_syns and coord not in added:
                        coupler_patch.add_qubit(coord[0], coord[1], role=role)
                        added.add(coord)
                coupler_patch.stabilizers.append({
                    'pauli': {tuple(q): P for q, P in ch['pauli'].items()},
                    'type': 'MIXED',
                    'syn_coord': syn,
                    'kf': {'flag': tuple(kf['flag']),
                           'shared': tuple(kf['shared']),
                           'orient': kf['orient']},
                })
                continue
            if syn not in native_syns and syn not in added:
                coupler_patch.add_qubit(syn[0], syn[1],
                                        role=('syndrome_z' if typ == 'Z'
                                              else 'syndrome_x'))
                added.add(syn)
            coupler_patch.stabilizers.append({
                'pauli': {tuple(q): P for q, P in ch['pauli'].items()},
                'type': typ if typ in ('X', 'Z') else 'MIXED',
                'syn_coord': syn,
            })
        coupler_patch.routed_layout = lay          # introspection for the protocol layer
        coupler_patch.subset_route = r
