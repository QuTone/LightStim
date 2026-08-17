# protocols/ppm/lowering.py
"""Pure PPM lowering kernel.

``lower_ppm()`` turns ONE Pauli-product-measurement request into a
:class:`LoweringPlan` — a declarative description of the construction —
without touching the system, builder, or tracker.  ``apply_plan()``
registers the plan's coupler with a :class:`~lightstim.ir.qec_system.QECSystem`
in a separate step.  The experiment driver
(:class:`~.sequential.SequentialPPMExperiment`) is one consumer of this
kernel; a logical-level compiler is the intended other.

Scope and honesty notes:

- The ancilla route is EXPLICIT and required (``()`` for cell-adjacent
  targets).  There is no auto-router; nothing here pretends otherwise.
- Pauli letters: X and Z only.  A ``Y`` target raises
  :class:`UnsupportedPauliError`: measuring a Y factor on a rotated
  surface-code patch needs a twist defect or a Y-wall construction that
  this stack does not implement (it cannot be silently rewritten as X/Z —
  that would be a different quantum instrument).
- Cell-adjacent two-patch requests are classified by the four-row seam
  rule table against the LIVE system (the classification is a live probe,
  so ``system`` is required for them).
- ``LoweringPlan.certificate`` carries the algebraic acceptance oracle of
  the accepted layout (commutation, the joint IS measured, no single
  logical and no proper subset product is exposed — the "true weight-w
  measurement, not w-1 pairwise parities" guarantee — expected rank
  change, no weight-1 logical, plus circuit-level checks when the layout
  hosts no stretched check).  Wall plans currently carry ``None``: the
  wall construction is validated by its own rule-table laws and
  end-to-end distance tests, not by this oracle — a known gap, tracked in
  the PR's capability table.
"""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
from lightstim.utils.linear_algebra import solve_linear_decomposition

from .spec import PatchSpec, cell_index
from .coupler import (route_and_build, RotatedRoutedMultiPatchCoupler,
                      SubsetRoute)
from .seam_rules import (RotatedSeamWallCoupler, SeamRuleError, classify_seam,
                         patch_view, wall_spec)

__all__ = [
    "PPMRequest", "LoweringPlan", "LoweringCertificate", "PPMOutcome",
    "UnsupportedPauliError", "lower_ppm", "apply_plan",
    "is_cell_adjacent_pair", "joint_pauli_vector", "record_parity",
]

_CONJ = {'X': 'Z', 'Z': 'X'}


class UnsupportedPauliError(ValueError):
    """A requested Pauli letter has no implemented lowering construction."""


@dataclass(frozen=True)
class PPMRequest:
    """One joint Pauli-product measurement, declaratively.

    Args:
        targets: ``((patch_name, 'X'|'Z'), ...)`` — one entry per target
            patch (each patch appears once; the single-seam attach law).
        route: explicit corridor cells on the coarse grid; ``()`` for
            cell-adjacent targets.  Required — there is no auto-router.
        construction: ``'auto'`` (rule table decides) | ``'wall'`` |
            ``'merge'`` — adjacent-pair override only.
        schedule: per-request override of the merged-round schedule
            (``'bent'`` | ``'diagonal'``), else the policy decides.
    """
    targets: Tuple[Tuple[str, str], ...]
    route: Tuple[Tuple[int, int], ...]
    construction: str = 'auto'
    schedule: Optional[str] = None

    def __post_init__(self):
        if self.route is None:
            raise ValueError(
                "PPMRequest: route is required — pass the explicit corridor "
                "cells, or () for cell-adjacent targets (no auto-router "
                "exists in this stack)")
        if len(self.targets) < 2:
            raise ValueError(
                "PPMRequest: a Pauli-product measurement needs at least "
                "two distinct target patches")
        names = [name for name, _ in self.targets]
        if len(set(names)) != len(names):
            raise ValueError(
                f"PPMRequest: target patches must be unique; got {names}")
        if self.construction not in ('auto', 'wall', 'merge'):
            raise ValueError(
                "PPMRequest: construction must be 'auto', 'wall', or "
                f"'merge'; got {self.construction!r}")
        if self.schedule not in (None, 'bent', 'diagonal'):
            raise ValueError(
                "PPMRequest: schedule must be None, 'bent', or 'diagonal'; "
                f"got {self.schedule!r}")
        for name, letter in self.targets:
            if letter == 'Y':
                raise UnsupportedPauliError(
                    f"target ({name!r}, 'Y'): measuring a Y factor needs a "
                    f"twist defect or Y-wall construction that this stack "
                    f"does not implement; it cannot be silently treated as "
                    f"X/Z (that is a different quantum instrument). "
                    f"Supported letters: X, Z.")
            if letter not in ('X', 'Z'):
                raise ValueError(
                    f"target ({name!r}, {letter!r}): Pauli letter must be "
                    f"'X' or 'Z'")

    @property
    def bus(self) -> str:
        """Majority letter — the corridor's stabilizer type."""
        n_x = sum(1 for _, P in self.targets if P == 'X')
        return 'X' if n_x >= len(self.targets) - n_x else 'Z'


@dataclass(frozen=True)
class LoweringCertificate:
    """Named algebraic/circuit acceptance items of the accepted layout."""
    items: Mapping[str, bool]

    @property
    def ok(self) -> bool:
        return bool(self.items) and all(self.items.values())

    @property
    def measures_exactly_the_product(self) -> bool:
        """The joint is measured, and neither any single logical nor any
        proper subset product leaks — a true weight-w instrument."""
        return (self.items.get('joint', False)
                and self.items.get('no_single', False)
                and self.items.get('no_subjoint', False))


@dataclass
class LoweringPlan:
    """Declarative result of lowering one PPM request.

    ``apply_plan(system, plan, name)`` performs the registration; the
    driver (or compiler backend) then activates, runs the merged rounds
    under ``schedule``, deactivates, and reads the corridor out in
    ``corridor_readout_basis``.
    """
    request: PPMRequest
    kind: str                                  # 'corridor' | 'wall'
    schedule: str                              # 'bent' | 'diagonal'
    specs: Tuple[PatchSpec, ...]               # live-orientation-stamped
    conj_names: FrozenSet[str]
    # corridor plans:
    route_result: Optional[SubsetRoute] = None
    # wall plans:
    wall: Optional[dict] = None
    # adjacent-pair rule-table row (live probe), when applicable:
    rule: Optional[object] = None
    certificate: Optional[LoweringCertificate] = None

    @property
    def bus(self) -> str:
        return self.request.bus

    @property
    def corridor_init_basis(self) -> str:
        """New corridor data qubits initialize (and read out) in the
        conjugate of the bus letter."""
        return _CONJ[self.bus]

    corridor_readout_basis = corridor_init_basis

    @property
    def merged_checks(self) -> Tuple[dict, ...]:
        """The merged-window check set this plan installs (corridor plans;
        the wall coupler materializes its apparatus at registration)."""
        if self.route_result is not None and self.route_result.layout is not None:
            return tuple(self.route_result.layout.checks)
        return ()

    @property
    def has_stretched_checks(self) -> bool:
        return any(c.get('kf') for c in self.merged_checks)


def _pick_schedule(request: PPMRequest, policy: str, *,
                   is_wall: bool, collinear: bool) -> str:
    """Merged-round schedule: a wall or any corridor bend forces the whole
    merged block onto the diagonal schedule; straight corridors stay bent."""
    if is_wall:
        if request.schedule == 'bent' or policy == 'bent':
            raise SeamRuleError(
                "a wall step cannot run the bent schedule: the bent chunk "
                "cannot express stretched (kf) checks — use 'diagonal'")
        return 'diagonal'
    if request.schedule is not None:
        return request.schedule
    if policy in ('bent', 'diagonal'):
        return policy
    return 'bent' if collinear else 'diagonal'


def is_cell_adjacent_pair(specs_by_name, request) -> bool:
    """Exactly two targets on edge-adjacent coarse cells — direct-seam
    territory of the rule table; everything else is a routed corridor."""
    names = [nm for nm, _ in request.targets]
    if len(names) != 2:
        return False
    cells = [cell_index(specs_by_name[nm].origin,
                        specs_by_name[nm].distance, seam=True)
             for nm in names]
    (a1, b1), (a2, b2) = cells
    return abs(a1 - a2) + abs(b1 - b2) == 1


def lower_ppm(specs: List[PatchSpec], request: PPMRequest, *,
              system=None, conj_names: FrozenSet[str] = frozenset(),
              schedule_policy: str = 'auto') -> LoweringPlan:
    """Lower one PPM request to a :class:`LoweringPlan`.  Pure: no
    system/builder/tracker mutation.

    Args:
        specs: every patch, stamped with its LIVE orientation.
        request: the PPM request (explicit route, X/Z letters).
        system: the live :class:`QECSystem` — required for cell-adjacent
            pairs, whose rule-table classification probes the REAL
            registered patches (not a reconstruction).
        conj_names: the colour-swapped subset of the targets.
        schedule_policy: experiment-wide policy ('auto'|'bent'|'diagonal').
    """
    if schedule_policy not in ('auto', 'bent', 'diagonal'):
        raise ValueError(
            "lower_ppm: schedule_policy must be 'auto', 'bent', or "
            f"'diagonal'; got {schedule_policy!r}")
    by_name = {s.name: s for s in specs}
    if len(by_name) != len(specs):
        raise ValueError("lower_ppm: patch specifications must have unique names")
    unknown = [nm for nm, _ in request.targets if nm not in by_name]
    if unknown:
        raise ValueError(f"lower_ppm: unknown target patch(es) {unknown}")

    adjacent_pair = is_cell_adjacent_pair(by_name, request)
    if not adjacent_pair and request.construction != 'auto':
        raise ValueError(
            "lower_ppm: construction overrides apply only to cell-adjacent "
            "two-patch requests")

    if adjacent_pair:
        if system is None:
            raise ValueError(
                "lower_ppm: a cell-adjacent pair is classified by the seam "
                "rule table against the LIVE system (live probe) — pass "
                "system=")
        names = [nm for nm, _ in request.targets]
        views = {nm: patch_view(system, nm) for nm in names}
        tgt = dict(request.targets)
        rule = classify_seam(views[names[0]], tgt[names[0]],
                             views[names[1]], tgt[names[1]])
        cons = request.construction if request.construction != 'auto' \
            else rule.construction
        if cons == 'wall':
            spec = wall_spec(views[names[0]], views[names[1]], tgt)
            return LoweringPlan(
                request=request, kind='wall',
                schedule=_pick_schedule(request, schedule_policy,
                                        is_wall=True, collinear=True),
                specs=tuple(specs), conj_names=frozenset(conj_names),
                wall=spec, rule=rule)
        plan = _lower_corridor(specs, request, conj_names, by_name,
                               schedule_policy, force_collinear=True)
        return LoweringPlan(
            request=request, kind=plan.kind, schedule=plan.schedule,
            specs=plan.specs, conj_names=plan.conj_names,
            route_result=plan.route_result, certificate=plan.certificate,
            rule=rule)

    return _lower_corridor(specs, request, conj_names, by_name,
                           schedule_policy, force_collinear=False)


def _lower_corridor(specs, request, conj_names, by_name, schedule_policy,
                    *, force_collinear):
    r = route_and_build(specs, list(request.targets), seam=True,
                        route=list(request.route),
                        conj_names=frozenset(conj_names))
    if r.status != 'ok':
        raise ValueError(
            f"lower_ppm: route_and_build failed: {r.status} — {r.message}")
    has_kf = any(ch.get('kf')
                 for ch in (r.layout.checks if r.layout is not None else ()))
    if force_collinear:
        # adjacent-pair merge rows: zero-cell corridor, collinear by
        # construction; the emission layer re-checks kf on the layout.
        schedule = _pick_schedule(request, schedule_policy,
                                  is_wall=False, collinear=True)
    else:
        names = [nm for nm, _ in request.targets]
        cells = [cell_index(by_name[nm].origin, by_name[nm].distance,
                            seam=True) for nm in names] \
            + [tuple(c) for c in (r.tree or ())]
        collinear = (len({a for a, _ in cells}) == 1
                     or len({b for _, b in cells}) == 1)
        schedule = _pick_schedule(request, schedule_policy,
                                  is_wall=has_kf, collinear=collinear)
    cert = (LoweringCertificate(items=dict(r.certificate))
            if r.certificate is not None else None)
    return LoweringPlan(
        request=request, kind='corridor', schedule=schedule,
        specs=tuple(specs), conj_names=frozenset(conj_names),
        route_result=r, certificate=cert)


@dataclass(frozen=True)
class PPMOutcome:
    """PROTOCOL OUTPUT of one applied PPM: the physical measurement-record
    parity of the requested joint product (review §3).

    ``records_*`` are sorted absolute measurement indices whose XOR equals
    the joint's pinned parity at capture time, or ``None`` when the joint
    is a free coin there — e.g. before the merge of a request that
    anti-commutes with the tracked state.  A ``None`` is a legitimately
    random outcome, NOT an error.

    This is deliberately distinct from an EVALUATION observable: whether
    and how a protocol output enters a deterministic, decodable quantity
    (an OBSERVABLE_INCLUDE, a closure detector, a feed-forward dependency)
    is the caller's evaluation choice, made from input states, these
    record parities, and the final measurement bases.
    """
    step: int
    targets: Tuple[Tuple[str, str], ...]
    records_pre_merge: Optional[Tuple[int, ...]]
    records_post_split: Optional[Tuple[int, ...]]


def joint_pauli_vector(system, targets) -> np.ndarray:
    """[X|Z] GF(2) vector of the joint logical named by ``targets``,
    assembled from the system's registered logical representatives."""
    n = system.num_qubits
    v = np.zeros(2 * n, dtype=np.uint8)
    for nm, letter in targets:
        matches = [
            rec for rec in system.logical_ops
            if rec.get('patch_name') == nm and rec.get('type') == letter
        ]
        if len(matches) != 1:
            raise ValueError(
                "joint_pauli_vector: expected exactly one registered "
                f"{letter} logical representative for patch {nm!r}; "
                f"found {len(matches)}")
        for q, pp in matches[0]['pauli'].items():
            if pp in ('X', 'Y'):
                v[q] ^= 1
            if pp in ('Z', 'Y'):
                v[n + q] ^= 1
    return v


def record_parity(tracker, vec) -> Optional[List[int]]:
    """Read-only: if ``vec`` lies in the span of the tracker's
    record-carrying rows, return the sorted XOR of the pinning rows'
    banked records (the operator's deterministic reconstruction); else
    ``None`` — a free outcome, not a parity check.

    Rows carrying the UNMEASURED sentinel are excluded from the basis:
    a closure through one XORs in an unbanked, per-shot-random gauge
    (measured 2026-08-05 on a parallel batch window), so excluding them
    only removes WRONG reconstructions.
    """
    stab, logs = tracker.stabilizers, tracker.logicals
    # deferred allocation can grow the qubit count between refreshes;
    # symplectic rows are [X | Z] halves, so widen by re-placing each half.
    wide = max(stab.matrix.shape[1],
               logs.matrix.shape[1] if logs.matrix.shape[0] > 0 else 0,
               vec.shape[0]) // 2

    def _widen(M):
        n0 = M.shape[1] // 2
        if n0 == wide:
            return M
        P = np.zeros((M.shape[0], 2 * wide), dtype=M.dtype)
        P[:, :n0] = M[:, :n0]
        P[:, wide:wide + n0] = M[:, n0:]
        return P

    if vec.shape[0] != 2 * wide:
        v = np.zeros(2 * wide, dtype=vec.dtype)
        n0 = vec.shape[0] // 2
        v[:n0] = vec[:n0]
        v[wide:wide + n0] = vec[n0:]
        vec = v
    if logs.matrix.shape[0] > 0:
        M = np.vstack([_widen(stab.matrix), _widen(logs.matrix)])
        recs_of = stab.records + logs.records
    else:
        M = _widen(stab.matrix)
        recs_of = list(stab.records)
    n_stab = stab.matrix.shape[0]
    rows_ok = [k for k in range(M.shape[0])
               if k >= n_stab
               or (recs_of[k] and UNMEASURED_STAB_RECORD not in recs_of[k])]
    if len(rows_ok) != M.shape[0]:
        M = M[rows_ok]
        recs_of = [recs_of[k] for k in rows_ok]
    if M.shape[0] == 0:
        return None
    out = solve_linear_decomposition(basis=M, targets=vec.reshape(1, -1),
                                     reduce_weight=True)
    coeffs = out[0]
    if coeffs is None:
        return None
    row = np.asarray(coeffs[0], dtype=np.uint8)
    if not np.array_equal((row.astype(int) @ M.astype(int)) % 2,
                          vec.astype(int)):
        return None
    recs = set()
    for j in np.nonzero(row)[0]:
        recs ^= set(recs_of[j])
    return sorted(recs)


def apply_plan(system, plan: LoweringPlan, name: str) -> None:
    """Register the plan's coupler with the system (the ONLY mutation of
    this module's API; activation/rounds/split stay with the caller)."""
    target_names = [nm for nm, _ in plan.request.targets]
    if plan.kind == 'wall':
        system.register_coupler(
            RotatedSeamWallCoupler(), patch_names=target_names, name=name,
            spec=plan.wall, occupied=frozenset(system.index_map))
        return
    system.register_coupler(
        RotatedRoutedMultiPatchCoupler(),
        patch_names=target_names, name=name, specs=list(plan.specs),
        target=list(plan.request.targets), subset_route=plan.route_result,
        seam=True, minority_names=plan.conj_names)
