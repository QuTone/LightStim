"""Rotated **multi-patch** bent joint lattice-surgery measurement — ``M(X̄₁·X̄₃·Z̄₂)``.

The three-patch generalization of the two-patch bent ``M(X̄₁·Z̄₂)`` in :mod:`.bent_layout`,
transcribed from ``rotated_3patch.png``.  Three logical patches take part, each an explicit
:class:`.PatchSpec` whose ``origin`` / ``distance`` / ``orientation`` / ``measured_logical``
genuinely drive the geometry (exactly like the two-patch generator — **not** a fixed template):

* ``p1`` contributes **X̄₁** — placed on the **left** of a horizontal band,
* ``p2`` contributes **Z̄₂** — placed on the **right** of the band,
* ``p3`` contributes **X̄₃** — placed on a vertical **arm** hanging below the band.

The construction is a **"T"**: ``place_patch`` puts all three patches at their origins; a
horizontal **band** is derived between ``p1`` and ``p2`` (with the same even-gap rule as the
two-patch bus), a vertical **arm** is derived between ``p3`` and the band, and a single **bent
mixed (XZ) domain wall** re-types the X-side (``p1`` + ``p3`` + the bus, everything left of
``p2``) and joins it to the Z-patch ``p2``.  Changing any origin moves the patches and
regenerates the band / arm / wall / re-typing / readout; an incompatible placement raises
:class:`.BentLayoutError` with a concrete geometric reason — never a hard-coded limit.

The construction realizes **only the triple joint** ``X̄₁·X̄₃·Z̄₂`` — no single logical and (the
subtle part) **no sub-joint** such as ``X̄₁·Z̄₂``.  With two X-logicals on one connected X-side,
a naive maximal code makes ``X̄₁`` and ``X̄₃`` homologous and measures only the sub-joint;
:func:`_select_joint_checks` keeps them independent by forbidding every proper sub-product from
the stabilizer span until the **full** joint lands in it.  The acceptance oracle (``verify``)
gates the result, as in the two-patch case.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from .bent_layout import (
    PatchSpec, BentLayoutError, place_patch,
    _bent_plaquettes, _symplectic, _gf2_rank, _in_span, _commute,
    _gf2_independent, _no_tick_collision,
)


def xxz_patches(distance, offset=0):
    """A ready-made ``[p1, p2, p3]`` PatchSpec set for the X̄₁X̄₃Z̄₂ joint at any odd ``d``.

    ``offset`` is the vertical staircase: the number of rows ``p2`` (the band) sits **below**
    ``p1``.  ``offset = 0`` is the row-aligned layout of ``rotated_3patch.png`` (``p1`` level with
    the band); ``offset > 0`` lowers the band + arm so ``p1`` sits higher, the bent/staircase
    layout of ``rotated_3patch_2.png`` (``offset = d - 1`` is the full staircase, ``p1.bottom``
    touching the band top).  ``p1`` (X̄₁) is always fused to the top of the X̄₃ trunk; ``p2`` (Z̄₂)
    is a ``d``-wide connector to the right; ``p3`` (X̄₃) hangs one row below the band.  Callers may
    instead pass their own origins to :func:`build_rotated_bent_xxz_layout`.
    """
    d = distance
    return [
        PatchSpec("p1", (1, 5), d, "X", "X_vertical"),                          # X̄₁ vertical, fused to trunk top
        PatchSpec("p2", (4 * d + 1, 5 + 2 * offset), d, "Z", "X_horizontal"),   # Z̄₂ vertical, band lowered by offset
        PatchSpec("p3", (2 * d - 1, 2 * d + 7 + 2 * offset), d, "X", "X_horizontal"),  # X̄₃ horizontal, arm below band
    ]


def _arrange_xxz(p1, p2, p3):
    """Validate the three PLACED patches form a **trunk + band** "T"/staircase and return geometry.

    A single vertical **trunk** at ``p3``'s columns runs from ``p1`` (top) down to ``p3`` (bottom),
    carrying both X-side logicals; a horizontal **band** at ``p2``'s rows branches off the trunk to
    the right and meets ``p2`` across the mixed wall.  ``p1`` (X̄₁) fuses to the trunk's left at its
    bus-facing column.  Geometry is read from the placed coordinates (coordinate-aware), and this
    one rule covers **both** the row-aligned case (``p1`` level with the band, ``rotated_3patch.png``)
    **and** the vertically-offset / staircase case (``p1`` raised above the band,
    ``rotated_3patch_2.png``) — the trunk simply extends up to wherever ``p1`` sits:

      * **trunk**: ``p3``'s columns × rows ``[p1.top .. p3.top]`` (the vertical spine);
      * **band**: rows ``p2.rows`` × columns ``[p3.left .. p2.right]``; the trunk↔``p2`` gap is
        ``hgap = p2.left - p3.right = 2 + 2k`` (``k>=0`` connector columns);
      * the trunk↔``p3`` gap is ``armgap = p3.top - p2.bottom = 2 + 2m`` (``m>=0`` connector rows);
      * the **mixed wall** falls automatically at the X↔Z re-typing boundary just left of ``p2``.

    The vertical offset between ``p1`` and ``p2`` is ``p2.top - p1.top`` (0 = row-aligned, >0 =
    staircase); both are valid as long as the band stays within the trunk's row span.

    Returns ``(data, retype, band_rows, x1, x3, z2, k, m)`` where ``data`` is the full data set,
    ``retype`` the X-side data set, and ``x1/x3/z2`` the bus-facing logical representatives.  Raises
    :class:`.BentLayoutError` (concrete reason) when the placement cannot host the construction —
    ``p1`` not seated above the trunk, an odd/zero/negative gap, the band outside the trunk span, or
    ``p3`` not on the X-side.  Any common translation of a valid placement is also valid.
    """
    cols = lambda p: sorted({c for c, _ in p["data"]})
    rows = lambda p: sorted({r for _, r in p["data"]})
    p1c, p1r = cols(p1), rows(p1)
    p2c, p2r = cols(p2), rows(p2)
    p3c, p3r = cols(p3), rows(p3)

    reasons = []
    if p1c[-1] != p3c[0]:
        reasons.append(f"p1's right column {p1c[-1]} must equal p3's left column {p3c[0]} "
                       f"(X̄₁ fuses to the top of the X̄₃ trunk)")
    hgap = p2c[0] - p3c[-1]
    if hgap < 2 or hgap % 2 != 0:
        reasons.append(f"horizontal gap p2.left-p3.right = {hgap}, expected positive even 2+2k "
                       f"(2 = minimal band, 2+2k = longer connector)")
    if p2r[0] < p1r[0]:
        reasons.append(f"p2/band top {p2r[0]} must be >= p1 top {p1r[0]} "
                       f"(p1 sits at or above the band — offset is downward)")
    armgap = p3r[0] - p2r[-1]
    if armgap < 2 or armgap % 2 != 0:
        reasons.append(f"vertical gap p3.top-band.bottom = {armgap}, expected positive even 2+2m "
                       f"(2 = minimal arm, 2+2m = taller arm)")
    if p2r[-1] > p3r[0]:
        reasons.append(f"band bottom {p2r[-1]} must stay within the trunk (above p3.top {p3r[0]})")
    if p3c[-1] >= p2c[0]:
        reasons.append(f"p3 (the X̄₃ arm) must lie left of the Z-patch p2 (p3 right col {p3c[-1]}, "
                       f"p2 left col {p2c[0]}); it sits on the X-side")
    if reasons:
        raise BentLayoutError("patches cannot host a bent X̄₁X̄₃Z̄₂ joint: " + "; ".join(reasons)
                              + ". (Any common translation of a valid placement is also valid.)")

    trunk = {(x, y) for x in p3c for y in range(p1r[0], p3r[0] + 2, 2)}        # vertical spine p1..p3
    band = {(x, y) for x in range(p3c[0], p2c[-1] + 2, 2) for y in p2r}        # horizontal band at p2's rows
    data = sorted(set(p1["data"]) | set(p2["data"]) | set(p3["data"]) | trunk | band)
    retype = {c for c in data if c[0] < p2c[0]}            # X-side = everything left of the Z-patch
    x1 = [(p1c[-1], y) for y in p1r]                       # bus-facing right edge of p1 (vertical)
    z2 = [(p2c[0], y) for y in p2r]                        # bus-facing left edge of p2 (vertical)
    x3 = [(x, p3r[0]) for x in p3c]                        # bus-facing top edge of p3 (horizontal)
    return data, retype, p2r, x1, x3, z2, (hgap - 2) // 2, (armgap - 2) // 2


def _proper_subproducts(sv, logicals):
    """Symplectic vectors of every PROPER non-empty subset product of the measured logicals.

    ``logicals`` is a list of ``(measured_pauli, support)``.  These are exactly the operators that
    must stay OUT of the stabilizer span so that only the FULL joint is measured.
    """
    vecs = [sv({c: P for c in sup}) for P, sup in logicals]
    n = len(vecs)
    out = []
    for r in range(1, n):
        for comb in itertools.combinations(range(n), r):
            v = np.zeros_like(vecs[0])
            for k in comb:
                v ^= vecs[k]
            out.append(v)
    return out


def _select_joint_checks(data, plaqs, logicals, seed=0, max_trials=8000):
    """Pick a valid stabilizer code that measures exactly the joint ``∏ᵢ P̄ᵢ`` — or return ``None``.

    The unambiguous weight-4 plaquettes (bulk + mixed wall) are forced.  The weight-2/3 boundary
    plaquettes are selected so the code is commuting, GF(2)-independent, keeps every PROPER
    sub-product of ``logicals`` out of the span, and puts the FULL joint into the span with the
    expected ``len(logicals) - 1`` residual logical count.  Boundary orderings are drawn with a
    fixed RNG (deterministic, reproducible) until one satisfies the joint target.  Returns the kept
    check list, or ``None`` if no ordering works within ``max_trials`` (the caller raises).
    """
    sv, n = _symplectic(data)
    joint = np.zeros(2 * n, np.uint8)
    for P, sup in logicals:
        joint ^= sv({c: P for c in sup})
    preserve = _proper_subproducts(sv, logicals)
    target_nlog = len(logicals) - 1

    forced = [p for p in plaqs if len(p["pauli"]) >= 4]
    boundary = [p for p in plaqs if len(p["pauli"]) < 4]
    F = [sv(p["pauli"]) for p in forced]

    def build(order):
        rows = list(F)
        kept = list(forced)
        for p in order:
            v = sv(p["pauli"])
            if not all(_commute(v, r, n) for r in rows):
                continue
            if not _gf2_independent(rows, v):
                continue
            if any(_in_span(rows + [v], L) for L in preserve):
                continue
            rows.append(v)
            kept.append(p)
        return rows, kept

    rng = np.random.RandomState(seed)
    for _ in range(max_trials):
        order = list(boundary)
        rng.shuffle(order)
        rows, kept = build(order)
        if _in_span(rows, joint) and len(data) - _gf2_rank(rows) == target_nlog:
            return kept
    return None


def _readout_chain(data, checks, logicals):
    """Check syndromes whose GF(2) product equals the joint ``∏ᵢ P̄ᵢ`` (for highlighting/readout)."""
    from ....utils.linear_algebra import solve_linear_decomposition
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


@dataclass
class MultiPatchJointLayout:
    """A rotated multi-patch bent joint-measurement layout (``M(X̄₁·X̄₃·Z̄₂)``).

    Attributes
    ----------
    distance:   code distance ``d`` of the participating patches.
    data:       sorted list of data-qubit coords ``(col, row)`` (data on (odd, odd)).
    checks:     stabilizer dicts ``{"syn", "type", "pauli", "corners"}`` with ``type`` in
                ``{"X", "Z", "M"}`` (M = mixed XZ domain wall).
    logicals:   list of ``(name, measured_pauli, support)`` — one per measured logical
                (``("p1","X",…)``, ``("p3","X",…)``, ``("p2","Z",…)``).
    x_observable: the X-type logical read out by the X-memory circuit (X̄₁).
    readout_chain: the check-syndrome set whose product equals the joint.
    specs:      the input PatchSpecs (provenance).
    """
    distance: int
    data: list
    checks: list
    logicals: list
    x_observable: list
    readout_chain: set = field(default_factory=set)
    specs: list = field(default_factory=list)

    def build_circuit(self, rounds=None, p=0.0):
        """No-MPP gate-level syndrome-extraction circuit (X-memory experiment on X̄₁)."""
        from .bent_joint_se import RotatedBentJointMeasurement
        if rounds is None:
            rounds = self.distance
        return RotatedBentJointMeasurement(list(self.data), self.checks,
                                           list(self.x_observable)).circuit(rounds=rounds, p=p)

    def verify(self, rounds=None):
        """The nine acceptance checks; returns a dict of booleans (all True == a valid joint)."""
        sv, n = _symplectic(self.data)
        S = [sv(ch["pauli"]) for ch in self.checks]
        singles = [sv({c: P for c in sup}) for _, P, sup in self.logicals]
        joint = np.zeros(2 * n, np.uint8)
        for v in singles:
            joint ^= v
        N = len(self.logicals)
        subs = []
        for r in range(1, N):
            for comb in itertools.combinations(range(N), r):
                v = np.zeros(2 * n, np.uint8)
                for k in comb:
                    v ^= singles[k]
                subs.append(v)
        commute = all(_commute(S[i], S[j], n) for i in range(len(S)) for j in range(i + 1, len(S)))
        twist = any(P == "Y" for ch in self.checks for P in ch["pauli"].values())
        out = dict(
            commute=commute,
            joint=_in_span(S, joint),
            no_single_or_subjoint=not any(_in_span(S, v) for v in subs),
            no_twist=not twist,
            logical_count=len(self.data) - _gf2_rank(S) == N - 1,
        )
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


def build_rotated_bent_xxz_layout(patches, routing="auto", seed=0, max_trials=8000):
    """Build the rotated three-patch bent joint-measurement layout ``M(X̄₁·X̄₃·Z̄₂)``.

    ``patches`` is ``[p1, p2, p3]`` (the role order), each a :class:`.PatchSpec`:
    ``p1`` (``measured_logical="X"``) contributes X̄₁ on the band-left, ``p2`` (``"Z"``)
    contributes Z̄₂ on the band-right, ``p3`` (``"X"``) contributes X̄₃ on the arm.  The geometry
    is generated **from the patches' actual origins** — both X-patches and the Z-patch are placed
    as real ``distance``×``distance`` rotated patches, the connecting band + arm bus is derived
    from their positions, and the data / CSS + mixed (XZ) wall stabilizers / re-typing / readout
    chain follow.  Changing an origin moves and regenerates the whole layout; the convenience
    helper :func:`xxz_patches` returns origins that reproduce ``rotated_3patch.png`` at any ``d``.

    Raises :class:`.BentLayoutError` (with a concrete geometric reason) if the requested placement
    cannot host the "T" or if no boundary selection measures the full joint — it does not silently
    fall back to a fixed layout or mis-measure a sub-joint.
    """
    if routing != "auto":
        raise NotImplementedError(f"only routing='auto' is currently supported, got {routing!r}")
    if len(patches) != 3:
        raise ValueError(f"expected exactly 3 patches [p1(X), p2(Z), p3(X)], got {len(patches)}")
    if [s.measured_logical for s in patches] != ["X", "Z", "X"]:
        raise ValueError("expected measured_logical order [p1='X', p2='Z', p3='X'] "
                         f"(X̄₁, Z̄₂, X̄₃), got {[s.measured_logical for s in patches]}")
    d = patches[0].distance
    if any(s.distance != d for s in patches):
        raise ValueError("all three patches must share the same distance")

    P1, P2, P3 = (place_patch(s) for s in patches)
    data, retype, _, x1, x3, z2, _, _ = _arrange_xxz(P1, P2, P3)
    logicals = [("p1", "X", x1), ("p3", "X", x3), ("p2", "Z", z2)]
    log_pairs = [(P, sup) for _, P, sup in logicals]

    sv, n = _symplectic(data)
    L = [sv({c: P for c in sup}) for P, sup in log_pairs]
    for phase in (0, 1):                                   # pick the parity under which the logicals commute
        plaqs = _bent_plaquettes(data, retype, phase)
        forced = [sv(p["pauli"]) for p in plaqs if len(p["pauli"]) >= 4]
        if any(not _commute(v, f, n) for v in L for f in forced):
            continue                                       # wrong global X/Z parity for these logicals
        checks = _select_joint_checks(data, plaqs, log_pairs, seed=seed, max_trials=max_trials)
        if checks is None:
            continue
        for c in checks:
            c["corners"] = sorted(c["pauli"])
        readout = _readout_chain(data, checks, log_pairs)
        return MultiPatchJointLayout(distance=d, data=data, checks=checks, logicals=logicals,
                                     x_observable=x1, readout_chain=readout, specs=list(patches))

    raise BentLayoutError(
        "could not assemble a valid X̄₁X̄₃Z̄₂ joint code for these patches "
        "(no boundary selection measured the full triple joint — the placement may be incompatible "
        "with a single bent domain wall keeping X̄₁ and X̄₃ independent).")
