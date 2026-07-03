"""Parameterized rotated bent (XZ) joint-measurement layout generator.

Given the two logical patches that participate in a joint measurement M(X̄₁·Z̄₂),
auto-build the data qubits, ordinary CSS stabilizers, mixed (XZ) domain-wall
stabilizers, boundary trim/replacement, readout chain, and the no-MPP gate-level
syndrome-extraction circuit — reusing the existing rotated-code coordinate /
stabilizer convention (data on (odd,odd); placement via shift; orientation via
transpose_coords). See docs/superpowers/specs/2026-06-26-rotated-bent-xz-generator-design.md.
"""

from dataclasses import dataclass, field

import numpy as np

from .code_patch import RotatedSurfaceCode


@dataclass(frozen=True)
class PatchSpec:
    """A logical patch that contributes one measured logical to the joint.

    origin:           bus-facing corner data coord, in the library (1,1)-corner convention.
    distance:         odd code distance.
    measured_logical: "X" | "Z" — which logical this patch contributes to the joint.
    orientation:      "X_horizontal" (X̄ runs along a row) | "X_vertical" (X̄ runs up a column).
    """
    name: str
    origin: tuple
    distance: int
    measured_logical: str
    orientation: str


def place_patch(spec):
    """Build one rotated patch placed/oriented per ``spec``; return coord-keyed dicts.

    Returns dict(data=set[(col,row)], checks=list[checkdict], x_support, z_support).
    Reuses RotatedSurfaceCode: build at the (1,1) corner, transpose for X_horizontal,
    then shift so the corner lands at ``origin``.
    """
    if spec.orientation not in ("X_horizontal", "X_vertical"):
        raise ValueError(f"orientation must be X_horizontal|X_vertical, got {spec.orientation!r}")
    code = RotatedSurfaceCode(distance=spec.distance)
    if spec.orientation == "X_horizontal":
        code.transpose_coords()                       # default X̄ vertical -> horizontal
    code.shift_coords(spec.origin[0] - 1, spec.origin[1] - 1)
    qc = code.qubit_coords
    data = {tuple(qc[i]) for i in code.data_indices}
    checks = []
    for s in code.stabilizers:
        pauli = {tuple(qc[q]): P for q, P in s["pauli"].items()}
        checks.append({"syn": tuple(s["syn_coord"]), "type": s["type"],
                       "pauli": pauli, "corners": sorted(pauli)})
    x_support = next(sorted(tuple(qc[q]) for q in lo["pauli"])
                     for lo in code.logical_ops if lo["type"] == "X")
    z_support = next(sorted(tuple(qc[q]) for q in lo["pauli"])
                     for lo in code.logical_ops if lo["type"] == "Z")
    return dict(data=data, checks=checks, x_support=x_support, z_support=z_support)


# -----------------------------------------------------------------------------
# Bent (XZ) joint-measurement layout
# -----------------------------------------------------------------------------

_FLIP = {"X": "Z", "Z": "X"}


class BentLayoutError(ValueError):
    """Raised when the requested patch placement cannot host a bent XZ joint measurement
    (e.g. the two patches are misaligned, or their parity/seam endpoints are incompatible).
    The message states the concrete geometric reason — it is NOT a generator hard-coding limit."""


@dataclass
class BentLayout:
    """A rotated bent (XZ) joint-measurement layout.

    Attributes
    ----------
    distance:  the code distance ``d`` of the two participating patches.
    data:      sorted list of data-qubit coords ``(col, row)`` (data on (odd,odd)).
    checks:    list of stabilizer dicts ``{"syn", "type", "pauli", "corners"}`` where
               ``type`` is "X" | "Z" | "M" (mixed XZ domain-wall) and ``pauli`` maps
               each corner coord to its Pauli letter.
    x_logical: the X-bar support of patch 1 (bus-facing boundary representative).
    z_logical: the Z-bar support of patch 2 (bus-facing boundary representative).
    """
    distance: int
    data: list
    checks: list
    x_logical: list
    z_logical: list
    readout_chain: set = field(default_factory=set)
    specs: list = field(default_factory=list)

    def build_circuit(self, rounds=None, p=0.0):
        """No-MPP gate-level syndrome-extraction circuit for this layout (X-memory experiment)."""
        from .bent_joint_se import RotatedBentJointMeasurement
        if rounds is None:
            rounds = self.distance
        return RotatedBentJointMeasurement(list(self.data), self.checks,
                                           list(self.x_logical)).circuit(rounds=rounds, p=p)

    def verify(self, rounds=None):
        """Run the eight acceptance checks; returns a dict of booleans (all True == valid)."""
        sv, n = _symplectic(self.data)
        S = [sv(ch["pauli"]) for ch in self.checks]
        comm = lambda a, b: int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0
        commute = all(comm(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S)))
        twist = any(P == "Y" for ch in self.checks for P in ch["pauli"].values())
        vXZ = sv({**{c: "X" for c in self.x_logical}, **{c: "Z" for c in self.z_logical}})
        vX = sv({c: "X" for c in self.x_logical})
        vZ = sv({c: "Z" for c in self.z_logical})
        out = dict(commute=commute, joint=_in_span(S, vXZ),
                   no_single=not _in_span(S, vX) and not _in_span(S, vZ),
                   no_twist=not twist, one_logical=len(self.data) - _gf2_rank(S) == 1)
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


def _arrange(p1, p2, d):
    """Validate the two PLACED patches form a routed bent L and return ``(connector, cut, k, m)``.

    Geometry (coordinate-aware): a horizontal Z-side **band** spans ``p1.left .. p2.right`` across
    ``p2``'s rows (the Z-patch is its right end); a **vertical bus column** at ``p1``'s columns runs
    down from the band to the X-patch (``p1``, its bottom end).  Both segments may be minimal or
    **extended/routed**:
      * horizontal: ``hgap = p2.left - p1.right = 2 + 2k`` (``k>=0``; ``k=0`` minimal, larger routes
        the Z-patch further right — a longer band);
      * vertical: ``vgap = p1.bottom - p2.top = 2 + 2m`` (``m>=0``; ``m=0`` minimal, larger routes
        the X-patch further down — a taller bus column).
    The mixed seam stays at the X-arm width either way (``#mixed = d`` for any ``k, m``).

    Raises :class:`BentLayoutError` with the concrete reason only when geometrically impossible:
    a gap that is zero/negative (overlap) or **odd** (the patches sit on incompatible odd-lattice
    parities).  Any common translation of a valid placement is also valid.
    """
    p1c = sorted({c for c, _ in p1["data"]}); p1r = sorted({r for _, r in p1["data"]})
    p2c = sorted({c for c, _ in p2["data"]}); p2r = sorted({r for _, r in p2["data"]})
    hgap = p2c[0] - p1c[-1]
    vgap = p1r[0] - p2r[-1]
    reasons = []
    if hgap < 2 or hgap % 2 != 0:
        reasons.append(f"horizontal gap p2.left-p1.right = {hgap}, expected positive even 2+2k "
                       f"(2 = minimal, 2+2k = extended band)")
    if vgap < 2 or vgap % 2 != 0:
        reasons.append(f"vertical gap p1.bottom-p2.top = {vgap}, expected positive even 2+2m "
                       f"(2 = minimal, 2+2m = extended bus column)")
    if reasons:
        raise BentLayoutError("patches cannot host a bent XZ seam: " + "; ".join(reasons)
                              + ". (Any common translation of a valid placement is also valid.)")
    band = {(c, r) for c in range(p1c[0], p2c[-1] + 1, 2) for r in p2r}        # horizontal Z band
    vcol = {(c, r) for c in p1c for r in range(p2r[-1] + 2, p1r[0], 2)}        # vertical bus to the X-arm
    cut = (p1c[0], p2r[0])                    # band corner opposite the bus
    return band | vcol, cut, (hgap - 2) // 2, (vgap - 2) // 2


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


def _check_vec(check, idx, n):
    v = np.zeros(2 * n, np.uint8)
    for c, P in check["pauli"].items():
        if P in ("X", "Y"):
            v[idx[c]] ^= 1
        if P in ("Z", "Y"):
            v[n + idx[c]] ^= 1
    return v


def _commute(a, b, n):
    return int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0


def _gf2_independent(rows, v):
    """True iff ``v`` is NOT in the GF(2) span of ``rows`` (rows assumed independent)."""
    M = np.array([r.copy() for r in rows] + [v.copy()], np.uint8)
    r = 0
    for c in range(M.shape[1]):
        piv = next((k for k in range(r, len(M)) if M[k, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for k in range(len(M)):
            if k != r and M[k, c]:
                M[k] ^= M[r]
        r += 1
    return r == len(rows) + 1


def _select_checks(data, plaqs, preserve):
    """Maximal commuting, GF(2)-independent, logical-preserving stabilizer set.

    All weight>=3 plaquettes (bulk + weight-3 corner + mixed seam) are forced.  The weight-2
    boundary plaquettes are then added greedily (ascending syn) only if they commute with, are
    independent of, and do NOT pull ANY logical in ``preserve`` into the stabilizer span.
    ``preserve`` is a list of symplectic vectors (one per logical that must survive).  Requiring
    the logicals to survive fixes the boundary-phase tiebreak with no position-specific heuristic,
    at any patch placement and for any number of patches.
    """
    sv, n = _symplectic(data)
    forced = [ch for ch in plaqs if len(ch["pauli"]) >= 3]
    rest = sorted((ch for ch in plaqs if len(ch["pauli"]) == 2), key=lambda c: c["syn"])

    kept = list(forced)
    rows = [sv(ch["pauli"]) for ch in forced]
    for ch in rest:
        v = sv(ch["pauli"])
        if not all(_commute(v, r, n) for r in rows):
            continue
        if not _gf2_independent(rows, v):
            continue
        if any(_in_span(rows + [v], L) for L in preserve):
            continue
        rows.append(v)
        kept.append(ch)
    return kept


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


def _gf2_rank(rows):
    if not len(rows):
        return 0
    M = np.array([r.copy() for r in rows], np.uint8)
    r = 0
    for c in range(M.shape[1]):
        piv = next((k for k in range(r, len(M)) if M[k, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for k in range(len(M)):
            if k != r and M[k, c]:
                M[k] ^= M[r]
        r += 1
    return r


def _in_span(rows, t):
    return _gf2_rank(list(rows) + [t]) == _gf2_rank(list(rows))


def _readout_chain(data, checks, x_logical, z_logical):
    """The set of check syndromes whose GF(2) product == the joint logical X̄₁·Z̄₂."""
    from ....utils.linear_algebra import solve_linear_decomposition
    sv, _ = _symplectic(data)
    basis = np.array([sv(ch["pauli"]) for ch in checks], np.uint8)
    target = sv({**{c: "X" for c in x_logical}, **{c: "Z" for c in z_logical}}).reshape(1, -1)
    co, dep, _ = solve_linear_decomposition(basis=basis, targets=target, reduce_weight=True)
    if not dep[0]:
        return set()
    return {checks[i]["syn"] for i in np.where(co[0])[0]}


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


def _algebraic_valid(data, checks, x_logical, z_logical):
    """The five algebraic acceptance checks (commute / joint / no-single / no-twist / one-logical)."""
    sv, n = _symplectic(data)
    S = [sv(ch["pauli"]) for ch in checks]
    if any(not _commute(S[i], S[j], n) for i in range(len(S)) for j in range(i + 1, len(S))):
        return False
    if any(P == "Y" for ch in checks for P in ch["pauli"].values()):
        return False
    vXZ = sv({**{c: "X" for c in x_logical}, **{c: "Z" for c in z_logical}})
    vX = sv({c: "X" for c in x_logical})
    vZ = sv({c: "Z" for c in z_logical})
    if not _in_span(S, vXZ) or _in_span(S, vX) or _in_span(S, vZ):
        return False
    return len(data) - _gf2_rank(S) == 1


def build_rotated_bent_xz_layout(specs, bend="auto"):
    """Build a rotated bent (XZ) joint-measurement layout from the two logical patches.

    ``specs`` is ``[p1, p2]`` where ``p1`` (``measured_logical="X"``) contributes X̄₁ and
    ``p2`` (``measured_logical="Z"``) contributes Z̄₂.  The geometry is generated **from the
    patches' actual ``origin``s**: both are placed as real ``distance``×``distance`` rotated
    patches, the corner bus connecting them is derived from their positions, and the data /
    CSS + mixed (XZ) stabilizers / readout chain follow.  Changing an origin moves and
    regenerates the whole layout.

    Raises :class:`BentLayoutError` (with a concrete geometric reason) if the requested
    placement cannot host a bent XZ seam — it does not silently fall back to a fixed layout.
    """
    if bend != "auto":
        raise NotImplementedError(f"only bend='auto' is currently supported, got {bend!r}")
    p1_spec, p2_spec = specs
    d = p1_spec.distance
    if p2_spec.distance != d:
        raise ValueError("both patches must share the same distance")
    if (p1_spec.measured_logical, p2_spec.measured_logical) != ("X", "Z"):
        raise ValueError("expected specs = [X-patch (measured_logical='X'), Z-patch ('Z')]")

    p1 = place_patch(p1_spec)
    p2 = place_patch(p2_spec)
    connector, cut, _k, _m = _arrange(p1, p2, d)      # _k/_m = extra horizontal/vertical bus length
    data = sorted((set(p1["data"]) | set(p2["data"]) | connector) - {cut})
    retype = set(p1["data"])                          # the X-arm is the re-typed side
    x_logical = sorted(p1["x_support"])
    z_logical = sorted(p2["z_support"])

    sv, _ = _symplectic(data)
    preserve = [sv({c: "X" for c in x_logical}), sv({c: "Z" for c in z_logical})]
    for phase in (0, 1):                               # phase absorbs the position-dependent parity
        plaqs = _bent_plaquettes(data, retype, phase)
        checks = _select_checks(data, plaqs, preserve)
        if _algebraic_valid(data, checks, x_logical, z_logical):
            readout = _readout_chain(data, checks, x_logical, z_logical)
            return BentLayout(distance=d, data=data, checks=checks, x_logical=x_logical,
                              z_logical=z_logical, readout_chain=readout, specs=list(specs))

    raise BentLayoutError(
        "could not assemble a valid single-logical stabilizer code for these patches "
        "(seam endpoints / parity incompatible with a bent XZ domain wall).")
