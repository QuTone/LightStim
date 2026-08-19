"""The four-row merge rule table as production code (single-seam Phase 1).

Includes the Handbook Sec. 10.4 alternating-spacing end-cap rule in
``wall_spec``.

For a joint Pauli-product measurement between two cell-adjacent patches, the
merge construction is chosen by (measurement kind x patch types):

====  ==========  =========  =============================================
row   measured    types      seam construction
====  ==========  =========  =============================================
1     same Pauli  same       plain merge (checkerboard continues; the seam
                             line is DATA) — no stretched stabilizer
2     same Pauli  different  stretched-stabilizer wall, uniform dominoes
                             (FIG1 family)
3     mixed       different  stretched-stabilizer wall, mixed dominoes
                             (FIG6 family)
4     mixed       same       recoloured single-cell mixed checks — no
                             stretched stabilizer (corridor mechanism)
====  ==========  =========  =============================================

VOCABULARY: **textbook/conjugate = the weight-2
lobe POSITIONS** of a patch — colours never enter the type.  The red/blue
**colours** (checkerboard phase) are a separate fact; ``rotate_90`` swaps
colours and keeps positions, ``litinski`` moves positions and keeps colours.

THE UNIFIED WALL LAW (machine-verified byte-identical to all four enumerated
wall cases at d = 3 and 5, both axes): the wall's dominoes, orient
alternation, relay side and end lobe are ONE closed form of (seam axis, the
two patches' checkerboard phases, d, origin) — see ``_wall_checks``.  A
domino's feet on each side carry that side's outward virtual-plaquette
colour; the flag aux A sits on the Z-foot side; ``orient = '+'`` iff A is on
the hi side; the relay side and end-lobe end follow the far side's colour at
the first interior column.

HARD INVARIANTS: kf metadata and coord-keyed paulis are ABSOLUTE coordinates
(``shift_coords`` / ``_translate_record`` do not translate them), so a
``WallSpec`` is generated at the live coordinates and the coupler must be
registered at offset (0, 0).  Multi-patch rule: the type class with FEWER
patches carries the stretched stabilizers.
"""
from dataclasses import dataclass

from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode

from .placement import RotatedSurfacePPMLayoutError

_FLIP = {'X': 'Z', 'Z': 'X'}
_FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}

TEXTBOOK, CONJUGATE = 'textbook', 'conjugate'


def _plaquette_type(cx, cy, phase=0):
    """Checkerboard type at an even-coordinate plaquette center."""
    return 'X' if (((cx + cy) // 2) + phase) % 2 else 'Z'


class SeamRuleError(RotatedSurfacePPMLayoutError):
    """A seam violates the rule table (wrong orientation, unreadable patch,
    or a live geometry that contradicts the type model)."""


# -----------------------------------------------------------------------------
# SECTION: live probes — positions (type), colours (phase), orientation
# -----------------------------------------------------------------------------

def lobe_positions(system, name):
    """Absolute syn coords of the patch's ACTIVE weight-2 checks (its type
    signature; sample between merges — paused facing lobes are invisible)."""
    return frozenset(
        tuple(int(round(v)) for v in system.stabilizers[u]['syn_coord'])
        for u in system.active_stabilizer_indices
        for s in [system.stabilizers[u]]
        if s.get('patch_name') == name and len(s['data_indices']) == 2)


def _reference_lobes(d):
    """Local weight-2 positions of the two types (origin-(1,1) frame):
    textbook = a plain RotatedSurfaceCode, conjugate = its transpose."""
    code = RotatedSurfaceCode(distance=d)
    textbook = frozenset(
        tuple(int(round(v)) for v in s['syn_coord'])
        for s in code.stabilizers if len(s['pauli']) == 2)
    return textbook, frozenset((y, x) for x, y in textbook)


def patch_type(lobes, origin, d):
    """TEXTBOOK or CONJUGATE from absolute weight-2 positions (colours are
    irrelevant by definition)."""
    ox, oy = origin
    local = frozenset((x - ox + 1, y - oy + 1) for x, y in lobes)
    textbook, conjugate = _reference_lobes(d)
    if local == textbook:
        return TEXTBOOK
    if local == conjugate:
        return CONJUGATE
    raise SeamRuleError(
        f"patch at origin {origin} has weight-2 positions {sorted(local)} "
        f"matching neither the textbook nor the conjugate layout for d={d}")


def detect_layout(system, name, d, origin):
    """(lr_type, tb_type, phase) of the live patch — boundary colours and the
    checkerboard phase, read off its records."""
    ox, oy = origin
    lr = tb = None
    bulk_type = None
    for uid in sorted(system.active_stabilizer_indices):
        st = system.stabilizers[uid]
        if st.get('patch_name') != name:
            continue
        sx, sy = (int(round(st['syn_coord'][0])),
                  int(round(st['syn_coord'][1])))
        if len(st['data_indices']) == 2:
            if sx < ox or sx > ox + 2 * d - 2:
                lr = st['type']
            elif sy < oy or sy > oy + 2 * d - 2:
                tb = st['type']
        elif len(st['data_indices']) == 4 and bulk_type is None:
            bulk_type = (sx, sy, st['type'])
    if not (lr and tb and bulk_type):
        raise SeamRuleError(f"could not read the live layout of {name!r}")
    sx, sy, t = bulk_type
    phase = 0 if t == _plaquette_type(sx, sy, 0) else 1
    return lr, tb, phase


@dataclass(frozen=True)
class PatchView:
    """Everything the rule table needs to know about one live patch.
    ``type`` is POSITIONS (textbook/conjugate); ``phase``/``lr``/``tb`` are
    COLOURS; ``orientation`` is the live logical orientation."""
    name: str
    origin: tuple
    distance: int
    lobes: frozenset
    phase: int
    lr: str
    tb: str
    orientation: str

    @property
    def type(self):
        return patch_type(self.lobes, self.origin, self.distance)

    @property
    def cell(self):
        d = self.distance
        return (self.origin[0] // (2 * d + 2), self.origin[1] // (2 * d + 2))


def patch_view(system, name):
    """Build a :class:`PatchView` from the live system (origin and d derived
    from the patch's data qubits; orientation from the X̄ logical record)."""
    owner = system.index_to_owner_map
    coords = [tuple(int(round(v)) for v in system.qubit_coords[q])
              for q in system.data_indices if owner.get(q) == name]
    if not coords:
        raise SeamRuleError(f"patch {name!r} has no data qubits in the system")
    xs = sorted({x for x, _ in coords})
    ys = sorted({y for _, y in coords})
    d = len(xs)
    if len(ys) != d:
        raise SeamRuleError(
            f"patch {name!r} is not square ({len(xs)}x{len(ys)})")
    origin = (xs[0], ys[0])
    lr, tb, phase = detect_layout(system, name, d, origin)
    # X̄ connects the two X boundaries: lr == 'X' means X̄ runs horizontally
    orientation = 'X_horizontal' if lr == 'X' else 'X_vertical'
    return PatchView(name=name, origin=origin, distance=d,
                     lobes=lobe_positions(system, name), phase=phase,
                     lr=lr, tb=tb, orientation=orientation)


# -----------------------------------------------------------------------------
# SECTION: classification (the rule table) and the multi-patch wall selector
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeRule:
    row: int                  # 1..4
    construction: str         # 'merge' | 'wall'
    same_pauli: bool
    same_type: bool
    axis: str                 # 'vertical' | 'horizontal' seam line direction
    reason: str


def _seam_axis(a, b):
    """The seam LINE direction between two cell-adjacent views ('vertical' =
    patches side by side, 'horizontal' = stacked).  Raises unless adjacent,
    aligned and equal distance."""
    if a.distance != b.distance:
        raise SeamRuleError(
            f"{a.name} (d={a.distance}) and {b.name} (d={b.distance}) differ")
    (ax, ay), (bx, by) = a.cell, b.cell
    if abs(ax - bx) + abs(ay - by) != 1:
        raise SeamRuleError(
            f"{a.name} at cell {a.cell} and {b.name} at cell {b.cell} are "
            f"not cell-adjacent — this seam API is for direct adjacency; "
            f"distant patches go through the routed corridor")
    return 'vertical' if ay == by else 'horizontal'


def _check_parallel(view, pauli, axis):
    """The measured logical must run PARALLEL to the seam line."""
    direction = ('horizontal' if (view.orientation == 'X_horizontal')
                 == (pauli == 'X') else 'vertical')
    if direction != axis:
        raise SeamRuleError(
            f"{view.name}: the measured {pauli}̄ runs {direction} but the "
            f"seam line is {axis} — the measured logical must run PARALLEL "
            f"to the seam (rotate the patch or re-place it)")


def classify_seam(a, pa, b, pb):
    """The rule-table row for measuring ``pa ⊗ pb`` across the seam between
    cell-adjacent patches ``a`` and ``b`` (:class:`PatchView` each)."""
    axis = _seam_axis(a, b)
    _check_parallel(a, pa, axis)
    _check_parallel(b, pb, axis)
    same_pauli = (pa == pb)
    same_type = (a.type == b.type)
    # self-check: different types <=> the facing lobes ALIGN on the same
    # along-seam positions (same type <=> interleaved) — verified truth table
    near, far = (a, b) if a.cell <= b.cell else (b, a)
    k = 0 if axis == 'horizontal' else 1   # along-seam coordinate index
    facing_near = {q[k] for q in near.lobes
                   if q[1 - k] == near.origin[1 - k] + 2 * near.distance - 1}
    facing_far = {q[k] for q in far.lobes
                  if q[1 - k] == far.origin[1 - k] - 1}
    aligned = facing_near == facing_far
    if aligned == same_type:
        raise SeamRuleError(
            f"type model and live geometry disagree on the {a.name}-{b.name} "
            f"seam: types {'equal' if same_type else 'differ'} but facing "
            f"lobes {'align' if aligned else 'interleave'}")
    row = {(True, True): 1, (True, False): 2,
           (False, False): 3, (False, True): 4}[(same_pauli, same_type)]
    construction = 'wall' if row in (2, 3) else 'merge'
    reason = (f"row {row}: {'same' if same_pauli else 'mixed'} Pauli + "
              f"{'same' if same_type else 'different'} type -> {construction}")
    return MergeRule(row=row, construction=construction, same_pauli=same_pauli,
                     same_type=same_type, axis=axis, reason=reason)


def assign_walls(views, target, seams):
    """Which patches' seams carry the stretched stabilizers: group the target
    patches by TYPE and pick the class with FEWER patches (e.g. 1 textbook +
    3 conjugate -> the textbook patch's seams get the walls).  Tie (2-2) ->
    the lexicographically smaller name set.  Returns {seam: bool} — a seam
    carries a wall iff it touches a selected patch."""
    types = {nm: views[nm].type for nm, _ in target}
    tb = sorted(nm for nm, t in types.items() if t == TEXTBOOK)
    cj = sorted(nm for nm, t in types.items() if t == CONJUGATE)
    if not tb or not cj:
        return {seam: False for seam in seams}
    if len(tb) != len(cj):
        chosen = set(min((tb, cj), key=len))
    else:
        chosen = set(min(tb, cj))           # deterministic 2-2 tiebreak
    return {seam: bool(set(seam) & chosen) for seam in seams}


# -----------------------------------------------------------------------------
# SECTION: the unified wall generator + the seam-wall coupler
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SeamFrame:
    axis: str          # seam LINE direction: 'vertical' | 'horizontal'
    near: int          # facing data line of the near patch (normal coord)
    lo: int
    seam: int
    hi: int
    far: int           # facing data line of the far patch
    span: tuple        # (first, last) along-seam plaquette coordinate


def _wall_checks(d, phi_near, phi_far, x0, y0, axis, sgn_override=None):
    """The unified wall law.  ``(x0, y0)`` = the NEAR patch's data origin;
    ``axis`` = 'horizontal' when the seam line runs horizontally (patches
    stacked, dominoes vertical) — matching :func:`_seam_axis`'s naming.

    Byte-identical to the four hand-specified wall cases at d = 3 and 5.
    Returns records in the form
    [(B, {foot: Pauli}, {'flag': A, 'shared': S, 'orient': '+'|'-'})]."""
    # P maps (along, normal) -> (x, y); for a horizontal seam the normal is y
    P = (lambda a, n: (a, n)) if axis == 'horizontal' else (lambda a, n: (n, a))
    a0, n0 = (x0, y0) if axis == 'horizontal' else (y0, x0)
    near = n0 + 2 * (d - 1)
    lo, seam, hi, far = near + 1, near + 2, near + 3, near + 4
    cs = [a0 - 1 + k for k in range(0, 2 * d + 1, 2)]
    inter = cs[1:-1]

    def cn(c):
        return _plaquette_type(*P(c, lo), phi_near)

    def cf(c):
        return _plaquette_type(*P(c, hi), phi_far)

    # cap end: the alternating-spacing rule (Handbook Sec. 10.4) decides —
    # wall_spec passes it in via sgn_override.  The far-side-colour proxy
    # below coincides with the slot rule on the four legacy families but
    # diverges on transposed row-3 (cap crammed next to existing corner
    # lobes, three checks fighting one corner -> 50% random detectors).
    sgn = (sgn_override if sgn_override is not None
           else (1 if cf(inter[0]) == 'Z' else -1))
    out = []
    for c in inter:
        t, b = cf(c), cn(c)
        A, B = (P(c, hi), P(c, lo)) if t == 'Z' else (P(c, lo), P(c, hi))
        out.append((B, {P(c - 1, far): t, P(c + 1, far): t,
                        P(c - 1, near): b, P(c + 1, near): b},
                    {'flag': A, 'shared': P(c + sgn, seam),
                     'orient': '+' if A == P(c, hi) else '-'}))
    e = cs[0] if sgn == 1 else cs[-1]
    ca = e + 2 * sgn
    tl, bl = _FLIP[cf(ca)], _FLIP[cn(ca)]
    A, B = (P(e, hi), P(e, lo)) if tl == 'Z' else (P(e, lo), P(e, hi))
    out.append((B, {P(e + sgn, far): tl, P(e + sgn, near): bl},
                {'flag': A, 'shared': P(e + sgn, seam),
                 'orient': '+' if A == P(e, hi) else '-'}))
    return out


@dataclass(frozen=True)
class WallSpec:
    checks: tuple      # legacy records ((B, {foot: P}, kf), ...) — ABSOLUTE coords
    retire: frozenset  # facing weight-2 syn coords to pause at merge
    frame: SeamFrame = None

    @staticmethod
    def from_records(records, retire):
        """A WallSpec from a legacy record list (frozen test specs)."""
        return WallSpec(
            checks=tuple((tuple(B), tuple(sorted(p.items())),
                          tuple(sorted(kf.items())))
                         for B, p, kf in records),
            retire=frozenset(retire))

    def as_list(self):
        return [(B, dict(pauli), dict(kf)) for B, pauli, kf in self.checks]


def seam_frame(a, b):
    axis = _seam_axis(a, b)
    near_view, far_view = (a, b) if a.cell <= b.cell else (b, a)
    k = 0 if axis == 'horizontal' else 1   # along-seam coordinate index
    n0 = near_view.origin[1 - k]
    near = n0 + 2 * (near_view.distance - 1)
    a0 = near_view.origin[k]
    return SeamFrame(axis=axis, near=near, lo=near + 1, seam=near + 2,
                     hi=near + 3, far=near + 4,
                     span=(a0 - 1, a0 - 1 + 2 * near_view.distance))


def wall_spec(a, b, target):
    """The wall between cell-adjacent views ``a`` and ``b`` for the measured
    Paulis ``target = {name: 'X'|'Z'}``.  Verifies the rule table sends this
    seam to a wall (rows 2/3) and that the domino colours are consistent
    with the measured Paulis; raises SeamRuleError otherwise."""
    rule = classify_seam(a, target[a.name], b, target[b.name])
    if rule.construction != 'wall':
        raise SeamRuleError(
            f"the {a.name}-{b.name} seam takes the row-{rule.row} merge, "
            f"not a wall ({rule.reason})")
    near_view, far_view = (a, b) if a.cell <= b.cell else (b, a)
    d = near_view.distance
    frame = seam_frame(a, b)
    # Handbook 10.4 alternating-spacing rule ("add one, skip one gap"): the
    # end cap may only sit at a wall end whose candidate position is at
    # least TWO gaps from every existing boundary lobe on that end row —
    # one gap away means the slot is taken and nothing may be added there.
    k = 0 if rule.axis == 'horizontal' else 1
    a0 = near_view.origin[k]
    end_lo, end_hi = a0 - 1, a0 - 1 + 2 * d
    all_lobes = set(near_view.lobes) | set(far_view.lobes)

    def _blocked(e):
        return any(L[k] == e and
                   min(abs(L[1 - k] - frame.lo),
                       abs(L[1 - k] - frame.hi)) == 2
                   for L in all_lobes)

    free = [s for s, e in ((1, end_lo), (-1, end_hi)) if not _blocked(e)]
    if not free:
        raise SeamRuleError(
            f"the {a.name}-{b.name} wall has NO free end slot for its cap "
            f"(alternating-spacing rule, Handbook Sec. 10.4): both end rows "
            f"already carry a boundary lobe one gap away")
    sgn_override = free[0] if len(free) == 1 else None
    checks = _wall_checks(d, near_view.phase, far_view.phase,
                          near_view.origin[0], near_view.origin[1],
                          rule.axis, sgn_override=sgn_override)
    k = 0 if rule.axis == 'horizontal' else 1
    retire = frozenset(
        q for v in (near_view, far_view) for q in v.lobes
        if q[1 - k] in (frame.lo, frame.hi))
    return WallSpec(checks=tuple((B, tuple(sorted(pauli.items())), tuple(sorted(kf.items())))
                                 for B, pauli, kf in
                                 ((B, p, kf) for B, p, kf in checks)),
                    retire=retire, frame=frame)


def _wallspec_records(spec):
    """checks back as mutable (B, pauli_dict, kf_dict) records."""
    return [(B, dict(pauli), dict(kf)) for B, pauli, kf in spec.checks]


# -----------------------------------------------------------------------------
# SECTION: the seam-wall coupler (production twin of the test harness classes)
# -----------------------------------------------------------------------------

from lightstim.ir.coupler import LogicalCouplerProtocol     # noqa: E402


class RotatedSeamWallCoupler(LogicalCouplerProtocol):
    """Registers a :class:`WallSpec` as an IR coupler: MIXED+kf stabilizer
    records at ABSOLUTE coordinates (register at offset (0, 0) only), the
    three-qubit apparatus per check (flag -> syndrome_z, shared -> syndrome_z,
    B -> syndrome_x; coords already owned by a patch are skipped), and the
    facing lobes in ``conflicting_stabilizer_coords`` so ``activate_coupler``
    pauses them at merge.  Merged rounds MUST use
    ``DiagonalSurfaceCodeExtractionBlock`` (the bent chunk raises on kf)."""
    EXPECTED_PATCH_COUNT = 2

    def _build_coupler_geometry(self, coupler_patch, patches, *, spec,
                                occupied=frozenset()):
        seen = set(occupied)
        for B, pauli, kf in _wallspec_records(spec):
            for q, role in ((B, 'syndrome_x'), (kf['flag'], 'syndrome_z'),
                            (kf['shared'], 'syndrome_z')):
                if q in seen:
                    continue
                seen.add(q)
                coupler_patch.add_qubit(q[0], q[1], role=role)
            coupler_patch.stabilizers.append(
                {'pauli': dict(pauli), 'type': 'MIXED', 'syn_coord': B,
                 'kf': dict(kf)})
        coupler_patch.conflicting_stabilizer_coords = set(spec.retire)
        coupler_patch.wall_spec = spec
        coupler_patch.num_logicals = 0
