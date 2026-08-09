"""Closed-form stage layouts of the Litinski patch rotation (Fig 45 steps 4→9).

Every stage boundary comes from ONE rule, ``rect_boundary``: on each edge of a
rectangular footprint, weight-2 lobes sit exactly where the outward virtual
plaquette's checkerboard type equals that edge's boundary type.  Verified
check-by-check against 800/1600-dpi crops of arXiv:1808.02892 Fig 45
(tests/test_grow_patch.py, test_move_corners.py, test_shrink_patch.py,
test_litinski_rotation.py), and cross-validated for rule generality against
Kishony-Fowler's diagonal-scheduling paper Fig 5 (their 2d+1 frame).

Stage map (bar-local y-up coords; the bar is (2d-1) rows × d cols, the original
patch occupies the TOP d rows and the rotation slides it to the BOTTOM and back):

* start (panels 1/4)   = ``rect_boundary(d, d, 'X', 'Z')``  — X̄ horizontal;
* grown (panel 5)      = builder completion of ``grown_forced`` (mixed
  convention: kept start checks + the far-end corner-pair structure);
* moved (panel 6)      = grown with the left long edge flipped to Z;
* shrunk (panel 7)     = ``rect_boundary(d, d, 'Z', 'X')``  — ROTATED, at the
  bar's bottom d×d;
* bar-2 (panel 8)      = ``rect_boundary(d, 2d-1, 'Z', 'X')`` — the rotated
  convention extended upward (convention-preserving, no corner pair);
* final (panel 9)      = ``rect_boundary(d, d, 'Z', 'X')`` again, at the TOP
  tile — identical local layout to panel 7, translated back home.
"""
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch


def ptype(cx, cy, phase=0):
    return 'X' if (((cx + cy) // 2) + phase) % 2 else 'Z'


def rect_boundary(d_z, d_x, lr, tb, phase=0):
    """Closed-form boundary of a ``d_z``-col × ``d_x``-row rectangular patch whose
    left/right edges carry type ``lr`` and top/bottom edges type ``tb``: lobes sit
    exactly where the outward virtual plaquette's checkerboard type matches."""
    w2 = {}
    for cy in range(2, 2 * d_x, 2):
        for cx, sx in ((0, 1), (2 * d_z, 2 * d_z - 1)):
            if ptype(cx, cy, phase) == lr:
                w2[(cx, cy)] = (lr, [(sx, cy - 1), (sx, cy + 1)])
    for cx in range(2, 2 * d_z, 2):
        for cy, sy in ((0, 1), (2 * d_x, 2 * d_x - 1)):
            if ptype(cx, cy, phase) == tb:
                w2[(cx, cy)] = (tb, [(cx - 1, sy), (cx + 1, sy)])
    return w2


def start_w2(d, phase=0):
    """Paper-q2 orientation d×d start boundary (Fig 45 panels 1/4)."""
    return rect_boundary(d, d, 'X', 'Z', phase)


def grown_forced(d, phase=0):
    """Anchors for the grown bar (panel 5): the start's non-growth-edge checks
    shifted to the top of the bar frame; the rule sweep completes the rest."""
    dy = 2 * (d - 1)
    out = {}
    for sc, (t, sup) in start_w2(d, phase).items():
        if sc[1] == 0:                       # growth-edge (bottom) lobes drop
            continue
        out[(sc[0], sc[1] + dy)] = (t, sorted((x, y + dy) for x, y in sup))
    return out


def grown_patch(d, phase=0):
    """The grown (2d-1)×d reference geometry (== Fig 45 panel 5)."""
    return RuleBasedRectPatch(distance_x=2 * d - 1, distance_z=d,
                              forced=grown_forced(d, phase), phase=phase)


def _w2_of(patch):
    out = {}
    for s in patch.stabilizers:
        if len(s['data_indices']) == 2:
            sc = tuple(int(round(v)) for v in s['syn_coord'])
            out[sc] = (s['type'], sorted(tuple(int(round(v)) for v in patch.qubit_coords[i])
                                         for i in s['data_indices']))
    return out


def moved_w2(d, phase=0):
    """Panel-6 boundary: the grown bar's w2 with the left long edge flipped to Z."""
    out = {sc: v for sc, v in _w2_of(grown_patch(d, phase)).items() if sc[0] != 0}
    for cy in range(2, 2 * (2 * d - 1), 2):
        if ptype(0, cy, phase) == 'Z':
            out[(0, cy)] = ('Z', [(1, cy - 1), (1, cy + 1)])
    return out


def shrunk_w2(d, phase=0):
    """Panel-7 (and panel-9) boundary: the ROTATED d×d convention."""
    return rect_boundary(d, d, 'Z', 'X', phase)


def bar2_w2(d, phase=0):
    """Panel-8 boundary: the rotated convention on the full (2d-1)×d bar."""
    return rect_boundary(d, 2 * d - 1, 'Z', 'X', phase)


def grown_w2_full(d, phase=0):
    """The COMPLETE grown-bar boundary (panel 5), as the rules complete it."""
    return _w2_of(grown_patch(d, phase))


def transpose_w2(w2):
    """Transpose a boundary dict (x <-> y).  ``ptype`` is symmetric under
    transposition, so the checkerboard phase is unchanged, and Pauli letters
    are untouched (coordinate relabeling).  Identity used by the ±x growth
    variants: transpose_w2(rect_boundary(a, b, lr, tb, ph))
    == rect_boundary(b, a, tb, lr, ph)."""
    return {(cy, cx): (t, sorted((y, x) for x, y in sup))
            for (cx, cy), (t, sup) in w2.items()}


def flip_w2(w2, Y):
    """Mirror a boundary dict vertically (y → Y − y).  Used to build the stage
    geometries for a +y growth direction (e.g. stim-rendering frames, which draw
    y downward): the completed layouts are flipped and passed fully-forced, because
    the rule sweep's lexicographic order is not mirror-symmetric."""
    return {(cx, Y - cy): (t, sorted((x, Y - y) for x, y in sup))
            for (cx, cy), (t, sup) in w2.items()}
