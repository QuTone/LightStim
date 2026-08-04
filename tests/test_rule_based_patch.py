"""Rule-based patch builder vs the two published grown-patch layouts.

Litinski (arXiv:1808.02892) Fig 45 step 5, read check-by-check off 800/1600-dpi
crops: the grown 3×5 bar keeps q2's three non-growth-edge boundary checks verbatim
and carries the same-type corner pair on the bottom-left (X) corner — X@(0,2) and
X@(2,0) sharing corner data qubit (1,1) — with a single Z@(6,2) bottom-right.

Kishony-Fowler (diagonal-scheduling paper) Fig 5b, extracted from the PDF's vector
graphics: d=5, conjugate orientation (X̄ vertical), grows RIGHT through the Z
boundary to 5×11 (2d+1), corner pair at the far bottom-right (X) corner, and the
mid-top-edge topological corner pinned to the destination sub-patch's boundary.

Both layouts must emerge from bulk + ``forced`` (kept live checks, plus destination
boundary anchors where the free edge has slack) + the corner/sweep rules — with no
per-paper hints.  The eager per-corner tile selection this file once guarded with a
``pair_type`` parameter is gone: corners already covered by a selected check are not
anchored, which is what makes both papers come out of the same rules.
"""
import numpy as np
import pytest

from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch

pytestmark = pytest.mark.smoke

D = 3
DY = 2 * (D - 1)

# Paper q2 starting boundary, 3×3 local coords, y up (verified against Fig 45 panel 1)
PAPER_Q2_W2 = {
    (0, 2): ('X', [(1, 1), (1, 3)]),   # left, lower half
    (4, 0): ('Z', [(3, 1), (5, 1)]),   # bottom, right half (growth edge -> dropped)
    (6, 4): ('X', [(5, 3), (5, 5)]),   # right, upper half
    (2, 6): ('Z', [(1, 5), (3, 5)]),   # top, left half
}
PAPER_Q2_BULK = {
    (2, 4): ('X', [(1, 3), (3, 3), (1, 5), (3, 5)]),
    (4, 4): ('Z', [(3, 3), (5, 3), (3, 5), (5, 5)]),
    (2, 2): ('Z', [(1, 1), (3, 1), (1, 3), (3, 3)]),
    (4, 2): ('X', [(3, 1), (5, 1), (3, 3), (5, 3)]),
}

# Fig 45 panel 5 ground truth: all 14 checks of the grown bar
PANEL5_BULK = {
    (2, 8): 'X', (4, 8): 'Z', (2, 6): 'Z', (4, 6): 'X',
    (2, 4): 'X', (4, 4): 'Z', (2, 2): 'Z', (4, 2): 'X',
}
PANEL5_W2 = {
    (2, 10): ('Z', [(1, 9), (3, 9)]),
    (6, 8):  ('X', [(5, 7), (5, 9)]),
    (0, 6):  ('X', [(1, 5), (1, 7)]),
    (0, 2):  ('X', [(1, 1), (1, 3)]),   # corner pair member
    (2, 0):  ('X', [(1, 1), (3, 1)]),   # corner pair member
    (6, 2):  ('Z', [(5, 1), (5, 3)]),
}


def _forced():
    out = {}
    for sc, (t, sup) in PAPER_Q2_W2.items():
        if sc == (4, 0):
            continue
        out[(sc[0], sc[1] + DY)] = (t, sorted((x, y + DY) for x, y in sup))
    return out


def _layout(patch):
    out = {}
    for s in patch.stabilizers:
        sc = tuple(int(round(v)) for v in s['syn_coord'])
        sup = sorted(tuple(int(round(v)) for v in patch.qubit_coords[i])
                     for i in s['data_indices'])
        out[sc] = (s['type'], sup)
    return out


def _grown():
    return RuleBasedRectPatch(distance_x=2 * D - 1, distance_z=D,
                              forced=_forced(), phase=0)


def _gf2_rank(M):
    M = M.copy() % 2
    r = 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[[i for i in range(M.shape[0]) if i != r and M[i, c]]] ^= M[r]
        r += 1
    return r


def test_step5_layout_matches_paper():
    lay = _layout(_grown())
    expect = {}
    for (cx, cy), t in PANEL5_BULK.items():
        expect[(cx, cy)] = (t, sorted((cx + dx, cy + dy)
                                      for dx in (-1, 1) for dy in (-1, 1)))
    for sc, (t, sup) in PANEL5_W2.items():
        expect[sc] = (t, sorted(sup))
    assert lay == expect


# ---- Kishony-Fowler Fig 5b (d=5, conjugate orientation, grow right to 2d+1) ----

KF_A_KEPT_W2 = {
    (0, 2):  ('Z', [(1, 1), (1, 3)]),
    (0, 6):  ('Z', [(1, 5), (1, 7)]),
    (2, 10): ('X', [(1, 9), (3, 9)]),
    (6, 10): ('X', [(5, 9), (7, 9)]),
    (4, 0):  ('X', [(3, 1), (5, 1)]),
    (8, 0):  ('X', [(7, 1), (9, 1)]),
}
KF_DEST_TOP_Z = {
    (16, 10): ('Z', [(15, 9), (17, 9)]),
    (20, 10): ('Z', [(19, 9), (21, 9)]),
}
KF_B_W2 = {
    (0, 2): ('Z', [(1, 1), (1, 3)]), (0, 6): ('Z', [(1, 5), (1, 7)]),
    (2, 10): ('X', [(1, 9), (3, 9)]), (6, 10): ('X', [(5, 9), (7, 9)]),
    (10, 10): ('X', [(9, 9), (11, 9)]),
    (16, 10): ('Z', [(15, 9), (17, 9)]), (20, 10): ('Z', [(19, 9), (21, 9)]),
    (4, 0): ('X', [(3, 1), (5, 1)]), (8, 0): ('X', [(7, 1), (9, 1)]),
    (12, 0): ('X', [(11, 1), (13, 1)]), (16, 0): ('X', [(15, 1), (17, 1)]),
    (20, 0): ('X', [(19, 1), (21, 1)]),
    (22, 2): ('X', [(21, 1), (21, 3)]), (22, 6): ('X', [(21, 5), (21, 7)]),
}


def test_kf_fig5b_layout_matches_paper():
    patch = RuleBasedRectPatch(distance_x=5, distance_z=11,
                               forced={**KF_A_KEPT_W2, **KF_DEST_TOP_Z}, phase=1)
    lay = _layout(patch)
    expect = {}
    for cx in range(2, 22, 2):
        for cy in range(2, 10, 2):
            t = 'X' if (((cx + cy) // 2) + 1) % 2 else 'Z'
            expect[(cx, cy)] = (t, sorted((cx + dx, cy + dy)
                                          for dx in (-1, 1) for dy in (-1, 1)))
    for sc, (t, sup) in KF_B_W2.items():
        expect[sc] = (t, sorted(sup))
    assert lay == expect


def test_step5_algebra_and_exhaustive_distance():
    patch = _grown()
    lay = _layout(patch)
    coords = sorted({tuple(int(round(v)) for v in patch.qubit_coords[i])
                     for i in patch.data_indices})
    n = len(coords)
    idx = {q: i for i, q in enumerate(coords)}

    def vec(t, sup):
        v = np.zeros(2 * n, dtype=np.uint8)
        for q in sup:
            v[idx[q] + (0 if t == 'X' else n)] = 1
        return v

    S = np.array([vec(t, sup) for t, sup in lay.values()], dtype=np.uint8)

    def commute(u, v):
        return (int(u[:n] @ v[n:]) + int(u[n:] @ v[:n])) % 2 == 0

    assert all(commute(S[i], S[j])
               for i in range(len(S)) for j in range(i + 1, len(S)))
    assert _gf2_rank(S) == n - 1          # k = 1

    # anchoring: exactly 7 of the 8 original q2 checks survive verbatim
    old = {}
    for sc, (t, sup) in {**PAPER_Q2_W2, **PAPER_Q2_BULK}.items():
        old[(sc[0], sc[1] + DY)] = (t, sorted((x, y + DY) for x, y in sup))
    kept = sum(1 for sc, rec in old.items() if lay.get(sc) == rec)
    assert kept == 7

    # TRUE distance by full GF(2) exhaustion per CSS sector (d = min(dx, dz))
    Hx = np.array([vec(t, sup)[:n] for t, sup in lay.values() if t == 'X'],
                  dtype=np.uint8)
    Hz = np.array([vec(t, sup)[n:] for t, sup in lay.values() if t == 'Z'],
                  dtype=np.uint8)

    def sector_distance(H_other, G_same):
        rk = _gf2_rank(G_same)
        best = None
        for bits in range(1, 1 << n):
            e = np.array([(bits >> i) & 1 for i in range(n)], dtype=np.uint8)
            if best is not None and int(e.sum()) >= best:
                continue
            if ((H_other @ e) % 2).any():
                continue
            if _gf2_rank(np.vstack([G_same, e])) == rk:
                continue
            best = int(e.sum())
        return best

    assert min(sector_distance(Hz, Hx), sector_distance(Hx, Hz)) == D
