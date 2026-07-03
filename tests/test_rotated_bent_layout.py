import json
import pathlib

import numpy as np
import pytest

FIX = pathlib.Path(__file__).parent / "fixtures" / "bent_xz_golden_d3.json"


def load_golden():
    g = json.load(open(FIX))
    data = [tuple(c) for c in g["data"]]
    checks = []
    for ch in g["checks"]:
        pauli = {tuple(int(v) for v in k.split(",")): p for k, p in ch["pauli"].items()}
        checks.append({"syn": tuple(ch["syn"]), "type": ch["type"], "pauli": pauli,
                       "corners": sorted(pauli)})
    return dict(distance=g["distance"], data=data, checks=checks,
                x_logical=[tuple(c) for c in g["x_logical"]],
                z_logical=[tuple(c) for c in g["z_logical"]],
                readout_chain={tuple(c) for c in g["readout_chain"]})


def symplectic(checks, data):
    idx = {c: i for i, c in enumerate(sorted(data))}
    n = len(idx)
    rows = []
    for ch in checks:
        v = np.zeros(2 * n, np.uint8)
        for c, P in ch["pauli"].items():
            if P in ("X", "Y"):
                v[idx[c]] ^= 1
            if P in ("Z", "Y"):
                v[n + idx[c]] ^= 1
        rows.append(v)
    return np.array(rows, np.uint8), idx, n


def gf2_rank(rows):
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


def in_span(rows, t):
    return gf2_rank(list(rows) + [t]) == gf2_rank(list(rows))


def _vec(support_pauli, idx, n):
    v = np.zeros(2 * n, np.uint8)
    for c, P in support_pauli.items():
        if P in ("X", "Y"):
            v[idx[c]] ^= 1
        if P in ("Z", "Y"):
            v[n + idx[c]] ^= 1
    return v


def acceptance(data, checks, x_logical, z_logical):
    """The six algebraic acceptance checks (commute/joint/no-single/no-twist/one-logical/n-mixed)."""
    S, idx, n = symplectic(checks, data)
    def comm(a, b):
        return int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0
    bad = sum(1 for i in range(len(S)) for j in range(i + 1, len(S)) if not comm(S[i], S[j]))
    twist = any(P == "Y" for ch in checks for P in ch["pauli"].values())
    rank = gf2_rank(list(S))
    vXZ = _vec({**{c: "X" for c in x_logical}, **{c: "Z" for c in z_logical}}, idx, n)
    vX = _vec({c: "X" for c in x_logical}, idx, n)
    vZ = _vec({c: "Z" for c in z_logical}, idx, n)
    return dict(commute=bad == 0, joint=in_span(S, vXZ),
                no_single=not in_span(S, vX) and not in_span(S, vZ),
                no_twist=not twist, one_logical=len(data) - rank == 1,
                n_mixed=sum(c["type"] == "M" for c in checks))


def test_golden_passes_acceptance():
    g = load_golden()
    a = acceptance(g["data"], g["checks"], g["x_logical"], g["z_logical"])
    assert a == {"commute": True, "joint": True, "no_single": True,
                 "no_twist": True, "one_logical": True, "n_mixed": 3}


from lightstim.qec_code.surface_code.rotated.bent_layout import PatchSpec, place_patch


def test_place_patch_xhorizontal_d3():
    p = place_patch(PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"))
    assert p["data"] == {(c, r) for c in (1, 3, 5) for r in (7, 9, 11)}
    # X_horizontal => X-bar is a horizontal row; bus-facing (bottom) row 7
    assert sorted(p["x_support"]) == [(1, 7), (3, 7), (5, 7)]
    assert all(ch["type"] in ("X", "Z") and len(ch["pauli"]) in (2, 4) for ch in p["checks"])


def test_place_patch_p2_zpatch_d3():
    p = place_patch(PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal"))
    assert p["data"] == {(c, r) for c in (7, 9, 11) for r in (1, 3, 5)}
    # X_horizontal => Z-bar vertical; bus-facing (left) col 7
    assert sorted(p["z_support"]) == [(7, 1), (7, 3), (7, 5)]


from lightstim.qec_code.surface_code.rotated.bent_layout import build_rotated_bent_xz_layout


def _canon(checks):
    return sorted((c["type"], tuple(sorted(c["pauli"].items()))) for c in checks)


from lightstim.qec_code.surface_code.rotated.bent_layout import BentLayoutError


def test_generator_canonical_d3_valid_and_figure_consistent():
    # The generator uses REAL full d=3 patches, so canonical = the figure-grounded golden
    # PLUS the Z-patch's far column (the golden hand-layout trimmed it). It must (a) pass full
    # verify, (b) carry the same measured logicals + 3 mixed seam checks, (c) contain the golden.
    g = load_golden()
    lay = build_rotated_bent_xz_layout(
        [PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"),
         PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal")])
    assert all(lay.verify().values())
    assert sorted(lay.x_logical) == sorted(g["x_logical"])
    assert sorted(lay.z_logical) == sorted(g["z_logical"])
    assert sum(c["type"] == "M" for c in lay.checks) == 3
    assert set(g["data"]) <= set(lay.data)            # golden is a sub-layout (full p2 adds col 11)


def test_generator_is_coordinate_aware():
    # Changing the patch origins must MOVE/REGENERATE data and checks (not a fixed template).
    A = build_rotated_bent_xz_layout(
        [PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"),
         PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal")])
    B = build_rotated_bent_xz_layout(
        [PatchSpec("p1", (3, 9), 3, "X", "X_horizontal"),     # same config, translated +(2,2)
         PatchSpec("p2", (9, 3), 3, "Z", "X_horizontal")])
    assert list(A.data) != list(B.data)
    assert _canon(A.checks) != _canon(B.checks)
    assert all(A.verify().values()) and all(B.verify().values())


def test_geometrically_impossible_placements_raise():
    # Genuinely impossible placements -> clear geometric reason (NOT a hard-coding failure):
    #   (1,7)/(5,1): zero horizontal gap (overlap);  (1,7)/(8,1): odd horizontal gap (parity);
    #   (1,5)/(7,1): zero vertical gap (X-arm overlaps the band).
    for o1, o2 in [((1, 7), (5, 1)), ((1, 7), (8, 1)), ((1, 5), (7, 1))]:
        with pytest.raises(BentLayoutError):
            build_rotated_bent_xz_layout(
                [PatchSpec("p1", o1, 3, "X", "X_horizontal"),
                 PatchSpec("p2", o2, 3, "Z", "X_horizontal")])


# (p1_origin, p2_origin, k, m): minimal, horizontal-only, vertical-only, and diagonal routed buses.
@pytest.mark.parametrize("o1,o2,k,m", [
    ((1, 7), (7, 1), 0, 0),     # minimal
    ((1, 7), (11, 1), 2, 0),    # horizontal extended only
    ((1, 11), (7, 1), 0, 2),    # vertical extended only
    ((1, 11), (11, 1), 2, 2),   # both (diagonal routed bus)
    ((1, 9), (9, 1), 1, 1),
])
def test_routed_bus_horizontal_and_vertical(o1, o2, k, m):
    # The bus may extend horizontally (k) AND vertically (m); both are valid routed buses,
    # not impossible. Each must build, pass full verify, and keep #mixed == d.
    lay = build_rotated_bent_xz_layout(
        [PatchSpec("p1", o1, 3, "X", "X_horizontal"),
         PatchSpec("p2", o2, 3, "Z", "X_horizontal")])
    assert all(lay.verify().values()), (o1, o2, lay.verify())
    assert sum(c["type"] == "M" for c in lay.checks) == 3      # seam stays at the X-arm width


import pytest


@pytest.mark.parametrize("d", [5, 7])
def test_generator_scales(d):
    p1 = PatchSpec("p1", (1, 2 * d + 1), d, "X", "X_horizontal")
    p2 = PatchSpec("p2", (2 * d + 1, 1), d, "Z", "X_horizontal")
    lay = build_rotated_bent_xz_layout([p1, p2])
    a = acceptance(lay.data, lay.checks, lay.x_logical, lay.z_logical)
    assert a["commute"] and a["joint"] and a["no_single"] and a["no_twist"] and a["one_logical"], (d, a)
    assert a["n_mixed"] == d
    assert len(lay.checks) == len(lay.data) - 1


def _layout_d3():
    return build_rotated_bent_xz_layout(
        [PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"),
         PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal")])


def test_readout_chain_product_is_joint_logical():
    lay = _layout_d3()
    assert len(lay.readout_chain) > 0
    # the product of the readout-chain checks must equal the joint logical X̄₁·Z̄₂
    chain = [c for c in lay.checks if c["syn"] in lay.readout_chain]
    S, idx, n = symplectic(chain, lay.data)
    prod = np.zeros(2 * n, np.uint8)
    for v in S:
        prod ^= v
    vXZ = _vec({**{c: "X" for c in lay.x_logical}, **{c: "Z" for c in lay.z_logical}}, idx, n)
    assert np.array_equal(prod, vXZ)


def test_verify_all_pass_d3():
    v = _layout_d3().verify()
    assert all(v.values()), v


def test_circuit_no_mpp_and_deterministic_d3():
    c = _layout_d3().build_circuit(rounds=3, p=0.0)
    assert c.num_observables == 1
    assert "MPP" not in str(c)
    det, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert not det.any() and not obs.any()


@pytest.mark.parametrize("d", [5, 7])
def test_verify_all_pass_scaled(d):
    lay = build_rotated_bent_xz_layout(
        [PatchSpec("p1", (1, 2 * d + 1), d, "X", "X_horizontal"),
         PatchSpec("p2", (2 * d + 1, 1), d, "Z", "X_horizontal")])
    v = lay.verify()
    assert all(v.values()), (d, v)
