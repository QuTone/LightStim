"""Tests for the general N-patch rotated joint measurement ``M(∏ᵢ P̄ᵢ)``.

One API (``build_rotated_multi_patch_joint_layout``) drives every case from explicit ``PatchSpec``
origins.  The 3-patch ``M(X̄₁X̄₃Z̄₂)`` and a 4-patch ``M(X̄₁X̄₃X̄₄Z̄₂)`` are *examples*, not special
cases.  The decisive properties: the full joint is measured, NO proper sub-product is, the residual
logical count is ``N-1``, there is no weight-1 leftover logical, and impossible placements raise.
"""

import itertools

import numpy as np
import pytest

from lightstim.qec_code.surface_code.rotated import (
    build_rotated_multi_patch_joint_layout, PatchSpec, BentLayoutError,
    diagnose_multi_patch_joint)
from lightstim.qec_code.surface_code.rotated.bent_layout import _symplectic, _in_span, _gf2_rank


def _joint_props(lay):
    """Re-derive the joint algebra from scratch: joint in span, no single, no proper sub-product."""
    sv, n = _symplectic(lay.data)
    S = [sv(ch["pauli"]) for ch in lay.checks]
    vecs = [sv({c: P for c in sup}) for _, P, sup in lay.logicals]
    N = len(vecs)
    joint = np.zeros(2 * n, np.uint8)
    for v in vecs:
        joint ^= v
    subs = []
    for r in range(1, N):
        for comb in itertools.combinations(range(N), r):
            x = np.zeros(2 * n, np.uint8)
            for k in comb:
                x ^= vecs[k]
            subs.append(x)
    return dict(joint=_in_span(S, joint),
                no_subproduct=not any(_in_span(S, x) for x in subs),
                logical_count=len(lay.data) - _gf2_rank(S) == N - 1)


# --- 3-patch example (fast) --------------------------------------------------

P3_D3 = [PatchSpec("p1", (1, 1), 3, "X", "X_vertical"),
         PatchSpec("p3", (5, 7), 3, "X", "X_horizontal"),
         PatchSpec("p2", (13, 1), 3, "Z", "X_horizontal")]
T3 = [("p1", "X"), ("p3", "X"), ("p2", "Z")]


def test_3patch_measures_triple_joint_only():
    lay = build_rotated_multi_patch_joint_layout(P3_D3, target=T3)
    assert lay.N == 3
    p = _joint_props(lay)
    assert p["joint"] and p["no_subproduct"] and p["logical_count"]


def test_3patch_all_eleven_checks_pass():
    lay = build_rotated_multi_patch_joint_layout(P3_D3, target=T3)
    res = lay.verify()
    assert all(res.values()), res
    assert "no_weight1_logical" in res and "no_subjoint" in res


def test_target_defaults_to_measured_logical():
    # omitting target reads (name, measured_logical) from the PatchSpecs.
    a = build_rotated_multi_patch_joint_layout(P3_D3)
    b = build_rotated_multi_patch_joint_layout(P3_D3, target=T3)
    assert a.data == b.data and len(a.checks) == len(b.checks)


def test_coordinate_aware_translation():
    base = build_rotated_multi_patch_joint_layout(P3_D3, target=T3)
    sx, sy = 8, 6
    moved = build_rotated_multi_patch_joint_layout(
        [PatchSpec(s.name, (s.origin[0] + sx, s.origin[1] + sy), s.distance,
                   s.measured_logical, s.orientation) for s in P3_D3], target=T3)
    assert sorted((x - sx, y - sy) for x, y in moved.data) == base.data    # exactly translated
    assert all(_joint_props(moved).values())


# --- failure modes -----------------------------------------------------------

@pytest.mark.parametrize("specs, target, needle", [
    ([PatchSpec("p1", (1, 1), 3, "X", "X_vertical"), PatchSpec("p3", (3, 3), 3, "X", "X_horizontal"),
      PatchSpec("p2", (13, 1), 3, "Z", "X_horizontal")], None, "overlap"),
    ([PatchSpec("p1", (2, 2), 3, "X", "X_vertical"), PatchSpec("p3", (6, 8), 3, "X", "X_horizontal"),
      PatchSpec("p2", (14, 2), 3, "Z", "X_horizontal")], None, "must be odd"),
])
def test_impossible_placements_raise_concrete_reason(specs, target, needle):
    with pytest.raises(BentLayoutError) as exc:
        build_rotated_multi_patch_joint_layout(specs, target=target)
    assert needle in str(exc.value)


def test_unsupported_routing_raises():
    with pytest.raises(NotImplementedError):
        build_rotated_multi_patch_joint_layout(P3_D3, target=T3, routing="manhattan")


# --- 4-patch example (genuinely N-patch, not a 3-special-case) ----------------

def test_4patch_M_x1x3x4z2():
    lay = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (1, 1), 3, "X", "X_vertical"), PatchSpec("p4", (5, 7), 3, "X", "X_vertical"),
         PatchSpec("p3", (9, 13), 3, "X", "X_horizontal"), PatchSpec("p2", (17, 7), 3, "Z", "X_horizontal")],
        target=[("p1", "X"), ("p3", "X"), ("p4", "X"), ("p2", "Z")])
    assert lay.N == 4
    assert all(lay.verify().values())                       # 4-body joint, #data-rank == 3, no weight-1
    p = _joint_props(lay)
    assert p["joint"] and p["no_subproduct"] and p["logical_count"]


def test_offset_3patch_auto_non_template():
    # the decisive non-template case: p1 and p3 do NOT share a column edge (p1.right=9 != p3.left=5),
    # they are offset by 14 rows.  The auto-router derives the fusion corridor from the coordinates;
    # there is no staircase to match.  This is the placement a template validator would reject.
    lay = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (1, 1), 5, "X", "X_vertical"), PatchSpec("p3", (5, 15), 5, "X", "X_horizontal"),
         PatchSpec("p2", (21, 15), 5, "Z", "X_horizontal")],
        target=[("p1", "X"), ("p3", "X"), ("p2", "Z")])
    assert lay.N == 3
    assert all(lay.verify().values())
    assert all(_joint_props(lay).values())


def test_explicit_routing_graph_matches_auto():
    # an explicit fusion tree (routing={'type':'tree','edges':[...]}) builds the same valid joint as
    # the auto-router for a 4-patch placement, and a list of edges is accepted equivalently.
    specs = [PatchSpec("p1", (1, 1), 3, "X", "X_vertical"), PatchSpec("p4", (5, 7), 3, "X", "X_vertical"),
             PatchSpec("p3", (9, 13), 3, "X", "X_horizontal"), PatchSpec("p2", (17, 7), 3, "Z", "X_horizontal")]
    tgt = [("p1", "X"), ("p4", "X"), ("p3", "X"), ("p2", "Z")]
    auto = build_rotated_multi_patch_joint_layout(specs, target=tgt, routing="auto")
    graph = build_rotated_multi_patch_joint_layout(
        specs, target=tgt, routing={"type": "tree", "edges": [("p1", "p4"), ("p4", "p3")]})
    as_list = build_rotated_multi_patch_joint_layout(specs, target=tgt, routing=[("p1", "p4"), ("p4", "p3")])
    assert auto.data == graph.data == as_list.data
    assert all(graph.verify().values()) and all(_joint_props(graph).values())


def test_3patch_user_d5_origins():
    # the rotated_3patch_2.png d=5 origins the user gave, via the general API.
    lay = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (1, 5), 5, "X", "X_vertical"), PatchSpec("p2", (21, 11), 5, "Z", "X_horizontal"),
         PatchSpec("p3", (9, 23), 5, "X", "X_horizontal")],
        target=[("p1", "X"), ("p3", "X"), ("p2", "Z")])
    assert len(lay.data) == 120
    assert all(lay.verify().values())


def test_four_xside_patches_raise_not_mismeasure():
    # single-Z scope: a fusion tree + single Z-band keeps <=3 X-side patches independent.  Four
    # X-side patches (a 5-patch joint) close a sub-product, so the generator RAISES (detected via
    # the full-plaquette-span pre-check) rather than silently mis-measuring a sub-product.
    with pytest.raises(BentLayoutError) as exc:
        build_rotated_multi_patch_joint_layout(
            [PatchSpec("p1", (1, 1), 3, "X", "X_vertical"), PatchSpec("p4", (5, 7), 3, "X", "X_vertical"),
             PatchSpec("p5", (9, 13), 3, "X", "X_vertical"), PatchSpec("p3", (13, 19), 3, "X", "X_horizontal"),
             PatchSpec("p2", (21, 13), 3, "Z", "X_horizontal")])
    assert "sub-product" in str(exc.value)


# --- multiple Z-side patches: M(Z̄₁Z̄₂X̄₃) (two Z onto one X-side bus, each via its own wall) ------

MULTIZ_D3 = [PatchSpec("p1", (1, 1),  3, "Z", "X_horizontal"),    # Z̄₁  (top-left)
             PatchSpec("p2", (13, 9), 3, "Z", "X_horizontal"),    # Z̄₂  (mid-right)
             PatchSpec("p3", (7, 17), 3, "X", "X_horizontal")]    # X̄₃  (bottom of the X-bus)
MULTIZ_T = [("p1", "Z"), ("p2", "Z"), ("p3", "X")]


def test_multi_z_builds_and_passes_all_checks():
    lay = build_rotated_multi_patch_joint_layout(MULTIZ_D3, target=MULTIZ_T)
    assert lay.N == 3
    res = lay.verify()
    assert all(res.values()), res
    # each Z-patch has its OWN mixed (XZ) wall -> at least two distinct mixed-wall seams
    mixed_cols = {c["syn"][0] for c in lay.checks if c["type"] == "M"}
    assert len(mixed_cols) >= 2


def test_multi_z_measures_triple_joint_only():
    # the decisive multi-Z property: Z̄₁Z̄₂X̄₃ is measured, but no single and NO pairwise sub-product.
    lay = build_rotated_multi_patch_joint_layout(MULTIZ_D3, target=MULTIZ_T)
    sv, n = _symplectic(lay.data)
    S = [sv(ch["pauli"]) for ch in lay.checks]
    L = {nm: sv({c: P for c in sup}) for nm, P, sup in lay.logicals}
    z1, z2, x3 = L["p1"], L["p2"], L["p3"]
    assert _in_span(S, z1 ^ z2 ^ x3)                              # full triple joint measured
    assert not _in_span(S, z1) and not _in_span(S, z2) and not _in_span(S, x3)
    assert not _in_span(S, z1 ^ z2) and not _in_span(S, z1 ^ x3) and not _in_span(S, z2 ^ x3)
    assert len(lay.data) - _gf2_rank(S) == 2                      # N-1 residual logicals
    p = _joint_props(lay)
    assert p["joint"] and p["no_subproduct"] and p["logical_count"]


def test_z_patch_enclosed_by_bus_raises():
    # a Z-patch ENCLOSED inside the X-bus — in the trunk columns AND rows, between two fused X-patches
    # — cannot host a mixed wall, so it is reported as overlapping the bus (a Z beside / above / below
    # the bus is fine; only one buried inside it is rejected).
    with pytest.raises(BentLayoutError) as exc:
        build_rotated_multi_patch_joint_layout(
            [PatchSpec("p1", (1, 1), 3, "X", "X_horizontal"), PatchSpec("p4", (1, 13), 3, "X", "X_horizontal"),
             PatchSpec("p2", (1, 7), 3, "Z", "X_horizontal")], target=[("p1", "X"), ("p4", "X"), ("p2", "Z")])
    assert "overlap" in str(exc.value)


def test_z_patch_attaches_from_above_or_below():
    # #7: a Z-patch may attach to the bus from ANY side, including above/below (sharing the trunk
    # columns) through a horizontal mixed wall — not only left/right.
    for zorigin in ((1, 1), (1, 13)):                            # Z above, then Z below the X-bus
        lay = build_rotated_multi_patch_joint_layout(
            [PatchSpec("x", (1, 7), 3, "X", "X_horizontal"), PatchSpec("z", zorigin, 3, "Z", "X_horizontal")],
            target=[("z", "Z"), ("x", "X")])
        assert all(lay.verify().values())
        assert _joint_props(lay)["no_subproduct"]


def test_pure_same_type_two_patch_joints():
    # #6: a joint need not mix X and Z.  Pure-X M(X̄₁X̄₂) and pure-Z M(Z̄₁Z̄₂) are legal same-type
    # merges (no mixed walls); pure-Z uses a Z-memory circuit.  Both measure exactly the pair-joint.
    xx = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (1, 1), 3, "X", "X_horizontal"), PatchSpec("p2", (1, 7), 3, "X", "X_horizontal")],
        target=[("p1", "X"), ("p2", "X")])
    zz = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (1, 1), 3, "Z", "X_horizontal"), PatchSpec("p2", (1, 7), 3, "Z", "X_horizontal")],
        target=[("p1", "Z"), ("p2", "Z")])
    for lay in (xx, zz):
        assert all(lay.verify().values()), lay.verify()
        assert sum(c["type"] == "M" for c in lay.checks) == 0       # same-type merge: no mixed walls
        assert _joint_props(lay)["joint"] and _joint_props(lay)["no_subproduct"]


def test_pure_three_same_type_is_rejected():
    # three parallel same-type patches bare-fused over-measure the pairwise products -> rejected
    # (a clean 3-body same-type parity needs an opposite-type ancilla bus, which is future work).
    with pytest.raises(BentLayoutError):
        build_rotated_multi_patch_joint_layout(
            [PatchSpec("p1", (1, 1), 3, "X", "X_horizontal"), PatchSpec("p2", (1, 7), 3, "X", "X_horizontal"),
             PatchSpec("p3", (1, 13), 3, "X", "X_horizontal")], target=[("p1", "X"), ("p2", "X"), ("p3", "X")])


# --- XXZZ: M(Z̄₁Z̄₂X̄₃X̄₄) (two X + two Z) — the rotated_4_patch.png cross --------------------------

# X̄₃/X̄₄ on a vertical trunk, Z̄₁/Z̄₂ on the horizontal arm; the Z-walls sit BETWEEN X̄₃ and X̄₄ in
# row-space, which keeps X̄₃ and X̄₄ independent (they are NOT homologous despite a shared trunk).
XXZZ_CROSS = [PatchSpec("p4", (9, 1),  3, "X", "X_horizontal"),    # X̄₄  (top of trunk)
              PatchSpec("p3", (9, 21), 3, "X", "X_horizontal"),    # X̄₃  (bottom of trunk)
              PatchSpec("p1", (1, 11), 3, "Z", "X_horizontal"),    # Z̄₁  (left arm, own wall)
              PatchSpec("p2", (17, 11), 3, "Z", "X_horizontal")]   # Z̄₂  (right arm, own wall)
XXZZ_T = [("p1", "Z"), ("p2", "Z"), ("p3", "X"), ("p4", "X")]


def test_xxzz_cross_measures_full_joint_only():
    # two X + two Z: the 4-body joint is measured and NO proper sub-product is — crucially X̄₃X̄₄ and
    # Z̄₁Z̄₂ are excluded (the same-type pairs are NOT homologous in the cross routing).
    lay = build_rotated_multi_patch_joint_layout(XXZZ_CROSS, target=XXZZ_T)
    assert lay.N == 4
    res = lay.verify()
    assert all(res.values()), res
    sv, n = _symplectic(lay.data)
    S = [sv(ch["pauli"]) for ch in lay.checks]
    L = {nm: sv({c: P for c in sup}) for nm, P, sup in lay.logicals}
    z1, z2, x3, x4 = L["p1"], L["p2"], L["p3"], L["p4"]
    assert _in_span(S, z1 ^ z2 ^ x3 ^ x4)                         # full 4-body joint measured
    assert not _in_span(S, x3 ^ x4)                               # X̄₃X̄₄ NOT measured (not homologous)
    assert not _in_span(S, z1 ^ z2)                               # Z̄₁Z̄₂ NOT measured
    assert not _in_span(S, z1 ^ x4) and not _in_span(S, z2 ^ x3)  # mixed pairs NOT measured
    assert len(lay.data) - _gf2_rank(S) == 3                      # N-1 residual logicals
    assert _joint_props(lay)["no_subproduct"]                     # no proper sub-product at all


def test_xxzz_adjacent_x_collapse_is_rejected():
    # the failure mode (forbidden placement): X̄₃,X̄₄ fused ADJACENTLY with no wall between them become
    # homologous, so X̄₃X̄₄ enters the span — the generator RAISES a clear, specific error (it names
    # the collapsing sub-product and the rule) rather than mis-measuring.
    with pytest.raises(BentLayoutError) as exc:
        build_rotated_multi_patch_joint_layout(
            [PatchSpec("p3", (7, 17), 3, "X", "X_horizontal"), PatchSpec("p4", (7, 23), 3, "X", "X_horizontal"),
             PatchSpec("p1", (1, 1), 3, "Z", "X_horizontal"), PatchSpec("p2", (13, 9), 3, "Z", "X_horizontal")],
            target=XXZZ_T)
    msg = str(exc.value)
    assert "forbidden placement" in msg and "X3X4" in msg and "sub-product" in msg


# --- diagnostic mode (draw/explain a failing candidate without raising) --------------------------

def test_diagnose_pass_and_fail_identify_subproduct():
    # the XXZZ cross diagnoses clean; the adjacent-X bad routing diagnoses the SPECIFIC sub-product
    # X̄₃X̄₄ in span, with a non-empty stabilizer chain — and never raises.
    good = diagnose_multi_patch_joint(XXZZ_CROSS, target=XXZZ_T)
    assert good.ok and good.joint_in_span and not good.failing_subproducts
    assert len(good.joint_chain) > 0

    bad = diagnose_multi_patch_joint(
        [PatchSpec("p3", (7, 17), 3, "X", "X_horizontal"), PatchSpec("p4", (7, 23), 3, "X", "X_horizontal"),
         PatchSpec("p1", (1, 1), 3, "Z", "X_horizontal"), PatchSpec("p2", (13, 9), 3, "Z", "X_horizontal")],
        target=XXZZ_T)
    assert not bad.ok and not bad.joint_in_span
    labels = [lbl for lbl, _, _ in bad.failing_subproducts]
    assert "X3X4" in labels                                      # X̄₃X̄₄ collapses
    chain = bad.failing_subproducts[labels.index("X3X4")][2]
    assert len(chain) > 0 and len(bad.data) > 0                  # drawable candidate + closing chain


def test_non_trunk_x_group_is_routed_not_rejected():
    # A non-trunk fused X-group (x5,x6 stacked off to the side) is a LEGAL candidate: the router must
    # route it and let the GF(2) oracle judge, not reject it as "not supported".  Here the oracle is
    # clean (no sub-product), so legality is geometry+oracle, never an implementation limit.
    P = [PatchSpec("x3", (11, 1), 3, "X", "X_horizontal"), PatchSpec("x4", (11, 29), 3, "X", "X_horizontal"),
         PatchSpec("x5", (3, 11), 3, "X", "X_horizontal"), PatchSpec("x6", (3, 17), 3, "X", "X_horizontal"),
         PatchSpec("z1", (19, 9), 3, "Z", "X_horizontal"), PatchSpec("z2", (19, 21), 3, "Z", "X_horizontal")]
    T = [("z1", "Z"), ("z2", "Z"), ("x3", "X"), ("x4", "X"), ("x5", "X"), ("x6", "X")]
    dg = diagnose_multi_patch_joint(P, target=T)
    assert len(dg.data) > 0                                       # routed, not hard-rejected
    assert "not supported" not in dg.reason and "non-trunk" not in dg.reason
    assert dg.ok and not dg.failing_subproducts                  # oracle: only the full joint, no sub-product


# --- genuine N-patch (N>4): the scalable 1X+(N-1)Z family + a 2X+4Z 6-patch ----------------------

def test_5patch_one_x_four_z_joint_only():
    # 5-patch M(X̄₀ Z̄₁Z̄₂Z̄₃Z̄₄): one X-patch, four Z-patches each via its own mixed wall.  The full
    # 5-body joint is measured and NO proper sub-product is (the scalable family — Z-side scales).
    patches = [PatchSpec("x0", (9, 21), 3, "X", "X_horizontal"),
               PatchSpec("z1", (1, 1), 3, "Z", "X_horizontal"), PatchSpec("z2", (17, 5), 3, "Z", "X_horizontal"),
               PatchSpec("z3", (1, 9), 3, "Z", "X_horizontal"), PatchSpec("z4", (17, 13), 3, "Z", "X_horizontal")]
    target = [("x0", "X"), ("z1", "Z"), ("z2", "Z"), ("z3", "Z"), ("z4", "Z")]
    lay = build_rotated_multi_patch_joint_layout(patches, target=target)
    assert lay.N == 5
    assert all(lay.verify().values())
    p = _joint_props(lay)
    assert p["joint"] and p["no_subproduct"] and p["logical_count"]
    # one mixed (XZ) wall per Z-patch -> several mixed-wall plaquettes
    assert sum(c["type"] == "M" for c in lay.checks) >= 4


@pytest.mark.slow
def test_3x_2z_side_arm_joint_only():
    # rotated_3X.png: M(Z̄₁Z̄₂X̄₃X̄₄X̄₅) — THREE X-patches.  X̄₄(top)/X̄₃(bottom) on a vertical trunk,
    # Z̄₁/Z̄₂ mixed walls between them, and X̄₅ a perpendicular (X_vertical) side-arm fused to the bus.
    # Three X-patches do NOT collapse here: X̄₅ runs perpendicular to X̄₃/X̄₄, so num_X>=3 is fine; the
    # GF(2) oracle confirms only the full 5-body joint is measured.
    patches = [PatchSpec("p4", (11, 1), 3, "X", "X_horizontal"), PatchSpec("p3", (11, 29), 3, "X", "X_horizontal"),
               PatchSpec("p5", (3, 23), 3, "X", "X_vertical"),
               PatchSpec("p1", (3, 9), 3, "Z", "X_horizontal"), PatchSpec("p2", (19, 17), 3, "Z", "X_horizontal")]
    target = [("p1", "Z"), ("p2", "Z"), ("p3", "X"), ("p4", "X"), ("p5", "X")]
    lay = build_rotated_multi_patch_joint_layout(patches, target=target)
    assert lay.N == 5 and sum(P == "X" for _, P, _ in lay.logicals) == 3   # genuinely 3 X-patches
    assert all(lay.verify().values())
    p = _joint_props(lay)
    assert p["joint"] and p["no_subproduct"] and p["logical_count"]


_EXPLICIT_6 = [PatchSpec("p4", (11, 1), 3, "X", "X_horizontal"), PatchSpec("p3", (11, 29), 3, "X", "X_horizontal"),
               PatchSpec("p5", (3, 23), 3, "X", "X_vertical"), PatchSpec("p6", (3, 17), 3, "X", "X_vertical"),
               PatchSpec("p1", (3, 9), 3, "Z", "X_horizontal"), PatchSpec("p2", (19, 17), 3, "Z", "X_horizontal")]
_EXPLICIT_T = [("p1", "Z"), ("p2", "Z"), ("p3", "X"), ("p4", "X"), ("p5", "X"), ("p6", "X")]


def test_explicit_routing_illegal_cases_report_reason():
    # explicit routing is honoured exactly; illegal graphs raise concrete reasons (these all fail in
    # parsing / routing, before the oracle, so they are fast).
    with pytest.raises(BentLayoutError) as e1:                # x_edge joining an X and a Z
        build_rotated_multi_patch_joint_layout(_EXPLICIT_6, target=_EXPLICIT_T,
                                               routing={"type": "explicit", "x_edges": [("p4", "p1")]})
    assert "two X-side patches" in str(e1.value)
    with pytest.raises(BentLayoutError) as e2:                # wrong declared wall side
        build_rotated_multi_patch_joint_layout(
            _EXPLICIT_6, target=_EXPLICIT_T,
            routing={"type": "explicit", "x_edges": [("p4", "p3"), ("p5", "trunk"), ("p6", "trunk")],
                     "z_attachments": [("p1", "right_wall"), ("p2", "right_wall")]})
    assert "p1" in str(e2.value) and "left" in str(e2.value) and "right" in str(e2.value)
    with pytest.raises(BentLayoutError) as e3:                # side-arm AND fusion edge for the same patch
        build_rotated_multi_patch_joint_layout(_EXPLICIT_6, target=_EXPLICIT_T,
                                               routing={"type": "explicit", "x_edges": [("p4", "p3"), ("p4", "trunk")]})
    assert "side-arm" in str(e3.value)


@pytest.mark.slow
def test_explicit_routing_builds_and_verifies():
    # the upstream-compiler form: the routing is taken exactly as given and still passes the oracle.
    lay = build_rotated_multi_patch_joint_layout(
        _EXPLICIT_6, target=_EXPLICIT_T,
        routing={"type": "explicit", "x_edges": [("p4", "p3"), ("p5", "trunk"), ("p6", "trunk")],
                 "z_attachments": [("p1", "left_wall"), ("p2", "right_wall")]})
    assert lay.N == 6 and all(lay.verify().values())
    assert _joint_props(lay)["no_subproduct"]


@pytest.mark.slow
def test_4x_2z_no_weight1_after_repair():
    # 4 X + 2 Z (X̄₅,X̄₆ a fused non-trunk group, X̄₆ opposite Z̄₂).  This placement first exposed a
    # stabilizer-SELECTION weakness — 6 corner weight-1 leftover logicals — but a weight-1-free code
    # exists (a boundary "killer" plaquette exists at each corner).  The swap-repair in selection
    # finds it, so the layout is fully clean: joint measured, NO sub-product, AND no weight-1.
    patches = [PatchSpec("p4", (11, 1), 3, "X", "X_horizontal"), PatchSpec("p3", (11, 29), 3, "X", "X_horizontal"),
               PatchSpec("p5", (3, 23), 3, "X", "X_vertical"), PatchSpec("p6", (3, 17), 3, "X", "X_vertical"),
               PatchSpec("p1", (3, 9), 3, "Z", "X_horizontal"), PatchSpec("p2", (19, 17), 3, "Z", "X_horizontal")]
    target = [("p1", "Z"), ("p2", "Z"), ("p3", "X"), ("p4", "X"), ("p5", "X"), ("p6", "X")]
    lay = build_rotated_multi_patch_joint_layout(patches, target=target)
    res = lay.verify()
    assert res["no_weight1_logical"], "swap-repair should clear the corner weight-1 logicals"
    assert all(res.values()), res
    assert _joint_props(lay)["no_subproduct"]


@pytest.mark.slow
def test_6patch_two_x_four_z_joint_only():
    # 6-patch M(X̄₅X̄₆ Z̄₁Z̄₂Z̄₃Z̄₄): two X-patches at the bus ends + four Z-patches.  Slow (the GF(2)
    # selection search grows with the data set), so it is marked slow.
    patches = [PatchSpec("x5", (9, 1), 3, "X", "X_horizontal"), PatchSpec("x6", (9, 31), 3, "X", "X_horizontal"),
               PatchSpec("z1", (1, 5), 3, "Z", "X_horizontal"), PatchSpec("z2", (17, 11), 3, "Z", "X_horizontal"),
               PatchSpec("z3", (1, 17), 3, "Z", "X_horizontal"), PatchSpec("z4", (17, 23), 3, "Z", "X_horizontal")]
    target = [("x5", "X"), ("x6", "X"), ("z1", "Z"), ("z2", "Z"), ("z3", "Z"), ("z4", "Z")]
    lay = build_rotated_multi_patch_joint_layout(patches, target=target)
    assert lay.N == 6
    assert all(lay.verify().values())
    assert _joint_props(lay)["no_subproduct"]
