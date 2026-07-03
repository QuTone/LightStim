"""Tests for the rotated three-patch bent joint measurement ``M(X̄₁·X̄₃·Z̄₂)``.

Figure-grounded (``rotated_3patch.png``): a band carrying X̄₁ (left) and Z̄₂ (right) with an
arm carrying X̄₃, joined by a single bent mixed domain wall.  The construction is driven by
explicit :class:`PatchSpec` origins (coordinate-aware, like the two-patch generator), and the
decisive property is that it measures the **triple** joint and NOT any sub-joint (the failure
mode where X̄₁ and X̄₃ collapse to homologous and only ``X̄₁·Z̄₂`` is measured).
"""

import numpy as np
import pytest

from lightstim.qec_code.surface_code.rotated import (
    build_rotated_bent_xxz_layout, xxz_patches, PatchSpec, BentLayoutError)
from lightstim.qec_code.surface_code.rotated.bent_layout import _symplectic, _in_span, _gf2_rank


def _spans(lay):
    sv, n = _symplectic(lay.data)
    S = [sv(ch["pauli"]) for ch in lay.checks]
    (_, _, x1), (_, _, x3), (_, _, z2) = lay.logicals
    vX1 = sv({c: "X" for c in x1})
    vX3 = sv({c: "X" for c in x3})
    vZ2 = sv({c: "Z" for c in z2})
    return S, dict(joint=_in_span(S, vX1 ^ vX3 ^ vZ2),
                   X1=_in_span(S, vX1), X3=_in_span(S, vX3), Z2=_in_span(S, vZ2),
                   X1Z2=_in_span(S, vX1 ^ vZ2), X3Z2=_in_span(S, vX3 ^ vZ2),
                   X1X3=_in_span(S, vX1 ^ vX3))


# --- figure reproduction -----------------------------------------------------

def test_xxz_d3_data_layout_matches_figure():
    # d=3 reproduces rotated_3patch.png: band 9×3 (=27) + arm 3×4 (=12) = 39 data, 3 mixed.
    lay = build_rotated_bent_xxz_layout(xxz_patches(3))
    assert len(lay.data) == 39
    assert sum(c["type"] == "M" for c in lay.checks) == 3
    assert [name for name, _, _ in lay.logicals] == ["p1", "p3", "p2"]
    assert [P for _, P, _ in lay.logicals] == ["X", "X", "Z"]
    # the three logical strings land exactly where the figure draws them
    (_, _, x1), (_, _, x3), (_, _, z2) = lay.logicals
    assert sorted(x1) == [(5, 5), (5, 7), (5, 9)]          # X̄₁ vertical, x=5
    assert sorted(x3) == [(5, 13), (7, 13), (9, 13)]       # X̄₃ horizontal, y=13
    assert sorted(z2) == [(13, 5), (13, 7), (13, 9)]       # Z̄₂ vertical, x=13


def test_xxz_explicit_patchspecs_equal_helper():
    # the PatchSpec interface (user-facing) and the convenience helper give the same layout.
    explicit = build_rotated_bent_xxz_layout([
        PatchSpec("p1", (1, 5),  3, "X", "X_vertical"),
        PatchSpec("p2", (13, 5), 3, "Z", "X_horizontal"),
        PatchSpec("p3", (5, 13), 3, "X", "X_horizontal"),
    ])
    assert explicit.data == build_rotated_bent_xxz_layout(xxz_patches(3)).data


# --- the joint property (the whole point) ------------------------------------

def test_xxz_measures_triple_joint_only():
    lay = build_rotated_bent_xxz_layout(xxz_patches(3))
    _, sp = _spans(lay)
    assert sp["joint"] is True
    assert not sp["X1"] and not sp["X3"] and not sp["Z2"]          # no single logical collapsed
    assert not sp["X1Z2"] and not sp["X3Z2"] and not sp["X1X3"]    # no sub-joint measured


def test_xxz_d3_all_nine_acceptance_checks_pass():
    lay = build_rotated_bent_xxz_layout(xxz_patches(3))
    res = lay.verify()
    assert all(res.values()), res


def test_xxz_logical_count_is_N_minus_1():
    lay = build_rotated_bent_xxz_layout(xxz_patches(3))
    S, _ = _spans(lay)
    assert len(lay.data) - _gf2_rank(S) == 2                      # 3 patches, one joint -> 2 left


def test_xxz_no_twist_and_three_mixed_at_d3():
    lay = build_rotated_bent_xxz_layout(xxz_patches(3))
    assert not any(P == "Y" for c in lay.checks for P in c["pauli"].values())
    assert sum(c["type"] == "M" for c in lay.checks) == 3        # #mixed == distance


def test_xxz_circuit_is_noiseless_deterministic():
    lay = build_rotated_bent_xxz_layout(xxz_patches(3))
    circ = lay.build_circuit(rounds=3, p=0.0)
    assert "MPP" not in str(circ)
    det, obs = circ.compile_detector_sampler(seed=0).sample(500, separate_observables=True)
    assert not det.any() and not obs.any()                       # deterministic memory experiment
    dem = circ.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circ.num_detectors


# --- coordinate-awareness (genuinely origin-driven, not a fixed template) -----

def test_xxz_translation_regenerates_and_stays_valid():
    base = build_rotated_bent_xxz_layout(xxz_patches(3))
    sx, sy = 6, 4
    moved = build_rotated_bent_xxz_layout([
        PatchSpec("p1", (1 + sx, 5 + sy),  3, "X", "X_vertical"),
        PatchSpec("p2", (13 + sx, 5 + sy), 3, "Z", "X_horizontal"),
        PatchSpec("p3", (5 + sx, 13 + sy), 3, "X", "X_horizontal"),
    ])
    assert sorted((x - sx, y - sy) for x, y in moved.data) == base.data   # exactly translated
    assert all(_spans(moved)[1][k] is (k == "joint") for k in ("joint", "X1", "X3", "Z2"))


def test_xxz_wider_connector_changes_data():
    # moving p2 two columns further right lengthens the band -> a strictly larger data set.
    narrow = build_rotated_bent_xxz_layout(xxz_patches(3))
    wide = build_rotated_bent_xxz_layout([
        PatchSpec("p1", (1, 5),  3, "X", "X_vertical"),
        PatchSpec("p2", (15, 5), 3, "Z", "X_horizontal"),
        PatchSpec("p3", (5, 13), 3, "X", "X_horizontal"),
    ])
    assert len(wide.data) > len(narrow.data)
    assert _spans(wide)[1]["joint"] is True


@pytest.mark.parametrize("patches, needle", [
    # p1 not fused to the trunk top (p1.right != p3.left)
    ([PatchSpec("p1", (1, 5), 3, "X", "X_vertical"), PatchSpec("p2", (15, 5), 3, "Z", "X_horizontal"),
      PatchSpec("p3", (7, 13), 3, "X", "X_horizontal")], "fuses to the top"),
    # odd horizontal band gap (p2.left - p3.right odd)
    ([PatchSpec("p1", (1, 5), 3, "X", "X_vertical"), PatchSpec("p2", (14, 5), 3, "Z", "X_horizontal"),
      PatchSpec("p3", (5, 13), 3, "X", "X_horizontal")], "horizontal gap"),
    # p2/band placed ABOVE p1 (negative offset)
    ([PatchSpec("p1", (1, 5), 3, "X", "X_vertical"), PatchSpec("p2", (13, 1), 3, "Z", "X_horizontal"),
      PatchSpec("p3", (5, 13), 3, "X", "X_horizontal")], "at or above the band"),
    # arm gap too small (p3 touching the band bottom)
    ([PatchSpec("p1", (1, 5), 3, "X", "X_vertical"), PatchSpec("p2", (13, 5), 3, "Z", "X_horizontal"),
      PatchSpec("p3", (5, 9), 3, "X", "X_horizontal")], "vertical gap"),
])
def test_xxz_impossible_placements_raise_concrete_reason(patches, needle):
    with pytest.raises(BentLayoutError) as exc:
        build_rotated_bent_xxz_layout(patches)
    assert needle in str(exc.value)


# --- vertical offset / staircase (rotated_3patch_2.png) ------------------------

@pytest.mark.parametrize("offset", [1, 2])
def test_xxz_offset_staircase_measures_triple_joint(offset):
    # the bent/staircase layout (p1 raised above the band) still measures exactly the triple joint.
    lay = build_rotated_bent_xxz_layout(xxz_patches(3, offset=offset))
    (_, _, x1), (_, _, x3), (_, _, z2) = lay.logicals
    p1_top = min(r for _, r in x1)
    band_top = min(r for _, r in z2)
    assert (band_top - p1_top) // 2 == offset                    # p1 genuinely raised by `offset` rows
    _, sp = _spans(lay)
    assert sp["joint"] and not sp["X1"] and not sp["X3"] and not sp["Z2"]
    assert not sp["X1Z2"] and not sp["X3Z2"] and not sp["X1X3"]
    assert all(lay.verify().values())


def test_xxz_offset_explicit_figure2_origins():
    # the exact rotated_3patch_2.png origins build a valid offset joint (45 data, +2-row staircase).
    lay = build_rotated_bent_xxz_layout([
        PatchSpec("p1", (1, 3),  3, "X", "X_vertical"),
        PatchSpec("p2", (13, 7), 3, "Z", "X_horizontal"),
        PatchSpec("p3", (5, 15), 3, "X", "X_horizontal"),
    ])
    assert len(lay.data) == 45
    assert all(lay.verify().values())


def test_xxz_offset_zero_unchanged_from_figure1():
    # offset=0 is byte-identical to the original row-aligned figure-1 helper output.
    assert build_rotated_bent_xxz_layout(xxz_patches(3, 0)).data == \
           build_rotated_bent_xxz_layout(xxz_patches(3)).data


def test_xxz_wrong_measured_logical_order_rejected():
    with pytest.raises(ValueError):
        build_rotated_bent_xxz_layout([
            PatchSpec("p1", (1, 5),  3, "Z", "X_vertical"),       # should be X
            PatchSpec("p2", (13, 5), 3, "Z", "X_horizontal"),
            PatchSpec("p3", (5, 13), 3, "X", "X_horizontal"),
        ])


@pytest.mark.slow
def test_xxz_scales_to_d5():
    lay = build_rotated_bent_xxz_layout(xxz_patches(5))
    assert len(lay.data) == 105                                  # full d×d patches + d-wide connector
    assert sum(c["type"] == "M" for c in lay.checks) == 5        # #mixed == distance
    assert all(lay.verify().values())
