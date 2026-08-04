"""Circuit-level verification of shrink_patch (Litinski Fig 45 step 6→7).

Ground truth read off 800/1600-dpi crops of Fig 45 panel 7 (the shrunk, ROTATED
q2 at the bottom tile, bar-local coords):

* bulk (2,4)X, (4,4)Z, (2,2)Z, (4,2)X — all four kept verbatim;
* lobes X(2,0), Z(6,2), Z(0,4) kept; NEW top lobe X(4,6) = the cut-line bulk
  X(4,6) truncated to {(3,5),(5,5)} (its value bridges via the X readouts);
* the cut-line Z bulk (2,6) does NOT survive (Z info destroyed by X readout);
* boundary walk left Z / top X / right Z / bottom X → X̄ VERTICAL, Z̄ HORIZONTAL:
  the patch is ROTATED relative to the original q2 (X̄ horizontal).

Sequence under test: start → grow (|0⟩) → move_corners → X-readout of the top
d−1 rows → shrink_patch → d rounds → readout.  All phases d diagonal rounds.
Gates: p=0 determinism, kept-uid anchoring, the §3.2 coverage criterion (asserted
inside shrink_patch itself), graphlike distance == d for both bases.
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.noise.config import NoiseConfig

from tests.test_grow_patch import paper_start_w2, grown_forced, ptype
from tests.test_move_corners import moved_w2, grown_patch

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


# the closed-form rotated-convention square (== panel 7 AND panel 9's local layout)
from lightstim.qec_code.surface_code.rotated.litinski_layouts import shrunk_w2   # noqa: F401


def _shrunk_w2_by_truncation(d):
    """Independent derivation: the moved bar's w2 fully inside the bottom d×d, plus
    the truncated X lobes on the cut line (cy = 2d, X checkerboard positions)."""
    out = {sc: v for sc, v in moved_w2(d).items() if sc[1] < 2 * d
           and all(y < 2 * d for _, y in v[1])}
    for cx in range(2, 2 * d, 2):
        if ptype(cx, 2 * d) == 'X':
            out[(cx, 2 * d)] = ('X', [(cx - 1, 2 * d - 1), (cx + 1, 2 * d - 1)])
    return out


def test_truncation_derivation_equals_closed_form():
    """Physically shrinking the moved bar must land exactly on the closed-form
    rotated-convention square — two independent constructions, one layout."""
    for d in (3, 5):
        a = {sc: (t, sorted(map(tuple, sup)))
             for sc, (t, sup) in _shrunk_w2_by_truncation(d).items()}
        b = {sc: (t, sorted(map(tuple, sup)))
             for sc, (t, sup) in shrunk_w2(d).items()}
        assert a == b


PANEL7_W2 = {   # Fig 45 panel 7, hand-read ground truth (d=3, bar-local coords)
    (2, 0): ('X', [(1, 1), (3, 1)]),
    (6, 2): ('Z', [(5, 1), (5, 3)]),
    (0, 4): ('Z', [(1, 3), (1, 5)]),
    (4, 6): ('X', [(3, 5), (5, 5)]),
}


def test_shrunk_boundary_matches_panel7():
    assert shrunk_w2(3) == {sc: (t, sorted(map(tuple, sup)))
                            for sc, (t, sup) in PANEL7_W2.items()}


def test_shrunk_geometry_is_rotated():
    """The shrunk code's logicals have SWAPPED orientation vs the original q2."""
    for d in (3, 5):
        patch = RuleBasedRectPatch(distance_x=d, distance_z=d,
                                   forced=shrunk_w2(d), phase=0)
        assert len(patch.stabilizers) == d * d - 1
        xs = {tuple(int(round(v)) for v in patch.qubit_coords[i])
              for i in patch.logical_ops[0]['data_indices']}
        zs = {tuple(int(round(v)) for v in patch.qubit_coords[i])
              for i in patch.logical_ops[1]['data_indices']}
        assert len({x for x, y in xs}) == 1     # X̄ vertical (one column)
        assert len({y for x, y in zs}) == 1     # Z̄ horizontal (one row)


def build_full_shrink(d, basis):
    dy = 2 * (d - 1)
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=paper_start_w2(d), phase=0),
                     offset=(0, dy), name="q2")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: basis for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    new_data, _ = system.grow_patch("q2", grown_patch(d), offset=(0, 0))
    builder.initialize({q: 'Z' for q in new_data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    system.move_corners("q2", RuleBasedRectPatch(distance_x=2 * d - 1, distance_z=d,
                                                 forced=moved_w2(d), phase=0),
                        offset=(0, 0))
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    # ---- SHRINK: X-readout of the top d−1 rows, then retire them ----
    removed_q = [q for q in sorted(system.data_indices)
                 if system.index_to_owner_map.get(q) == "q2"
                 and system.qubit_coords[q][1] > 2 * d - 1]
    builder.apply_data_readout(final_measurements={q: 'X' for q in removed_q})
    shrunk_geom = RuleBasedRectPatch(distance_x=d, distance_z=d,
                                     forced=shrunk_w2(d), phase=0)
    old_active = {u for u in system.active_stabilizer_indices
                  if system.stabilizers[u].get('patch_name') == "q2"}
    retired = system.shrink_patch("q2", shrunk_geom, offset=(0, 0))
    assert sorted(retired) == removed_q
    kept = len(old_active & set(system.active_stabilizer_indices))
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    live = [q for q in sorted(system.data_indices)
            if q in system.qubit_coords and q not in retired
            and system.index_to_owner_map.get(q) == "q2"]
    builder.apply_data_readout({q: basis for q in live})
    return builder, kept


def _check(d, basis, kept_expect):
    builder, kept = build_full_shrink(d, basis)
    assert kept == kept_expect
    det, obs = builder.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d} {basis}: detector fired at p=0"
    assert not obs.any(), f"d={d} {basis}: observable not deterministic at p=0"
    noisy = builder.build_noisy_circuit(noise_params=NP, noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_shrink_d3(basis):
    # shrunk code: 8 checks; kept-uid = 4 bulk + 3 lobes = 7; new = truncated X(4,6)
    _check(3, basis, kept_expect=7)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_shrink_d5(basis):
    # 24 checks; kept-uid = 16 bulk + 6 lobes = 22; new = 2 truncated X lobes
    _check(5, basis, kept_expect=22)
