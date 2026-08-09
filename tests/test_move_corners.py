"""Circuit-level verification of move_corners (Litinski Fig 45 step 5→6).

Ground truth read check-by-check off 800/1600-dpi crops of Fig 45 panel 6
(d=3 grown bar after the corner movement):

* ALL 8 bulk checks unchanged (Fig 40b: only boundary stabilizers change);
* top Z(2,10), right-upper X(6,8), bottom X(2,0), right-lower Z(6,2) kept verbatim;
* the LEFT long edge flips X→Z: X(0,6), X(0,2) retire; Z(0,8), Z(0,4) appear.

Topological effect: the same-type corner wrap moves from the bottom-left (X pair
of panel 5) to the top-left (Z pair: Z(0,8)+Z(2,10) share (1,9)) — matching the
transposed structure of Kishony-Fowler Fig 5 b→c (they flip the long bottom edge
of their horizontal bar).  No data qubits are added or removed.

Sequence under test: start (d rounds) → grow (|0⟩ region, d rounds) →
move_corners (d rounds, no init) → readout.  Gates: p=0 determinism, uid
anchoring across BOTH deformations, graphlike distance == d for both bases.
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.noise.config import NoiseConfig

from tests.test_grow_patch import paper_start_w2, grown_forced

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


# the layout generators live in the library now; re-exported for the notebook
from lightstim.qec_code.surface_code.rotated.litinski_layouts import (   # noqa: F401
    grown_patch, moved_w2)


PANEL6_W2 = {   # Fig 45 panel 6, hand-read ground truth (d=3)
    (2, 10): ('Z', [(1, 9), (3, 9)]),
    (6, 8):  ('X', [(5, 7), (5, 9)]),
    (0, 8):  ('Z', [(1, 7), (1, 9)]),
    (0, 4):  ('Z', [(1, 3), (1, 5)]),
    (2, 0):  ('X', [(1, 1), (3, 1)]),
    (6, 2):  ('Z', [(5, 1), (5, 3)]),
}


def test_moved_boundary_matches_panel6():
    assert moved_w2(3) == {sc: (t, sorted(map(tuple, sup)))
                           for sc, (t, sup) in PANEL6_W2.items()}


def test_moved_geometry_is_valid_code():
    d = 5
    patch = RuleBasedRectPatch(distance_x=2 * d - 1, distance_z=d,
                               forced=moved_w2(d), phase=0)
    n = len(patch.data_indices)
    assert len(patch.stabilizers) == n - 1
    # bulk identical to the grown bar's
    grown_bulk = {tuple(map(int, s['syn_coord'])): s['type']
                  for s in grown_patch(d).stabilizers if len(s['data_indices']) == 4}
    moved_bulk = {tuple(map(int, s['syn_coord'])): s['type']
                  for s in patch.stabilizers if len(s['data_indices']) == 4}
    assert moved_bulk == grown_bulk


def build_rotation_prefix(d, basis):
    """start → grow → move_corners, d diagonal rounds each, then readout."""
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
    kept_grow = len(system.active_stabilizer_indices)

    moved_geom = RuleBasedRectPatch(distance_x=2 * d - 1, distance_z=d,
                                    forced=moved_w2(d), phase=0)
    old_active = set(system.active_stabilizer_indices)
    md, _ = system.move_corners("q2", moved_geom, offset=(0, 0))
    kept_move = len(old_active & set(system.active_stabilizer_indices))
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    builder.apply_data_readout({q: basis for q in sorted(system.data_indices)})
    return builder, kept_move


def _check(d, basis, kept_expect):
    builder, kept_move = build_rotation_prefix(d, basis)
    assert kept_move == kept_expect     # bulk + unchanged boundary keep their uid
    det, obs = builder.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d} {basis}: detector fired at p=0"
    assert not obs.any(), f"d={d} {basis}: observable not deterministic at p=0"
    noisy = builder.build_noisy_circuit(noise_params=NP, noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_move_corners_d3(basis):
    _check(3, basis, kept_expect=12)    # 14 checks, 2 left-edge lobes retire


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_move_corners_d5(basis):
    _check(5, basis, kept_expect=40)    # 44 checks, 4 left-edge lobes retire
