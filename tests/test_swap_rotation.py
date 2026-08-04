"""SWAP-network 90-degree rotation (the physically honest rotate_90).

The rotation moves data states with nearest-neighbour SWAPs (hot-potato ring
shifts through the patch's own ancillas), keeps the patch's TYPE (weight-2
lobe positions invariant — the user's textbook/conjugate classification) and
swaps the logical orientations.  Spec layer = the user's four-edge rule;
execution layer = ring shifts; the module asserts spec == execution and gate
locality internally.  Here: p=0 cleanliness, bridging across the network,
type preservation, and full graphlike distance for both bases at d=3 and 5.
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.qec_code.surface_code.rotated.litinski_layouts import rect_boundary
from lightstim.qec_code.surface_code.rotated.swap_rotation import (
    assert_nearest_neighbor)
from lightstim.qec_code.surface_code.rotated.operation import (
    RotatedSurfaceCodeLogicalOpSet,
)

_OPS = RotatedSurfaceCodeLogicalOpSet()


def _rotate(system, builder, name):
    """Rotate via the LogicalOpSet adapter (the op under test)."""
    return _OPS.swap_rotation(builder, system.patches[name][0])
from lightstim.noise.config import NoiseConfig

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def lobe_positions(system, name):
    """The patch's weight-2 syn coords — the user's type invariant."""
    return {tuple(int(round(v)) for v in system.stabilizers[u]['syn_coord'])
            for u in system.active_stabilizer_indices
            for s in [system.stabilizers[u]]
            if s.get('patch_name') == name and len(s['data_indices']) == 2}


def build_case(d, basis):
    """Textbook patch, d rounds SE, SWAP rotation, d rounds SE, readout."""
    system = QECSystem()
    system.add_patch(
        RuleBasedRectPatch(distance_x=d, distance_z=d,
                           forced=rect_boundary(d, d, 'Z', 'X'), phase=0),
        offset=(0, 0), name="q")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()

    data = sorted(system.data_indices)
    builder.initialize({q: basis for q in data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    lobes_before = lobe_positions(system, "q")
    _rotate(system, builder, "q")
    lobes_after = lobe_positions(system, "q")
    assert lobes_after == lobes_before, \
        "SWAP rotation must not move the weight-2 positions (type invariant)"

    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)
    builder.apply_data_readout(final_measurements={q: basis for q in data})
    return builder


def _check(d, basis):
    builder = build_case(d, basis)
    c = builder.circuit
    assert c.num_observables == 1
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d} {basis}: detector fired at p=0"
    assert not obs.any(), f"d={d} {basis}: observable not deterministic"
    noisy = builder.build_noisy_circuit(noise_params=NP,
                                       noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_swap_rotation_memory(d, basis):
    _check(d, basis)


def test_whole_circuit_is_nearest_neighbor():
    """Locality lint over the full circuit including the SWAP network."""
    system = QECSystem()
    system.add_patch(
        RuleBasedRectPatch(distance_x=3, distance_z=3,
                           forced=rect_boundary(3, 3, 'Z', 'X'), phase=0),
        offset=(0, 0), name="q")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: 'Z' for q in data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=3)
    _rotate(system, builder, "q")
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=3)
    builder.apply_data_readout(final_measurements={q: 'Z' for q in data})
    assert_nearest_neighbor(builder.circuit, system.qubit_coords)
