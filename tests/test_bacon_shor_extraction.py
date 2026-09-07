"""Canonical geometric schedule and equivalence to generic gauge extraction."""

import numpy as np
import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock


def _system(distance=3):
    system = QECSystem()
    system.add_patch(BaconShorCode(distance=distance), name="bs")
    return system


def test_default_bacon_shor_schedule_has_uniform_geometric_directions(monkeypatch):
    # Freeze the public d=3 schedule independently of any coloring algorithm.
    # This also distinguishes it from the generic schedule's mixed Z directions.
    def forbidden_coloring(*args, **kwargs):
        raise AssertionError("Dedicated extraction must not invoke edge coloring")

    monkeypatch.setattr(
        "lightstim.qec_code.generic_css.gauge_SE_block.color_bipartite_edges", forbidden_coloring,
    )
    experiment = MemoryExperiment(qec_patch=BaconShorCode(distance=3))
    assert experiment.block_class is BaconShorCodeExtractionBlock
    block = experiment.block_class(experiment.system)
    cx_targets = [[t.value for t in op.targets_copy()] for op in block.circuit if op.name == "CX"]
    assert cx_targets == [
        [9, 0, 10, 1, 11, 3, 12, 4, 13, 6, 14, 7],
        [9, 1, 10, 2, 11, 4, 12, 5, 13, 7, 14, 8],
        [0, 15, 1, 16, 2, 17, 3, 18, 4, 19, 5, 20],
        [3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20],
    ]
    assert block.cnot_depth == 4


@pytest.mark.parametrize("distance", [2, 3, 5])
@pytest.mark.parametrize("order", [("X", "Z"), ("Z", "X"), ("Z", "Z", "X"), ("X",)])
def test_dedicated_and_generic_have_the_same_signed_measurement_instrument(distance, order):
    system = _system(distance)
    dedicated = BaconShorCodeExtractionBlock(system, basis_order=order)
    generic = GenericCSSGaugeExtractionBlock(system, basis_order=order)
    assert dedicated.depth_x == dedicated.depth_z == generic.depth_x == generic.depth_z == 2
    assert dedicated.circuit.num_measurements == generic.circuit.num_measurements
    # All signed stabilizer flows, including input/output/record correlations,
    # span the same measurement instrument despite different CNOT layer order.
    for left, right in [(dedicated.circuit, generic.circuit), (generic.circuit, dedicated.circuit)]:
        assert all(right.has_flow(flow, unsigned=False) for flow in left.flow_generators())


@pytest.mark.parametrize("theta,transpose", [(0, False), (np.pi/2, False), (np.pi/4, True)])
def test_geometric_schedule_uses_global_indices_and_patch_orientation(theta, transpose):
    system = _system(2)
    patch = BaconShorCode(distance=3)
    if transpose:
        patch.transpose_coords()
    patch.rotate_coords(theta, center=(0, 0))
    system.add_patch(patch, name="placed", offset=(20.125, 30.25))
    # Only measure the second patch; the first ensures global != local indices.
    system.active_gauge_indices = {i for i, g in enumerate(system.gauges) if g["patch_name"] == "placed"}
    block = BaconShorCodeExtractionBlock(system)
    local = BaconShorCodeExtractionBlock(_system(3))
    mapping = system.local_to_global_map["placed"]
    assert block.x_layers == [[(mapping[a], mapping[q]) for a, q in layer] for layer in local.x_layers]
    assert block.z_layers == [[(mapping[a], mapping[q]) for a, q in layer] for layer in local.z_layers]
    for basis, physical in zip(block.basis_order, block.measurement_blocks):
        gauges = [g for g in system.active_gauges if g["type"] == basis]
        for measurement, gauge in enumerate(gauges):
            pauli = stim.PauliString(system.num_qubits)
            for q, p in gauge["pauli"].items():
                pauli[q] = p
            assert physical.has_flow(stim.Flow(input=pauli, measurements=[measurement]), unsigned=False)


@pytest.mark.parametrize("order", [(), ("Y",)])
def test_dedicated_rejects_invalid_basis_order(order):
    with pytest.raises(ValueError, match="nonempty sequence"):
        BaconShorCodeExtractionBlock(_system(), basis_order=order)


def test_dedicated_does_not_silently_measure_an_incomplete_gauge():
    system = _system()
    gauge = system.gauges[0]
    # A valid weight-2 CSS support but not this ancilla's two local neighbors.
    gauge["data_indices"] = [0, 2]
    gauge["pauli"] = {0: "X", 2: "X"}
    with pytest.raises(ValueError, match="declared Bacon-Shor X neighbors"):
        BaconShorCodeExtractionBlock(system)


def test_dedicated_requires_a_measurement_ancilla():
    system = _system()
    system.gauges[0]["syn_idx"] = None
    with pytest.raises(ValueError, match="registered syndrome ancilla"):
        BaconShorCodeExtractionBlock(system)
