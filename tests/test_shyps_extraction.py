"""Fixed cyclic offsets, physical gauge readout and patch-frame registration."""

import numpy as np
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock
from lightstim.qec_code.shyps import SHYPSCode, SHYPSCodeExtractionBlock


def _system(r=3):
    system = QECSystem()
    system.add_patch(SHYPSCode(r), name="shyps")
    return system


@pytest.mark.parametrize("r,offsets", [(3, (0, 2, 3)), (4, (0, 1, 4)), (5, (0, 2, 5))])
def test_each_cnot_layer_uses_one_cyclic_offset_without_coloring(r, offsets, monkeypatch):
    def forbidden_coloring(*args, **kwargs):
        raise AssertionError("Dedicated extraction must not invoke edge coloring")

    monkeypatch.setattr(
        "lightstim.qec_code.generic_css.gauge_SE_block.color_bipartite_edges",
        forbidden_coloring,
    )
    system = _system(r)
    patch = system.patches["shyps"][0]
    mapping = system.local_to_global_map["shyps"]
    extraction = SHYPSCodeExtractionBlock(system)
    m = 2**r - 1
    assert patch.gauge_offsets == offsets
    assert extraction.depth_x == extraction.depth_z == 3
    assert extraction.cnot_depth == 6
    assert extraction.circuit.num_measurements == 2 * m**2
    assert extraction.circuit.num_detectors == extraction.circuit.num_observables == 0

    data_positions = {mapping[q]: position for position, q in patch.data_qubits.items()}
    records = {gauge["syn_idx"]: gauge for gauge in system.active_gauges}
    for basis, layers, block in zip(
        ("X", "Z"), (extraction.x_layers, extraction.z_layers), extraction.measurement_blocks,
    ):
        cnot_instructions = [op for op in block if op.name == "CX"]
        assert len(cnot_instructions) == 3
        measured_supports = {}
        for offset, layer, instruction in zip(offsets, layers, cnot_instructions):
            targets = [target.value for target in instruction.targets_copy()]
            assert len(targets) == len(set(targets)) == 2 * m**2
            pairs = list(zip(targets[::2], targets[1::2]))
            assert pairs == [(a, q) if basis == "X" else (q, a) for a, q in layer]
            for ancilla, data in layer:
                gauge = records[ancilla]
                i, j = gauge["product_index"]
                row, column = data_positions[data]
                assert gauge["type"] == basis
                if basis == "X":
                    assert column == j and (row - i) % m == offset
                else:
                    assert row == i and (column - j) % m == offset
                measured_supports.setdefault(ancilla, set()).add(data)
        for ancilla, support in measured_supports.items():
            assert support == set(records[ancilla]["pauli"])


@pytest.mark.parametrize("order", [("X", "Z"), ("Z", "X"), ("Z", "Z", "X"), ("X",)])
def test_dedicated_and_generic_have_the_same_signed_measurement_instrument(order):
    system = _system()
    dedicated = SHYPSCodeExtractionBlock(system, basis_order=order)
    generic = GenericCSSGaugeExtractionBlock(system, basis_order=order)
    assert dedicated.cnot_depth == 3 * len(order)
    assert len(dedicated.measurement_blocks) == len(order)
    assert dedicated.circuit.num_measurements == generic.circuit.num_measurements
    for left, right in [(dedicated.circuit, generic.circuit), (generic.circuit, dedicated.circuit)]:
        assert all(right.has_flow(flow, unsigned=False) for flow in left.flow_generators())


def test_schedule_keeps_semantic_offsets_after_placement_and_rotation():
    system = _system()
    patch = SHYPSCode(4, shift=(3, 5))
    patch.transpose_coords()
    patch.rotate_coords(np.pi / 2, center=(0, 0))
    system.add_patch(patch, name="placed", offset=(100.125, 80.25))
    system.active_gauge_indices = {
        i for i, gauge in enumerate(system.gauges) if gauge["patch_name"] == "placed"
    }
    extraction = SHYPSCodeExtractionBlock(system)
    local = SHYPSCodeExtractionBlock(_system(4))
    mapping = system.local_to_global_map["placed"]
    for actual, expected in [(extraction.x_layers, local.x_layers), (extraction.z_layers, local.z_layers)]:
        assert actual == [[(mapping[a], mapping[q]) for a, q in layer] for layer in expected]


@pytest.mark.parametrize("order", [(), ("Y",)])
def test_invalid_basis_order_is_rejected(order):
    with pytest.raises(ValueError, match="nonempty sequence"):
        SHYPSCodeExtractionBlock(_system(), basis_order=order)


def test_inconsistent_gauge_support_is_rejected():
    system = _system()
    gauge = system.gauges[0]
    gauge["data_indices"] = [0, 1, 2]
    gauge["pauli"] = {q: "X" for q in gauge["data_indices"]}
    with pytest.raises(ValueError, match="declared cyclic offsets"):
        SHYPSCodeExtractionBlock(system)


def test_wrong_gauge_ancilla_is_rejected():
    system = _system()
    system.gauges[0]["syn_idx"] = system.gauges[1]["syn_idx"]
    with pytest.raises(ValueError, match="product index does not match"):
        SHYPSCodeExtractionBlock(system)
