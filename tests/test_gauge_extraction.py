"""Physical instrument checks for the generic CSS gauge extraction circuit."""

import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock


def _system(distance=3):
    system = QECSystem()
    system.add_patch(BaconShorCode(distance=distance), name="bs", offset=(10, 20))
    return system


def _pauli(record, n):
    result = stim.PauliString(n)
    for qubit, basis in record["pauli"].items():
        result[qubit] = basis
    return result


@pytest.mark.parametrize("distance", [2, 3, 4])
@pytest.mark.parametrize("order", [("X", "Z"), ("Z", "X"), ("Z", "Z", "X"), ("X",)])
def test_gauge_extraction_measures_declared_generators_and_preserves_bare_logicals(distance, order):
    system = _system(distance)
    block = GenericCSSGaugeExtractionBlock(system, basis_order=order)
    assert len(block.measurement_blocks) == len(order)
    assert sum(block.measurement_blocks, stim.Circuit()) == block.circuit

    for basis, physical in zip(order, block.measurement_blocks):
        gauges = sorted(
            (g for g in system.active_gauges if g["type"] == basis),
            key=lambda g: g["syn_idx"],
        )
        assert physical[-1].name == ("MX" if basis == "X" else "M")
        assert physical.num_measurements == len(gauges)
        for record_index, gauge in enumerate(gauges):
            assert physical.has_flow(
                stim.Flow(input=_pauli(gauge, system.num_qubits), measurements=[record_index]),
                unsigned=False,
            )
        for logical in system.logical_ops:
            pauli = _pauli(logical, system.num_qubits)
            assert physical.has_flow(stim.Flow(input=pauli, output=pauli), unsigned=False)

    for logical in system.logical_ops:
        pauli = _pauli(logical, system.num_qubits)
        assert block.circuit.has_flow(stim.Flow(input=pauli, output=pauli), unsigned=False)

    # Every color is an actual physical matching, including both endpoint types.
    for layer in block.x_layers + block.z_layers:
        endpoints = [q for edge in layer for q in edge]
        assert len(endpoints) == len(set(endpoints))
    assert block.cnot_depth == sum(block.depth_x if b == "X" else block.depth_z for b in order)


@pytest.mark.parametrize("order", [(), ("Y",), ("X", "invalid")])
def test_gauge_extraction_rejects_empty_or_non_css_orders(order):
    with pytest.raises(ValueError, match="nonempty sequence"):
        GenericCSSGaugeExtractionBlock(_system(), basis_order=order)


def test_gauge_extraction_rejects_accidental_same_basis_ancilla_reuse():
    system = _system()
    x_gauges = [g for g in system.gauges if g["type"] == "X"]
    x_gauges[1]["syn_idx"] = x_gauges[0]["syn_idx"]
    with pytest.raises(ValueError, match="distinct syndrome ancilla"):
        GenericCSSGaugeExtractionBlock(system)


def test_gauge_extraction_accepts_redundant_generators_with_separate_ancillas():
    # A third X generator is a center product of two existing X gauges.
    # Its record must still measure that product correctly and independently.
    patch = BaconShorCode(distance=2)
    patch.add_qubit(5, 5, role="syndrome_x")
    patch.create_stim_gauge(
        {patch.qubit_coords[q]: "X" for q in patch.data_indices},
        syn_coord=(5, 5),
        type="X",
    )
    system = QECSystem()
    system.add_patch(patch, name="redundant")
    block = GenericCSSGaugeExtractionBlock(system, basis_order=("X",))
    measured = sorted(
        (g for g in system.active_gauges if g["type"] == "X"),
        key=lambda g: g["syn_idx"],
    )
    for i, gauge in enumerate(measured):
        assert block.circuit.has_flow(
            stim.Flow(input=_pauli(gauge, system.num_qubits), measurements=[i]),
            unsigned=False,
        )
