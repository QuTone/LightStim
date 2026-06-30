import numpy as np

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.BB_code import BBCode
from lightstim.qec_code.kasai_code import KasaiCode, KasaiCodeExtractionBlock


def _expected_stabilizer_row(stab, num_qubits):
    row = np.zeros(2 * num_qubits, dtype=np.uint8)
    data = np.array(stab["data_indices"], dtype=int)
    if stab["type"] == "X":
        row[data] = 1
    elif stab["type"] == "Z":
        row[num_qubits + data] = 1
    else:
        raise AssertionError(f"Unexpected stabilizer type {stab['type']!r}")
    return row


def test_kasai_generic_coloration_depth_and_backpropagation():
    code = KasaiCode.from_preset("chen_p96")
    system = QECSystem()
    system.add_patch(code, name="kasai")

    block = KasaiCodeExtractionBlock(system)

    assert block.depth_x == 12
    assert block.depth_z == 12
    assert block.cnot_depth == 24

    for inst in block.circuit:
        if inst.name == "CNOT":
            qubits = [target.value for target in inst.targets_copy()]
            assert len(qubits) == len(set(qubits))

    back_paulis, syn_indices = CircuitBuilder._get_back_propagated_pauli(
        block.circuit,
        system.num_qubits,
    )
    stabilizer_by_syn = {stab["syn_idx"]: stab for stab in system.active_stabilizers}

    for row, syn_idx in zip(back_paulis, syn_indices):
        expected = _expected_stabilizer_row(stabilizer_by_syn[syn_idx], system.num_qubits)
        assert np.array_equal(row.astype(np.uint8), expected)


def test_memory_experiment_uses_generic_css_fallback():
    code = BBCode(
        l=6,
        m=6,
        A=[[3, 0], [0, 1], [0, 2]],
        B=[[0, 3], [1, 0], [2, 0]],
    )
    system = QECSystem()
    system.add_patch(code, name="bb")

    circuit = MemoryExperiment(
        qec_system=system,
        rounds=1,
        noise_params=None,
        basis="Z",
    ).build()

    assert circuit.num_detectors > 0
    assert circuit.num_observables == 12
