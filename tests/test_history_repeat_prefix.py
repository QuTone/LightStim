"""History after an SE prefix must see the tracker's entire logical frame."""

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker


class _ThreeQubitPatch(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        for qubit in range(3):
            self.add_qubit(qubit, 0, "data")
        for qubit in (0, 2):
            self.create_stim_stabilizer({(qubit, 0): "Z"}, type="Z")
        self.create_stim_logical({(1, 0): "Z"}, "Z")
        self.create_stim_logical({(1, 0): "X"}, "X")
        self.num_logicals = 1


def _check_current_state(builder):
    tracker = builder.tracker
    for tableau in (tracker.stabilizers, tracker.logicals):
        for row, records in zip(tableau.matrix, tableau.records):
            assert all(record >= 0 for record in records)
            pauli = stim.PauliString.from_numpy(
                xs=np.asarray(row[:tracker.num_qubits], dtype=np.bool_),
                zs=np.asarray(row[tracker.num_qubits:], dtype=np.bool_),
            )
            assert builder.circuit.has_flow(
                stim.Flow(output=pauli, measurements=records), unsigned=True,
            ), (str(pauli), records)


def _build(rounds, batched):
    system = QECSystem()
    system.add_patch(_ThreeQubitPatch(), name="test")
    builder = CircuitBuilder(SyndromeTracker(3, 1), system)
    builder.initialize({0: "Z", 1: "Z", 2: "Z"}, 3)
    builder.tracker.rebase_stabilizers_onto_code_basis(system)
    blocks = (stim.Circuit("RX 0\nMX 0"), stim.Circuit("CX 0 1\nM 0"))
    body = blocks[0] + blocks[1]
    for _ in range(1 if batched else rounds):
        builder.apply_syndrome_extraction(
            body, rounds=rounds if batched else 1, measurement_blocks=blocks,
        )
        _check_current_state(builder)
    builder.apply_syndrome_extraction(stim.Circuit("CX 1 2\nM 2"))
    _check_current_state(builder)
    builder.apply_data_readout({0: "Z", 1: "Z", 2: "Z"})
    builder.to_stim_circuit()
    return builder


def _rows(circuit):
    offset, detectors, observables = 0, [], {}
    for instruction in circuit.flattened():
        if instruction.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            row = set(offset + target.value for target in instruction.targets_copy())
            if instruction.name == "DETECTOR":
                detectors.append(row)
            else:
                index = int(instruction.gate_args_copy()[0])
                observables.setdefault(index, set()).symmetric_difference_update(row)
        offset += instruction.num_measurements
    return detectors + list(observables.values())


def _rank(rows):
    pivots = {}
    for source in rows:
        row = set(source)
        while row:
            pivot = max(row)
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row.symmetric_difference_update(pivots[pivot])
    return len(pivots)


@pytest.mark.parametrize("rounds", [3, 5])
def test_later_history_includes_the_entire_repeated_logical_frame(rounds):
    explicit = _build(rounds, False)
    batched = _build(rounds, True)
    expected_relation = tuple(range(1, 2 * rounds, 2)) + (2 * rounds,)
    assert batched.tracker.logical_history[0].records == expected_relation
    assert batched.tracker.logical_history == explicit.tracker.logical_history
    for builder in (explicit, batched):
        circuit = builder.circuit
        for relation in builder.tracker.logical_history:
            assert circuit.has_flow(stim.Flow(measurements=relation.records), unsigned=True)
        # Public circuit export emits every archived relation by default.
        assert circuit.num_observables == 2
        circuit.detector_error_model()
        det, obs = circuit.compile_detector_sampler(seed=17).sample(
            128, separate_observables=True,
        )
        assert not det.any() and not obs.any()
    a, b = _rows(explicit.circuit), _rows(batched.circuit)
    assert _rank(a) == _rank(b) == _rank(a + b)
