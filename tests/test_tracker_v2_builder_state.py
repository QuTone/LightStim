"""Builder state stays consistent across registration and buffered rounds."""
from types import SimpleNamespace

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("initial_patches", [0, 1])
def test_dynamic_subsystem_registration_preserves_preparation_requirements(basis, initial_patches):
    system = QECSystem()
    if initial_patches:
        system.add_patch(BaconShorCode(distance=3), name="first")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    system.register_tracker(tracker)
    system.register_builder(builder)

    def extract():
        block = BaconShorCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(
            block.circuit, measurement_blocks=block.measurement_blocks)

    if initial_patches:
        builder.initialize({q: basis for q in system.data_indices}, system.num_qubits)
        extract()
    existing_data = set(system.data_indices)
    system.add_patch(BaconShorCode(distance=3), name="new", offset=(20, 0))
    builder.initialize({q: basis for q in set(system.data_indices) - existing_data},
                       system.num_qubits)
    extract()
    assert tracker.logicals.count == initial_patches + 1
    builder.apply_data_readout({q: basis for q in system.data_indices})
    circuit = builder.to_stim_circuit()
    assert circuit.num_observables == initial_patches + 1
    assert tracker.expected_num_logicals == 0
    circuit.detector_error_model()
    assert not circuit.compile_detector_sampler(seed=91).sample(
        32, append_observables=True).any()


def _postselected_transport(rounds, *, buffered):
    # Two known Z checks on data 0/1. Each extraction measures data 0 via
    # syndrome 2 and moves data 1's state onto data 0 for the next round.
    checks = [{"pauli": {0: "Z"}}, {"pauli": {1: "Z"}}]
    system = SimpleNamespace(
        num_qubits=3, data_indices={0, 1}, syndrome_indices={2},
        active_syndrome_indices={2}, active_qubit_indices={0, 1},
        qubit_coords={0: (0, 0), 1: (1, 0), 2: (2, 0)},
        active_stabilizers=checks, active_stabilizer_indices={0, 1},
        stabilizers=checks, effective_stabilizer=checks.__getitem__,
    )
    tracker = SyndromeTracker(3, post_select_detector_coords={(2, 0, 0)})
    builder = CircuitBuilder(tracker, system)
    builder.initialize({0: "Z", 1: "Z"}, 3)
    builder.apply_mid_data_readout({1: "Z"})
    tracker.post_select_row_indices = {
        i for i, row in enumerate(tracker.stabilizers.matrix) if row[3 + 1]
    }
    extraction = stim.Circuit("R 2\nSWAP 0 2 0 1\nM 2")
    if buffered:
        builder.apply_syndrome_extraction(extraction, rounds=rounds)
    else:
        for index in range(rounds):
            if index:
                builder.circuit.append("TICK")
            builder.apply_syndrome_extraction(extraction)
    return builder.to_stim_circuit()


def _selected_records(circuit):
    cursor = 0
    selected = []
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR" and instruction.tag == "post-select":
            selected.append(tuple(sorted(cursor + target.value
                                         for target in instruction.targets_copy())))
        cursor += instruction.num_measurements
    return selected


@pytest.mark.parametrize("rounds", [2, 4])
def test_buffered_postselection_uses_absolute_measurement_records(rounds):
    buffered = _postselected_transport(rounds, buffered=True)
    explicit = _postselected_transport(rounds, buffered=False)
    # m0 measures the selected initial Z1. It closes at m2, after two
    # extraction rounds, even when that round is built in a local fragment.
    selected = _selected_records(buffered)
    assert (0, 2) in selected
    assert selected == _selected_records(explicit)
    assert len(selected) == len(set(selected))
    for circuit in (buffered, explicit):
        circuit.detector_error_model()
        assert circuit.has_all_flows([
            stim.Flow(measurements=list(records)) for records in selected
        ], unsigned=False)
        assert not circuit.compile_detector_sampler(seed=19).sample(32).any()
