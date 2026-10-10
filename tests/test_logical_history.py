"""Historical parity dependencies are separate from the live logical tableau.

The independent oracle here is Stim's physical-circuit flow generator, not
the tracker's own decomposition or its observable-count bookkeeping.
"""

import contextlib
import io

import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker, UNMEASURED_STAB_RECORD
from lightstim.ir.logical_history import LOGICAL_HISTORY_TAG, logical_history_observable_indices
from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment


def _word(records):
    word = 0
    for record in records:
        word ^= 1 << record
    return word


def _rank(rows):
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return len(pivots)


def _annotation_rows(circuit):
    detectors, observables = [], {}
    offset = 0
    for instruction in circuit.flattened():
        if instruction.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            row = _word(
                offset + target.value for target in instruction.targets_copy()
            )
            if instruction.name == "DETECTOR":
                detectors.append(row)
            else:
                key = int(instruction.gate_args_copy()[0])
                observables[key] = observables.get(key, 0) ^ row
        offset += instruction.num_measurements
    return detectors, observables


def _all_fixed_input_parities(circuit):
    rows = [
        _word(flow.measurements_copy())
        for flow in circuit.flow_generators()
        if flow.input_copy().weight == 0 and flow.output_copy().weight == 0
    ]
    # For these initialized, terminally measured circuits, the record-only
    # generators already form a complete basis. Check this independently
    # rather than silently assuming the extracted subset is complete.
    assert _rank(rows) == circuit.count_determined_measurements()
    return rows


def _assert_complete_parity_space(circuit):
    detectors, observables = _annotation_rows(circuit)
    actual = detectors + list(observables.values())
    expected = _all_fixed_input_parities(circuit)
    assert _rank(actual) == _rank(expected)
    assert _rank(actual + expected) == _rank(expected)


def _two_patch_zz(rounds, *, teleport=False, first_readout=None):
    experiment = TwoPatchLSExperiment(
        patch1_config={"distance": 3},
        patch2_config={"distance": 3},
        offset=(0, 6),
        interaction_type="ZZ",
        initial_state_patch1="Z",
        initial_state_patch2="X" if teleport else "Z",
        measure_state_patch1=first_readout or ("X" if teleport else "Z"),
        measure_state_patch2="Z",
        rounds=rounds,
        noise_params=None,
        rotate_patch1=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = experiment.build()
    return experiment, circuit


@pytest.mark.parametrize("rounds", [1, 3, 5])
def test_joint_zz_history_closes_missing_parity_without_changing_native_targets(rounds):
    experiment, circuit = _two_patch_zz(rounds)
    tracker = experiment.tracker
    # Remove only the explicitly tagged new targets to inspect preserved
    # native target supports. The public build has already exported history.
    before = stim.Circuit()
    for instruction in circuit:
        if not (isinstance(instruction, stim.CircuitInstruction)
                and instruction.name == "OBSERVABLE_INCLUDE"
                and instruction.tag == LOGICAL_HISTORY_TAG):
            before.append(instruction)
    native_detectors, native_observables = _annotation_rows(before)
    native_rows = native_detectors + list(native_observables.values())
    expected = _all_fixed_input_parities(circuit)

    assert circuit.num_observables == 3
    assert logical_history_observable_indices(circuit) == (2,)
    assert _rank(native_rows) + 1 == _rank(expected)
    assert len(tracker.logical_history) == 1
    relation = tracker.logical_history[0]
    assert relation.logical_indices == (0, 1)
    assert relation.measurement_index == max(relation.records)
    assert circuit.has_flow(stim.Flow(measurements=relation.records), unsigned=True)
    if rounds == 1:
        assert circuit.num_detectors == 48
        assert _rank(native_rows) == 50
        assert _rank(expected) == 51
        assert relation.records == (48, 49, 50)
    if rounds == 5:
        # The pre-surgery memory contains compressed repeated rounds. The
        # archived relation must still use absolute, correctly shifted indices.
        assert any(isinstance(inst, stim.CircuitRepeatBlock) for inst in circuit)

    added = tracker.append_logical_history_observables(circuit)
    assert added == []
    assert circuit.num_observables == 3
    assert circuit[:len(before)] == before
    detectors, observables = _annotation_rows(circuit)
    assert detectors == native_detectors
    assert {key: observables[key] for key in native_observables} == native_observables
    assert observables[2] == _word(relation.records)
    _assert_complete_parity_space(circuit)

    # The reference convention is handled by Stim's measurement-to-detection
    # conversion; adding a deterministic historical target produces no p=0 flips.
    dets, obs = circuit.compile_detector_sampler(seed=19).sample(
        16, separate_observables=True
    )
    assert not dets.any()
    assert not obs.any()
    after = circuit.copy()
    assert tracker.append_logical_history_observables(circuit) == []
    assert circuit == after


@pytest.mark.parametrize("rounds, expected_rank", [(1, 49), (2, 100), (5, 253)])
def test_teleportation_random_joint_measurement_does_not_create_extra_target(
    rounds, expected_rank
):
    experiment, circuit = _two_patch_zz(rounds, teleport=True)
    assert circuit.num_observables == 1
    assert circuit.count_determined_measurements() == expected_rank
    assert experiment.tracker.logical_history == []
    before = circuit.copy()
    assert experiment.tracker.append_logical_history_observables(circuit) == []
    assert circuit == before
    _assert_complete_parity_space(circuit)


def test_joint_history_survives_incompatible_terminal_readout():
    experiment, circuit = _two_patch_zz(1, first_readout="X")
    # Reading patch 1 in X prevents its initial logical Z from closing at the
    # end, but cannot erase a deterministic ZZ relation measured earlier.
    assert len(experiment.tracker.logical_history) == 1
    relation = experiment.tracker.logical_history[0]
    assert relation.records == (48, 49, 50)
    assert circuit.has_flow(stim.Flow(measurements=relation.records), unsigned=True)
    assert len(logical_history_observable_indices(circuit)) == 1
    assert experiment.tracker.append_logical_history_observables(circuit) == []
    _assert_complete_parity_space(circuit)


def _measure_one_retained_qubit(tracker, circuit):
    tracker.process_mid_measurement(
        circuit=circuit,
        forward_symplectic_matrix=np.eye(2, dtype=np.uint8),
        back_propagated_paulis=np.array([[0, 1]], dtype=np.uint8),
        reset_paulis=None,
        measurement_qubit_indices=[0],
        measurement_bases=["Z"],
        measurement_coords=[(0, 0)],
        discarded_measurement_qubit_indices=set(),
    )


def test_logical_dependency_is_captured_with_empty_stabilizer_bank():
    tracker = SyndromeTracker(1, 1)
    tracker.logicals.matrix = np.array([[0, 1]], dtype=np.uint8)
    tracker.logicals.records = [[]]
    circuit = stim.Circuit("R 0\nM 0")
    _measure_one_retained_qubit(tracker, circuit)
    assert circuit.num_detectors == circuit.num_observables == 0
    assert len(tracker.logical_history) == 1
    assert tracker.logical_history[0].records == (0,)
    assert tracker.logical_history[0].logical_indices == (0,)
    assert tracker.append_logical_history_observables(circuit) == [0]
    _assert_complete_parity_space(circuit)


def test_new_random_measurement_is_not_a_logical_history_relation():
    tracker = SyndromeTracker(1, 1)
    tracker.logicals.matrix = np.array([[1, 0]], dtype=np.uint8)
    tracker.logicals.records = [[]]
    circuit = stim.Circuit("RX 0\nM 0")
    _measure_one_retained_qubit(tracker, circuit)
    assert tracker.logical_history == []
    assert tracker.expected_num_logicals == 0
    assert tracker.append_logical_history_observables(circuit) == []
    _assert_complete_parity_space(circuit)


def test_unknown_logical_record_is_not_silently_treated_as_zero():
    tracker = SyndromeTracker(1, 1)
    tracker.logicals.matrix = np.array([[0, 1]], dtype=np.uint8)
    tracker.logicals.records = [[UNMEASURED_STAB_RECORD]]
    circuit = stim.Circuit("R 0\nM 0")
    with pytest.raises(ValueError, match="[Uu]nknown|[Ss]entinel|[Uu]nmeasured"):
        _measure_one_retained_qubit(tracker, circuit)
    assert tracker.logical_history == []
    assert circuit.num_detectors == circuit.num_observables == 0
