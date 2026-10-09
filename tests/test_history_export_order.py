"""A finished protocol has the same task IDs in every output copy."""

import contextlib
import io
from types import SimpleNamespace

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.experiment import QECExperiment
from lightstim.ir.logical_history import (
    LOGICAL_HISTORY_TAG,
    LogicalHistoryRelation,
    logical_history_observable_indices,
)
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment


def _builder(reserved_ids):
    tracker = SyndromeTracker(2)
    tracker.total_measurements = 2
    for _ in range(reserved_ids):
        tracker.allocate_observable()
    tracker.logical_history = [LogicalHistoryRelation((0,), 0, (0,))]
    system = SimpleNamespace(
        index_map={(0, 0): 0, (1, 0): 1},
        data_coords=[(0, 0), (1, 0)],
    )
    builder = CircuitBuilder(tracker, system)
    builder.circuit = stim.Circuit(
        "R 0 1\nTICK\nX 0\nTICK\nM 0 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )
    return builder


def _observable_annotations(circuit):
    return [
        (int(inst.gate_args_copy()[0]), tuple(inst.targets_copy()), inst.tag)
        for inst in circuit.flattened()
        if inst.name == "OBSERVABLE_INCLUDE"
    ]


def _finish_copy(builder, circuit, noise_params):
    experiment = SimpleNamespace(
        builder=builder,
        noise_params=noise_params,
        noise_model="circuit_level_with_idling",
    )
    return QECExperiment._inject_noise(experiment, circuit)


@pytest.mark.parametrize("reserved_ids", [1, 3])
@pytest.mark.parametrize("buffer_first", [True, False], ids=["buffer-first", "copy-first"])
@pytest.mark.parametrize(
    "noise_params",
    [None, NoiseConfig(), NoiseConfig(p_1q=0.01, p_idle=0.02)],
    ids=["clean", "zero-noise", "nonzero-noise"],
)
def test_preexport_copies_keep_the_same_ids_in_every_finalization_order(
    reserved_ids, buffer_first, noise_params
):
    builder = _builder(reserved_ids)
    original = builder.circuit.copy()
    copies = [original.copy() for _ in range(3)]

    if buffer_first:
        finished_buffer = builder.to_stim_circuit().copy()
    first_output = _finish_copy(builder, copies[0], noise_params)
    if not buffer_first:
        finished_buffer = builder.to_stim_circuit().copy()
    outputs = [first_output] + [
        _finish_copy(builder, circuit, noise_params) for circuit in copies[1:]
    ]

    expected_annotations = [
        (0, (stim.target_rec(-1),), ""),
        (reserved_ids, (stim.target_rec(-2),), LOGICAL_HISTORY_TAG),
    ]
    for circuit in [finished_buffer, *copies, *outputs]:
        assert logical_history_observable_indices(circuit) == (reserved_ids,)
        assert circuit.num_observables == reserved_ids + 1
        assert _observable_annotations(circuit) == expected_annotations
        circuit.detector_error_model()

    for supplied, output in zip(copies, outputs):
        assert supplied[:len(original)] == original
        assert output.reference_sample().tolist() == [True, False]
        has_noise = any(inst.name == "DEPOLARIZE1" for inst in output)
        assert has_noise == bool(noise_params and noise_params.p_1q)
        # Reusing an already finalized clean copy cannot append another task.
        before = supplied.copy()
        assert _finish_copy(builder, supplied, noise_params) == output
        assert supplied == before

    assert builder.to_stim_circuit() == finished_buffer
    assert builder.circuit[:len(original)] == original


def test_tracker_export_reuses_ids_across_identical_complete_raw_copies():
    builder = _builder(reserved_ids=3)
    copies = [builder.circuit.copy() for _ in range(3)]
    for circuit in copies:
        assert builder.tracker.append_logical_history_observables(circuit) == [3]
        assert logical_history_observable_indices(circuit) == (3,)
        assert circuit.num_observables == 4
        before = circuit.copy()
        assert builder.tracker.append_logical_history_observables(circuit) == []
        assert circuit == before
    assert copies[0] == copies[1] == copies[2]


@pytest.mark.parametrize("direct_tracker", [False, True], ids=["builder", "tracker"])
def test_supplied_native_id_takes_precedence_over_a_previous_history_id(direct_tracker):
    builder = _builder(reserved_ids=1)

    def finish(circuit):
        if direct_tracker:
            builder.tracker.append_logical_history_observables(circuit)
            return circuit
        return _finish_copy(builder, circuit, None)

    raw = builder.circuit.copy()
    finished_buffer = builder.to_stim_circuit().copy()
    supplied = raw.copy()
    supplied.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    native = supplied.copy()

    output = finish(supplied)
    assert output[:len(native)] == native
    assert logical_history_observable_indices(output) == (2,)
    assert output.num_observables == 3
    assert _observable_annotations(output) == [
        (0, (stim.target_rec(-1),), ""),
        (1, (stim.target_rec(-1),), ""),
        (2, (stim.target_rec(-2),), LOGICAL_HISTORY_TAG),
    ]
    assert builder.circuit == finished_buffer
    # A copy-specific native annotation must not change another copy's IDs.
    assert finish(raw) == finished_buffer


def test_copy_first_export_does_not_reserve_ids_in_the_builder():
    builder = _builder(reserved_ids=1)
    supplied = builder.circuit.copy()
    supplied.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 3)
    output = _finish_copy(builder, supplied, None)
    assert logical_history_observable_indices(output) == (4,)
    assert output.num_observables == 5

    finished = builder.to_stim_circuit()
    assert logical_history_observable_indices(finished) == (1,)
    assert finished.num_observables == 2
    assert builder.tracker.allocate_observable() == 2


def test_reservations_after_export_remain_reserved_when_history_id_is_reused():
    builder = _builder(reserved_ids=1)
    raw = builder.circuit.copy()
    builder.to_stim_circuit()
    assert builder.tracker.allocate_observable() == 2

    assert builder.tracker.append_logical_history_observables(raw) == [1]
    assert logical_history_observable_indices(raw) == (1,)
    assert raw.num_observables == 2
    # The unannotated reservation remains occupied, without creating an empty
    # column just because this same history relation was exported again.
    assert builder.tracker.allocate_observable() == 3


def test_two_patch_zz_saved_preexport_copy_has_exactly_the_three_original_tasks(monkeypatch):
    saved = []
    original_to_stim = CircuitBuilder.to_stim_circuit

    def save_complete_preexport_copy(builder, **kwargs):
        saved.append(builder.circuit.copy())
        return original_to_stim(builder, **kwargs)

    monkeypatch.setattr(CircuitBuilder, "to_stim_circuit", save_complete_preexport_copy)
    experiment = TwoPatchLSExperiment(
        patch1_config={"distance": 3},
        patch2_config={"distance": 3},
        offset=(0, 6),
        interaction_type="ZZ",
        initial_state_patch1="Z",
        initial_state_patch2="Z",
        measure_state_patch1="Z",
        measure_state_patch2="Z",
        rounds=1,
        noise_params=None,
        rotate_patch1=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        finished = experiment.build().copy()
    assert len(saved) == 1
    raw = saved[0]
    assert logical_history_observable_indices(raw) == ()
    assert raw.num_observables == 2

    supplied = raw.copy()
    noisy = experiment.builder.build_noisy_circuit(
        NoiseConfig(p_1q=0.001, p_2q=0.001), circuit=supplied
    )
    for circuit in [finished, supplied, noisy]:
        assert logical_history_observable_indices(circuit) == (2,)
        assert circuit.num_observables == 3
        assert _observable_annotations(circuit) == _observable_annotations(finished)
        circuit.detector_error_model()
    assert supplied == finished
    assert experiment.builder.circuit == finished
    assert any(inst.name in {"DEPOLARIZE1", "DEPOLARIZE2"} for inst in noisy)
