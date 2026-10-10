"""Finalization and noise injection respect supplied complete output copies."""

from types import SimpleNamespace

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.experiment import QECExperiment
from lightstim.ir.logical_history import LogicalHistoryRelation, logical_history_observable_indices
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig


def _builder():
    tracker = SyndromeTracker(2)
    tracker.total_measurements = 2
    tracker.total_observables = 1
    tracker.logical_history = [LogicalHistoryRelation((0,), 0, (0,))]
    system = SimpleNamespace(index_map={(0, 0): 0, (1, 0): 1},
                             data_coords=[(0, 0), (1, 0)])
    builder = CircuitBuilder(tracker, system)
    builder.circuit = stim.Circuit("R 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
    return builder


def _output_copy():
    # Same tracked record stream, a different fixed ideal logical value, and
    # one auxiliary qubit to test the moment-noise universe of the copy.
    return stim.Circuit("R 0 1 2\nTICK\nX 0\nTICK\nM 0 1\n"
                        "OBSERVABLE_INCLUDE(0) rec[-1]")


@pytest.mark.parametrize("noisy", [False, True])
def test_experiment_finalizes_the_supplied_copy_and_preserves_the_buffer(noisy):
    builder = _builder()
    original = builder.circuit.copy()
    experiment = SimpleNamespace(
        builder=builder,
        noise_params=NoiseConfig(p_1q=0.01, p_idle=0.02) if noisy else None,
        noise_model="circuit_level_with_idling",
    )
    outputs = []
    for _ in range(2):
        supplied = _output_copy()
        supplied_original = supplied.copy()
        result = QECExperiment._inject_noise(experiment, supplied)
        assert logical_history_observable_indices(result) == (1,)
        assert result.num_observables == 2
        assert result.num_qubits == 3
        assert supplied[:len(supplied_original)] == supplied_original
        assert result.reference_sample().tolist() == [True, False]
        native = [inst for inst in result if inst.name == "OBSERVABLE_INCLUDE"
                  and inst.gate_args_copy() == [0.0]]
        assert len(native) == 1 and native[0].targets_copy() == [stim.target_rec(-1)]
        result.detector_error_model()
        if noisy:
            # Qubit 2 exists only in the supplied circuit, but participates in
            # its idle-noise universe. Noise must not use the build buffer.
            assert any(inst.name == "DEPOLARIZE1" and stim.GateTarget(2) in inst.targets_copy()
                       for inst in result)
        else:
            assert result is supplied
        outputs.append(result)
    assert outputs[0] == outputs[1]
    assert builder.circuit == original
    assert builder.tracker.total_observables == 1


def test_supplied_copy_with_untracked_records_is_rejected_before_mutation():
    builder = _builder()
    supplied = _output_copy() + stim.Circuit("M 2")
    original = supplied.copy()
    with pytest.raises(ValueError, match="complete circuit"):
        builder.build_noisy_circuit(NoiseConfig(), circuit=supplied)
    assert supplied == original
    assert builder.tracker.total_observables == 1
