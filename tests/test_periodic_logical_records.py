"""Repeated syndrome extraction preserves growing logical record expressions."""

import copy

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.color_code import ColorCode, ColorCodeBellFlaggingBlock


def test_record_transition_touching_previous_round_declines_without_mutation(monkeypatch):
    process = CircuitBuilder._process_measurement_blocks
    compress = CircuitBuilder._try_compress_steady_rounds
    probes = []
    decisions = []

    def nonlocal_probe(self, **kwargs):
        tracker = kwargs.get("tracker")
        previous_end = None if tracker is None else tracker.total_measurements
        process(self, **kwargs)
        if tracker is not None:
            # Inject a recurrence involving an older record. This test checks
            # the shortcut's boundary, not the physical validity of that rule.
            records = set(tracker.logicals.records[0])
            records.symmetric_difference_update({previous_end - 1})
            tracker.logicals.records = [sorted(records)]
            probes.append(previous_end)

    def observe(self, **kwargs):
        before = copy.deepcopy(self.tracker)
        result = compress(self, **kwargs)
        decisions.append(result)
        assert result is None
        assert self.tracker.total_measurements == before.total_measurements
        assert self.tracker.meas_rec_to_idx_map == before.meas_rec_to_idx_map
        for name in ("stabilizers", "logicals"):
            actual, expected = getattr(self.tracker, name), getattr(before, name)
            np.testing.assert_array_equal(actual.matrix, expected.matrix)
            assert actual.records == expected.records
        return result

    monkeypatch.setattr(CircuitBuilder, "_process_measurement_blocks", nonlocal_probe)
    monkeypatch.setattr(CircuitBuilder, "_try_compress_steady_rounds", observe)
    MemoryExperiment(qec_patch=ColorCode(distance=5),
                     extraction_block_class=ColorCodeBellFlaggingBlock,
                     rounds=5).build()
    assert len(decisions) == 1
    assert len(probes) == 2


@pytest.mark.parametrize("mode", ["reference", "persistent"])
@pytest.mark.parametrize("distance,rounds,basis", [
    (5, 3, "Z"), (5, 9, "Z"), (5, 5, "X"), (7, 5, "Z"), (11, 5, "Z"),
])
def test_color_memory_matches_explicit_rounds_exactly(
    monkeypatch, mode, distance, rounds, basis,
):
    if mode == "persistent":
        pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native" if mode == "persistent" else "reference")
    compress = CircuitBuilder._try_compress_steady_rounds
    readout = CircuitBuilder.apply_data_readout
    snapshots = []
    accepted_deltas = []

    def observe_compression(self, **kwargs):
        previous = set(self.tracker.logicals.records[0])
        result = compress(self, **kwargs)
        assert result is not None
        accepted_deltas.append(previous ^ set(self.tracker.logicals.records[0]))
        return result

    def observe_readout(self, *args, **kwargs):
        snapshots.append(copy.deepcopy(self.tracker))
        return readout(self, *args, **kwargs)

    def build():
        experiment = MemoryExperiment(
            qec_patch=ColorCode(distance=distance),
            extraction_block_class=ColorCodeBellFlaggingBlock,
            rounds=rounds,
            basis=basis,
            noise_params=NoiseConfig(p_1q=0.001, p_2q=0.002,
                                     p_meas=0.003, p_reset=0.001),
        )
        return experiment.build()

    monkeypatch.setattr(CircuitBuilder, "_try_compress_steady_rounds", observe_compression)
    monkeypatch.setattr(CircuitBuilder, "apply_data_readout", observe_readout)
    compressed = build()
    monkeypatch.setattr(CircuitBuilder, "_try_compress_steady_rounds", lambda self, **kwargs: None)
    explicit = build()

    assert len(accepted_deltas) == 1
    if distance == 5 and basis == "Z":
        assert len(accepted_deltas[0]) == (rounds - 2) * (distance - 1)
    assert any(isinstance(instruction, stim.CircuitRepeatBlock)
               for instruction in compressed)
    # This checks all noise locations, detector rows, and observable targets,
    # not just noiseless samples or equality modulo detector relations.
    assert compressed.flattened() == explicit.flattened()
    compressed.detector_error_model()

    compressed_state, explicit_state = snapshots
    assert compressed_state.total_measurements == explicit_state.total_measurements
    for name in ("stabilizers", "logicals"):
        actual, expected = getattr(compressed_state, name), getattr(explicit_state, name)
        np.testing.assert_array_equal(actual.matrix, expected.matrix)
        assert actual.records == expected.records
        for records in actual.records:
            for record in records:
                assert compressed_state.meas_rec_to_idx_map[record] == explicit_state.meas_rec_to_idx_map[record]
