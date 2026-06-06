import numpy as np
import pytest
import stim

from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline, list_decoders
from lightstim.simulation.decoder_backend._accounting import count_batch
from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices
from lightstim.simulation.decoder_backend.external import ExternalDecoder
from lightstim.simulation.decoder_backend.registry import register_decoder
from lightstim.simulation.simulator import ExperimentTask, QECSimulator


def _simple_observable_circuit(error_probability: float = 0.0) -> stim.Circuit:
    circuit = stim.Circuit()
    if error_probability:
        circuit.append("X_ERROR", [0], error_probability)
    circuit.append("M", [0])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


@pytest.mark.smoke
def test_cpu_decoder_does_not_import_cudaq_qec():
    """CPU decoders must not eagerly import cudaq_qec.

    Regression for the NVML lock contention bug where every multiprocessing
    worker that grabbed a CPU decoder also pulled in cudaq_qec, which forks
    two nvidia-smi probes per import. Hundreds of concurrent workers could
    saturate the NVML global lock and hang the driver. cudaq_qec must stay
    lazy — only loaded when the user actually requests a GPU decoder.
    """
    import subprocess
    import sys

    script = (
        "import sys; "
        "from lightstim.simulation.decoder_backend.registry import get_decoder; "
        "d = get_decoder('pymatching', backend='cpu'); "
        "print('cudaq_qec_imported=' + ('yes' if 'cudaq_qec' in sys.modules else 'no'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "cudaq_qec_imported=no" in result.stdout, (
        f"cudaq_qec was imported by the CPU path:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_multiprocess_unknown_decoder_raises_in_parent():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("definitely_missing"),
        max_shots=10,
        max_errors=1,
        batch_size=5,
        num_workers=2,
        print_progress=False,
    )

    with pytest.raises(ValueError, match="Unknown decoder 'definitely_missing'"):
        pipeline.run(_simple_observable_circuit())


def test_legacy_nvidia_gpu_backend_is_disabled():
    simulator = QECSimulator(backend="nvidia_gpu", num_workers=1)

    with pytest.raises(NotImplementedError, match="placeholder decoder"):
        simulator.run_batch([ExperimentTask(_simple_observable_circuit())])


def test_relay_bp_registered_and_runs():
    """Relay-BP (sinter-native, Pattern A) registers and decodes when installed."""
    pytest.importorskip("relay_bp")

    assert "relay-bp" in list_decoders()
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("relay-bp"),
        max_shots=200,
        max_errors=10_000,
        batch_size=100,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.1))
    assert stats.shots > 0


def test_tesseract_registered_and_runs():
    """Tesseract (sinter-native, Pattern A, lazy import) registers and decodes.

    importorskip imports tesseract_decoder; if the installed wheel doesn't match
    your CPU it may fail to import here — build tesseract_decoder from source in
    that case.
    """
    pytest.importorskip("tesseract_decoder")

    assert "tesseract" in list_decoders()
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("tesseract", params={"det_beam": 40}),
        max_shots=200,
        max_errors=10_000,
        batch_size=100,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.1))
    assert stats.shots > 0


def test_list_decoders_deduplicates_canonical_names():
    names = list_decoders()

    assert names == sorted(set(names))


def test_post_selection_can_reject_all_shots_without_decoding():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=5,
        max_errors=1,
        batch_size=5,
        num_workers=1,
        post_select_detector_indices=[0],
        print_progress=False,
    )

    stats = pipeline.run(_simple_observable_circuit(error_probability=1.0))

    assert stats.shots == 5
    assert stats.post_selected_shots == 0
    assert stats.errors == 0
    assert stats.post_selection_rate == 0.0


# --------------------------------------------------------------------------- #
# ExternalDecoder facade
# --------------------------------------------------------------------------- #
#
# For _simple_observable_circuit the DEM has one error mechanism that flips the
# single detector and the single observable, so H and obs_matrix are both the
# 1x1 identity. A perfect decoder therefore just echoes the syndrome: as a
# "correction" it equals the error column, and as "observables" it equals the
# flipped observable.


class _PerfectSingleExternal(ExternalDecoder):
    """Correction output via decode_single only — exercises the single->batch bridge."""

    output_type = "correction"

    def decode_single(self, syndrome):
        return syndrome.astype(np.uint8), None  # flag=None => converged


class _PerfectBatchExternal(ExternalDecoder):
    """Correction output via decode_batch only."""

    output_type = "correction"

    def decode_batch(self, syndromes):
        return syndromes.astype(np.uint8), None


class _ObservablesExternal(ExternalDecoder):
    """Direct observable-flip output — skips the obs_matrix multiply."""

    output_type = "observables"

    def decode_single(self, syndrome):
        return syndrome.astype(np.uint8), None


class _FlakyExternal(ExternalDecoder):
    """Predicts perfectly but flags *every* shot as a failed decode."""

    output_type = "observables"

    def decode_batch(self, syndromes):
        n = syndromes.shape[0]
        return syndromes.astype(np.uint8), np.zeros(n, dtype=bool)  # all failed


register_decoder("test-ext-single", _PerfectSingleExternal)
register_decoder("test-ext-batch", _PerfectBatchExternal)
register_decoder("test-ext-obs", _ObservablesExternal)
register_decoder("test-ext-flaky", _FlakyExternal)


@pytest.mark.parametrize(
    "name", ["test-ext-single", "test-ext-batch", "test-ext-obs"]
)
def test_external_decoder_registered_and_corrects(name):
    assert name in list_decoders()
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(name),
        max_shots=500,
        max_errors=10_000,
        batch_size=250,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.2))
    # Echo decoder is exact for this code: zero residual logical errors.
    assert stats.shots >= 500
    assert stats.errors == 0


def test_external_decoder_missing_output_type_raises():
    class _NoOutputType(ExternalDecoder):
        def decode_single(self, syndrome):
            return syndrome, None

    with pytest.raises(ValueError, match="output_type"):
        _NoOutputType().compile_decoder_for_dem(
            dem=_simple_observable_circuit().detector_error_model()
        )


def test_external_decoder_no_decode_method_raises():
    class _NoDecode(ExternalDecoder):
        output_type = "correction"

    with pytest.raises(NotImplementedError, match="decode_batch or decode_single"):
        _NoDecode().compile_decoder_for_dem(
            dem=_simple_observable_circuit().detector_error_model()
        )


def test_failure_flag_policy_error_counts_every_shot():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("test-ext-flaky", on_decode_failure="error"),
        max_shots=300,
        max_errors=10_000,
        batch_size=300,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.2))
    # Predictions are perfect, but every shot is flagged failed => all count as errors.
    assert stats.post_selected_shots == stats.shots
    assert stats.errors == stats.post_selected_shots
    assert stats.logical_error_rate == 1.0


def test_failure_flag_policy_discard_empties_denominator():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("test-ext-flaky", on_decode_failure="discard"),
        max_shots=300,
        max_errors=10_000,
        batch_size=300,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.2))
    assert stats.shots >= 300
    assert stats.post_selected_shots == 0
    assert stats.errors == 0


def test_failure_flag_policy_ignore_trusts_prediction():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("test-ext-flaky", on_decode_failure="ignore"),
        max_shots=300,
        max_errors=10_000,
        batch_size=300,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.2))
    assert stats.post_selected_shots == stats.shots
    assert stats.errors == 0


def test_failure_flag_policy_error_multiprocess():
    """Same 'error' policy through the multi-worker code path in worker.py."""
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("test-ext-flaky", on_decode_failure="error"),
        max_shots=400,
        max_errors=10_000,
        batch_size=100,
        num_workers=2,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.2))
    assert stats.post_selected_shots > 0
    assert stats.errors == stats.post_selected_shots


# --------------------------------------------------------------------------- #
# count_batch flag accounting (unit-level)
# --------------------------------------------------------------------------- #


def _pack(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")


def test_count_batch_flags_error_policy():
    obs = np.array([[0], [0], [0]], dtype=np.uint8)
    pred = _pack(np.array([[0], [0], [0]], dtype=np.uint8))  # all predictions correct
    flags = np.array([True, False, True])  # shot 1 failed
    kept, errors = count_batch(
        obs_filtered=obs,
        pred_packed=pred,
        post_select_corrected_observable_indices=None,
        target_observable_indices=None,
        flags=flags,
        on_decode_failure="error",
    )
    assert kept == 3
    assert errors == 1  # only the failed shot


def test_count_batch_flags_discard_policy():
    obs = np.array([[0], [0], [0]], dtype=np.uint8)
    pred = _pack(np.array([[1], [0], [0]], dtype=np.uint8))  # shot 0 is a real error
    flags = np.array([True, False, False])  # shots 1,2 failed
    kept, errors = count_batch(
        obs_filtered=obs,
        pred_packed=pred,
        post_select_corrected_observable_indices=None,
        target_observable_indices=None,
        flags=flags,
        on_decode_failure="discard",
    )
    assert kept == 1  # failed shots dropped from denominator
    assert errors == 1  # the surviving shot 0 mispredicts


def test_dem_to_matrices_repeated_targets_cancel():
    """stim treats a target listed an even number of times as parity (cancel),
    so dem_to_matrices must XOR rather than assign."""
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D0 D1 L0 L0
        error(0.2) D1 D2 L1
        """
    )
    H, obs, priors = dem_to_matrices(dem)
    # col 0: D0 listed twice -> cancels; D1 stays; L0 twice -> cancels.
    assert H[:, 0].tolist() == [0, 1, 0]
    assert obs[:, 0].tolist() == [0, 0]
    # col 1: ordinary single targets.
    assert H[:, 1].tolist() == [0, 1, 1]
    assert obs[:, 1].tolist() == [0, 1]
    assert priors.tolist() == [0.1, 0.2]


def test_dem_to_matrices_circuit_dem_is_binary():
    """Regression: circuit-generated DEMs (no repeated targets) stay 0/1."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=5e-3,
    )
    H, obs, _ = dem_to_matrices(circuit.detector_error_model())
    assert set(np.unique(H).tolist()) <= {0, 1}
    assert set(np.unique(obs).tolist()) <= {0, 1}


def test_count_batch_failed_shot_survives_post_decode_ps_under_error_policy():
    """A decode-failed shot that post-decode PS would reject must still count as a
    logical error (and stay in the denominator) under on_decode_failure='error' —
    not silently vanish to (kept=0, errors=0)."""
    obs = np.array([[1], [0]], dtype=np.uint8)            # shot0 obs=1, shot1 obs=0
    pred = _pack(np.array([[0], [0]], dtype=np.uint8))    # corrected: shot0=1 (rejected), shot1=0
    flags = np.array([False, True])                       # shot0 failed to decode

    kept, errors = count_batch(
        obs_filtered=obs,
        pred_packed=pred,
        post_select_corrected_observable_indices=[0],
        target_observable_indices=None,
        flags=flags,
        on_decode_failure="error",
    )
    assert (kept, errors) == (2, 1)  # failed shot0 counted; converged shot1 kept, no error

    # discard heralds the failed shot away; ignore trusts the prediction (PS rejects shot0).
    discard = count_batch(
        obs_filtered=obs, pred_packed=pred,
        post_select_corrected_observable_indices=[0], target_observable_indices=None,
        flags=flags, on_decode_failure="discard",
    )
    ignore = count_batch(
        obs_filtered=obs, pred_packed=pred,
        post_select_corrected_observable_indices=[0], target_observable_indices=None,
        flags=flags, on_decode_failure="ignore",
    )
    assert discard == (1, 0)
    assert ignore == (1, 0)


def test_count_batch_no_flags_matches_plain_counting():
    obs = np.array([[0], [1], [0]], dtype=np.uint8)
    pred = _pack(np.array([[0], [0], [0]], dtype=np.uint8))  # shot 1 mispredicted
    kept, errors = count_batch(
        obs_filtered=obs,
        pred_packed=pred,
        post_select_corrected_observable_indices=None,
        target_observable_indices=None,
        flags=None,
        on_decode_failure="error",
    )
    assert kept == 3
    assert errors == 1
