"""Sampling must follow the physical circuit, independently of worker count."""

from threading import Lock
from types import SimpleNamespace

import pytest
import stim

from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline
from lightstim.simulation.decoder_backend.worker import _decode_worker_cpu


pytestmark = pytest.mark.smoke


@pytest.fixture
def forbid_dem_sampling(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Simulation must sample the circuit, not the decoder DEM")

    monkeypatch.setattr(stim.DetectorErrorModel, "compile_sampler", fail)


def _observable_circuit():
    return stim.Circuit("""
        R 0
        X_ERROR(0.25) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)


def _gauge_circuit():
    return stim.Circuit("""
        RX 0
        R 1
        M 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)


def _pipeline(num_workers, **kwargs):
    return SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=256,
        max_errors=257,
        batch_size=64,
        num_workers=num_workers,
        print_progress=False,
        progress_poll_interval_sec=0.01,
        **kwargs,
    )


def test_single_worker_does_not_sample_dem(forbid_dem_sampling):
    stats = _pipeline(1).run(_observable_circuit())
    assert stats.shots == stats.post_selected_shots == 256
    assert stats.errors == 0


def test_parallel_worker_does_not_sample_dem(forbid_dem_sampling):
    # Invoke the worker directly so the guard also works on spawn-based platforms.
    shots, kept, errors, completed = [SimpleNamespace(value=0) for _ in range(4)]
    _decode_worker_cpu(
        circuit=_observable_circuit(),
        decoder_name="pymatching",
        decoder_params={},
        decoder_backend="cpu",
        batch_size=64,
        max_shots=256,
        max_errors=257,
        post_select_indices=[],
        post_select_observable_indices=None,
        post_select_corrected_observable_indices=None,
        target_observable_indices=None,
        shots_counter=shots,
        post_counter=kept,
        errors_counter=errors,
        lock=Lock(),
        completed_counter=completed,
    )
    assert shots.value == completed.value == kept.value == 256
    assert errors.value == 0


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("error_probability", [0.25, 1.0])
def test_correlated_post_selection(num_workers, error_probability):
    circuit = stim.Circuit(f"""
        R 0 1
        E({error_probability}) X0 X1
        M 0 1
        DETECTOR[post-select] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    stats = _pipeline(num_workers).run(circuit)

    assert stats.shots == 256
    assert stats.errors == 0
    if error_probability == 1.0:
        assert stats.post_selected_shots == 0
    else:
        assert 0 < stats.post_selected_shots < stats.shots


@pytest.mark.parametrize("num_workers", [1, 2])
def test_gauge_detector_option_reaches_all_workers(num_workers):
    with pytest.warns(UserWarning, match="allow_gauge_detectors=True"):
        stats = _pipeline(num_workers, allow_gauge_detectors=True).run(_gauge_circuit())

    assert stats.shots == stats.post_selected_shots == 256
    assert stats.errors == 0


@pytest.mark.parametrize("num_workers", [1, 2])
def test_invalid_circuit_does_not_return_successful_stats(num_workers):
    exception = ValueError if num_workers == 1 else RuntimeError
    message = "non-deterministic detectors" if num_workers == 1 else "Simulation worker failed"
    with pytest.raises(exception, match=message):
        _pipeline(num_workers).run(_gauge_circuit())
