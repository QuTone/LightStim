"""Clifft adapter checks for physical sampling and raw record semantics."""

import importlib.util
import os
from queue import Empty
import sys

import numpy as np
import pytest

from lightstim.simulation.clifft_backend import ClifftSampler


pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def clifft_python():
    executable = os.environ.get("CLIFFT_PYTHON")
    if executable:
        return executable
    if sys.version_info >= (3, 12) and importlib.util.find_spec("clifft") is not None:
        return sys.executable
    pytest.skip("Set CLIFFT_PYTHON to a Python environment with Clifft installed")


@pytest.mark.parametrize(("gate", "y_sign"), [("T", 1), ("T_DAG", -1)])
def test_true_t_expectations_and_measurements(clifft_python, monkeypatch, gate, y_sign):
    monkeypatch.setenv("CLIFFT_PYTHON", "missing-clifft-python")
    circuit = f"""
        RX 0
        {gate}[physical] 0
        EXP_VAL X0
        EXP_VAL Y0
        MX 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    with ClifftSampler(circuit, python_executable=clifft_python, seed=29) as sampler:
        batch = sampler.sample_batch(2048)

    assert batch.detectors.shape == (2048, 0)
    assert batch.measurements.shape == batch.observables.shape == (2048, 1)
    assert batch.expectations.shape == (2048, 2)
    np.testing.assert_array_equal(batch.measurements, batch.observables)
    amplitude = 1 / np.sqrt(2)
    np.testing.assert_allclose(batch.expectations[:, 0], amplitude, atol=1e-12, rtol=0)
    np.testing.assert_allclose(batch.expectations[:, 1], y_sign * amplitude, atol=1e-12, rtol=0)
    mean_x = 1 - 2 * batch.measurements.mean()
    assert mean_x == pytest.approx(amplitude, abs=0.06)


def test_raw_parities_are_kept_without_selection(clifft_python, monkeypatch):
    monkeypatch.setenv("CLIFFT_PYTHON", clifft_python)
    circuit = """
        R 0
        X 0
        M 0
        DETECTOR[post-select](1, 2, 3) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    with ClifftSampler(circuit) as sampler:
        batch = sampler.sample_batch(8)

    expected = np.ones((8, 1), dtype=np.uint8)
    np.testing.assert_array_equal(batch.measurements, expected)
    np.testing.assert_array_equal(batch.detectors, expected)
    np.testing.assert_array_equal(batch.observables, expected)
    assert batch.expectations.shape == (8, 0)


def test_measurement_noise_flips_record_without_flipping_state(clifft_python):
    with ClifftSampler("R 0\nM(1) 0\nM 0", python_executable=clifft_python) as sampler:
        batch = sampler.sample_batch(8)

    np.testing.assert_array_equal(batch.measurements, [[1, 0]] * 8)
    assert batch.detectors.shape == batch.observables.shape == batch.expectations.shape == (8, 0)


def test_seeded_batches_repeat_across_samplers(clifft_python):
    circuit = "RX 0\nT 0\nMX 0"
    results = []
    for _ in range(2):
        with ClifftSampler(circuit, python_executable=clifft_python, seed=401) as sampler:
            with pytest.raises(ValueError, match="shots"):
                sampler.sample_batch(0)
            results.append([sampler.sample_batch(256).measurements for _ in range(2)])
        with pytest.raises(RuntimeError, match="closed"):
            sampler.sample_batch(1)
        sampler.close()

    first_sampler, second_sampler = results
    for left, right in zip(first_sampler, second_sampler):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first_sampler[0], first_sampler[1])


def test_missing_clifft_dependency_is_reported(monkeypatch, tmp_path):
    # Make the worker see a missing dependency even when Clifft is installed.
    (tmp_path / "sitecustomize.py").write_text("import sys\nsys.modules['clifft'] = None\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("CLIFFT_PYTHON", raising=False)

    with pytest.raises(RuntimeError, match="(?i)clifft"):
        ClifftSampler("R 0\nM 0")


def test_peak_rank_limit_is_reported(clifft_python):
    with pytest.raises(RuntimeError, match="peak_rank"):
        ClifftSampler("RX 0\nT 0\nMX 0", python_executable=clifft_python, max_peak_rank=0)


def test_sampling_timeout_closes_worker(clifft_python, monkeypatch):
    def response_timeout(*, timeout):
        raise Empty

    with ClifftSampler("R 0\nM 0", python_executable=clifft_python) as sampler:
        process = sampler._process
        # Exercise the sampling timeout path after compilation without relying on timing.
        monkeypatch.setattr(sampler._responses, "get", response_timeout)
        with pytest.raises(TimeoutError, match="sampling"):
            sampler.sample_batch(1)
        assert process.poll() is not None
        with pytest.raises(RuntimeError, match="closed"):
            sampler.sample_batch(1)
