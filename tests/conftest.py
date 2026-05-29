"""
Pytest configuration and shared fixtures.

Test tiers (via markers):
  smoke       — <60s total, run on every commit / PR.
  integration — <5min total, run before merge. Default if no marker.
  slow        — >5min, run manually or on release.

CI command:
  pytest tests/ -m "not slow" --timeout=60 -q
"""
import io
import contextlib
import pytest
import stim
import numpy as np


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast tests (<60s total), safe for every commit")
    config.addinivalue_line("markers", "integration: moderate tests (<5min), run before merge")
    config.addinivalue_line("markers", "slow: long-running tests (>5min), run manually")


# ── Shared assertion helpers (importable by all test modules) ─────────────────

def assert_valid_circuit(circuit: stim.Circuit):
    assert circuit.num_qubits > 0, "circuit has no qubits"
    assert circuit.num_detectors > 0, "circuit has no detectors"
    assert circuit.num_observables > 0, "circuit has no observables"


def assert_noiseless(circuit: stim.Circuit, shots: int = 200):
    """Noiseless circuit must produce zero detection events and zero logical errors."""
    dets, obs = circuit.compile_detector_sampler(seed=42).sample(
        shots=shots, separate_observables=True
    )
    assert not np.any(dets), "unexpected detection events in noiseless circuit"
    assert not np.any(obs),  "unexpected logical errors in noiseless circuit"


def assert_dem_valid(circuit: stim.Circuit):
    """DEM construction must succeed and have matching detector/observable counts."""
    dem = circuit.flattened().detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == circuit.num_observables


def build_quiet(fn):
    """Suppress stdout while building a circuit (protocols print progress)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()
