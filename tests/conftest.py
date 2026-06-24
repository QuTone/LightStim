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
import subprocess
import sys
import pytest
import stim
import numpy as np


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast tests (<60s total), safe for every commit")
    config.addinivalue_line("markers", "integration: moderate tests (<5min), run before merge")
    config.addinivalue_line("markers", "slow: long-running tests (>5min), run manually")


# ── Safe native-import guard (importable by all test modules) ─────────────────

def importorskip_safe(module_name: str, reason: str | None = None):
    """Like ``pytest.importorskip``, but survives a native import that *aborts*
    the interpreter instead of raising ``ImportError``.

    Some prebuilt C-extension wheels (e.g. ``tesseract_decoder``, ``relay_bp``)
    are compiled with CPU instructions the host may not support. Importing such
    a wheel raises SIGILL ("illegal instruction"), which kills the whole process
    and core-dumps *below* the Python exception layer — so ``try/except`` and
    ``pytest.importorskip`` cannot catch it, and the entire pytest session dies
    (exit 132). GitHub's ``ubuntu-latest`` pool mixes CPU generations, so this
    surfaces as a flaky, runner-dependent CI failure.

    We first probe the import in a throwaway subprocess. If that subprocess
    crashes or exits non-zero (missing module, SIGILL, segfault, timeout) we
    skip; only on a clean probe do we import the module for real in-process —
    safe, because the same binary on the same CPU just imported cleanly.
    """
    try:
        probe = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.skip(reason or f"{module_name!r} import probe timed out")

    if probe.returncode != 0:
        pytest.skip(
            reason
            or f"{module_name!r} is not safely importable here "
            f"(import probe exited {probe.returncode}); a prebuilt wheel that "
            f"doesn't match this CPU can abort the process with SIGILL — build "
            f"{module_name!r} from source on this machine to run this test."
        )
    return pytest.importorskip(module_name)


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
