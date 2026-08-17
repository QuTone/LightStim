"""Smoke test for the checked-in exact-MLE benchmark."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks" / "mle"))

from run_mle import run_preset


def test_core_mle_benchmark_agrees_with_direct_milp():
    result = run_preset("surface-d5", shots=1)
    assert result["shots"] == 1
    assert result["objective_agreement"]
    assert result["max_objective_delta"] <= 1e-6
