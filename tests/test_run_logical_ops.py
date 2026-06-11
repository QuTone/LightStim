"""
Tests for the §7.1 logical-Pauli tasks in benchmarks/logical_ops/run_logical_ops.py.

Unit-level only (no simulation): task construction, metadata schema,
checkpoint-key uniqueness.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "logical_ops"))

from run_logical_ops import ALL_GATES, _ck_key, build_tasks


@pytest.mark.smoke
class TestPauliTasks:

    def test_pauli_gate_registered(self):
        assert "pauli" in ALL_GATES

    def test_build_pauli_tasks_shape_and_meta(self):
        tasks = build_tasks("pauli", distances=[3], p_values=[1e-3], rounds=2,
                            num_layers_list=[0, 2])
        # 2 pauli/basis pairings × 2 modes × 2 layer counts × 1 d × 1 p
        assert len(tasks) == 8

        subs = {m["sub_experiment"] for _, m in tasks}
        assert subs == {
            "PX_physical_L0", "PX_physical_L2", "PX_frame_L0", "PX_frame_L2",
            "PZ_physical_L0", "PZ_physical_L2", "PZ_frame_L0", "PZ_frame_L2",
        }

        for circuit, meta in tasks:
            assert circuit.num_detectors > 0
            assert meta["gate"] == "pauli"
            assert meta["rounds"] == meta["d"], "pauli experiment is memory-like: rounds = d"

        keys = {_ck_key(m) for _, m in tasks}
        assert len(keys) == len(tasks), "checkpoint keys must be unique per task"


@pytest.mark.smoke
class TestPauliPlotParsing:

    def test_parse_pauli_subexperiments(self):
        import pandas as pd
        from plot_logical_ops import parse_pauli_subexperiments
        df = pd.DataFrame({
            "sub_experiment": ["PX_physical_L0", "PZ_frame_L8"],
            "logical_error_rate": [0.1, 0.2],
        })
        out = parse_pauli_subexperiments(df)
        assert list(out["pauli"]) == ["X", "Z"]
        assert list(out["mode"]) == ["physical", "frame"]
        assert list(out["layers"]) == [0, 8]
