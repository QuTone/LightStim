"""MLE integration tests for the logical-circuits benchmark runner."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNNER_DIR = REPO / "benchmarks" / "logical_circuits"
sys.path.insert(0, str(RUNNER_DIR))

from run_logical_circuits import (
    _BELL_COLS,
    _BELL_RESULT_KEYS,
    _ck_key,
    _decoder_config,
    _ensure_result_schema,
)


def test_logical_runner_mle_config_propagates_controls():
    cfg = _decoder_config(
        "mle-ilp", mle_time_limit=2.5, on_decode_failure="discard"
    )
    assert cfg.name == "mle-ilp"
    assert cfg.backend == "cpu"
    assert cfg.params == {"time_limit": 2.5}
    assert cfg.on_decode_failure == "discard"


def test_logical_checkpoint_key_includes_decoder_controls():
    base = {
        "experiment": "bell_tele", "protocol": "tg", "state": "Z",
        "routing_mult": 1, "d": 3, "rounds": "pre=3 mid=1 post=1",
        "p": 1e-3, "decoder": "mle-ilp", "decoder_time_limit": 0.0,
        "on_decode_failure": "error",
    }
    assert _ck_key(base, _BELL_RESULT_KEYS) != _ck_key(
        {**base, "decoder": "pymatching"}, _BELL_RESULT_KEYS
    )
    assert _ck_key(base, _BELL_RESULT_KEYS) != _ck_key(
        {**base, "decoder_time_limit": 1.0}, _BELL_RESULT_KEYS
    )
    assert _ck_key(base, _BELL_RESULT_KEYS) != _ck_key(
        {**base, "on_decode_failure": "discard"}, _BELL_RESULT_KEYS
    )


def test_logical_legacy_csv_migration(tmp_path):
    path = tmp_path / "legacy.csv"
    old_columns = [
        column for column in _BELL_COLS
        if column not in {"decoder_time_limit", "on_decode_failure"}
    ]
    pd.DataFrame([{column: 0 for column in old_columns}]).to_csv(path, index=False)

    _ensure_result_schema(path, _BELL_COLS)

    migrated = pd.read_csv(path)
    assert list(migrated.columns) == _BELL_COLS
    assert migrated["decoder_time_limit"].iloc[0] == 0.0
    assert migrated["on_decode_failure"].iloc[0] == "error"


def test_logical_cli_advertises_mle_controls():
    result = subprocess.run(
        [sys.executable, str(RUNNER_DIR / "run_logical_circuits.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "mle-ilp" in result.stdout
    assert "--mle-time-limit" in result.stdout
    assert "--on-decode-failure" in result.stdout
