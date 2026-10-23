"""End-to-end benchmark accounting and append semantics."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "benchmarks/logical_ops/run_rotated_enlarge.py"
pytestmark = pytest.mark.smoke


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_cli_quick_sweep_records_counts_config_and_appends_changed_budget(tmp_path):
    output = tmp_path / "enlarge.csv"
    command = [sys.executable, str(RUNNER), "--quick", "--p-values", "0", "0.001",
               "--max-shots", "17", "--max-errors", "100", "--batch-size", "8",
               "--workers", "1", "--output", str(output)]
    subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True, timeout=60)
    rows = _read(output)
    assert len(rows) == 8
    assert {(r["target_distance"], r["basis"], r["p"]) for r in rows} == {
        (d, b, p) for d in ("5", "7") for b in ("X", "Z") for p in ("0.0", "0.001")
    }
    assert len({r["run_id"] for r in rows}) == 1
    assert len({r["row_id"] for r in rows}) == 8
    for row in rows:
        assert row["shots"] == row["accepted"] == "17"
        assert row["p_idle"] == "0.0"
        errors = int(row["errors"])
        assert 0 <= errors <= 17
        assert float(row["logical_error_rate"]) == errors / 17
        assert 0 <= float(row["ci95_low"]) <= errors / 17 <= float(row["ci95_high"]) <= 1
        assert len(row["circuit_sha256"]) == 64
        config = json.loads(row["config_json"])
        assert config["sampling"]["max_shots"] == 17
        assert config["protocol"]["target_distance"] == int(row["target_distance"])
        assert config["noise"]["p_idle"] == 0
        assert config["decoder"]["name"] == "pymatching"
        if row["p"] == "0.0":
            assert errors == 0
            # z^2/(n+z^2) is the Wilson upper endpoint for zero errors.
            assert float(row["ci95_high"]) == pytest.approx(1.96**2 / (17 + 1.96**2))

    # The same path must accept a different budget instead of silently skipping.
    subprocess.run(command + ["--target-distances", "5", "--bases", "Z",
                               "--p-values", "0", "--max-shots", "19"],
                   cwd=REPO, check=True, capture_output=True, text=True, timeout=30)
    appended = _read(output)
    assert appended[:8] == rows
    assert len(appended) == 9
    assert appended[-1]["shots"] == appended[-1]["accepted"] == "19"
    assert appended[-1]["errors"] == "0"
    assert appended[-1]["run_id"] != rows[0]["run_id"]


@pytest.mark.parametrize("arguments", [
    ["--source-distance", "4"], ["--target-distances", "3"],
    ["--rounds", "0"], ["--p-values", "nan"], ["--p-idle", "-0.1"],
    ["--max-shots", "0"], ["--workers", "0"],
])
def test_cli_rejects_invalid_configuration_without_output(tmp_path, arguments):
    output = tmp_path / "invalid.csv"
    result = subprocess.run([sys.executable, str(RUNNER), "--quick", *arguments,
                             "--output", str(output)], cwd=REPO, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 2
    assert not output.exists()


def test_cli_rejects_incompatible_existing_csv(tmp_path):
    output = tmp_path / "other.csv"
    output.write_text("old,header\n1,2\n")
    result = subprocess.run([sys.executable, str(RUNNER), "--quick", "--output", str(output)],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "different CSV schema" in result.stderr
    assert output.read_text() == "old,header\n1,2\n"
