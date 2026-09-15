"""H-family CLI accounting, complete-detector acceptance, and checkpoints."""

import csv
import math
from pathlib import Path
import subprocess
import sys

import pytest
import stim

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "benchmarks/memory/run_h_code.py"
sys.path.insert(0, str(RUNNER.parent))
from run_h_code import sample_postselected_memory

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("fault,shots,accepted,errors", [
    ("", 37, 37, 0),
    ("X_ERROR(1) 1", 16, 16, 16),
    ("X_ERROR(1) 2", 37, 0, 0),
])
def test_postselection_counts_all_detectors_and_logicals(fault, shots, accepted, errors):
    # The last data readout also supplies the last detector. A fault there
    # must reject the shot; an error in the second logical must be counted.
    c = stim.Circuit(f"R 0 1 2\n{fault}\nM 0 1 2\n"
                     "DETECTOR rec[-3]\nDETECTOR rec[-1]\n"
                     "OBSERVABLE_INCLUDE(0) rec[-3]\nOBSERVABLE_INCLUDE(1) rec[-2]")
    result = sample_postselected_memory(
        c, max_shots=37, max_errors=7, batch_size=16, seed=98,
    )
    assert (result["shots"], result["accepted"], result["errors"]) == (shots, accepted, errors)
    assert result["rejected"] == shots - accepted
    if accepted:
        assert result["logical_error_rate"] == errors / accepted
    else:
        assert math.isnan(result["logical_error_rate"])


def test_cli_sweeps_and_resumes_complete_configurations(tmp_path):
    output = tmp_path / "memory.csv"
    command = [sys.executable, str(RUNNER), "--n", "6", "8", "--basis", "X", "Z",
               "--se-circuits", "dedicated", "coloration", "--p-values", "0",
               "--max-shots", "37", "--max-errors", "7", "--batch-size", "16",
               "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {(r["n_data"], r["basis"], r["se_circuit"]) for r in rows} == {
        (n, b, s) for n in ("6", "8") for b in ("X", "Z") for s in ("dedicated", "coloration")
    }
    for row in rows:
        assert int(row["k"]) == int(row["n_data"]) - 4
        assert int(row["n_total"]) == int(row["n_data"]) + 4
        assert row["shots"] == row["accepted"] == "37"
        assert row["errors"] == row["rejected"] == "0"
        assert row["postselection"] == "all_detectors_including_readout"
        assert row["decoder_name"] == "none"
    before = output.read_bytes()
    resumed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    assert resumed.stdout.count("Skip completed:") == 8
    assert output.read_bytes() == before
    # Noise settings belong to the checkpoint key, even at zero gate noise.
    subprocess.run(command + ["--p-idle", "0.001"], check=True, capture_output=True, timeout=30)
    with output.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 16


@pytest.mark.parametrize("arguments", [["--n", "7"], ["--rounds", "0"],
                                      ["--p-values", "nan"], ["--max-shots", "0"]])
def test_cli_rejects_invalid_configuration_before_output(tmp_path, arguments):
    output = tmp_path / "invalid.csv"
    result = subprocess.run([sys.executable, str(RUNNER), *arguments, "--output", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert not output.exists()
