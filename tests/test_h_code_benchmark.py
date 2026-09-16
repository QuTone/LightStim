"""H-family CLI accounting, complete-detector acceptance, and checkpoints."""

import csv
import math
from pathlib import Path
import subprocess
import sys

import pytest
import stim

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "benchmarks/memory/run_memory.py"
sys.path.insert(0, str(RUNNER.parent))
from run_memory import sample_postselected_memory, build_circuit, _task_key

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
    command = [sys.executable, str(RUNNER), "--codes", "h_code", "--h-n", "6", "8",
               "--mode", "full_postselection", "--basis", "X", "Z",
               "--h-se-circuits", "dedicated", "coloration", "--p-values", "0",
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
        assert row["mode"] == "full_postselection"
        assert row["h_n"] == row["n_data"]
        assert row["distance"] == "2"
        assert row["decoder_name"] == "none"
    before = output.read_bytes()
    resumed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    assert "8 skipped" in resumed.stdout
    assert output.read_bytes() == before
    # Noise settings belong to the checkpoint key, even at zero gate noise.
    subprocess.run(command + ["--p-idle", "0.001"], check=True, capture_output=True, timeout=30)
    with output.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 16


@pytest.mark.parametrize("arguments", [["--h-n", "7"], ["--rounds", "0"],
                                      ["--p-values", "nan"], ["--max-shots", "0"],
                                      ["--mode", "full_postselection", "--decoder", "mwpf"]])
def test_cli_rejects_invalid_configuration_before_output(tmp_path, arguments):
    output = tmp_path / "invalid.csv"
    result = subprocess.run([sys.executable, str(RUNNER), "--codes", "h_code",
                             *arguments, "--output", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize("n", [6, 8, 12])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("schedule", ["dedicated", "coloration"])
def test_unified_h_code_builder(n, basis, schedule):
    c, n_data, n_total, k = build_circuit(
        "h_code", distance=2, h_n=n, p=0, basis=basis, se_circuit=schedule,
    )
    assert (n_data, n_total, k) == (n, n + 4, n - 4)
    assert c.num_detectors == 8
    dets, obs = c.compile_detector_sampler(seed=98).sample(32, separate_observables=True)
    assert not dets.any() and not obs.any()


def _size_and_mode_rows():
    base = dict(code="h_code", distance=2, p=0.001, basis="Z", rounds=2,
                se_circuit="dedicated", noise_model="circuit_level", decoder_name="none")
    return [{**base, "h_n": n, "mode": mode, "p": p, "logical_error_rate": p}
            for n in [6, 8] for mode in ["decode", "full_postselection"]
            for p in [0.001, 0.002]]


def test_unified_checkpoint_keeps_sizes_and_modes_separate():
    assert len({_task_key(row) for row in _size_and_mode_rows()}) == 8


def test_plot_keeps_sizes_and_modes_separate():
    import pandas as pd
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("seaborn")
    from plot_memory import plot_ler_vs_p

    fig, ax = plt.subplots()
    try:
        plot_ler_vs_p(pd.DataFrame(_size_and_mode_rows()), ax)
        assert len(ax.lines) == 4
        assert len({line.get_label() for line in ax.lines}) == 4
        assert all(len(line.get_xdata()) == 2 for line in ax.lines)
    finally:
        plt.close(fig)


def test_surface_code_uses_same_postselection_mode_and_checkpoint(tmp_path):
    import pandas as pd

    output = tmp_path / "surface.csv"
    command = [sys.executable, str(RUNNER), "--codes", "rotated_sc", "--distances", "3",
               "--p-values", "0.001", "--max-shots", "37", "--max-errors", "100",
               "--batch-size", "16", "--num-workers", "1", "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    before = pd.read_csv(output).iloc[0]
    assert before["mode"] == "decode" and before["decoder_name"] == "pymatching"
    postselected = command + ["--mode", "full_postselection"]
    subprocess.run(postselected, check=True, capture_output=True, timeout=30)
    rows = pd.read_csv(output)
    assert len(rows) == 2
    assert rows.iloc[1]["mode"] == "full_postselection"
    assert rows.iloc[1]["shots"] == 37
    assert rows.iloc[1]["accepted"] + rows.iloc[1]["rejected"] == 37
    saved = output.read_bytes()
    subprocess.run(postselected, check=True, capture_output=True, timeout=30)
    assert output.read_bytes() == saved
