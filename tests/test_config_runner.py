import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import stim

from lightstim.config import ExperimentConfig, expand_sweep, load_config
from lightstim.config.cli import run_cli
from lightstim.config.runner import _sample, sample_full_postselection


ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "examples/config/rotated-surface-memory.json"


@pytest.mark.parametrize("runner", ["memory", "logical_ops", "logical_circuits", "state_injection", "cross_ls"])
def test_runner_config_overrides_and_checkpoint(tmp_path, runner):
    csv_path = tmp_path / "run.csv"
    cmd = [
        sys.executable, str(ROOT / "benchmarks" / runner / f"run_{runner}.py"),
        "--config", str(MEMORY), "--distances", "3", "--p-values", "0",
        "--basis", "X", "--seed", "37", "--max-shots", "7", "--batch-size", "4",
        "--num-workers", "1", "--output", str(csv_path),
    ]
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)
    initial = csv_path.read_bytes()
    if runner == "memory":
        run_cli(cmd[2:])
        assert csv_path.read_bytes() == initial
    with csv_path.open(newline="") as src:
        row, = csv.DictReader(src)
    assert (row["shots"], row["accepted"], row["errors"], row["seed"]) == ("7", "7", "0", "37")
    assert row["basis"] == "X"
    assert row["rounds"] == row["distance"] == "3"
    assert json.loads(row["sweep_coordinates"]) == {"p": 0, "distance": 3}
    assert len(row["config_sha256"]) == 64


def test_cli_overrides_named_noise_axis_and_zipped_parameters(tmp_path, monkeypatch):
    spec = load_config(ROOT / "examples/config/lattice-surgery-cnot.json").spec
    spec["parameters"]["rate"] = spec["parameters"].pop("p")
    spec["sweep"]["axes"]["rate"] = spec["sweep"]["axes"].pop("p")
    spec["noise"]["p"] = {"$param": "rate"}
    path = tmp_path / "cnot.json"
    path.write_text(json.dumps(spec))
    runs = []
    monkeypatch.setattr("lightstim.config.cli.run_experiment",
                        lambda config, output, **kw: runs.extend(expand_sweep(config)))
    run_cli(["--config", str(path), "--distances", "3", "--p-values", "0.004",
             "--param", 'control_init="X"'])
    assert len(runs) == len(spec["sweep"]["zip"][0]["target_init"])
    assert [point["parameters"]["target_init"] for point in runs] == spec["sweep"]["zip"][0]["target_init"]
    for point in runs:
        assert point.coordinates["control_init"] == "X"
        assert point.coordinates["rate"] == point["noise"]["p"] == 0.004
        assert point["simulation"] == spec["simulation"]


def test_cnot_without_a_target_observable_cannot_record_a_measured_zero(tmp_path, capsys):
    output = tmp_path / "cnot.csv"
    config = ROOT / "examples/config/lattice-surgery-cnot.json"
    with pytest.raises(SystemExit) as exc:
        run_cli(["--config", str(config), "--quick", "--param", 'control_meas="X"',
                 "--param", 'target_meas="X"', "--num-workers", "1", "--output", str(output)])
    assert exc.value.code == 2
    assert "at least one target observable" in capsys.readouterr().err
    assert not output.exists()


def test_empty_target_selection_cannot_record_a_measured_zero():
    spec = load_config(MEMORY).spec
    spec["simulation"]["target_observable_indices"] = []
    circuit = stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    with pytest.raises(ValueError, match="at least one target observable"):
        _sample(circuit, ExperimentConfig(spec).resolve())


@pytest.mark.parametrize("num_workers", [1, 3])
def test_postselection_counts_and_observable_policy(num_workers):
    circ = stim.Circuit("""
        R 0 1 2
        X_ERROR(1) 1 2
        M 0 1 2
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """)
    settings = dict(max_shots=7, max_errors=20, seed=17, batch_size=3, num_workers=num_workers)
    assert sample_full_postselection(circ, **settings)["errors"] == 7
    assert sample_full_postselection(circ, observable_failure_mode="parity", **settings)["errors"] == 0
    assert sample_full_postselection(circ, target_observable_indices=[1],
                                     observable_failure_mode="parity", **settings)["errors"] == 7
    circ.append("DETECTOR", [stim.target_rec(-1)])
    assert sample_full_postselection(circ, **settings)["accepted"] == 0


def test_decode_ignores_tags_and_hybrid_honors_them():
    circ = stim.Circuit("""
        R 0
        X_ERROR(0.5) 0
        M 0
        DETECTOR[post-select] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    spec = load_config(MEMORY).spec
    sim = spec["simulation"]
    sim.update(max_shots=100, num_workers=1)
    assert _sample(circ, ExperimentConfig(spec).resolve())["accepted"] == 100
    sim.update(mode="hybrid", postselection={"region": "tagged"})
    assert 0 < _sample(circ, ExperimentConfig(spec).resolve())["accepted"] < 100
    sim["postselection"] = {"detector_indices": [1]}
    with pytest.raises(ValueError, match="detector_indices"):
        _sample(circ, ExperimentConfig(spec).resolve())
