import csv
import json
from pathlib import Path

import pytest

from lightstim.config import ExperimentConfig, expand_sweep, export_benchmarks_json, load_config
from lightstim.config.results import append_result_row, config_digest, normalize_result_row, point_config_digest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/config"


def _write_rows(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, columns)
        writer.writeheader()
        writer.writerows(rows)


def _config_rows():
    source = load_config(EXAMPLES / "rotated-surface-memory.json")
    overrides = {
        "sweep": {"axes": {"distance": [3, 5], "p": [0.001, 0.002]}},
        "simulation": {**source["simulation"], "max_shots": 20},
    }
    effective = ExperimentConfig({**source.spec, **overrides})
    rows = []
    for point in expand_sweep(effective):
        rows.append(normalize_result_row({
            "protocol_id": effective["protocol_id"], "parameters": point["parameters"],
            "sweep_coordinates": point.coordinates, "config_overrides": overrides,
            "config_sha256": point_config_digest(effective, point["parameters"]),
            "run_config_sha256": config_digest(effective.spec), "shots": 20, "accepted": 18,
            "errors": 2, "logical_error_rate": 2 / 18, "logical_error_rate_per_round": 0.04,
            "seconds": 1.25,
        }))
    return source, effective, rows


def test_grouped_curves_and_effective_provenance(tmp_path):
    source, effective, rows = _config_rows()
    path = tmp_path / "run.csv"
    _write_rows(path, rows)
    output = tmp_path / "nested/benchmarks.json"
    document = export_benchmarks_json(path, source, output)
    assert json.loads(output.read_text()) == document
    assert document["provenance"] == {"kind": "measured", "config": effective.spec}
    series, = document["series"]
    assert series["group_by"] == "distance"
    assert series["fixed"] == {"basis": "Z"}
    assert len(series["points"]) == 4
    assert {p["parameters"]["rounds"] for p in series["points"]} == {3, 5}
    assert all(p["logical_error_rate_per_round"] == 0.04 for p in series["points"])
    assert all(p["acceptance"] == 0.9 for p in series["points"])


@pytest.mark.parametrize(("field", "value", "error"), [
    ("config_sha256", "0" * 64, "Compile-section digest"),
    ("run_config_sha256", "0" * 64, "Run digest"),
    ("sweep_coordinates", '{"p":0.001,"distance":7}', "outside the effective sweep"),
    ("sweep_coordinates", '{"p":0.002,"distance":3}', "resolved sweep coordinates"),
    ("accepted", 0, "Inconsistent|No accepted shots"),
])
def test_bad_point_cannot_replace_output(tmp_path, field, value, error):
    source, _, rows = _config_rows()
    rows[0][field] = value
    path = tmp_path / "run.csv"
    _write_rows(path, rows)
    target = path.with_name("benchmarks.json")
    target.write_text("previous result")
    with pytest.raises(ValueError, match=f"CSV line 2: .*({error})"):
        export_benchmarks_json(path, source)
    assert target.read_text() == "previous result"


def test_legacy_cnot_csv_maps_bases_and_preserves_columns(tmp_path):
    row = {
        "gate": "CNOT_LS_ZZ_XX", "d": 3, "rounds": 2, "p": 0.001,
        "init_basis": "XZ", "measure_basis": "ZZ", "post_selected_shots": 4,
        "shots": "5.0", "errors": 1, "logical_error_rate": 0.25, "decoder": "gpu_bposd",
    }
    path = tmp_path / "legacy.csv"
    _write_rows(path, [row])
    append_result_row(path, {**row, "init_basis": "XX", "measure_basis": "XX", "new_col": 1})
    with path.open() as src:
        r1, r2 = csv.DictReader(src)
    assert r1["gate"] == r2["gate"] == row["gate"]
    assert r1["new_col"] == "" and r2["new_col"] == "1"
    doc = export_benchmarks_json(path, EXAMPLES / "lattice-surgery-cnot.json")
    assert "legacy" in doc["provenance"]["note"]
    series1, series2 = doc["series"]
    assert series1["decoder"] == "bposd" and series1["fixed"]["backend"] == "gpu"
    assert series1["points"][0]["parameters"]["target_init"] == "Z"
    assert series2["points"][0]["parameters"]["target_init"] == "X"
    assert series1["points"][0]["accepted"] == 4
    assert "logical_error_rate_per_round" not in series1["points"][0]


@pytest.mark.parametrize(("protocol", "fields"), [
    ("rotated-surface-memory", {"code": "bb_144_12_12"}),
    ("rotated-surface-memory", {"gate": "memory", "init_basis": "X"}),
    ("rotated-surface-memory", {"gate": "state_injection"}),
    ("lattice-surgery-cnot", {"gate": "CNOT_trans"}),
    ("lattice-surgery-cnot", {"gate": "CNOT_LS_XX_ZZ"}),
    ("lattice-surgery-cnot", {"experiment": "distill_ls"}),
])
def test_legacy_protocol_conflicts_abort_export(tmp_path, protocol, fields):
    path = tmp_path / "legacy.csv"
    _write_rows(path, [{"d": 3, "p": 0.001, "shots": 10, "errors": 1,
                        "logical_error_rate": 0.1, **fields}])
    output = tmp_path / "benchmarks.json"
    output.write_text("previous export")
    with pytest.raises(ValueError, match="CSV line 2: Legacy"):
        export_benchmarks_json(path, EXAMPLES / f"{protocol}.json")
    assert output.read_text() == "previous export"


def test_legacy_logical_ops_memory_uses_recorded_basis(tmp_path):
    source = load_config(EXAMPLES / "rotated-surface-memory.json").spec
    source["protocol_id"] = "unrotated-memory"
    source["system"]["patches"][0]["code"] = "unrotated_sc"
    path = tmp_path / "legacy.csv"
    _write_rows(path, [{"gate": "memory", "init_basis": "X", "measure_basis": "X",
                        "d": 3, "p": 0.001, "shots": 10, "errors": 1, "logical_error_rate": 0.1}])
    doc = export_benchmarks_json(path, source)
    assert doc["series"][0]["points"][0]["parameters"]["basis"] == "X"


def test_distinguishes_rounds_and_zero_acceptance(tmp_path):
    source = load_config(EXAMPLES / "rotated-surface-memory.json")
    source.spec["sweep"]["axes"]["rounds"] = [1, 2]
    path = tmp_path / "raw.csv"
    row = {
        "distance": 3, "p": 0.001, "basis": "Z", "errors": 0, "shots": 5,
        "rounds": 1, "kept": 4, "ler": 0, "ps_mode": "pqrm_only",
    }
    _write_rows(path, [row, {**row, "rounds": 2}])
    doc = export_benchmarks_json(path, source)
    assert [s["fixed"]["rounds"] for s in doc["series"]] == [1, 2]
    assert doc["series"][0]["mode"] == "hybrid"
    assert doc["series"][0]["fixed"]["postselection_region"] == "pqrm_only"
    _write_rows(path, [{**row, "kept": 0}])
    with pytest.raises(ValueError, match="No accepted shots"):
        export_benchmarks_json(path, source)


def test_legacy_injection_mode_and_input_paths(tmp_path):
    assert normalize_result_row({"post_select_mode": "full_postselection"})["mode"] == "hybrid"
    assert normalize_result_row({"post_select_mode": "full_postselection"})["postselection_region"] == "injection"
    source = tmp_path / "benchmarks.json"
    original = (EXAMPLES / "rotated-surface-memory.json").read_bytes()
    source.write_bytes(original)
    path = tmp_path / "run.csv"
    path.touch()
    for target in (source, path):
        with pytest.raises(ValueError, match="differ from"):
            export_benchmarks_json(path, source, target)
    assert source.read_bytes() == original
