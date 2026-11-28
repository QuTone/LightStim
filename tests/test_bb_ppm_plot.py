"""Validate recorded scientific statistics and portable BB PPM plotting."""
import base64
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from xml.etree import ElementTree

import pytest

from benchmarks.logical_ops import plot_bb_ppm as plot


@pytest.fixture
def portable_results(tmp_path):
    shutil.copyfile(plot.DEFAULT_RESULTS, tmp_path / "results.csv")
    shutil.copyfile(plot.DEFAULT_RESULTS.with_suffix(".manifest.json"), tmp_path / "manifest.json")
    return tmp_path / "results.csv"


def rewrite(path, *, change_rows=None, change_manifest=None):
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if change_rows is not None:
        change_rows(manifest["rows"])
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=plot.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(manifest["rows"])
        manifest["results_csv"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if change_manifest is not None:
        change_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture
def zero_first_point_results(portable_results):
    """Exercise zero-count plotting independently of the recorded sample draw."""
    def zero_first(rows):
        n = rows[0]["shots"]
        rows[0].update(logical_failures=0, nonconverged=0, converged_correct=n,
                       converged_logical_failure=0, unconverged_correct_observables=0,
                       unconverged_logical_failure=0, ler=0.0, nonconvergence_rate=0.0,
                       ler_95_exact_two_sided_low=0.0,
                       ler_95_exact_two_sided_high=1 - .025 ** (1 / n),
                       ler_95_one_sided_upper=1 - .05 ** (1 / n))
    rewrite(portable_results, change_rows=zero_first)
    return portable_results


def test_recorded_results_are_portable_and_exact(portable_results):
    data = plot.read_results(portable_results)
    assert [(r["p"], r["shots"]) for r in data["results"]] == [
        (.0005, 1_000_000), (.001, 1_000_000), (.002, 100_000)]
    assert data["manifest"]["raw_shots_in_repository"] is False
    assert "/home/" not in json.dumps(data["manifest"])
    assert plot.read_results(portable_results.with_name("manifest.json")) == data


def test_published_notebook_identifies_its_displayed_results():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks/LogicalOps/logical_PPM_BB_code.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    results = plot.read_results(
        expected_experiment={"target": "Y1"},
        shots_by_p={.0005: 1_000_000, .001: 1_000_000, .002: 100_000},
    )
    outputs = [out["data"] for cell in notebook["cells"] for out in cell.get("outputs", [])
               if "data" in out]
    expected_png = plot.DEFAULT_RESULTS.with_name("ler.png").read_bytes()
    assert any(base64.b64decode("".join(out["image/png"])) == expected_png
               for out in outputs if "image/png" in out)

    tables = ["".join(out["text/html"]) for out in outputs if "text/html" in out]
    ler_table, = [table for table in tables if "LER (95% CI)" in table]
    body = ler_table.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    rows = ElementTree.fromstring(f"<tbody>{body}</tbody>").findall("tr")
    assert [[cell.text for cell in row.findall("td")] for row in rows] == (
        plot.result_table(results).astype(str).values.tolist()
    )


def test_csv_digest_and_manifest_rows_are_both_verified(zero_first_point_results):
    portable_results = zero_first_point_results
    portable_results.write_bytes(portable_results.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        plot.read_results(portable_results)
    rewrite(portable_results, change_manifest=lambda m: m["results_csv"].update(
        sha256=hashlib.sha256(portable_results.read_bytes()).hexdigest()))
    rewrite(portable_results, change_manifest=lambda m: m["rows"][0].update(nonconverged=1))
    with pytest.raises(ValueError, match="recorded statistics"):
        plot.read_results(portable_results)


@pytest.mark.parametrize("change,match", [
    ({"logical_failures": 1}, "counts"),
    ({"nonconverged": 1}, "counts"),
    ({"ler_95_exact_two_sided_high": 0}, "confidence interval"),
    ({"ler_95_one_sided_upper": .5}, "confidence interval"),
    ({"sum_decode_cpu_seconds": -1}, "CPU"),
    ({"ler": float("nan")}, "finite"),
    ({"source_manifest_sha256": "bad"}, "SHA256"),
])
def test_invalid_statistics_rejected_even_when_rehashed(zero_first_point_results, change, match):
    portable_results = zero_first_point_results
    rewrite(portable_results, change_rows=lambda rows: rows[0].update(change))
    with pytest.raises(ValueError, match=match):
        plot.read_results(portable_results)


@pytest.mark.parametrize("kwargs", [
    {"expected_plan_sha256": "0"*64},
    {"decoder_params": {"alpha": 1}},
    {"shots_by_p": {.0001: 50, .0005: 50, .001: 50}},
    {"expected_experiment": {"target": "X2 Z7 X12"}},
    {"expected_experiment": {"rounds": 9}},
    {"expected_experiment": {"noise_model": "different"}},
    {"expected_experiment": {"initial_state": "Z"}},
    {"expected_experiment": {"final_basis": "Z"}},
])
def test_incompatible_experiment_rejected(portable_results, kwargs):
    with pytest.raises(ValueError):
        plot.read_results(portable_results, **kwargs)


def test_matching_configuration_accepts_tuple_interval(portable_results):
    data = plot.read_results(portable_results)
    params = copy.deepcopy(data["parameters"])
    params["gamma_dist_interval"] = tuple(params["gamma_dist_interval"])
    result = plot.read_results(portable_results,
        expected_plan_sha256=data["experiment"]["compiled_plan_sha256"],
        decoder_params=params, shots_by_p=data["requested_shots_by_p"],
        expected_experiment={"target": "Y1", "rounds": 10,
                             "initial_state": "X", "final_basis": "X"})
    assert result == data


def test_partial_run_does_not_invent_missing_points(portable_results):
    rewrite(portable_results, change_rows=lambda rows: rows.pop(),
            change_manifest=lambda m: m.update(status="partial"))
    data = plot.read_results(portable_results,
                             shots_by_p={.0005: 1_000_000, .001: 1_000_000, .002: 100_000})
    assert len(data["results"]) == 2
    rewrite(portable_results, change_manifest=lambda m: m.update(status="complete"))
    with pytest.raises(ValueError, match="missing"):
        plot.read_results(portable_results)


def test_csv_sidecar_manifest_supported(portable_results):
    portable_results.with_name("manifest.json").rename(portable_results.with_suffix(".manifest.json"))
    assert len(plot.read_results(portable_results)["results"]) == 3


def test_connected_figure_preserves_zero_and_style(zero_first_point_results):
    import matplotlib as mpl

    portable_results = zero_first_point_results
    data = plot.read_results(portable_results)
    before = dict(mpl.rcParams)
    figure = plot.create_figure(data)
    assert dict(mpl.rcParams) == before
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    points = axes.containers[0].lines[0]
    assert list(points.get_xdata()) == [.0005, .001, .002]
    assert list(points.get_ydata()) == [row["ler"] for row in data["results"]]
    assert points.get_ydata()[0] == 0
    assert points.get_linestyle() == "-"
    assert axes.get_yscale() == "linear"
    intervals = axes.containers[0].lines[2][0].get_segments()
    assert intervals[0][0][1] == 0
    assert intervals[0][1][1] == data["results"][0]["ler_95_exact_two_sided_high"]


@pytest.mark.parametrize("grid", ["recorded", "dense", "partial", "single"])
def test_linear_noise_axis_and_rendered_tick_spacing(grid):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    data = plot.read_results()
    rows = data["results"]
    if grid == "dense":
        rows = [dict(rows[0], p=.0005 + i * 1e-6) for i in range(50)]
        data["requested_shots_by_p"] = {row["p"]: row["shots"] for row in rows}
    elif grid == "partial":
        rows = rows[:2]
        data["status"] = "partial"
    elif grid == "single":
        rows = rows[:1]
        data["requested_shots_by_p"] = {rows[0]["p"]: rows[0]["shots"]}
    data["results"] = rows

    figure = plot.create_figure(data)
    figure.set_dpi(100)  # The Notebook displays this 8-inch figure at 800 px.
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    assert len(figure.axes) == 1
    ax = figure.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xscale() == "linear"
    if grid == "recorded":
        assert ax.get_title() == "Gross Code Pauli Product Measurement"
        assert ax.get_xlabel() == "Physical Error Rate $p$"
        assert ax.get_ylabel() == "Logical Error Rate (LER)"
        assert ax.get_legend().get_texts()[0].get_text() == r"$\bar{Y}_{1}$"
        caption, = figure.texts
        assert "two-sided 95% Clopper–Pearson" in caption.get_text()
        bounds = caption.get_window_extent(canvas.get_renderer())
        assert figure.bbox.contains(*bounds.min) and figure.bbox.contains(*bounds.max)
        assert bounds.y1 < ax.xaxis.label.get_window_extent(canvas.get_renderer()).y0
    line, _, bars = ax.containers[0].lines
    assert list(line.get_xdata()) == [row["p"] for row in rows]
    assert list(line.get_ydata()) == [row["ler"] for row in rows]
    assert line.get_linestyle() == ("None" if grid == "partial" else "-")
    for row, segment in zip(rows, bars[0].get_segments()):
        assert segment[:, 1] == pytest.approx(
            [row["ler_95_exact_two_sided_low"], row["ler_95_exact_two_sided_high"]])

    left, right = ax.get_xlim()
    assert not ax.xaxis.get_offset_text().get_text()
    if grid == "recorded":
        assert [tick.label1.get_text() for tick in ax.xaxis.get_major_ticks()
                if left <= tick.get_loc() <= right] == [
            "5 × 10⁻⁴", "1 × 10⁻³", "1.5 × 10⁻³", "2 × 10⁻³",
        ]
    ticks = ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks()
    labels = [tick.label1.get_window_extent(canvas.get_renderer()) for tick in ticks
              if left <= tick.get_loc() <= right and tick.label1.get_visible()
              and tick.label1.get_text()]
    labels.sort(key=lambda box: box.x0)
    assert len(labels) >= 2
    assert all(left_box.x1 + 2 < right_box.x0 for left_box, right_box
               in zip(labels, labels[1:]))


def test_import_does_not_change_environment_or_import_science_backends():
    code = """
import os, sys
before = dict(os.environ)
from benchmarks.logical_ops import plot_bb_ppm
assert dict(os.environ) == before
assert not any(key in sys.modules for key in ('stim', 'relay_bp', 'matplotlib', 'IPython'))
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1])
