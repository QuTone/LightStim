"""Read, validate, and plot recorded BB PPM results without running experiments.

The CSV contains completed fixed-size cohorts. Its adjacent JSON manifest binds
the table to an experiment and records the provenance of the original archive.
This module does not sample, decode, manage processes, or write Notebooks.
Importing it changes neither thread settings nor plotting defaults.

Run from the repository root with ``python -m benchmarks.logical_ops.plot_bb_ppm
--input .../results.csv --output .../ler.png`` to export a figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_RESULTS = Path(__file__).parent / "reference" / "bb_ppm" / "y1_large_20261124" / "results.csv"
SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
COUNT_FIELDS = (
    "shots", "logical_failures", "nonconverged", "converged_correct",
    "converged_logical_failure", "unconverged_correct_observables",
    "unconverged_logical_failure",
)
FLOAT_FIELDS = (
    "p", "ler", "ler_95_exact_two_sided_low", "ler_95_exact_two_sided_high",
    "ler_95_one_sided_upper", "nonconvergence_rate", "sum_decode_cpu_seconds",
)
HASH_FIELDS = ("source_manifest_sha256", "decoder_manifest_sha256")
CSV_FIELDS = (
    "p", *COUNT_FIELDS, "ler", "ler_95_exact_two_sided_low",
    "ler_95_exact_two_sided_high", "ler_95_one_sided_upper",
    "nonconvergence_rate", "sum_decode_cpu_seconds", *HASH_FIELDS,
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _check_hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256: {label}.")


def _budgets(values):
    result = {}
    for key, count in values.items():
        p = float(key)
        if not math.isfinite(p) or not 0 < p < 1 or p in result:
            raise ValueError("Noise points must be unique probabilities between 0 and 1.")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Requested shot counts must be positive integers.")
        result[p] = count
    if not result:
        raise ValueError("At least one fixed shot budget is required.")
    return result


def _validate_rows(rows):
    """Validate count identities and recompute exact fixed-N binomial intervals."""
    from scipy.stats import beta

    seen = set()
    for row in rows:
        for key in COUNT_FIELDS:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"Invalid nonnegative count: {key}.")
        for key in FLOAT_FIELDS:
            if isinstance(row[key], bool) or not math.isfinite(row[key]):
                raise ValueError(f"Invalid finite value: {key}.")
        p, n, k = row["p"], row["shots"], row["logical_failures"]
        if not 0 < p < 1 or p in seen or n <= 0 or k > n:
            raise ValueError("Invalid or duplicate completed noise point.")
        seen.add(p)
        cc, cw, uc, uw = (row[key] for key in COUNT_FIELDS[3:])
        if (cc + cw + uc + uw != n or cw + uw != k
                or uc + uw != row["nonconverged"]):
            raise ValueError("Logical-failure and convergence counts are inconsistent.")
        expected = {
            "ler": k / n,
            "nonconvergence_rate": (uc + uw) / n,
            "ler_95_exact_two_sided_low": 0.0 if k == 0 else beta.ppf(.025, k, n-k+1),
            "ler_95_exact_two_sided_high": 1.0 if k == n else beta.ppf(.975, k+1, n-k),
            "ler_95_one_sided_upper": 1.0 if k == n else beta.ppf(.95, k+1, n-k),
        }
        for key, value in expected.items():
            if not 0 <= row[key] <= 1 or not math.isclose(
                    row[key], value, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError(f"Count-derived rate or exact confidence interval differs: {key}.")
        if row["sum_decode_cpu_seconds"] < 0:
            raise ValueError("Decode CPU time must be nonnegative.")
        for key in HASH_FIELDS:
            _check_hash(row[key], key)


def read_results(results_path=DEFAULT_RESULTS, *, manifest_path=None,
                 expected_plan_sha256=None, decoder_params=None, shots_by_p=None,
                 expected_experiment=None):
    """Return verified data; never read raw shots or execute an experiment.

    ``results_path`` is a CSV or its manifest JSON. A CSV uses
    ``<stem>.manifest.json`` when present, otherwise ``manifest.json`` beside it.
    ``expected_plan_sha256`` is the *compiled circuit plan* hash, not a campaign
    hash. ``expected_experiment`` may specify any subset of manifest experiment
    fields, for example target, rounds, initial_state, final_basis, noise_model.
    Partial runs expose only completed fixed-N points; expected shot budgets
    are compared against all requested points in the manifest.
    """
    path = Path(results_path)
    if path.suffix.lower() == ".json":
        if manifest_path is not None:
            raise ValueError("Specify the manifest only once.")
        manifest_path, path = path, None
    elif manifest_path is None:
        sidecar = path.with_suffix(".manifest.json")
        manifest_path = sidecar if sidecar.exists() else path.with_name("manifest.json")
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != 1
            or manifest.get("kind") != "bb_ppm_recorded_results"
            or manifest.get("status") not in {"complete", "partial"}):
        raise ValueError("Unsupported BB PPM result manifest.")
    csv_info = manifest["results_csv"]
    csv_name = csv_info["path"]
    if not isinstance(csv_name, str) or Path(csv_name).name != csv_name:
        raise ValueError("The manifest CSV must be a filename in the same directory.")
    declared_path = manifest_path.parent / csv_name
    if path is None:
        path = declared_path
    elif path.resolve() != declared_path.resolve():
        raise ValueError("The requested CSV differs from the manifest CSV.")
    csv_bytes = path.read_bytes()
    _check_hash(csv_info["sha256"], "results_csv.sha256")
    if _sha256(csv_bytes) != csv_info["sha256"]:
        raise ValueError("Result CSV SHA256 differs from its manifest.")
    reader = csv.DictReader(csv_bytes.decode("utf-8").splitlines())
    if reader.fieldnames != list(CSV_FIELDS):
        raise ValueError("Unexpected BB PPM CSV columns or column order.")
    rows = []
    for raw in reader:
        row = dict(raw)
        for key in COUNT_FIELDS:
            if not re.fullmatch(r"\d+", row[key] or ""):
                raise ValueError(f"Invalid integer CSV column: {key}.")
            row[key] = int(row[key])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key])
        rows.append(row)
    rows.sort(key=lambda row: row["p"])
    _validate_rows(rows)
    if rows != sorted(manifest["rows"], key=lambda row: row["p"]):
        raise ValueError("CSV rows differ from the manifest's recorded statistics.")
    experiment, decoder = manifest["experiment"], manifest["decoder"]
    resources = manifest["resources"]
    required = ("target", "compiled_plan_sha256", "clean_circuit_sha256",
                "noise_model", "rounds", "initial_state", "final_basis")
    if any(key not in experiment for key in required):
        raise ValueError("The experiment identity is incomplete.")
    if (decoder.get("name") != "relay-bp" or not isinstance(decoder.get("parameters"), dict)
            or type(experiment["rounds"]) is not int or experiment["rounds"] <= 0):
        raise ValueError("Invalid decoder or round-count identity.")
    for key in ("target", "noise_model", "initial_state", "final_basis"):
        if not isinstance(experiment[key], str) or not experiment[key]:
            raise ValueError(f"Invalid experiment identity: {key}.")
    for key in ("physical_qubits", "ticks", "detectors", "observables"):
        if type(resources[key]) is not int or resources[key] < (0 if key in {"ticks", "detectors"} else 1):
            raise ValueError(f"Invalid circuit resource: {key}.")
    for key in ("compiled_plan_sha256", "clean_circuit_sha256"):
        _check_hash(experiment[key], key)
    if (resources["plan_sha256"] != experiment["compiled_plan_sha256"]
            or resources["circuit_sha256"] != experiment["clean_circuit_sha256"]
            or resources["sequence"] != experiment["target"]):
        raise ValueError("Circuit resources differ from the experiment identity.")
    if expected_plan_sha256 is not None and expected_plan_sha256 != experiment["compiled_plan_sha256"]:
        raise ValueError("Saved results do not match the compiled circuit plan.")
    if decoder_params is not None and _canonical(decoder_params) != _canonical(decoder["parameters"]):
        raise ValueError("Decoder parameters differ from the recorded experiment.")
    for key, value in (expected_experiment or {}).items():
        if key not in experiment or _canonical(experiment[key]) != _canonical(value):
            raise ValueError(f"Recorded experiment differs: {key}.")
    requested = _budgets(manifest["requested_shots_by_p"])
    completed = {row["p"]: row["shots"] for row in rows}
    if any(requested.get(p) != n for p, n in completed.items()):
        raise ValueError("A completed cohort differs from its fixed shot budget.")
    if manifest["status"] == "complete" and completed != requested:
        raise ValueError("Complete results are missing requested noise points.")
    if shots_by_p is not None and _budgets(shots_by_p) != requested:
        raise ValueError("Shot budgets differ from the recorded experiment.")
    return {
        "results": rows, "resources": resources, "parameters": decoder["parameters"],
        "experiment": experiment, "status": manifest["status"],
        "requested_shots_by_p": requested, "manifest": manifest,
        "manifest_sha256": _sha256(manifest_bytes), "csv_sha256": _sha256(csv_bytes),
    }


def _scientific_tick(value, position):
    """Format each tick with its own power of ten."""
    if value == 0:
        return "0"
    coefficient, exponent = f"{value:.12e}".split("e")
    coefficient = coefficient.rstrip("0").rstrip(".")
    return f"{coefficient} × 10{str(int(exponent)).translate(SUPERSCRIPTS)}"


def create_figure(data):
    """Create a Figure with actual zero rates and exact 95% error bars.

    Connecting lines guide the eye; they are not a fitted error-rate model.
    Physical error probabilities are spaced linearly. LER uses a logarithmic
    scale when every displayed interval lies above zero and a linear scale
    when an interval includes zero.
    The one-sided 95% upper bound is reported separately in the display table.
    """
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    from lightstim.plot.styles import apply_paper_style, bold_ticks, CODES

    rows = sorted(data["results"], key=lambda row: row["p"])
    _validate_rows(rows)
    if not rows:
        raise ValueError("There are no completed fixed-N results to plot.")
    probabilities = [row["p"] for row in rows]
    rates = [row["ler"] for row in rows]
    lower = [row["ler_95_exact_two_sided_low"] for row in rows]
    upper = [row["ler_95_exact_two_sided_high"] for row in rows]
    errors = [[rate - low for rate, low in zip(rates, lower)],
              [high - rate for rate, high in zip(rates, upper)]]
    p_min, p_max = min(data["requested_shots_by_p"]), max(data["requested_shots_by_p"])
    padding = .08 * (p_max - p_min or p_min)
    with mpl.rc_context():
        apply_paper_style()
        figure = Figure(figsize=(8, 5.4), layout="constrained")
        figure.get_layout_engine().set(rect=(0, .10, 1, .90))
        ax = figure.subplots()
        ax.set_xlim(p_min - padding, p_max + padding)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(_scientific_tick)
        if min(lower) > 0:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-.04 * max(upper), min(1.05, 1.12 * max(upper)))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        target = re.sub(r"\b([XYZ])(\d+)\b", r"$\\bar{\1}_{\2}$", data["experiment"]["target"])
        ax.errorbar(probabilities, rates, yerr=errors, fmt="o",
                    color=CODES["bb_144_12_12"], capsize=4, label=target,
                    linestyle="-" if data["status"] == "complete" else "none")
        ax.set_title("Gross Code Pauli Product Measurement")
        ax.set_xlabel("Physical Error Rate $p$")
        ax.set_ylabel("Logical Error Rate (LER)")
        ax.legend(title="Target", loc="lower right")
        ax.tick_params(labelsize=10)
        ax.grid(alpha=.25)
        bold_ticks(ax)
        figure.text(.02, .02,
                    "Points show the fraction of shots with at least one logical observable error after Relay-BP decoding.\n"
                    "Error bars show two-sided 95% Clopper–Pearson confidence intervals.",
                    fontsize=9, fontweight="normal", va="bottom")
    return figure


def result_table(data):
    """Return a display table using p's power of ten for each row's probabilities."""
    import pandas as pd

    rows = []
    for row in data["results"]:
        exponent = math.floor(math.log10(row["p"]))
        scale = 10.0 ** exponent
        power = f" × 10{str(exponent).translate(SUPERSCRIPTS)}"
        value = row["ler"] / scale
        lower = row["ler_95_exact_two_sided_low"] / scale
        upper = row["ler_95_exact_two_sided_high"] / scale
        rows.append({
            "p": f"{row['p'] / scale:.4g}{power}",
            "Shots": row["shots"], "Failures": row["logical_failures"],
            "LER (95% CI)": f"[{value:.4g} (+{upper-value:.3g} / −{value-lower:.3g})]{power}",
            "Nonconverged": row["nonconverged"],
            "Valid & correct": row["converged_correct"],
            "95% one-sided upper bound": f"{row['ler_95_one_sided_upper'] / scale:.4g}{power}",
        })
    return pd.DataFrame(rows)


def main(argv=None):
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).parent / "results"
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=results_dir / "fig_bb_ppm.png")
    args = parser.parse_args(argv)
    data = read_results(args.input, manifest_path=args.manifest)
    figure = create_figure(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(result_table(data).to_string(index=False))
    print(f"Saved: {Path(os.path.relpath(args.output, Path.cwd())).as_posix()}")


if __name__ == "__main__":
    main()
