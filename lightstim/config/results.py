"""Common CSV columns and conversion to the Protocol Zoo benchmark format."""

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


# Keep the original memory runner's columns for its plotting and analysis scripts.
RESULT_COLUMNS = [
    "code", "distance", "h_n", "p", "basis", "rounds", "se_circuit", "mode",
    "p_idle", "p_1q", "p_idle_mode", "p_1q_mode", "detector_basis", "noise_model",
    "decoder_name", "decoder_params", "decoder_time_limit", "on_decode_failure",
    "layout", "block_class", "shots", "errors", "accepted", "rejected", "acceptance",
    "logical_error_rate", "seconds", "n_data", "n_total", "k", "osd_order",
    "max_iterations", "ms_scaling_factor", "seed", "observable_failure_mode",
    "p_2q", "p_meas", "p_reset", "noise_custom_params", "postselection_region",
    "block_failure_probability", "block_failure_se", "logical_failure_rate_per_round",
    "logical_failure_rate_per_round_se", "protocol_id", "backend", "config_sha256",
    "run_config_sha256", "sweep_coordinates", "parameters", "config_overrides",
]


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def point_config_digest(template, parameters):
    """Hash the same compile inputs as a Zoo preview bundle for this point."""
    section = {key: template[key] for key in ("system", "operations", "noise")}
    return config_digest({**section, "parameters": parameters})


def _count(value):
    # An older CSV rewritten by pandas can contain a value such as 1000.0.
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"Invalid shot/error count {value!r}") from None
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"Invalid shot/error count {value!r}")
    return int(number)


def normalize_result_row(row):
    """Add common names and keep protocol-specific columns and metadata."""
    result = dict.fromkeys(RESULT_COLUMNS)
    result.update(row)
    aliases = {
        "distance": ("d", "d_surf"),
        "basis": ("state", "inject_state"),
        "p": ("p_2q",),
        "decoder_name": ("decoder",),
        "logical_error_rate": ("ler", "block_failure_probability"),
        "logical_failure_rate_per_round": ("logical_error_rate_per_round",),
        "accepted": ("post_selected_shots", "kept"),
        "acceptance": ("post_selection_rate", "ps_rate"),
    }
    for key, names in aliases.items():
        if result[key] in (None, ""):
            result[key] = next((row[n] for n in names if row.get(n) not in (None, "")), None)
    if not result["mode"]:
        if row.get("post_select_mode"):
            legacy = row["post_select_mode"]
            result["mode"] = {"full_qec": "decode", "all_detectors": "full_postselection",
                              "full_postselection": "hybrid"}.get(legacy, legacy)
            if legacy in {"full_postselection", "hybrid"}:
                result["postselection_region"] = {"full_postselection": "injection", "hybrid": "logical_strip"}[legacy]
        elif row.get("ps_mode"):
            result["mode"] = "hybrid"
            result["postselection_region"] = row["ps_mode"]
    if result["decoder_name"] in {"cpu_bposd", "gpu_bposd"}:
        backend = result["decoder_name"].split("_")[0]
        if result["backend"] and result["backend"] != backend:
            raise ValueError(f"Decoder {result['decoder_name']} conflicts with backend {result['backend']}")
        result.update(decoder_name="bposd", backend=backend)
    if result["shots"] not in (None, "") and result["accepted"] not in (None, ""):
        shots, kept = _count(result["shots"]), _count(result["accepted"])
        result["rejected"] = shots - kept
        result["acceptance"] = kept / shots if shots else None
        if kept == 0:
            result["logical_error_rate"] = None
            result["block_failure_probability"] = None
    if result["block_failure_probability"] is None:
        result["block_failure_probability"] = result["logical_error_rate"]
    for key, value in result.items():
        if isinstance(value, (dict, list, tuple)):
            result[key] = canonical_json(value)
    return result


def upgrade_csv(path, result_columns, defaults_for=None):
    """Extend the header atomically and preserve each original field value."""
    if not path.exists() or not path.stat().st_size:
        return list(result_columns)
    with path.open(newline="", encoding="utf-8") as src:
        old = next(csv.reader(src), [])
    # Early benchmark checkpoints could store just the task input columns.
    if not set(old).intersection(result_columns) or len(set(old)) != len(old):
        raise ValueError(f"Cannot migrate {path}: invalid benchmark CSV header")
    # Migrations with defaults specify the desired schema order. Ordinary
    # appends retain the order of existing protocol-specific columns.
    order = [*result_columns, *old] if defaults_for else [*old, *result_columns]
    columns = list(dict.fromkeys(order))
    if columns != old:
        tmp = NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False)
        try:
            with tmp, path.open(newline="", encoding="utf-8") as src:
                writer = csv.DictWriter(tmp, columns)
                writer.writeheader()
                for row in csv.DictReader(src):
                    writer.writerow({**defaults_for(row), **row} if defaults_for else row)
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return columns


def append_result_row(path, row):
    """Append one completed point, retaining extra columns in existing files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = normalize_result_row(row)
    columns = upgrade_csv(path, row)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _point(row, parameters):
    point = {"parameters": parameters}
    for name in ("shots", "errors", "accepted"):
        if row.get(name) not in ("", None):
            point[name] = _count(row[name])
    if "shots" not in point or "errors" not in point:
        raise ValueError("Missing shots or errors in the benchmark CSV")
    kept = point.get("accepted", point["shots"])
    if not 0 <= point["errors"] <= kept <= point["shots"]:
        raise ValueError("Inconsistent shots, accepted or errors in the benchmark CSV")
    for source, name in (
        ("logical_error_rate", "logical_error_rate"), ("ci_low", "ci_low"),
        ("ci_high", "ci_high"), ("acceptance", "acceptance"), ("seconds", "seconds"),
        ("logical_failure_rate_per_round", "logical_error_rate_per_round"),
    ):
        if row.get(source) not in ("", None):
            value = float(row[source])
            if math.isfinite(value):
                if value < 0 or (name != "seconds" and value > 1):
                    raise ValueError(f"Invalid {source} in the benchmark CSV")
                point[name] = value
    if not kept:
        raise ValueError("No accepted shots; the Zoo requires a measured logical error rate")
    if "logical_error_rate" not in point:
        raise ValueError("Missing a finite logical error rate in the benchmark CSV")
    return point


def _sweep_axes(config):
    sweep = config.get("sweep", {})
    axes = dict(sweep.get("axes", {}))
    for group in sweep.get("zip", []):
        axes.update(group)
    return axes


def _legacy_parameters(row, config):
    values = dict(row)
    if row.get("gate") == "memory" and not values.get("basis"):
        values["basis"] = row.get("init_basis")
    # The legacy LS CNOT runner recorded the two patch bases in one column.
    for column, suffix in (("init_basis", "init"), ("measure_basis", "meas")):
        pair = row.get(column)
        if isinstance(pair, str) and len(pair) == 2:
            values[f"control_{suffix}"], values[f"target_{suffix}"] = pair
    overrides = {}
    axes = _sweep_axes(config)
    for key, default in config.resolve()["parameters"].items():
        value = values.get(key)
        if value not in (None, ""):
            overrides[key] = value if isinstance(default, str) else json.loads(str(value))
        elif key in axes:
            if len(set(axes[key])) != 1:
                raise ValueError(f"Cannot identify the swept parameter {key!r} in the legacy CSV")
            overrides[key] = axes[key][0]
    return config.resolve(overrides)["parameters"]


def _check_legacy_protocol(row, config):
    """Check protocol identifiers emitted by the repository's direct runners."""
    entrypoint = config["implementation"]["entrypoint"]
    memory = entrypoint == "lightstim.protocols.memory.MemoryExperiment"
    cnot = entrypoint == "lightstim.protocols.cnot_ls.CNOTLSExperiment"
    if not (memory or cnot):
        return
    if row.get("experiment") in {"bell_tele", "s_gate_tele", "distill_ls", "distill_tg"} or row.get("pqrm"):
        raise ValueError(f"Legacy experiment belongs to another protocol than {config['protocol_id']!r}")
    gate = row.get("gate")
    code = row.get("code")
    if gate:
        expected = "memory"
        if cnot:
            interactions = {item["id"]: item["interaction"] for item in config["system"].get("couplers", [])}
            order = [interactions[target] for op in config["operations"] if op["op"] == "activate_coupler"
                     for target in op["targets"] if target in interactions]
            expected = "CNOT_LS_" + "_".join(order)
        if gate != expected:
            raise ValueError(f"Legacy gate {gate!r} conflicts with the configured {expected!r}")
        # The historical logical-ops 'memory' gate used an unrotated surface patch.
        if memory and not code:
            code = "unrotated_sc"
    codes = {patch["code"] for patch in config["system"]["patches"]}
    if code and set(code.split(",")) != codes:
        raise ValueError(f"Legacy code {code!r} conflicts with the configured patch codes {sorted(codes)}")


def _json_object(row, name):
    value = json.loads(row.get(name) or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _recorded_parameters(row, effective):
    if row["parameters"]:
        parameters = _json_object(row, "parameters")
        if parameters.keys() != effective["parameters"].keys():
            raise ValueError("Recorded parameters differ from the effective config")
    else:
        parameters = _legacy_parameters(row, effective)
    if row["run_config_sha256"]:
        if not row["sweep_coordinates"] or not row["parameters"] or not row["config_sha256"]:
            raise ValueError("Missing sweep coordinates, parameters or config digest")
        coordinates = _json_object(row, "sweep_coordinates")
        if coordinates.keys() != _sweep_axes(effective).keys():
            raise ValueError("Recorded coordinates differ from the sweep axes")
        sweep = effective.get("sweep", {})
        for axis, choices in sweep.get("axes", {}).items():
            if coordinates[axis] not in choices:
                raise ValueError(f"{axis!r} is outside the effective sweep")
        for group in sweep.get("zip", []):
            if tuple(coordinates[k] for k in group) not in zip(*group.values()):
                raise ValueError("Coordinates do not form a point in the zipped sweep")
        if effective.resolve(coordinates)["parameters"] != parameters:
            raise ValueError("Recorded parameters differ from the resolved sweep coordinates")
    if row["config_sha256"] and point_config_digest(effective, parameters) != row["config_sha256"]:
        raise ValueError("Compile-section digest does not match the recorded parameters")
    return parameters


def _series_inputs(row, config, parameters, x_key):
    axes = _sweep_axes(config)
    variable = {x_key, "distance"}
    # rounds=d is a dependent coordinate, while an independent rounds sweep
    # defines different curves at the same (distance, p).
    for key in parameters:
        ref = config["parameters"].get(key)
        while key not in axes and isinstance(ref, dict):
            if ref["$param"] in variable:
                variable.add(key)
                break
            if ref["$param"] in axes:
                break
            ref = config["parameters"].get(ref["$param"])
    fixed = {k: v for k, v in parameters.items() if k not in variable}
    for key in ("backend", "postselection_region", "on_decode_failure", "observable_failure_mode"):
        if row.get(key):
            fixed[key] = row[key]
    for key in ("decoder_params", "noise_custom_params"):
        if row.get(key):
            value = _json_object(row, key)
            if value:
                fixed[key] = value
    if not row.get("run_config_sha256"):
        # Keep distinct legacy protocol variants in separate Zoo series.
        for key in ("gate", "sub_experiment", "experiment", "protocol", "injection_protocol",
                    "code", "se_circuit", "detector_basis", "method", "pqrm", "noise_mode",
                    "init_basis", "measure_basis", "state_prep", "basis", "routing_mult", "p_injected",
                    "decoder_time_limit", "osd_order", "max_iterations", "ms_scaling_factor"):
            if row.get(key) not in (None, "") and key not in parameters:
                fixed[key] = row[key]
        for key in ("p_1q", "p_2q", "p_idle", "p_meas", "p_reset"):
            value = row.get(key)
            if value in (None, "") or row.get("p") in (None, ""):
                continue
            if (row.get(key + "_mode") == "fixed" or (key == "p_1q" and row.get("pqrm"))
                    or float(value) != float(row["p"])):
                fixed[key] = float(value)
    return fixed


def export_benchmarks_json(csv_path, config, output=None):
    """Write a Zoo benchmarks document and return it.

    Each CSV row with run provenance must agree with the supplied source
    config plus its recorded CLI overrides. Older runner files use the source
    config to interpret their CSV columns and carry an explicit provenance note.
    """
    from .loader import ExperimentConfig, load_config

    source_path = Path(config) if isinstance(config, (str, Path)) else None
    csv_path = Path(csv_path)
    destination = Path(output) if output else csv_path.with_name("benchmarks.json")
    inputs = [csv_path] + ([source_path] if source_path else [])
    if any(destination.resolve() == path.resolve() for path in inputs):
        raise ValueError("Benchmark JSON output must differ from the source CSV and config")
    if source_path:
        config = load_config(source_path)
    elif not isinstance(config, ExperimentConfig):
        config = ExperimentConfig.from_dict(config)
    effective = None
    groups = {}
    has_legacy = False
    with csv_path.open(newline="", encoding="utf-8") as src:
        for line, raw in enumerate(csv.DictReader(src), start=2):
            try:
                row = normalize_result_row({k: v for k, v in raw.items() if v != ""})
                overrides = _json_object(row, "config_overrides")
                if effective is None:
                    effective = ExperimentConfig.from_dict({**config.spec, **overrides}) if overrides else config
                    effective_overrides = overrides
                    run_digest = config_digest(effective.spec)
                elif overrides != effective_overrides:
                    raise ValueError("Export one effective run configuration per benchmarks.json")
                if (effective["protocol_id"] != config["protocol_id"] or
                        row["protocol_id"] and row["protocol_id"] != effective["protocol_id"]):
                    raise ValueError("Row belongs to another protocol")
                if row["run_config_sha256"] and row["run_config_sha256"] != run_digest:
                    raise ValueError("Run digest does not match the supplied config and overrides")

                if not row["run_config_sha256"]:
                    _check_legacy_protocol(row, effective)
                parameters = _recorded_parameters(row, effective)
                has_legacy |= not bool(row["run_config_sha256"])
                decoder = row["decoder_name"] or effective["decoder"].get("name", "chain")
                mode = row["mode"] or effective["simulation"]["mode"]
                if mode not in {"decode", "full_postselection", "hybrid"}:
                    raise ValueError(f"Unknown mode {mode!r}")
                noise = row["noise_model"] or effective["noise"]["model"]
                rate = effective["noise"]["p"]
                x_key = rate["$param"] if isinstance(rate, dict) else "p"
                if x_key not in parameters:
                    parameters = {**parameters, x_key: float(row.get("p") or rate)}
                fixed = _series_inputs(row, effective, parameters, x_key)
                group_by = "distance"
                if group_by not in parameters:
                    group_by = next((k for k in fixed if k in parameters), "protocol_id")
                    if group_by == "protocol_id":
                        parameters = {**parameters, group_by: effective["protocol_id"]}
                key = (noise, decoder, mode, canonical_json(fixed))
                if key not in groups:
                    groups[key] = {
                        "id": f"series-{len(groups) + 1}",
                        "label": " · ".join([noise, decoder, mode, *(f"{k}={v}" for k, v in fixed.items())]),
                        "noise_model": noise, "decoder": decoder, "mode": mode, "fixed": fixed,
                        "group_by": group_by,
                        "points": [],
                    }
                groups[key]["points"].append(_point(row, parameters))
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
    if not groups:
        raise ValueError("Benchmark CSV contains no measured points")
    provenance = {"kind": "measured", "config": effective.spec}
    if has_legacy:
        provenance["note"] = "The supplied config describes legacy CSV rows whose run configuration digest was not recorded."
    document = {
        "schema_version": "qpz.benchmarks/v1-draft", "protocol_id": effective["protocol_id"],
        "provenance": provenance, "series": list(groups.values()),
        "x": {"param": x_key},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as tmp:
        try:
            json.dump(document, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
            tmp.close()
            os.replace(tmp.name, destination)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return document
