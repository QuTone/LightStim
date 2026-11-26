"""Command-line overrides shared by the installed package and benchmark scripts."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from lightstim.noise.config import NOISE_MODELS
from lightstim.simulation.decoder_backend.config import normalize_decoder

from .loader import ExperimentConfig, load_config
from .runner import run_experiment


def _object(text):
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def _parameter_overrides(spec, values):
    sweep = spec.get("sweep", {})
    axes = sweep.get("axes", {})
    zipped = {k for group in sweep.get("zip", []) for k in group}
    for key, items in values.items():
        if key not in spec["parameters"]:
            raise ValueError(f"No declared parameter {key!r} in {spec['protocol_id']}")
        spec["parameters"][key] = items[0]
        if key not in zipped and (len(items) > 1 or key in axes):
            axes[key] = items
    if axes:
        spec.setdefault("sweep", {})["axes"] = axes
    for group in sweep.get("zip", []):
        if not (group.keys() & values.keys()):
            continue
        lengths = {len(values[k] if k in values else v) for k, v in group.items()
                   if k not in values or len(values[k]) != 1}
        if len(lengths) > 1:
            raise ValueError(f"Overrides for zipped parameters {list(group)} must have matching lengths")
        n = next(iter(lengths), 1)
        for k in group.keys() & values.keys():
            group[k] = values[k] * n if len(values[k]) == 1 else values[k]


def _override_decoder(spec, args):
    if not any(getattr(args, key) is not None for key in (
        "decoder", "backend", "decoder_params", "on_decode_failure", "mle_time_limit",
        "osd_order", "max_iterations", "ms_scaling_factor",
    )):
        return
    decoder = spec["decoder"]
    if "chain" in decoder:
        if not args.decoder:
            raise ValueError("Override a chain stage in the config, or replace the chain using --decoder")
        decoder = spec["decoder"] = {}
    name = args.decoder or decoder["name"]
    backend = args.backend
    if backend is None and name not in {"gpu_bposd", "cpu_bposd", "nv-qldpc-decoder"}:
        backend = decoder.get("backend")
    name, backend = normalize_decoder(name, backend)
    if name != decoder.get("name"):
        decoder.pop("params", None)
    decoder.update(name=name, backend=backend)
    if args.on_decode_failure:
        decoder["on_decode_failure"] = args.on_decode_failure
    params = decoder.setdefault("params", {})
    params.update(args.decoder_params or {})
    for key in ("osd_order", "max_iterations", "ms_scaling_factor"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    if args.mle_time_limit is not None:
        if name != "mle-ilp" or args.mle_time_limit < 0:
            raise ValueError("--mle-time-limit needs mle-ilp and a non-negative limit")
        params["time_limit"] = args.mle_time_limit


def _effective_config(source, args):
    spec = deepcopy(source.spec)
    sim = spec["simulation"]
    if args.quick:
        sim["max_shots"] = min(100, sim["max_shots"])
        sim["max_errors"] = min(10, sim["max_errors"])
        sweep = spec.get("sweep", {})
        for group in [sweep.get("axes", {}), *sweep.get("zip", [])]:
            for name in group:
                group[name] = group[name][:1]
    values = {key: getattr(args, key) for key in ("distance", "rounds", "basis", "p")
              if getattr(args, key) is not None}
    if "p" in values:
        rate = spec["noise"]["p"]
        if isinstance(rate, dict):
            if rate.keys() != {"$param"}:
                raise ValueError(f"Override the affine noise rate with --param {rate['$param']}=JSON")
            values[rate["$param"]] = values.pop("p")
        else:
            # Give a literal noise.p a named axis so --p-values can sweep it.
            spec["parameters"].setdefault("p", values["p"][0])
            spec["noise"]["p"] = {"$param": "p"}
    for assignment in args.param or []:
        key, sep, raw = assignment.partition("=")
        if not sep:
            raise ValueError("--param requires NAME=JSON")
        value = json.loads(raw)
        items = value if isinstance(value, list) else [value]
        if not items or key in values:
            raise ValueError(f"Empty or duplicate override for {key!r}")
        values[key] = items
    _parameter_overrides(spec, values)
    if args.codes and set(args.codes) != {patch["code"] for patch in spec["system"]["patches"]}:
        raise ValueError("--codes conflicts with this protocol's system.patches")
    noise = spec["noise"]
    if args.noise_model:
        noise["model"] = args.noise_model
    for key in ("p_idle", "p_1q", "p_2q", "p_meas", "p_reset"):
        if getattr(args, key) is not None:
            noise.setdefault("overrides", {})[key] = getattr(args, key)
    if args.noise_custom_params is not None:
        noise.setdefault("custom_params", {}).update(args.noise_custom_params)
    _override_decoder(spec, args)
    for key in ("mode", "max_shots", "max_errors", "seed", "num_workers", "batch_size",
                "observable_failure_mode"):
        if getattr(args, key) is not None:
            sim[key] = getattr(args, key)
    if args.mode and args.mode != "hybrid":
        sim.pop("postselection", None)
    if args.postselection_region:
        sim.setdefault("postselection", {})["region"] = args.postselection_region
    if args.ps_mode:
        sim.update(mode="hybrid", postselection={"region": args.ps_mode})
    if args.inject_modes:
        if args.inject_modes == "full_qec":
            sim["mode"] = "decode"
            sim.pop("postselection", None)
        else:
            region = "injection" if args.inject_modes == "full_postselection" else "logical_strip"
            sim.update(mode="hybrid", postselection={"region": region})
    return ExperimentConfig.from_dict(spec)


def run_cli(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Run a LightStim experiment config",
                                     allow_abbrev=False)
    parser.add_argument("--config", required=True, metavar="JSON")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--output", type=Path, metavar="CSV")
    out.add_argument("--output-dir", type=Path)
    parser.add_argument("--distances", dest="distance", type=int, nargs="+")
    parser.add_argument("--rounds", "--rounds-values", type=int, nargs="+")
    parser.add_argument("--p-values", dest="p", type=float, nargs="+")
    parser.add_argument("--basis", "--states", "--inject-states", nargs="+")
    parser.add_argument("--param", action="append", metavar="NAME=JSON",
                        help="Override a declared parameter with one scalar or an array of sweep values")
    parser.add_argument("--codes", nargs="+")
    parser.add_argument("--noise-model", choices=NOISE_MODELS)
    for flag in ("p-idle", "p-1q", "p-2q", "p-meas", "p-reset"):
        parser.add_argument("--" + flag, type=float)
    parser.add_argument("--noise-custom-params", type=_object)
    parser.add_argument("--decoder")
    parser.add_argument("--backend", choices=("cpu", "gpu", "fpga"))
    parser.add_argument("--decoder-params", type=_object)
    parser.add_argument("--osd-order", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--ms-scaling-factor", type=float)
    parser.add_argument("--mle-time-limit", type=float)
    parser.add_argument("--on-decode-failure", choices=("error", "discard", "ignore"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mode", choices=("decode", "full_postselection", "hybrid"))
    modes.add_argument("--ps-mode", choices=("pqrm_only", "hybrid"))
    modes.add_argument("--inject-modes", choices=("full_qec", "full_postselection", "hybrid"))
    parser.add_argument("--postselection-region")
    parser.add_argument("--observable-failure-mode", choices=("any", "parity"))
    for flag in ("max-shots", "max-errors", "seed", "batch-size", "num-workers"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--quick", action="store_true",
                        help="Use the first coordinate on each axis and at most 100 shots and 10 errors")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        source = load_config(args.config)
        effective = _effective_config(source, args)
        path = args.output or (args.output_dir or Path.cwd()) / (source["protocol_id"] + ".csv")
        if path.resolve() == Path(args.config).resolve():
            raise ValueError("CSV output must differ from the source config")
        run_experiment(effective, path, source_config=source, progress=args.progress,
                       progress_interval=args.progress_interval)
    except (ValueError, OSError, NotImplementedError) as exc:
        parser.error(str(exc))
    print(path)
