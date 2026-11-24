"""Shared options and historical decoder presets for the benchmark scripts."""

import argparse
import json
from decimal import Decimal
from math import isfinite
from numbers import Integral, Real

from lightstim.noise.config import NOISE_MODELS, NOISE_RATES, NoiseConfig
from lightstim.simulation.decoder_backend.config import (
    DecoderConfig, PipelineConfig, normalize_decoder,
)


def json_object(value):
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError
        json.dumps(parsed, allow_nan=False)
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a finite JSON object") from exc


def add_run_options(parser):
    """Use the installed simulation pipeline's defaults at every CLI entry."""
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--noise-model", choices=NOISE_MODELS, default="circuit_level")
    parser.add_argument("--noise-custom-params", type=json_object, default={},
                        help="JSON of model-specific rates, e.g. for XZ_biased")
    parser.add_argument("--mle-time-limit", type=float, default=0.0,
                        help="Seconds per shot for mle-ilp (0 for unlimited)")
    parser.add_argument("--on-decode-failure", choices=["error", "discard", "ignore"],
                        default="error")
    parser.add_argument("--max-shots", type=int, default=PipelineConfig.max_shots)
    parser.add_argument("--max-errors", type=int, default=PipelineConfig.max_errors)
    parser.add_argument("--batch-size", type=int, default=PipelineConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=PipelineConfig.num_workers)
    parser.add_argument("--seed", type=int, default=PipelineConfig.seed)


def validate_run_options(parser, args):
    for field in ("max_shots", "max_errors", "batch_size", "num_workers"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")
    if not isfinite(args.mle_time_limit) or args.mle_time_limit < 0:
        parser.error("--mle-time-limit must be finite and non-negative")
    if args.mle_time_limit and args.decoder != "mle-ilp":
        parser.error("--mle-time-limit is only valid with --decoder mle-ilp")


def benchmark_noise(p, model="circuit_level", custom_params=None, **overrides):
    overrides = {key: val for key, val in overrides.items() if val is not None}
    # Historical command lines permit rates through 1; the v1 base-rate
    # shorthand is capped at 0.5 but individual channel rates allow 1.
    if p > 0.5:
        overrides = {**dict.fromkeys(NOISE_RATES, p),
                     **overrides}
        p = 0
    return NoiseConfig.from_spec({
        "model": model, "p": p, "overrides": overrides, "custom_params": custom_params or {},
    })


def benchmark_decoder(name, backend=None, *, mle_time_limit=0.0,
                      on_decode_failure="error", use_presets=True,
                      osd_order=10, max_iterations=1000, ms_scaling_factor=0):
    name, backend = normalize_decoder(name, backend)
    params = {}
    if name in ("bposd", "nv-qldpc-decoder") and use_presets:
        params = dict(max_iterations=max_iterations, osd_order=osd_order,
                      bp_method="min_sum", ms_scaling_factor=ms_scaling_factor,
                      osd_method="osd_cs")
        if backend == "gpu":
            params["use_osd"] = True
    elif name == "mwpf" and use_presets:
        params["cluster_node_limit"] = 50
    elif name == "mle-ilp":
        params["time_limit"] = mle_time_limit
    return DecoderConfig.from_spec({"name": name, "backend": backend,
                                    "params": params, "on_decode_failure": on_decode_failure})


def decoder_metadata(config):
    return {
        "decoder_name": config.name,
        "backend": config.backend,
        "decoder_params": json.dumps(config.params, sort_keys=True),
        "on_decode_failure": config.on_decode_failure,
        "decoder_time_limit": config.params.get("time_limit", 0.0),
    }


def checkpoint_rows(path):
    """Preserve float round-trips and the full seed even with empty legacy cells."""
    import pandas as pd
    return pd.read_csv(path, dtype={"seed": str}, float_precision="round_trip").to_dict("records")


def checkpoint_key(row, fields, defaults=None):
    """Compare selected input fields after a pandas CSV round-trip."""
    items = []
    for key in sorted(fields):
        value = row.get(key)
        if value is None or (isinstance(value, float) and value != value):
            value = (defaults or {}).get(key)
        if value is not None:
            if key == "seed":
                value = format(Decimal(str(value)).normalize(), "f")
            elif isinstance(value, Real) and not isinstance(value, (bool, Integral)):
                value = int(value) if value.is_integer() else value
            items.append((key, str(value)))
    return tuple(items)
