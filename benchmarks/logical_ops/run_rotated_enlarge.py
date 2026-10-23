"""Sweep rotated-code enlargement using the native LightStim simulation pipeline.

Each completed task is appended immediately. Every invocation samples every task;
existing rows are retained, never interpreted as a request to skip or resume.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.rotated_surface_enlarge import RotatedSurfaceEnlargeExperiment
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline


DEFAULT_OUTPUT = Path(__file__).parent / "results/rotated_enlarge_results.csv"
FIELDS = [
    "run_id", "row_id", "task_index", "completed_at", "source_distance",
    "target_distance", "basis", "rounds", "source_rounds", "direction", "p",
    "p_idle", "noise_model", "decoder", "max_shots", "max_errors", "batch_size",
    "num_workers", "shots", "accepted", "errors", "logical_error_rate",
    "ci95_low", "ci95_high", "seconds", "circuit_sha256", "config_json",
]


def wilson_interval(errors, accepted, z=1.96):
    """Binomial Wilson interval, including an upper bound when no errors occur."""
    if not accepted:
        return math.nan, math.nan
    rate = errors / accepted
    denominator = 1 + z * z / accepted
    center = (rate + z * z / (2 * accepted)) / denominator
    radius = z * math.sqrt(
        rate * (1 - rate) / accepted + z * z / (4 * accepted * accepted)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-distance", type=int, default=3)
    parser.add_argument("--target-distances", type=int, nargs="+")
    parser.add_argument("--bases", "--basis", nargs="+", choices=["X", "Z"],
                        default=["X", "Z"])
    parser.add_argument("--rounds", type=int, default=1,
                        help="Total target-code extraction rounds, including transition (default: 1)")
    parser.add_argument("--source-rounds", type=int, default=1)
    parser.add_argument("--p-values", type=float, nargs="+")
    parser.add_argument("--p-idle", type=float, default=0.0,
                        help="Absolute idle depolarization probability (default: 0)")
    parser.add_argument("--decoder", choices=["pymatching", "bposd", "mwpf"],
                        default="pymatching")
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--max-errors", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", "--num-workers", dest="num_workers", type=int,
                        default=1)
    parser.add_argument("--quick", action="store_true",
                        help="Two target sizes and two p values; 1000 shots/10 errors (explicit options win)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.target_distances is None:
        args.target_distances = [args.source_distance + 2, args.source_distance + 4]
    if args.p_values is None:
        args.p_values = [1e-3, 3e-3] if args.quick else [5e-4, 1e-3, 3e-3, 1e-2]
    for key, normal, quick in [("max_shots", 1_000_000, 1000),
                               ("max_errors", 100, 10), ("batch_size", 1000, 250)]:
        if getattr(args, key) is None:
            setattr(args, key, quick if args.quick else normal)
    if args.source_distance < 3 or args.source_distance % 2 == 0:
        parser.error("--source-distance must be odd and at least 3")
    if any(d <= args.source_distance or d % 2 == 0 for d in args.target_distances):
        parser.error("--target-distances must be odd and larger than --source-distance")
    for name in ["rounds", "source_rounds", "max_shots", "max_errors", "batch_size", "num_workers"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in [*args.p_values, args.p_idle]):
        parser.error("noise probabilities must be finite and in [0, 1]")
    if args.output.exists() and args.output.stat().st_size:
        with args.output.open(newline="") as stream:
            if next(csv.reader(stream)) != FIELDS:
                parser.error(f"existing output has a different CSV schema: {args.output}")
    return args


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    args = parse_args(argv)
    run_id = uuid.uuid4().hex
    tasks = list(itertools.product(args.target_distances, args.bases, args.p_values))
    decoder = DecoderConfig(args.decoder)
    sampling = dict(max_shots=args.max_shots, max_errors=args.max_errors,
                    batch_size=args.batch_size, num_workers=args.num_workers)
    pipeline = SimulationPipeline(decoder_config=decoder, print_progress=False, **sampling)
    print(f"Run {run_id}: {len(tasks)} tasks; append to {args.output}", flush=True)
    for index, (target, basis, p) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] d={args.source_distance}->{target}, "
              f"basis={basis}, p={p:g}", flush=True)
        noise = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=args.p_idle)
        experiment_config = dict(distance=args.source_distance, target_distance=target,
                                 basis=basis, rounds=args.rounds, source_rounds=args.source_rounds,
                                 direction="+x,+y", noise_model="circuit_level")
        circuit = RotatedSurfaceEnlargeExperiment(
            **experiment_config, noise_params=noise,
        ).build()
        config = dict(experiment="RotatedSurfaceEnlargeExperiment", schema_version=1,
                      protocol=experiment_config, noise=asdict(noise),
                      decoder=asdict(decoder), sampling=sampling,
                      allow_gauge_detectors=False, target_observable_indices=None,
                      seed_policy="native pipeline: 0 for single worker; process ID based for multiple workers",
                      quick=args.quick)
        stats = pipeline.run(circuit, json_metadata=config)
        low, high = wilson_interval(stats.errors, stats.post_selected_shots)
        rate = stats.errors / stats.post_selected_shots if stats.post_selected_shots else math.nan
        row = dict(
            run_id=run_id, row_id=uuid.uuid4().hex, task_index=index,
            completed_at=datetime.now(timezone.utc).isoformat(),
            source_distance=args.source_distance, target_distance=target, basis=basis,
            rounds=args.rounds, source_rounds=args.source_rounds, direction="+x,+y",
            p=p, p_idle=args.p_idle, noise_model="circuit_level", decoder=args.decoder,
            **sampling, shots=stats.shots, accepted=stats.post_selected_shots,
            errors=stats.errors, logical_error_rate=rate, ci95_low=low, ci95_high=high,
            seconds=stats.seconds, circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
            config_json=json.dumps(config, sort_keys=True, separators=(",", ":")),
        )
        append_result(args.output, row)
        print(f"  errors={stats.errors}/{stats.post_selected_shots}, LER={rate:.6g}, "
              f"95% CI=[{low:.6g}, {high:.6g}], {stats.seconds:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
