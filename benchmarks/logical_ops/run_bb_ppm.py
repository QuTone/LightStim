"""Run a portable BB PPM fixed-shot sweep with per-shot resume checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.logical_ops.bb_ppm_runner import (
    DEFAULT_OUTPUT, DEFAULT_P_VALUES, DEFAULT_RUN_DIR, available_workers, run_campaign,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Two p points and two shots per point; explicit options win")
    parser.add_argument("--p-values", type=float, nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", help="One fixed count for all points, or one per p")
    parser.add_argument("--workers", type=int, default=available_workers())
    parser.add_argument("--target", nargs="+", default=["X2", "Z7", "X12"], help="One joint measurement, e.g. X2 Z7 X12")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--basis", choices=("X", "Z"), default="X")
    parser.add_argument("--seed", type=int, default=0, help="Physical sample seed; decoder seed stays zero")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true", help="Refuse an existing run instead of resuming it")
    args = parser.parse_args(argv)
    args.target = [factor for argument in args.target for factor in argument.split()]
    args.p_values = args.p_values or (DEFAULT_P_VALUES[:2] if args.quick else DEFAULT_P_VALUES)
    args.shots = args.shots or [2 if args.quick else 50]
    if len(args.shots) == 1:
        args.shots *= len(args.p_values)
    if len(args.shots) != len(args.p_values) or any(n <= 0 for n in args.shots):
        parser.error("--shots requires one positive count or one per p")
    if not 1 <= args.workers <= available_workers():
        parser.error(f"--workers must be between 1 and {available_workers()}")
    if args.rounds < 1 or not 0 <= args.seed < 2**64:
        parser.error("--rounds must be positive and --seed an unsigned 64-bit integer")
    if len(set(args.p_values)) != len(args.p_values) or any(not 0 < p < .5 for p in args.p_values):
        parser.error("--p-values must be unique and in (0, .5)")
    if any(re.fullmatch(r"[XYZ](?:[1-9]|1[0-2])", value.upper()) is None for value in args.target):
        parser.error("--target expects Pauli factors X1 through Z12")
    return args


def main(argv=None):
    args = parse_args(argv)
    from lightstim.protocols.bb_ppm import BBPPMStep
    try:
        step = BBPPMStep(tuple((int(value[1:]), value[0].upper()) for value in args.target))
        run_campaign(sequence=[step], p_values=args.p_values, shots=args.shots, workers=args.workers,
                     rounds=args.rounds, initial_states=args.basis, final_measure_states=args.basis,
                     seed=args.seed, run_dir=args.run_dir, output=args.output, resume=not args.no_resume,
                     print_progress=True)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"BB PPM benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
