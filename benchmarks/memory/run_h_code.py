"""H-family memory LER with full detector postselection and no decoding.

Sweep even code sizes, X/Z bases, extraction schedules, and physical error
rates. Accept only shots with every detector zero, including final readout;
an accepted shot fails if any logical observable flips. Reuse plot_memory.py
for the output CSV. This benchmarks bare memory, not H6 encoded preparation.
"""

import argparse
import csv
import math
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode, HCodeExtractionBlock
from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock


SE_BLOCKS = {
    "dedicated": HCodeExtractionBlock,
    "coloration": GenericCSSColorationExtractionBlock,
}
KEY_COLUMNS = (
    "code", "n_data", "p", "basis", "rounds", "se_circuit", "noise_model",
    "p_idle", "p_idle_mode", "postselection", "seed", "max_shots", "max_errors", "batch_size",
)
RESULT_COLUMNS = (*KEY_COLUMNS, "distance", "n_total", "k", "block_class",
                  "decoder_name", "p_1q", "p_1q_mode",
                  "shots", "accepted", "rejected", "errors", "acceptance",
                  "logical_error_rate", "seconds")


def sample_postselected_memory(circuit, *, max_shots, max_errors, batch_size, seed):
    """Error-targeted collection, capped at max_shots, checked after each batch."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = accepted = errors = 0
    while shots < max_shots and errors < max_errors:
        dets, obs = sampler.sample(
            min(batch_size, max_shots - shots), separate_observables=True,
        )
        keep = ~dets.any(axis=1)
        shots += len(keep)
        accepted += int(keep.sum())
        errors += int(obs[keep].any(axis=1).sum())
    return {
        "shots": shots, "accepted": accepted, "rejected": shots - accepted,
        "errors": errors, "acceptance": accepted / shots if shots else math.nan,
        "logical_error_rate": errors / accepted if accepted else math.nan,
    }


def _task_key(row):
    return tuple(str(row[column]) for column in KEY_COLUMNS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=int, default=[6], help="Even data-qubit counts >= 6")
    parser.add_argument("--basis", nargs="+", choices=("X", "Z"), default=["Z", "X"])
    parser.add_argument("--se-circuits", nargs="+", choices=tuple(SE_BLOCKS), default=["dedicated"])
    parser.add_argument("--p-values", nargs="+", type=float, default=[0.002, 0.004, 0.008, 0.016])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--p-idle", type=float, default=None, help="Idle error rate; defaults to each swept p")
    parser.add_argument("--noise-model", choices=("circuit_level", "phenomenological", "code_capacity"), default="circuit_level")
    parser.add_argument("--max-shots", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/h_code_postselection.csv")
    parser.add_argument("--quick", action="store_true", help="Smoke sweep: n=6,8; two p values; at most 1000 shots/task")
    args = parser.parse_args(argv)
    if args.quick:
        args.n = [6, 8]
        args.p_values = [0.004, 0.008]
        args.max_shots = min(args.max_shots, 1000)
    if any(n < 6 or n % 2 for n in args.n):
        parser.error("--n must contain even integers >= 6")
    for name in ("rounds", "max_shots", "max_errors", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")
    rates = [*args.p_values, *([] if args.p_idle is None else [args.p_idle])]
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in rates):
        parser.error("Noise probabilities must be finite and between 0 and 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists() and args.output.stat().st_size:
        with args.output.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(RESULT_COLUMNS):
                parser.error("Output CSV has a different schema; choose a new --output file")
            done = {_task_key(row) for row in reader}

    for n, basis, schedule, p in product(args.n, args.basis, args.se_circuits, args.p_values):
        task = {
            "code": f"h_{n}", "n_data": n, "p": p, "basis": basis,
            "rounds": args.rounds, "se_circuit": schedule,
            "noise_model": args.noise_model,
            "p_idle": p if args.p_idle is None else args.p_idle,
            "p_idle_mode": "sweep" if args.p_idle is None else "fixed",
            "postselection": "all_detectors_including_readout", "seed": args.seed,
            "max_shots": args.max_shots, "max_errors": args.max_errors,
            "batch_size": args.batch_size,
        }
        key = _task_key(task)
        label = f"H{n} {basis} {schedule} p={p:g}"
        if key in done:
            print(f"Skip completed: {label}", flush=True)
            continue
        print(f"Run: {label}", flush=True)
        start = time.perf_counter()
        block = SE_BLOCKS[schedule]
        circuit = MemoryExperiment(
            qec_patch=HCode(n), extraction_block_class=block,
            rounds=args.rounds, basis=basis, noise_model=args.noise_model,
            noise_params=NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=task["p_idle"]),
        ).build()
        circuit.detector_error_model(decompose_errors=False)
        result = sample_postselected_memory(
            circuit, max_shots=args.max_shots, max_errors=args.max_errors,
            batch_size=args.batch_size, seed=args.seed,
        )
        row = {
            **task, **result, "distance": 2, "n_total": circuit.num_qubits, "k": n - 4,
            "block_class": block.__name__, "decoder_name": "none",
            "p_1q": p,
            "p_1q_mode": "sweep", "seconds": time.perf_counter() - start,
        }
        needs_header = not args.output.exists() or args.output.stat().st_size == 0
        with args.output.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        done.add(key)
        print(f"  LER={result['logical_error_rate']:.6g}, accepted={result['accepted']}/{result['shots']}, "
              f"errors={result['errors']}", flush=True)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
