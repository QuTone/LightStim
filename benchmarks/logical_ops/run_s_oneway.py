#!/usr/bin/env python3
"""
S Gate One-Way Benchmark — Unrotated Surface Code

S†_L is noiseless (restoration only); the measured LER is the single-S-gate
logical error rate directly — no post-hoc division by 2 needed.

Circuit:  |+⟩ → SE → S_L (noisy) → SE → S†_L (noiseless) → SE → MX

Usage:
    venv/bin/python benchmarks/logical_ops/run_s_oneway.py [options]

    --quick              Reduced sweep for fast iteration
    --distances N [N ...]  Override distance list (e.g. --distances 3 5 7)
    --max-shots N
    --max-errors N
    --num-workers N
    --decoder NAME       bposd (default) | pymatching | mwpf | nv-qldpc-decoder

Output:
    benchmarks/logical_ops/results/fig1_s_oneway_raw.csv
"""

import sys
import io
import argparse
import contextlib
from pathlib import Path
from itertools import product
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lightstim.protocols.fold_transversal import build_s_oneway_circuit
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig

# =============================================================================
# Sweep configuration
# =============================================================================

FULL_SWEEP = {
    "distance": [3, 5, 7],
    "p": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    "rounds": 2,
}

QUICK_SWEEP = {
    "distance": [3, 5],
    "p": [1e-4, 1e-3, 5e-3],
    "rounds": 2,
}

PIPELINE_DEFAULTS = {
    "max_errors": 100,
    "max_shots": 1_000_000_000,
    "num_workers": 32,
}

CHECKPOINT_KEYS = ["gate", "sub_experiment", "d", "p"]

OUT_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = OUT_DIR / "fig1_s_oneway_raw.csv"

# =============================================================================
# Task builder
# =============================================================================

def build_s_oneway_tasks(sweep: dict) -> List[ExperimentTask]:
    tasks = []
    for d, p in product(sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_s_oneway_circuit(d, rounds=sweep["rounds"], noise_params=noise)
        tasks.append(ExperimentTask(circuit, json_metadata={
            "gate": "S",
            "sub_experiment": "S_oneway",
            "init_basis": "X",
            "measure_basis": "X",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
        }))
    return tasks

# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_completed_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    missing = [k for k in CHECKPOINT_KEYS if k not in df.columns]
    if missing:
        return set()
    return set(zip(*(df[k] for k in CHECKPOINT_KEYS)))


def filter_tasks(tasks: List[ExperimentTask], completed: set) -> List[ExperimentTask]:
    remaining, skipped = [], 0
    for t in tasks:
        m = t.json_metadata
        key = (m["gate"], m["sub_experiment"], m["d"], m["p"])
        if key in completed:
            skipped += 1
        else:
            remaining.append(t)
    if skipped:
        print(f"  Skipping {skipped} already-completed tasks (checkpoint).")
    return remaining


def append_results(df_new: pd.DataFrame, csv_path: Path) -> None:
    if csv_path.exists():
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="S Gate One-Way Benchmark — Unrotated SC")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sweep for fast iteration")
    parser.add_argument("--distances", type=int, nargs="+", default=None,
                        help="Override distances (e.g. --distances 3 5 7)")
    parser.add_argument("--p-values", type=float, nargs="+", default=None,
                        help="Override p values (e.g. --p-values 1e-3 2e-3 5e-3)")
    parser.add_argument("--max-shots", type=int, default=PIPELINE_DEFAULTS["max_shots"])
    parser.add_argument("--max-errors", type=int, default=PIPELINE_DEFAULTS["max_errors"])
    parser.add_argument("--num-workers", type=int, default=PIPELINE_DEFAULTS["num_workers"])
    parser.add_argument("--decoder", type=str, default="bposd",
                        choices=["bposd", "pymatching", "mwpf", "nv-qldpc-decoder"],
                        help="Decoder backend (default: bposd)")
    args = parser.parse_args()

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    if args.distances:
        sweep = dict(sweep, distance=args.distances)
    if args.p_values:
        sweep = dict(sweep, p=args.p_values)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("S Gate One-Way Benchmark — Unrotated Surface Code")
    print("S†_L noiseless; LER = single-gate S error rate directly")
    print(f"Mode       : {'quick' if args.quick else 'full'}")
    print(f"Distances  : {sweep['distance']}")
    print(f"p values   : {sweep['p']}")
    print(f"rounds     : {sweep['rounds']}")
    print(f"max_shots  : {args.max_shots}")
    print(f"max_errors : {args.max_errors}")
    print(f"num_workers: {args.num_workers}")
    print(f"decoder    : {args.decoder}")
    print(f"Output     : {CSV_PATH}")
    print("=" * 60)

    print("\nBuilding tasks...")
    all_tasks = build_s_oneway_tasks(sweep)
    completed = load_completed_keys(CSV_PATH)
    tasks = filter_tasks(all_tasks, completed)

    if not tasks:
        print(f"All {len(all_tasks)} tasks already done — nothing to run.")
        return

    n_obs = sum(t.circuit.num_observables for t in tasks)
    print(f"{len(tasks)}/{len(all_tasks)} tasks to run, {n_obs} total observables\n")

    backend = "gpu" if args.decoder == "nv-qldpc-decoder" else "cpu"
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(args.decoder, backend=backend),
        max_errors=args.max_errors,
        max_shots=args.max_shots,
        num_workers=args.num_workers,
        print_progress=True,
    )

    for j, task in enumerate(tasks):
        meta = task.json_metadata
        print(f"[{j+1}/{len(tasks)}] S_oneway  d={meta['d']}  p={meta['p']:.1e}", flush=True)
        stats = pipeline.run(task.circuit, meta)
        row = {
            **meta,
            "shots":               stats.shots,
            "post_selected_shots": stats.post_selected_shots,
            "post_selection_rate": stats.post_selection_rate,
            "errors":              stats.errors,
            "logical_error_rate":  stats.logical_error_rate,
            "seconds":             stats.seconds,
            "decoder":             stats.decoder,
        }
        append_results(pd.DataFrame([row]), CSV_PATH)
        print(f"    → LER={stats.logical_error_rate:.2e}  "
              f"({stats.errors}/{stats.shots:,} shots)", flush=True)

    print(f"\nDone. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
