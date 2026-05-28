"""
State Injection Benchmark Suite — Paper Figures 7-10.

Sweeps injection_protocol × inject_state × post_select_mode × distances × rounds × p.
Outputs CSV results with per-task checkpointing.

Usage:
    # Full run (all states, all protocols):
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py

    # Quick test (fewer shots, corner only, Z+Y states):
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py --quick

    # Specific state:
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py --inject-state Y

    # Specific protocol:
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py --protocol corner
"""

import sys
import io
import argparse
import contextlib
from pathlib import Path
from itertools import product

import pandas as pd

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lightstim.protocols.state_injection import StateInjectionExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = OUTPUT_DIR / "state_injection.csv"

# =============================================================================
# Sweep configuration
# =============================================================================

FULL_SWEEP = {
    "injection_protocol": ["corner", "middle"],
    "inject_state": ["Z", "X", "Y"],
    "post_select_mode": ["full_postselection", "full_qec", "hybrid"],
    "distance": [3, 5, 7],
    "rounds": [2, 3],
    "p": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
}

QUICK_SWEEP = {
    "injection_protocol": ["corner"],
    "inject_state": ["Z", "Y"],
    "post_select_mode": ["full_postselection", "full_qec", "hybrid"],
    "distance": [3, 5],
    "rounds": [2],
    "p": [1e-4, 5e-4, 1e-3],
}

PIPELINE_CONFIG = {
    "max_errors": 200,
    "max_shots": 100_000_000,
    "num_workers": 32,
}

# Checkpoint key columns: uniquely identify one simulation task
CHECKPOINT_KEYS = ["injection_protocol", "inject_state", "post_select_mode", "d", "rounds", "p"]

# =============================================================================
# Checkpoint helpers
# =============================================================================

def _ck_key(meta: dict) -> tuple:
    """Stable checkpoint key: float p normalized to 6-digit sci notation."""
    result = []
    for k in CHECKPOINT_KEYS:
        v = meta[k]
        if isinstance(v, float):
            result.append(f"{v:.6e}")
        else:
            result.append(str(v))
    return tuple(result)


def load_completed_keys() -> set:
    """Return set of checkpoint keys already in the CSV."""
    if not CSV_PATH.exists():
        return set()
    df = pd.read_csv(CSV_PATH)
    if any(k not in df.columns for k in CHECKPOINT_KEYS):
        return set()
    completed = set()
    for _, row in df.iterrows():
        completed.add(_ck_key(row.to_dict()))
    return completed


# =============================================================================
# Task building
# =============================================================================

def build_tasks(sweep: dict) -> list:
    """Build (circuit, metadata) list from sweep configuration."""
    tasks = []
    combos = list(product(
        sweep["injection_protocol"],
        sweep["inject_state"],
        sweep["post_select_mode"],
        sweep["distance"],
        sweep["rounds"],
        sweep["p"],
    ))

    print(f"Building {len(combos)} tasks...")
    for protocol, state, mode, d, r, p in combos:
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = StateInjectionExperiment(
                distance=d, rounds=r,
                inject_state=state,
                protocol=protocol,
                post_select_mode=mode,
                noise_params=noise,
            )
            circuit = exp.build()
        meta = {
            "injection_protocol": protocol,
            "inject_state": state,
            "post_select_mode": mode,
            "d": d,
            "rounds": r,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="State Injection Benchmark Suite — Paper Figures 7-10"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: corner only, Z+Y states, d=3,5, fewer p-values")
    parser.add_argument("--inject-state", type=str, default=None,
                        choices=["Z", "X", "Y"],
                        help="Run only one inject_state (default: all)")
    parser.add_argument("--protocol", type=str, default=None,
                        choices=["corner", "middle"],
                        help="Run only one injection protocol (default: all)")
    parser.add_argument("--max-shots", type=int, default=PIPELINE_CONFIG["max_shots"])
    parser.add_argument("--max-errors", type=int, default=PIPELINE_CONFIG["max_errors"])
    parser.add_argument("--num-workers", type=int, default=PIPELINE_CONFIG["num_workers"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP

    # Apply per-flag overrides
    if args.inject_state is not None:
        sweep = dict(sweep)
        sweep["inject_state"] = [args.inject_state]
    if args.protocol is not None:
        sweep = dict(sweep)
        sweep["injection_protocol"] = [args.protocol]

    max_shots = 100_000 if args.quick else args.max_shots
    max_errors = 20 if args.quick else args.max_errors

    print("=" * 60)
    print("State Injection Benchmark — Rotated Surface Code")
    print(f"Mode             : {'quick' if args.quick else 'full'}")
    print(f"injection_protocol: {sweep['injection_protocol']}")
    print(f"inject_state     : {sweep['inject_state']}")
    print(f"post_select_mode : {sweep['post_select_mode']}")
    print(f"distances        : {sweep['distance']}")
    print(f"rounds           : {sweep['rounds']}")
    print(f"p values         : {sweep['p']}")
    print(f"max_shots        : {max_shots}")
    print(f"max_errors       : {max_errors}")
    print(f"num_workers      : {args.num_workers}")
    print(f"Output           : {CSV_PATH}")
    print("=" * 60)

    # Build tasks and filter by checkpoint
    all_tasks = build_tasks(sweep)
    completed = load_completed_keys()

    pending = []
    skipped = 0
    for circuit, meta in all_tasks:
        if _ck_key(meta) in completed:
            skipped += 1
        else:
            pending.append((circuit, meta))

    if skipped:
        print(f"  Checkpoint: {skipped} tasks already done, skipping.")
    print(f"  Tasks: {len(pending)} to run / {len(all_tasks)} total")

    if not pending:
        print("All tasks already done.")
        return

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching", backend="cpu"),
        max_shots=max_shots,
        max_errors=max_errors,
        num_workers=args.num_workers,
        print_progress=True,
    )

    for j, (circuit, meta) in enumerate(pending):
        print(f"\n[{j+1}/{len(pending)}] protocol={meta['injection_protocol']} "
              f"state={meta['inject_state']} mode={meta['post_select_mode']} "
              f"d={meta['d']} r={meta['rounds']} p={meta['p']}", flush=True)
        stats = pipeline.run(circuit, meta)
        row = {
            **meta,
            "shots": stats.shots,
            "post_selected_shots": stats.post_selected_shots,
            "post_selection_rate": stats.post_selection_rate,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": stats.seconds,
            "decoder": stats.decoder,
        }
        # Append immediately — a kill/OOM never loses this result
        pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)
        print(f"  -> LER={stats.logical_error_rate:.2e} "
              f"({stats.errors}/{stats.shots:,})", flush=True)

    print(f"\nDone. Saved CSV: {CSV_PATH}")
    print("\nGenerate figures:")
    for fig_n in [7, 8, 9, 10]:
        print(f"  PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig{fig_n}.py")


if __name__ == "__main__":
    main()
