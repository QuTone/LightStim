#!/usr/bin/env python3
"""
Logical Operation Benchmark — Unrotated Surface Code (Fig 1)

Records raw per-sub-experiment data for:
  H gate          2 sub-exps: H|0>→MX, H|+>→MZ
  S gate          1 sub-exp:  S·S† roundtrip |+>→MX  (LER_per_gate = total/2)
  Transversal CNOT  5 sub-exps: ZZ→ZZ, ZX→ZX, XZ→XX, XZ→ZZ, XX→XX
  LS CNOT           5 sub-exps: same basis coverage, ancilla |+> protocol
  memory          Z-basis memory at d rounds — baseline for comparison

All averaging / per-gate LER is computed in post-processing — not here.

Usage:
    python eval/logical_op_benchmark/run_logical_ops.py [--quick] [--gate GATE]

    --quick       Reduced sweep for fast iteration
    --gate GATE   Run only one gate: H | S | CNOT_trans | CNOT_LS | memory
    --max-shots N
    --max-errors N
    --num-workers N

Output:
    eval/logical_op_benchmark/results/fig1_{gate}_raw.csv
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

from lightstim.protocols.fold_transversal import (
    build_gate_verification_circuit,
    build_s_roundtrip_circuit,
)
from lightstim.protocols.cnot_trans import CNOTTransExperiment
from lightstim.protocols.cnot_ls import CNOTLSExperiment
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)

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

# Checkpoint key columns: uniquely identify one simulation task in the CSV.
CHECKPOINT_KEYS = ["gate", "sub_experiment", "d", "p"]

OUT_DIR = Path(__file__).resolve().parent / "results"

# =============================================================================
# H gate tasks
# =============================================================================

def build_h_tasks(sweep: dict) -> List[ExperimentTask]:
    """
    Two sub-experiments covering both conjugate bases:
      H_ZtoX : init Z (|0>_L) -> H_L -> measure X   checks H|0>=|+>
      H_XtoZ : init X (|+>_L) -> H_L -> measure Z   checks H|+>=|0>
    LER_H = mean(LER_ZtoX, LER_XtoZ) computed in post.
    """
    tasks = []
    sub_exps = [
        ("H_ZtoX", "Z", "X"),
        ("H_XtoZ", "X", "Z"),
    ]
    for (sub, init_b, meas_b), d, p in product(sub_exps, sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_gate_verification_circuit(
                d, ["fold_transversal_hadamard"], init_b, meas_b,
                rounds=sweep["rounds"], unencode=False, noise_params=noise,
            )
        tasks.append(ExperimentTask(circuit, json_metadata={
            "gate": "H",
            "sub_experiment": sub,
            "init_basis": init_b,
            "measure_basis": meas_b,
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
        }))
    return tasks

# =============================================================================
# S gate tasks
# =============================================================================

def build_s_tasks(sweep: dict) -> List[ExperimentTask]:
    """
    S·S† roundtrip for fault-tolerant evaluation:
      init X (|+>_L) -> SE -> S_L -> SE -> S†_L -> SE -> transversal MX
    S·S† = I so X_L = +1 always. LER_per_gate = LER_total / 2.

    Direct S|+>->MY is avoided: unencode creates weight-0 DEM errors.
    """
    tasks = []
    for d, p in product(sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_s_roundtrip_circuit(d, rounds=sweep["rounds"], noise_params=noise)
        tasks.append(ExperimentTask(circuit, json_metadata={
            "gate": "S",
            "sub_experiment": "S_roundtrip",
            "init_basis": "X",
            "measure_basis": "X",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
        }))
    return tasks

# =============================================================================
# CNOT sub-experiment definitions (shared between trans and LS)
# =============================================================================
#
# CNOT logical action (ctrl=C, tgt=T):
#   Z_C -> Z_C x Z_T       X_C -> X_C
#   Z_T -> Z_T             X_T -> X_C x X_T
#
# Sub-exp  init(C,T)  meas(C,T)  observables (noiseless)
# ZZ_ZZ    Z  Z       Z  Z       Z_C=+1, Z_T=+1   (2 independent)
# ZX_ZX    Z  X       Z  X       Z_C=+1, X_T=+1   (2 independent)
# XZ_XX    X  Z       X  X       X_C x X_T=+1     (1 joint)
# XZ_ZZ    X  Z       Z  Z       Z_C x Z_T=+1     (1 joint)
# XX_XX    X  X       X  X       X_C=+1, X_T=+1   (2 independent)
#
# Average LER: mean over the 5 sub-experiments (after summing XZ_XX + XZ_ZZ).

CNOT_SUB_EXPERIMENTS = [
    # (label,     init_C, init_T, meas_C, meas_T)
    ("ZZ_ZZ",    "Z",    "Z",    "Z",    "Z"),
    ("ZX_ZX",    "Z",    "X",    "Z",    "X"),
    ("XZ_XX",    "X",    "Z",    "X",    "X"),
    ("XZ_ZZ",    "X",    "Z",    "Z",    "Z"),
    ("XX_XX",    "X",    "X",    "X",    "X"),
]

# =============================================================================
# Transversal CNOT tasks
# =============================================================================

def build_cnot_trans_tasks(sweep: dict) -> List[ExperimentTask]:
    """5 sub-experiments for transversal CNOT via CNOTTransExperiment."""
    tasks = []
    for (sub, ic, it, mc, mt), d, p in product(CNOT_SUB_EXPERIMENTS, sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = CNOTTransExperiment(
                code_patch_class=UnrotatedSurfaceCode,
                extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                code_params_control={"distance": d},
                offset_target=(2 * d + 2, 0),
                initial_basis_control=ic,
                initial_basis_target=it,
                measure_basis_control=mc,
                measure_basis_target=mt,
                rounds_before=sweep["rounds"],
                rounds_after=sweep["rounds"],
                noise_params=noise,
            )
            circuit = exp.build()
        tasks.append(ExperimentTask(circuit, json_metadata={
            "gate": "CNOT_trans",
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
        }))
    return tasks

# =============================================================================
# LS CNOT tasks
# =============================================================================

def build_cnot_ls_tasks(sweep: dict) -> List[ExperimentTask]:
    """
    5 sub-experiments for LS CNOT via CNOTLSExperiment.
    Protocol A: ancilla init |+> (X), measure Z.
    Layout: ancilla at origin, target at (+2d, 0), control at (0, +2d).
    """
    tasks = []
    for (sub, ic, it, mc, mt), d, p in product(CNOT_SUB_EXPERIMENTS, sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = CNOTLSExperiment(
                patch_configs={
                    "c": {"distance": d},
                    "t": {"distance": d},
                    "a": {"distance": d},
                },
                offset_ta=(2 * d, 0),
                offset_ca=(0, 2 * d),
                initial_state_dict={"a": "X", "c": ic, "t": it},
                measure_state_dict={"a": "Z", "c": mc, "t": mt},
                extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                rounds=sweep["rounds"],
                noise_params=noise,
            )
            circuit = exp.build()
        tasks.append(ExperimentTask(circuit, json_metadata={
            "gate": "CNOT_LS",
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
        }))
    return tasks

# =============================================================================
# Memory baseline tasks
# =============================================================================

def build_memory_tasks(sweep: dict) -> List[ExperimentTask]:
    """
    Z-basis memory experiment — baseline for comparison with gate LER.
    rounds = d (code distance), matching the standard memory benchmark convention.

    Note: the existing eval/memory_benchmark/ runs at p in [1e-3, 1.5e-2].
    This run covers the lower p range needed for Fig 1 comparison.
    """
    tasks = []
    for d, p in product(sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            system = QECSystem()
            system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch")
            exp = MemoryExperiment(
                qec_system=system,
                extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                rounds=d,
                noise_params=noise,
                noise_model="circuit_level",
                basis="Z",
            )
            circuit = exp.build()
        tasks.append(ExperimentTask(circuit, json_metadata={
            "gate": "memory",
            "sub_experiment": "memory_Z",
            "init_basis": "Z",
            "measure_basis": "Z",
            "d": d,
            "rounds": d,
            "p": p,
        }))
    return tasks

# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_completed_keys(csv_path: Path) -> set:
    """Return set of (gate, sub_experiment, d, p) tuples already in the CSV."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    missing = [k for k in CHECKPOINT_KEYS if k not in df.columns]
    if missing:
        return set()
    return set(zip(*(df[k] for k in CHECKPOINT_KEYS)))


def filter_tasks(tasks: List[ExperimentTask], completed: set) -> List[ExperimentTask]:
    """Drop tasks whose checkpoint key is already in `completed`."""
    remaining = []
    skipped = 0
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


def append_results(df_new: "pd.DataFrame", csv_path: Path) -> None:
    """Append new results to CSV, creating it if necessary."""
    if csv_path.exists():
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


# =============================================================================
# Registry
# =============================================================================

GATE_BUILDERS = {
    "H":          build_h_tasks,
    "S":          build_s_tasks,
    "CNOT_trans": build_cnot_trans_tasks,
    "CNOT_LS":    build_cnot_ls_tasks,
    "memory":     build_memory_tasks,
}

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Logical Op Benchmark — Unrotated SC")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sweep for fast iteration")
    parser.add_argument("--gate", choices=list(GATE_BUILDERS), default=None,
                        help="Run only one gate (default: all)")
    parser.add_argument("--max-shots", type=int, default=PIPELINE_DEFAULTS["max_shots"])
    parser.add_argument("--max-errors", type=int, default=PIPELINE_DEFAULTS["max_errors"])
    parser.add_argument("--num-workers", type=int, default=PIPELINE_DEFAULTS["num_workers"])
    parser.add_argument("--decoder", type=str, default="bposd",
                        choices=["bposd", "pymatching", "mwpf", "nv-qldpc-decoder"],
                        help="Decoder to use (default: bposd)")
    args = parser.parse_args()

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gates_to_run = [args.gate] if args.gate else list(GATE_BUILDERS)

    print("=" * 60)
    print("Logical Operation Benchmark — Unrotated Surface Code")
    print(f"Mode       : {'quick' if args.quick else 'full'}")
    print(f"Gates      : {gates_to_run}")
    print(f"Distances  : {sweep['distance']}")
    print(f"p values   : {sweep['p']}")
    print(f"rounds     : {sweep['rounds']}")
    print(f"max_shots  : {args.max_shots}")
    print(f"max_errors : {args.max_errors}")
    print(f"num_workers: {args.num_workers}")
    print(f"decoder    : {args.decoder}")
    print(f"Output     : {OUT_DIR}")
    print("=" * 60)

    backend = "gpu" if args.decoder == "nv-qldpc-decoder" else "cpu"
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(args.decoder, backend=backend),
        max_errors=args.max_errors,
        max_shots=args.max_shots,
        num_workers=args.num_workers,
        print_progress=True,
    )

    for gate in gates_to_run:
        print(f"\n{'─'*50}")
        csv_path = OUT_DIR / f"fig1_{gate.lower()}_raw.csv"

        print(f"Building {gate} tasks...")
        all_tasks = GATE_BUILDERS[gate](sweep)

        completed = load_completed_keys(csv_path)
        tasks = filter_tasks(all_tasks, completed)

        if not tasks:
            print(f"  All {len(all_tasks)} tasks already done — skipping.")
            continue

        n_obs = sum(t.circuit.num_observables for t in tasks)
        print(f"  {len(tasks)}/{len(all_tasks)} tasks to run, {n_obs} total observables")

        print(f"Running {gate} ({len(tasks)} tasks)...")
        for j, task in enumerate(tasks):
            meta = task.json_metadata
            print(f"  [{j+1}/{len(tasks)}] {meta.get('gate')} {meta.get('sub_experiment')} "
                  f"d={meta.get('d')} p={meta.get('p')}", flush=True)
            stats = pipeline.run(task.circuit, meta)
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
            append_results(pd.DataFrame([row]), csv_path)
            print(f"    → LER={stats.logical_error_rate:.2e} "
                  f"({stats.errors}/{stats.shots:,} shots)", flush=True)

        print(f"Saved: {csv_path}")

    print("\nDone. Run post-processing to average sub-experiments into per-gate LER.")


if __name__ == "__main__":
    main()
