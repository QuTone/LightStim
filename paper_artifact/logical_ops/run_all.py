"""
Logical Operations Benchmark Suite — Paper Figures 1-6.

Sweeps gate types × distances × physical error rates for the unrotated surface code.
Outputs CSV results for each gate/figure.

Usage:
    # Full run (all figures):
    PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py

    # Specific figure:
    PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --figure 1

    # Quick test (fewer shots, 2 distances, 3 p-values):
    PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --quick

    # Specific gate:
    PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --gate H
"""

import sys
import io
import argparse
import contextlib
from pathlib import Path
from itertools import product
from typing import List

import pandas as pd

# Ensure repo root is on sys.path
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

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

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
    "p": [1e-3, 5e-3],
    "rounds": 2,
}

# bposd handles non-CSS correlations; pymatching fails for S/CNOT
BPOSD_DECODER = DecoderConfig(name="bposd", backend="cpu")
PYMATCHING_DECODER = DecoderConfig(name="pymatching", backend="cpu")

# Checkpoint result columns — excluded from deduplication key
_CK_RESULT_COLS = frozenset({
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "seconds", "decoder",
})


def _ck_key(d: dict) -> tuple:
    """Stable checkpoint key: input fields only, floats normalized to 6-digit sci notation."""
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(d.items()) if k not in _CK_RESULT_COLS
    )


def _run_tasks(task_list, max_shots, max_errors, num_workers, checkpoint_path, decoder_config):
    """Run a list of (circuit, metadata) tuples with per-task checkpointing."""
    # Load existing checkpoint so already-done tasks are skipped on resume
    existing_records = []
    done_keys = set()
    cp = Path(checkpoint_path)
    if cp.exists():
        df_ck = pd.read_csv(cp)
        existing_records = df_ck.to_dict("records")
        for rec in existing_records:
            done_keys.add(_ck_key(rec))
        print(f"  Checkpoint: {len(done_keys)} tasks already done, skipping.")

    pending = [(c, m) for c, m in task_list if _ck_key(m) not in done_keys]
    n_skip = len(task_list) - len(pending)
    if n_skip:
        print(f"  Skipping {n_skip} completed tasks, {len(pending)} remaining.")

    pipeline = SimulationPipeline(
        decoder_config=decoder_config,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=1_000,
        num_workers=num_workers,
        print_progress=True,
    )

    new_records = []
    for i, (circuit, meta) in enumerate(pending):
        gate_label = meta.get("gate", "?")
        sub_label = meta.get("sub_experiment", "?")
        d_label = meta.get("d", "?")
        p_label = meta.get("p", "?")
        print(f"\n[{i+1}/{len(pending)}] {gate_label} {sub_label} d={d_label} p={p_label}",
              flush=True)

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
        new_records.append(row)

        # Persist immediately — a kill/OOM never loses this result
        pd.DataFrame([row]).to_csv(cp, mode="a", header=not cp.exists(), index=False)
        print(f"  -> LER={stats.logical_error_rate:.2e} ({stats.errors} errors, "
              f"{stats.shots:,} shots, {stats.seconds:.1f}s)", flush=True)

    all_records = existing_records + new_records
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# =============================================================================
# CNOT sub-experiment definitions (shared between trans and LS)
# =============================================================================

CNOT_SUB_EXPERIMENTS = [
    # (label,  init_C, init_T, meas_C, meas_T)
    ("ZZ_ZZ", "Z",    "Z",    "Z",    "Z"),
    ("ZX_ZX", "Z",    "X",    "Z",    "X"),
    ("XZ_XX", "X",    "Z",    "X",    "X"),
    ("XZ_ZZ", "X",    "Z",    "Z",    "Z"),
    ("XX_XX", "X",    "X",    "X",    "X"),
]

# =============================================================================
# Figure 1: LS CNOT ZZ->XX
# =============================================================================

def run_figure1(sweep, max_shots, max_errors, num_workers):
    """Figure 1: LS CNOT ZZ->XX — 5 sub-experiments."""
    print("=" * 60)
    print("FIGURE 1: LS CNOT ZZ->XX")
    print("=" * 60)

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
        meta = {
            "gate": "CNOT_LS",
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
            "figure": 1,
        }
        tasks.append((circuit, meta))

    return _run_tasks(
        tasks, max_shots, max_errors, num_workers,
        checkpoint_path=OUTPUT_DIR / "fig1_cnot_ls_zz_xx.csv",
        decoder_config=BPOSD_DECODER,
    )


# =============================================================================
# Figure 2: LS CNOT XX->ZZ
# =============================================================================

def run_figure2(sweep, max_shots, max_errors, num_workers):
    """Figure 2: LS CNOT XX->ZZ — 5 sub-experiments."""
    print("=" * 60)
    print("FIGURE 2: LS CNOT XX->ZZ")
    print("=" * 60)

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
        meta = {
            "gate": "CNOT_LS",
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
            "figure": 2,
        }
        tasks.append((circuit, meta))

    return _run_tasks(
        tasks, max_shots, max_errors, num_workers,
        checkpoint_path=OUTPUT_DIR / "fig2_cnot_ls_xx_zz.csv",
        decoder_config=BPOSD_DECODER,
    )


# =============================================================================
# Figure 3: Transversal CNOT
# =============================================================================

def run_figure3(sweep, max_shots, max_errors, num_workers):
    """Figure 3: Transversal CNOT — 5 sub-experiments."""
    print("=" * 60)
    print("FIGURE 3: Transversal CNOT")
    print("=" * 60)

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
        meta = {
            "gate": "CNOT_trans",
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
            "figure": 3,
        }
        tasks.append((circuit, meta))

    return _run_tasks(
        tasks, max_shots, max_errors, num_workers,
        checkpoint_path=OUTPUT_DIR / "fig3_cnot_trans.csv",
        decoder_config=BPOSD_DECODER,
    )


# =============================================================================
# Figure 4: H gate
# =============================================================================

def run_figure4(sweep, max_shots, max_errors, num_workers):
    """Figure 4: H gate — 2 sub-experiments (H_ZtoX and H_XtoZ)."""
    print("=" * 60)
    print("FIGURE 4: H gate")
    print("=" * 60)

    sub_exps = [
        ("H_ZtoX", "Z", "X"),
        ("H_XtoZ", "X", "Z"),
    ]

    tasks = []
    for (sub, init_b, meas_b), d, p in product(sub_exps, sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_gate_verification_circuit(
                d, ["fold_transversal_hadamard"], init_b, meas_b,
                rounds=sweep["rounds"], unencode=False, noise_params=noise,
            )
        meta = {
            "gate": "H",
            "sub_experiment": sub,
            "init_basis": init_b,
            "measure_basis": meas_b,
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
            "figure": 4,
        }
        tasks.append((circuit, meta))

    return _run_tasks(
        tasks, max_shots, max_errors, num_workers,
        checkpoint_path=OUTPUT_DIR / "fig4_h.csv",
        decoder_config=BPOSD_DECODER,
    )


# =============================================================================
# Figure 5: S gate (S_oneway: S then noiseless S†, measuring X)
# =============================================================================

def run_figure5(sweep, max_shots, max_errors, num_workers):
    """Figure 5: S gate via S_oneway — S then noiseless S†, measuring X."""
    print("=" * 60)
    print("FIGURE 5: S gate (S_oneway)")
    print("=" * 60)

    tasks = []
    for d, p in product(sweep["distance"], sweep["p"]):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_s_roundtrip_circuit(d, rounds=sweep["rounds"], noise_params=noise)
        meta = {
            "gate": "S",
            "sub_experiment": "S_oneway",
            "init_basis": "X",
            "measure_basis": "X",
            "d": d,
            "rounds": sweep["rounds"],
            "p": p,
            "figure": 5,
        }
        tasks.append((circuit, meta))

    return _run_tasks(
        tasks, max_shots, max_errors, num_workers,
        checkpoint_path=OUTPUT_DIR / "fig5_s.csv",
        decoder_config=BPOSD_DECODER,
    )


# =============================================================================
# Figure 6: Memory baseline
# =============================================================================

def run_figure6(sweep, max_shots, max_errors, num_workers):
    """Figure 6: Memory baseline — Z-basis memory, rounds=d."""
    print("=" * 60)
    print("FIGURE 6: Memory baseline (Unrotated SC, Z basis)")
    print("=" * 60)

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
        meta = {
            "gate": "memory",
            "sub_experiment": "Z_memory",
            "init_basis": "Z",
            "measure_basis": "Z",
            "d": d,
            "rounds": d,
            "p": p,
            "figure": 6,
        }
        tasks.append((circuit, meta))

    return _run_tasks(
        tasks, max_shots, max_errors, num_workers,
        checkpoint_path=OUTPUT_DIR / "fig6_memory.csv",
        decoder_config=PYMATCHING_DECODER,
    )


# =============================================================================
# Figure registry
# =============================================================================

FIGURE_RUNNERS = {
    1: run_figure1,
    2: run_figure2,
    3: run_figure3,
    4: run_figure4,
    5: run_figure5,
    6: run_figure6,
}

FIGURE_GATES = {
    1: "CNOT_LS",
    2: "CNOT_LS",
    3: "CNOT_trans",
    4: "H",
    5: "S",
    6: "memory",
}

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Logical Operations Benchmark Suite — Paper Figures 1-6"
    )
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4, 5, 6], default=None,
                        help="Run only a specific figure (default: all)")
    parser.add_argument("--gate", type=str, default=None,
                        choices=["CNOT_LS_ZZ_XX", "CNOT_LS_XX_ZZ", "CNOT_trans", "H", "S", "memory"],
                        help="Run only one gate (use with --figure or alone)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 distances, 2 p-values, fewer shots")
    parser.add_argument("--max-shots", type=int, default=1_000_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    max_errors = 20 if args.quick else args.max_errors
    max_shots = 100_000 if args.quick else args.max_shots

    print("=" * 60)
    print("Logical Operations Benchmark — Unrotated Surface Code")
    print(f"Mode       : {'quick' if args.quick else 'full'}")
    print(f"Distances  : {sweep['distance']}")
    print(f"p values   : {sweep['p']}")
    print(f"rounds     : {sweep['rounds']}")
    print(f"max_shots  : {max_shots}")
    print(f"max_errors : {max_errors}")
    print(f"num_workers: {args.num_workers}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)

    # Determine which figures to run
    if args.figure is not None:
        figures_to_run = [args.figure]
    elif args.gate is not None:
        gate_to_fig = {
            "CNOT_LS_ZZ_XX": 1,
            "CNOT_LS_XX_ZZ": 2,
            "CNOT_trans":    3,
            "H":             4,
            "S":             5,
            "memory":        6,
        }
        figures_to_run = [gate_to_fig[args.gate]]
    else:
        figures_to_run = list(FIGURE_RUNNERS.keys())

    for fig_n in figures_to_run:
        runner = FIGURE_RUNNERS[fig_n]
        runner(sweep, max_shots, max_errors, args.num_workers)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"Results in: {OUTPUT_DIR}")
    print("\nGenerate figures:")
    for fig_n in [1, 2, 3, 4, 5, 6]:
        print(f"  PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig{fig_n}.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
