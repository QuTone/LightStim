"""
General logical operations benchmark runner for LightStim.

Sweeps gate types × distances × physical error rates for the unrotated surface code.
Results are saved to a combined CSV with per-task checkpointing (append-on-complete).

Supported gates
---------------
    H             Fold-transversal Hadamard (2 sub-experiments: Z→X and X→Z)
    S             Fold-transversal S gate via S·S† roundtrip (1 sub-experiment)
    CNOT_trans    Transversal CNOT (5 sub-experiments)
    CNOT_LS_ZZ_XX Lattice Surgery CNOT, ZZ-XX protocol (5 sub-experiments)
    CNOT_LS_XX_ZZ Lattice Surgery CNOT, XX-ZZ protocol (5 sub-experiments)
    memory        Z-basis memory baseline (1 sub-experiment, rounds=d)

For state injection benchmarks, see benchmarks/state_injection/.

Decoders
--------
    bposd         CPU BP+OSD  (default for gates; handles non-CSS correlations)
    pymatching    CPU MWPM    (default for memory; sufficient for CSS memory)
    mwpf          CPU MWPF    (general purpose)

CSV output schema
-----------------
    gate, sub_experiment, init_basis, measure_basis, d, rounds, p,
    shots, post_selected_shots, post_selection_rate,
    errors, logical_error_rate, seconds, decoder

Usage
-----
    # All gates, default sweep:
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py

    # Single gate, custom sweep:
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py \\
        --gate H --distances 3 5 7 --p-values 5e-4 1e-3 2e-3 5e-3 1e-2

    # Quick test (fewer shots):
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py --quick

    # Custom output path:
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py \\
        --gate CNOT_trans --output benchmarks/logical_ops/results/cnot_trans.csv
"""

import argparse
import contextlib
import io
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root → lightstim importable

from lightstim.protocols.fold_transversal import (
    build_gate_verification_circuit,
    build_s_roundtrip_circuit,
)
from lightstim.protocols.cnot_trans import CNOTTransExperiment
from lightstim.protocols.cnot_ls import CNOTLSExperiment
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)

# ── Available gates ───────────────────────────────────────────────────────────

ALL_GATES = ["H", "S", "CNOT_trans", "CNOT_LS_ZZ_XX", "CNOT_LS_XX_ZZ", "memory"]

# CNOT sub-experiments (shared between transversal and LS variants)
# (label, init_control, init_target, meas_control, meas_target)
_CNOT_SUB_EXPERIMENTS = [
    ("ZZ_ZZ", "Z", "Z", "Z", "Z"),
    ("ZX_ZX", "Z", "X", "Z", "X"),
    ("XZ_XX", "X", "Z", "X", "X"),
    ("XZ_ZZ", "X", "Z", "Z", "Z"),
    ("XX_XX", "X", "X", "X", "X"),
]

# ── Circuit builders ──────────────────────────────────────────────────────────

def _build_h_tasks(distances, p_values, rounds):
    """H gate: 2 sub-experiments (Z→X and X→Z)."""
    tasks = []
    sub_exps = [
        ("H_ZtoX", "Z", "X"),
        ("H_XtoZ", "X", "Z"),
    ]
    for (sub, init_b, meas_b), d, p in product(sub_exps, distances, p_values):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_gate_verification_circuit(
                d, ["fold_transversal_hadamard"], init_b, meas_b,
                rounds=rounds, unencode=False, noise_params=noise,
            )
        meta = {
            "gate": "H",
            "sub_experiment": sub,
            "init_basis": init_b,
            "measure_basis": meas_b,
            "d": d,
            "rounds": rounds,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


def _build_s_tasks(distances, p_values, rounds):
    """S gate: S·S† roundtrip (LER_per_gate = total LER / 2)."""
    tasks = []
    for d, p in product(distances, p_values):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = build_s_roundtrip_circuit(d, rounds=rounds, noise_params=noise)
        meta = {
            "gate": "S",
            "sub_experiment": "S_roundtrip",
            "init_basis": "X",
            "measure_basis": "X",
            "d": d,
            "rounds": rounds,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


def _build_cnot_trans_tasks(distances, p_values, rounds):
    """Transversal CNOT: 5 sub-experiments."""
    tasks = []
    for (sub, ic, it, mc, mt), d, p in product(_CNOT_SUB_EXPERIMENTS, distances, p_values):
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
                rounds_before=rounds,
                rounds_after=rounds,
                noise_params=noise,
            )
            circuit = exp.build()
        meta = {
            "gate": "CNOT_trans",
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": rounds,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


def _build_cnot_ls_tasks(distances, p_values, rounds, protocol):
    """
    Lattice Surgery CNOT: 5 sub-experiments.

    protocol: 'ZZ_XX' — ancilla init |+> (X), measure Z
              'XX_ZZ' — ancilla init |0> (Z), measure X
    Both use the same CNOTLSExperiment; only the gate label differs.
    """
    tasks = []
    for (sub, ic, it, mc, mt), d, p in product(_CNOT_SUB_EXPERIMENTS, distances, p_values):
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
                rounds=rounds,
                noise_params=noise,
            )
            circuit = exp.build()
        gate_label = f"CNOT_LS_{protocol}"
        meta = {
            "gate": gate_label,
            "sub_experiment": sub,
            "init_basis": f"{ic}{it}",
            "measure_basis": f"{mc}{mt}",
            "d": d,
            "rounds": rounds,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


def _build_memory_tasks(distances, p_values):
    """Memory baseline: Z-basis, rounds = d."""
    tasks = []
    for d, p in product(distances, p_values):
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
            "sub_experiment": "memory_Z",
            "init_basis": "Z",
            "measure_basis": "Z",
            "d": d,
            "rounds": d,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


def build_tasks(gate: str, distances, p_values, rounds: int):
    """Dispatch to the appropriate circuit builder for a given gate."""
    if gate == "H":
        return _build_h_tasks(distances, p_values, rounds)
    if gate == "S":
        return _build_s_tasks(distances, p_values, rounds)
    if gate == "CNOT_trans":
        return _build_cnot_trans_tasks(distances, p_values, rounds)
    if gate == "CNOT_LS_ZZ_XX":
        return _build_cnot_ls_tasks(distances, p_values, rounds, "ZZ_XX")
    if gate == "CNOT_LS_XX_ZZ":
        return _build_cnot_ls_tasks(distances, p_values, rounds, "XX_ZZ")
    if gate == "memory":
        return _build_memory_tasks(distances, p_values)
    raise ValueError(f"Unknown gate: {gate!r}. Available: {ALL_GATES}")


# ── Decoder config ────────────────────────────────────────────────────────────

def _decoder_config(name: str) -> DecoderConfig:
    if name == "pymatching":
        return DecoderConfig(name="pymatching", backend="cpu")
    if name == "mwpf":
        return DecoderConfig(name="mwpf", backend="cpu",
                             params={"cluster_node_limit": 50})
    if name == "bposd":
        return DecoderConfig(name="bposd", backend="cpu")
    raise ValueError(f"Unknown decoder: {name!r}. Choose: bposd, pymatching, mwpf")


# ── Checkpointing ─────────────────────────────────────────────────────────────

_RESULT_COLS = frozenset({
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "seconds", "decoder",
})


def _ck_key(row: dict) -> tuple:
    """Stable checkpoint key from input-only fields (floats normalized to 6-digit sci)."""
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in _RESULT_COLS
    )


def _load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {_ck_key(r) for r in df.to_dict("records")}


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_tasks(task_list, decoder_cfg: DecoderConfig,
               max_shots: int, max_errors: int,
               num_workers: int, output_path: Path) -> None:
    """
    Run a list of (circuit, metadata) tuples with per-task checkpointing.
    Already-completed tasks (by checkpoint key) are skipped on resume.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = _load_done_keys(output_path)
    if done_keys:
        print(f"  Checkpoint: {len(done_keys)} task(s) already done, skipping.")

    pending = [(c, m) for c, m in task_list if _ck_key(m) not in done_keys]
    n_skip = len(task_list) - len(pending)
    if n_skip:
        print(f"  Skipping {n_skip} completed tasks, {len(pending)} remaining.")

    if not pending:
        return

    pipeline = SimulationPipeline(
        decoder_config=decoder_cfg,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=1_000,
        num_workers=num_workers,
        print_progress=True,
    )

    for i, (circuit, meta) in enumerate(pending):
        print(f"  [{i+1}/{len(pending)}] {meta.get('gate')} {meta.get('sub_experiment')} "
              f"d={meta.get('d')} p={meta.get('p'):.2e}", flush=True)

        t0 = time.perf_counter()
        stats = pipeline.run(circuit, meta)
        elapsed = time.perf_counter() - t0

        row = {
            **meta,
            "shots": stats.shots,
            "post_selected_shots": stats.post_selected_shots,
            "post_selection_rate": stats.post_selection_rate,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": elapsed,
            "decoder": stats.decoder,
        }
        # Persist immediately — a kill/OOM never loses this result
        pd.DataFrame([row]).to_csv(
            output_path, mode="a", header=not output_path.exists(), index=False,
        )
        print(f"  -> LER={stats.logical_error_rate:.2e} "
              f"({stats.errors} errors, {stats.shots:,} shots, {elapsed:.1f}s)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--gate", nargs="+", choices=ALL_GATES, default=None,
        metavar="GATE",
        help=f"Gate(s) to benchmark (default: all). Choices: {', '.join(ALL_GATES)}",
    )
    ap.add_argument(
        "--distances", nargs="+", type=int, default=[3, 5, 7],
        help="Code distances to sweep (default: 3 5 7)",
    )
    ap.add_argument(
        "--p-values", nargs="+", type=float,
        default=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        help="Physical error rate values (default: 5e-4 1e-3 2e-3 5e-3 1e-2)",
    )
    ap.add_argument(
        "--rounds", type=int, default=2,
        help="SE rounds for gate benchmarks (default: 2). Memory always uses rounds=d.",
    )
    ap.add_argument(
        "--decoder", choices=["bposd", "pymatching", "mwpf"], default=None,
        help="Decoder to use (default: bposd for gates, pymatching for memory)",
    )
    ap.add_argument("--max-shots",   type=int, default=1_000_000_000)
    ap.add_argument("--max-errors",  type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--quick", action="store_true",
        help="Quick mode: distances=[3,5], 2 p-values, max_shots=100k, max_errors=20",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output CSV path (default: results/logical_ops_results.csv)",
    )
    args = ap.parse_args()

    # Quick mode overrides
    if args.quick:
        distances  = [3, 5]
        p_values   = [1e-3, 5e-3]
        max_shots  = 100_000
        max_errors = 20
    else:
        distances  = args.distances
        p_values   = args.p_values
        max_shots  = args.max_shots
        max_errors = args.max_errors

    gates_to_run = args.gate if args.gate else ALL_GATES
    output_path  = Path(args.output) if args.output else \
        SCRIPT_DIR / "results" / "logical_ops_results.csv"

    print("=" * 60)
    print("Logical Operations Benchmark — Unrotated Surface Code")
    print(f"Mode       : {'quick' if args.quick else 'full'}")
    print(f"Gates      : {gates_to_run}")
    print(f"Distances  : {distances}")
    print(f"p values   : {p_values}")
    print(f"rounds     : {args.rounds} (gates); d (memory)")
    print(f"max_shots  : {max_shots:.0e}")
    print(f"max_errors : {max_errors}")
    print(f"num_workers: {args.num_workers}")
    print(f"Output     : {output_path}")
    print("=" * 60)

    for gate in gates_to_run:
        print(f"\n{'─' * 50}")
        print(f"Gate: {gate}")

        # Choose decoder: explicit flag > sensible default per gate
        if args.decoder is not None:
            decoder_name = args.decoder
        elif gate == "memory":
            decoder_name = "pymatching"
        else:
            decoder_name = "bposd"

        print(f"Decoder    : {decoder_name}")

        tasks = build_tasks(gate, distances, p_values, args.rounds)
        print(f"Tasks      : {len(tasks)}")

        _run_tasks(
            tasks,
            _decoder_config(decoder_name),
            max_shots, max_errors,
            args.num_workers,
            output_path,
        )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"Results → {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
