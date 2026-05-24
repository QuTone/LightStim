"""
State injection benchmark runner for LightStim.

Sweeps inject_state × injection_protocol × post_select_mode × distances × p values
for the unrotated surface code. Results saved to CSV with per-task checkpointing.

Supported inject_states : Z  X  Y
Supported protocols      : corner  middle
Supported modes          : full_postselection  full_qec  hybrid

CSV output schema
-----------------
    gate, sub_experiment, inject_state, injection_protocol, post_select_mode,
    d, rounds, p,
    shots, post_selected_shots, post_selection_rate,
    errors, logical_error_rate, seconds, decoder

Usage
-----
    # All states/protocols/modes, default sweep:
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/run_state_injection.py

    # Z state only, corner protocol:
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/run_state_injection.py \\
        --inject-states Z --inject-protocols corner

    # Full postselection mode only, custom p sweep:
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/run_state_injection.py \\
        --inject-modes full_postselection --p-values 1e-4 5e-4 1e-3 2e-3 5e-3 1e-2

    # Quick smoke test:
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/run_state_injection.py --quick
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

from lightstim.protocols.state_injection import StateInjectionExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)

# ── Task builder ───────────────────────────────────────────────────────────────

def build_tasks(distances, p_values, rounds, states, protocols, modes):
    tasks = []
    for state, protocol, mode, d, p in product(states, protocols, modes, distances, p_values):
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = StateInjectionExperiment(
                code_patch_class=UnrotatedSurfaceCode,
                extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                op_set_class=UnrotatedSurfaceCodeLogicalOpSet,
                distance=d,
                inject_state=state,
                post_select_mode=mode,
                rounds=rounds,
                protocol=protocol,
                noise_params=noise,
                noise_model="circuit_level",
            )
            circuit = exp.build()
        meta = {
            "gate": "state_injection",
            "sub_experiment": f"{state}_{protocol}_{mode}",
            "inject_state": state,
            "injection_protocol": protocol,
            "post_select_mode": mode,
            "d": d,
            "rounds": rounds,
            "p": p,
        }
        tasks.append((circuit, meta))
    return tasks


# ── Checkpointing ─────────────────────────────────────────────────────────────

_RESULT_COLS = frozenset({
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "seconds", "decoder",
})


def _ck_key(row: dict) -> tuple:
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in _RESULT_COLS
    )


def _load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {_ck_key(r) for r in pd.read_csv(path).to_dict("records")}


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_tasks(task_list, decoder_cfg: DecoderConfig,
               max_shots: int, max_errors: int,
               num_workers: int, output_path: Path) -> None:
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
        print(f"  [{i+1}/{len(pending)}] {meta['inject_state']} "
              f"{meta['injection_protocol']} {meta['post_select_mode']} "
              f"d={meta['d']} p={meta['p']:.2e}", flush=True)

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
        pd.DataFrame([row]).to_csv(
            output_path, mode="a", header=not output_path.exists(), index=False,
        )
        print(f"  -> LER={stats.logical_error_rate:.2e} "
              f"({stats.errors} errors, {stats.shots:,} shots, {elapsed:.1f}s)", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--inject-states", nargs="+", default=["Z", "X", "Y"],
        choices=["Z", "X", "Y"],
        help="States to inject (default: Z X Y)",
    )
    ap.add_argument(
        "--inject-protocols", nargs="+", default=["corner", "middle"],
        choices=["corner", "middle"],
        help="Injection protocols (default: corner middle)",
    )
    ap.add_argument(
        "--inject-modes", nargs="+",
        default=["full_postselection", "full_qec", "hybrid"],
        choices=["full_postselection", "full_qec", "hybrid"],
        help="Post-selection modes (default: all three)",
    )
    ap.add_argument(
        "--distances", nargs="+", type=int, default=[3, 5, 7],
        help="Code distances (default: 3 5 7)",
    )
    ap.add_argument(
        "--p-values", nargs="+", type=float,
        default=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        help="Physical error rate values",
    )
    ap.add_argument(
        "--rounds", type=int, default=2,
        help="SE rounds (default: 2)",
    )
    ap.add_argument(
        "--decoder", choices=["pymatching", "bposd", "mwpf"], default="pymatching",
        help="Decoder (default: pymatching)",
    )
    ap.add_argument("--max-shots",   type=int, default=1_000_000_000)
    ap.add_argument("--max-errors",  type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--quick", action="store_true",
        help="Quick mode: Z only, corner, full_postselection, d=[3,5], 2 p-values",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output CSV path (default: benchmarks/state_injection/results/state_injection_results.csv)",
    )
    args = ap.parse_args()

    if args.quick:
        states    = ["Z"]
        protocols = ["corner"]
        modes     = ["full_postselection"]
        distances = [3, 5]
        p_values  = [1e-3, 5e-3]
        max_shots  = 100_000
        max_errors = 20
    else:
        states    = args.inject_states
        protocols = args.inject_protocols
        modes     = args.inject_modes
        distances = args.distances
        p_values  = args.p_values
        max_shots  = args.max_shots
        max_errors = args.max_errors

    output_path = Path(args.output) if args.output else \
        SCRIPT_DIR / "results" / "state_injection_results.csv"

    decoder_cfg = DecoderConfig(name=args.decoder, backend="cpu")

    print("=" * 60)
    print("State Injection Benchmark — Unrotated Surface Code")
    print(f"Mode       : {'quick' if args.quick else 'full'}")
    print(f"States     : {states}")
    print(f"Protocols  : {protocols}")
    print(f"Modes      : {modes}")
    print(f"Distances  : {distances}")
    print(f"p values   : {p_values}")
    print(f"rounds     : {args.rounds}")
    print(f"Decoder    : {args.decoder}")
    print(f"max_shots  : {max_shots:.0e}")
    print(f"max_errors : {max_errors}")
    print(f"num_workers: {args.num_workers}")
    print(f"Output     : {output_path}")
    print("=" * 60)

    tasks = build_tasks(distances, p_values, args.rounds, states, protocols, modes)
    print(f"Total tasks: {len(tasks)}")

    _run_tasks(tasks, decoder_cfg, max_shots, max_errors, args.num_workers, output_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"Results → {output_path}")
    print("\nGenerate plots:")
    print("  PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
