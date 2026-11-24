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
import json
import contextlib
import io
import sys
import time
from itertools import product
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root → lightstim importable

from lightstim.protocols.state_injection import StateInjectionExperiment
from benchmarks._config import (
    add_run_options, benchmark_decoder, benchmark_noise, checkpoint_key, checkpoint_rows,
    decoder_metadata, validate_run_options,
)
from lightstim.config.results import append_result_row
from lightstim.config.runner import run_config_cli, sample_full_postselection
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)

# ── Task builder ───────────────────────────────────────────────────────────────

def build_tasks(distances, p_values, rounds, states, protocols, modes, *,
                noise_model="circuit_level", custom_params=None):
    tasks = []
    for state, protocol, mode, d, p in product(states, protocols, modes, distances, p_values):
        noise = benchmark_noise(p, noise_model, custom_params)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = StateInjectionExperiment(
                code_patch_class=UnrotatedSurfaceCode,
                extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                op_set_class=UnrotatedSurfaceCodeLogicalOpSet,
                distance=d,
                inject_state=state,
                post_select_mode="full_qec" if mode == "all_detectors" else mode,
                rounds=rounds,
                protocol=protocol,
                noise_params=noise,
                noise_model=noise_model,
            )
            circuit = exp.build()
        meta = {
            "sweep_coordinates": {"basis": state, "injection_protocol": protocol,
                                  "post_select_mode": mode, "distance": d, "p": p},
            "gate": "state_injection",
            "sub_experiment": f"{state}_{protocol}_{mode}",
            "inject_state": state,
            "injection_protocol": protocol,
            "post_select_mode": mode,
            "basis": state,
            "mode": {"all_detectors": "full_postselection", "full_qec": "decode"}.get(mode, "hybrid"),
            "postselection_region": {"full_postselection": "injection", "hybrid": "logical_strip"}.get(mode),
            "noise_model": noise_model,
            "noise_custom_params": json.dumps(custom_params or {}, sort_keys=True),
            "d": d,
            "rounds": rounds,
            "p": p, "code": "unrotated_sc",
            **dict.fromkeys(("p_1q", "p_2q", "p_meas", "p_reset", "p_idle"), p),
        }
        tasks.append((circuit, meta))
    return tasks


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _ck_key(row: dict) -> tuple:
    fields = ("gate", "sub_experiment", "inject_state", "injection_protocol", "post_select_mode",
              "d", "rounds", "p", "seed", "noise_model", "noise_custom_params",
              "decoder_name", "backend", "decoder_params", "on_decode_failure",
              "decoder_time_limit", "config_sha256")
    defaults = {"noise_model": "circuit_level", "noise_custom_params": "{}", "seed": 0,
                "backend": "cpu", "decoder_params": "{}", "on_decode_failure": "error",
                "decoder_time_limit": 0.0, "decoder_name": row.get("decoder")}
    return checkpoint_key(row, fields, defaults)


def _load_done_keys(path: Path) -> set:
    if not path.exists() or not path.stat().st_size:
        return set()
    return {_ck_key(r) for r in checkpoint_rows(path)}


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_tasks(task_list, decoder_cfg: DecoderConfig,
               max_shots: int, max_errors: int,
               num_workers: int, output_path: Path,
               *, batch_size: int = 1_000, seed: int = 0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_list = [
        (c, {**meta, "seed": seed, **(decoder_metadata(decoder_cfg) if meta["mode"] != "full_postselection"
                                     else {"decoder_name": "none"})})
        for c, meta in task_list
    ]
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
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        print_progress=True,
    ) if any(m["mode"] != "full_postselection" for _, m in pending) else None

    for i, (circuit, meta) in enumerate(pending):
        print(f"  [{i+1}/{len(pending)}] {meta['inject_state']} "
              f"{meta['injection_protocol']} {meta['post_select_mode']} "
              f"d={meta['d']} p={meta['p']:.2e}", flush=True)

        t0 = time.perf_counter()
        if meta["mode"] == "full_postselection":
            result = sample_full_postselection(circuit, max_shots=max_shots, max_errors=max_errors,
                                               batch_size=batch_size, seed=seed, num_workers=num_workers)
            result.update(post_selected_shots=result["accepted"], post_selection_rate=result["acceptance"])
        else:
            stats = pipeline.run(circuit, meta)
            result = {"shots": stats.shots, "post_selected_shots": stats.post_selected_shots,
                      "post_selection_rate": stats.post_selection_rate,
                      "errors": stats.errors, "logical_error_rate": stats.logical_error_rate}
        elapsed = time.perf_counter() - t0

        row = {
            **meta,
            **result, "seconds": elapsed, "decoder": meta["decoder_name"],
        }
        append_result_row(output_path, row)
        print(f"  -> LER={row['logical_error_rate']:.2e} "
              f"({row['errors']} errors, {row['shots']:,} shots, {elapsed:.1f}s)", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--inject-states", "--basis", nargs="+", default=["Z", "X", "Y"],
        choices=["Z", "X", "Y"],
        help="States to inject (default: Z X Y)",
    )
    ap.add_argument(
        "--inject-protocols", nargs="+", default=["corner", "middle"],
        choices=["corner", "middle"],
        help="Injection protocols (default: corner middle)",
    )
    modes_group = ap.add_mutually_exclusive_group()
    modes_group.add_argument("--mode", choices=["decode", "full_postselection", "hybrid"])
    ap.add_argument("--postselection-region", choices=["injection", "logical_strip"])
    modes_group.add_argument(
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
        "--decoder", choices=["pymatching", "bposd", "cpu_bposd", "gpu_bposd", "nv-qldpc-decoder",
                              "mwpf", "mle-ilp", "ionq-beam-search"],
        default="pymatching",
        help="Decoder (default: pymatching)",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="Quick mode: Z only, corner, full_postselection, d=[3,5], 2 p-values",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output CSV path (default: benchmarks/state_injection/results/state_injection_results.csv)",
    )
    add_run_options(ap)
    if run_config_cli(ap):
        return
    args = ap.parse_args()
    validate_run_options(ap, args)

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

    if args.postselection_region and args.mode != "hybrid":
        ap.error("--postselection-region requires --mode hybrid")
    if args.mode:
        region = args.postselection_region or "injection"
        modes = [{"decode": "full_qec", "full_postselection": "all_detectors",
                  "hybrid": "full_postselection" if region == "injection" else "hybrid"}[args.mode]]

    output_path = Path(args.output) if args.output else \
        SCRIPT_DIR / "results" / "state_injection_results.csv"

    decoder_cfg = benchmark_decoder(args.decoder, args.backend, use_presets=False,
                                    mle_time_limit=args.mle_time_limit, on_decode_failure=args.on_decode_failure)

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

    tasks = build_tasks(distances, p_values, args.rounds, states, protocols, modes,
                        noise_model=args.noise_model, custom_params=args.noise_custom_params)
    print(f"Total tasks: {len(tasks)}")

    _run_tasks(tasks, decoder_cfg, max_shots, max_errors, args.num_workers, output_path,
               batch_size=args.batch_size, seed=args.seed)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"Results → {output_path}")
    print("\nGenerate plots:")
    print("  PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
