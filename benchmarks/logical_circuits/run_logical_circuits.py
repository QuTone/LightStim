"""
General logical circuits benchmark runner for LightStim.

Experiments
-----------
    bell_tele   Bell-state teleportation (TG / ZZ-LS / XX-LS)
    routing     LER vs routing distance (ZZ-LS and XX-LS, d fixed)
    distill_ls  LS 7-to-1 |Y⟩ distillation (Steane)
    distill_tg  TG 7-to-1 |Y⟩ distillation (hypercube PQRM)

CSV output
----------
    bell_tele / routing → results/bell_tele_results.csv
        gate, protocol, state, routing_mult, d, rounds, p,
        shots, errors, logical_error_rate, decoder, seconds

    distill_ls / distill_tg → results/{distill_ls|distill_tg}_results.csv
        experiment, d, rounds, p_injected, noise_mode, p, p_in,
        shots, post_selected_shots, post_selection_rate,
        errors, logical_error_rate, decoder, seconds

All outputs use per-task checkpointing (append-on-complete).

Usage
-----
    # Quick smoke test:
    PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py --quick

    # Bell teleportation only:
    PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \\
        --experiment bell_tele --distances 3 5 7 --p-values 5e-4 1e-3 2e-3 5e-3

    # LS distillation, injection-only noise:
    PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \\
        --experiment distill_ls --noise-mode injection --p-injected 1e-3 5e-3 2e-2

    # All experiments (long):
    PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py --experiment all
"""
import argparse
import contextlib
import csv
import io
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root

from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)

# Bell tele / routing builders — directory name has a hyphen so use importlib
import importlib.util as _ilu


def _import_from(name: str, rel: str):
    path = SCRIPT_DIR / "bell-teleportation" / rel
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod_tg         = _import_from("run_tg",         "run_tg.py")
_mod_ls_zz      = _import_from("run_ls_zz",      "run_ls_zz.py")
_mod_ls_xx      = _import_from("run_ls_xx",      "run_ls_xx.py")
_mod_ls_zz_dist = _import_from("run_ls_zz_dist", "run_ls_zz_dist.py")
_mod_ls_xx_dist = _import_from("run_ls_xx_dist", "run_ls_xx_dist.py")

_build_tg         = _mod_tg.build_circuit
_build_ls_zz      = _mod_ls_zz.build_circuit
_build_ls_xx      = _mod_ls_xx.build_circuit
_build_ls_zz_dist = _mod_ls_zz_dist.build_circuit
_build_ls_xx_dist = _mod_ls_xx_dist.build_circuit

# Distillation builders
from lightstim.protocols.ls_distillation import (
    build_distillation_circuit as _build_ls_distill,
    inject_noise as _inject_ls,
    estimate_p_in as _estimate_p_in_ls,
    run_simulation as _run_ls_sim,
    _LS_MAGIC_NAMES,
)
from lightstim.protocols.tg_distillation import (
    build_distillation_circuit as _build_tg_distill,
    inject_noise as _inject_tg,
    estimate_p_in as _estimate_p_in_tg,
    run_simulation as _run_tg_sim,
    _TG_MAGIC_NAMES,
)


# ── Result columns ────────────────────────────────────────────────────────────

_BELL_COLS = [
    "experiment", "protocol", "state", "routing_mult", "d", "rounds", "p",
    "shots", "errors", "logical_error_rate", "decoder", "seconds",
]
_DISTILL_COLS = [
    "experiment", "d", "rounds", "p_injected", "noise_mode", "p", "p_in",
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "decoder", "seconds",
]
_BELL_RESULT_KEYS = frozenset({"shots", "errors", "logical_error_rate", "decoder", "seconds"})
_DISTILL_RESULT_KEYS = frozenset({
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "decoder", "seconds",
})


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _ck_key(row: dict, result_keys: frozenset) -> tuple:
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in result_keys
    )


def _load_done(path: Path, result_keys: frozenset) -> set:
    if not path.exists():
        return set()
    import pandas as pd
    df = pd.read_csv(path)
    return {_ck_key(r, result_keys) for r in df.to_dict("records")}


def _append_row(path: Path, row: dict, cols: list) -> None:
    header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if header:
            w.writeheader()
        w.writerow(row)


# ── Bell teleportation ─────────────────────────────────────────────────────────

_BELL_BUILDERS = {
    "tg":     (_build_tg,         "bposd"),
    "ls_zz":  (_build_ls_zz,      "pymatching"),
    "ls_xx":  (_build_ls_xx,      "pymatching"),
}

_ROUTING_BUILDERS = {
    "ls_zz": _build_ls_zz_dist,
    "ls_xx": _build_ls_xx_dist,
}


def _run_bell_tele(args, output_path: Path) -> None:
    done = _load_done(output_path, _BELL_RESULT_KEYS)
    protocols = args.protocols if args.protocols else list(_BELL_BUILDERS)

    pipeline_cache: dict = {}

    for protocol, d, state, p in product(protocols, args.distances, args.states, args.p_values):
        builder_fn, default_decoder = _BELL_BUILDERS[protocol]
        decoder_name = args.decoder or default_decoder

        row_proto = {
            "experiment": "bell_tele",
            "protocol": protocol,
            "state": state,
            "routing_mult": 1,
            "d": d,
            "rounds": f"pre={d} mid=1 post=1" if protocol == "tg" else f"pre={d} ls={d}",
            "p": p,
            "decoder": decoder_name,
        }
        if _ck_key(row_proto, _BELL_RESULT_KEYS) in done:
            print(f"  SKIP {protocol} {state} d={d} p={p:.0e}")
            continue

        print(f"  [{protocol}] state={state} d={d} p={p:.0e} decoder={decoder_name}")
        with contextlib.redirect_stdout(io.StringIO()):
            circuit, info = builder_fn(d, state)
        circuit_key = (protocol, d, state)
        noisy = _inject_bell(circuit, p)

        if decoder_name not in pipeline_cache:
            pipeline_cache[decoder_name] = SimulationPipeline(
                decoder_config=_decoder_config(decoder_name),
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                batch_size=10_000,
                num_workers=args.num_workers,
                print_progress=False,
            )
        pipeline = pipeline_cache[decoder_name]

        t0 = time.perf_counter()
        stats = pipeline.run(noisy)
        elapsed = time.perf_counter() - t0

        row = {
            **row_proto,
            "shots": stats.shots,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": round(elapsed, 2),
        }
        _append_row(output_path, row, _BELL_COLS)
        print(f"    LER={stats.logical_error_rate:.2e}  ({stats.errors}/{stats.shots:,})  {elapsed:.1f}s")


def _run_routing(args, output_path: Path) -> None:
    done = _load_done(output_path, _BELL_RESULT_KEYS)
    protocols = [p for p in (args.protocols or ["ls_zz", "ls_xx"]) if p in _ROUTING_BUILDERS]
    mults = args.mults or [2, 4, 8]

    pipeline_cache: dict = {}

    for protocol, d, state, mult, p in product(protocols, args.distances, args.states, mults, args.p_values):
        builder_fn = _ROUTING_BUILDERS[protocol]
        decoder_name = args.decoder or "pymatching"

        row_proto = {
            "experiment": "routing",
            "protocol": protocol,
            "state": state,
            "routing_mult": mult,
            "d": d,
            "rounds": f"pre={d} ls={d}",
            "p": p,
            "decoder": decoder_name,
        }
        if _ck_key(row_proto, _BELL_RESULT_KEYS) in done:
            print(f"  SKIP routing {protocol} state={state} d={d} mult={mult} p={p:.0e}")
            continue

        print(f"  [routing/{protocol}] state={state} d={d} mult={mult} p={p:.0e}")
        with contextlib.redirect_stdout(io.StringIO()):
            circuit, info = builder_fn(d, state, mult)
        noisy = _inject_bell(circuit, p)

        if decoder_name not in pipeline_cache:
            pipeline_cache[decoder_name] = SimulationPipeline(
                decoder_config=_decoder_config(decoder_name),
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                batch_size=10_000,
                num_workers=args.num_workers,
                print_progress=False,
            )
        pipeline = pipeline_cache[decoder_name]

        t0 = time.perf_counter()
        stats = pipeline.run(noisy)
        elapsed = time.perf_counter() - t0

        row = {
            **row_proto,
            "shots": stats.shots,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": round(elapsed, 2),
        }
        _append_row(output_path, row, _BELL_COLS)
        print(f"    LER={stats.logical_error_rate:.2e}  ({stats.errors}/{stats.shots:,})  {elapsed:.1f}s")


def _inject_bell(circuit, p: float):
    noise = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    injector = NoiseInjector.from_circuit_level(noise, list(range(circuit.num_qubits)))
    return injector.inject_noise(circuit)


# ── Distillation ──────────────────────────────────────────────────────────────

def _run_distillation(args, which: str, output_path: Path) -> None:
    done = _load_done(output_path, _DISTILL_RESULT_KEYS)

    if which == "ls":
        build_fn    = _build_ls_distill
        p_in_fn     = _estimate_p_in_ls
        magic_names = _LS_MAGIC_NAMES
        build_kwargs = {}
        obs_target   = ["W4"]
    else:
        build_fn    = _build_tg_distill
        p_in_fn     = _estimate_p_in_tg
        magic_names = _TG_MAGIC_NAMES
        build_kwargs = {"r": 1}
        obs_target   = ["W0"]

    noise_modes = args.noise_mode or ["injection"]
    p_injected_list = args.p_injected or [1e-3, 5e-3, 2e-2]
    p_list = args.p_values if args.p_values else [1e-3]
    decoder_name = args.decoder or "pymatching"

    for d in args.distances:
        rounds = d
        print(f"\n  Building d={d}, rounds={rounds}")
        with contextlib.redirect_stdout(io.StringIO()):
            circuit, info, system = build_fn(d, rounds, **build_kwargs)

        matrix, patch_names = build_obs_patch_matrix(circuit, system)
        T, target_obs, ps_obs = identify_distillation_observables(
            matrix, patch_names, obs_target
        )
        magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                        if owner in magic_names}
        magic_data = magic_qubits & system.data_indices

        print(f"  {info['num_qubits']} qubits, {info['num_detectors']} det, "
              f"{info['num_observables']} obs  target={target_obs} ps={ps_obs}")

        for mode in noise_modes:
            if mode == "injection":
                sweep = [(0.0, p_inj) for p_inj in p_injected_list]
            elif mode == "full":
                sweep = [(p, 0.0) for p in p_list]
            else:  # both
                sweep = [(p, p_inj) for p in p_list for p_inj in p_injected_list]

            for p, p_inj in sweep:
                row_proto = {
                    "experiment": f"distill_{which}",
                    "d": d,
                    "rounds": rounds,
                    "p_injected": p_inj,
                    "noise_mode": mode,
                    "p": p,
                    "decoder": decoder_name,
                }
                if _ck_key(row_proto, _DISTILL_RESULT_KEYS) in done:
                    print(f"  SKIP d={d} mode={mode} p={p:.0e} p_inj={p_inj:.0e}")
                    continue

                # Calibrate p_in (injection and both modes only)
                if mode in ("injection", "both"):
                    p_bg = p if mode == "both" else 0.0
                    p_in = p_in_fn(d, rounds, p_injected=p_inj,
                                   p_background=p_bg,
                                   max_shots=args.max_shots // 10,
                                   max_errors=50,
                                   batch_size=5_000)
                else:
                    p_in = float("nan")

                label = f"p={p:.0e} p_inj={p_inj:.0e}" if mode != "full" else f"p={p:.0e}"
                print(f"\n  [distill_{which}] d={d} mode={mode} {label}  p_in={p_in:.3e}")

                if which == "ls":
                    stats = _run_ls_sim(
                        circuit, magic_qubits, p, p_inj, mode,
                        ps_obs, target_obs, decoder_name,
                        args.max_shots, args.max_errors,
                        batch_size=50_000, num_workers=args.num_workers,
                        data_indices=magic_data,
                    )
                else:
                    stats = _run_tg_sim(
                        circuit, magic_qubits, p, p_inj, mode,
                        T, ps_obs, target_obs, decoder_name,
                        args.max_shots, args.max_errors,
                        num_workers=args.num_workers,
                        backend="cpu", batch_size=50_000,
                    )
                row = {
                    **row_proto,
                    "p_in": p_in,
                    "shots": stats.shots,
                    "post_selected_shots": stats.post_selected_shots,
                    "post_selection_rate": stats.post_selection_rate,
                    "errors": stats.errors,
                    "logical_error_rate": stats.logical_error_rate,
                    "seconds": round(stats.seconds, 2),
                }
                _append_row(output_path, row, _DISTILL_COLS)
                print(f"  p_out={stats.logical_error_rate:.2e}  "
                      f"PS_rate={stats.post_selection_rate:.2f}  "
                      f"({stats.errors}/{stats.post_selected_shots:,})  {stats.seconds:.1f}s")


# ── Decoder config ─────────────────────────────────────────────────────────────

def _decoder_config(name: str) -> DecoderConfig:
    if name == "pymatching":
        return DecoderConfig("pymatching", backend="cpu")
    if name == "bposd":
        return DecoderConfig("bposd", backend="cpu", params={
            "max_iterations": 1000, "osd_order": 10,
            "bp_method": "min_sum", "ms_scaling_factor": 0,
            "osd_method": "osd_cs",
        })
    if name == "mwpf":
        return DecoderConfig("mwpf", backend="cpu", params={"cluster_node_limit": 50})
    raise ValueError(f"Unknown decoder: {name!r}")


# ── CLI ────────────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = ["bell_tele", "routing", "distill_ls", "distill_tg"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--experiment", nargs="+",
        choices=ALL_EXPERIMENTS + ["all"],
        default=["all"],
        help="Experiment(s) to run (default: all)",
    )
    ap.add_argument(
        "--distances", nargs="+", type=int, default=[3, 5, 7],
        help="Code distances (default: 3 5 7)",
    )
    ap.add_argument(
        "--p-values", nargs="+", type=float,
        default=[5e-4, 1e-3, 2e-3, 5e-3],
        help="Physical error rates for bell_tele, routing, distill full/both (default: 5e-4 1e-3 2e-3 5e-3)",
    )
    ap.add_argument(
        "--states", nargs="+", choices=["X", "Z"], default=["X", "Z"],
        help="States for bell_tele / routing (default: X Z)",
    )
    ap.add_argument(
        "--protocols", nargs="+",
        choices=["tg", "ls_zz", "ls_xx"],
        default=None,
        help="Protocols for bell_tele (default: all three)",
    )
    ap.add_argument(
        "--mults", nargs="+", type=int, default=None,
        help="Routing multipliers for routing experiment (default: 2 4 8)",
    )
    ap.add_argument(
        "--p-injected", nargs="+", type=float, default=None,
        help="Injection noise rates for distillation (default: 1e-3 5e-3 2e-2)",
    )
    ap.add_argument(
        "--noise-mode", nargs="+",
        choices=["injection", "full", "both"],
        default=None,
        help="Noise mode for distillation (default: injection)",
    )
    ap.add_argument(
        "--decoder", default=None,
        choices=["pymatching", "bposd", "mwpf"],
        help="Override decoder for all experiments (default: per-experiment default)",
    )
    ap.add_argument("--max-shots",   type=int, default=1_000_000_000)
    ap.add_argument("--max-errors",  type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--quick", action="store_true",
        help="Quick mode: d=[3], 2 p-values, max_shots=100k, max_errors=20",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory for results CSVs (default: benchmarks/logical_circuits/results/)",
    )
    args = ap.parse_args()

    if args.quick:
        args.distances  = [3]
        args.p_values   = [1e-3, 5e-3]
        args.max_shots  = 100_000
        args.max_errors = 20
        if args.p_injected is None:
            args.p_injected = [5e-3, 2e-2]
        if args.noise_mode is None:
            args.noise_mode = ["injection"]

    experiments = ALL_EXPERIMENTS if "all" in args.experiment else args.experiment

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bell_csv    = out_dir / "bell_tele_results.csv"
    distill_ls_csv = out_dir / "distill_ls_results.csv"
    distill_tg_csv = out_dir / "distill_tg_results.csv"

    print("=" * 60)
    print("Logical Circuits Benchmark — Unrotated Surface Code")
    print(f"Experiments : {experiments}")
    print(f"Distances   : {args.distances}")
    print(f"p values    : {args.p_values}")
    print(f"max_shots   : {args.max_shots:.0e}")
    print(f"max_errors  : {args.max_errors}")
    print(f"num_workers : {args.num_workers}")
    print(f"Output dir  : {out_dir}")
    print("=" * 60)

    for exp in experiments:
        print(f"\n{'─'*50}")
        print(f"Experiment: {exp}")
        if exp == "bell_tele":
            _run_bell_tele(args, bell_csv)
        elif exp == "routing":
            _run_routing(args, bell_csv)
        elif exp == "distill_ls":
            _run_distillation(args, "ls", distill_ls_csv)
        elif exp == "distill_tg":
            _run_distillation(args, "tg", distill_tg_csv)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    for f in sorted(out_dir.glob("*.csv")):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
