"""
General logical circuits benchmark runner for LightStim.

Experiments
-----------
    bell_tele   Bell-state teleportation (TG / ZZ-LS / XX-LS)
    s_gate_tele Logical S-gate teleportation (ZZ-LS / transversal CNOT)
    distill_ls  LS 7-to-1 |Y⟩ distillation (Steane)
    distill_tg  TG 7-to-1 |Y⟩ distillation (hypercube PQRM)

CSV output
----------
    bell_tele → results/bell_tele_results.csv
        experiment, protocol, state, routing_mult, d, rounds, p,
        shots, errors, logical_error_rate, decoder, seconds

    s_gate_tele → results/s_gate_tele_results.csv
        experiment, code, method, state_prep, d, rounds, p,
        shots, errors, logical_error_rate, decoder, seconds

    distill_ls / distill_tg → results/{distill_ls|distill_tg}_results.csv
        experiment, d, rounds, p_injected, noise_mode, p, p_in,
        decoder, decoder_time_limit, on_decode_failure,
        shots, post_selected_shots, post_selection_rate,
        errors, logical_error_rate, seconds

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
import subprocess
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

# Bell teleportation builders — direct protocol imports
from lightstim.protocols.bell_teleportation import (
    BellTeleportTG,
    BellTeleportZZLS,
    BellTeleportXXLS,
)
from lightstim.protocols.gate_teleport import SGateTeleportExperiment


def _build_bell_circuit(protocol: str, d: int, state: str) -> "stim.Circuit":
    if protocol == "tg":
        return BellTeleportTG(distance=d, teleport_state=state).build()
    if protocol == "ls_zz":
        return BellTeleportZZLS(distance=d, teleport_state=state).build()
    if protocol == "ls_xx":
        return BellTeleportXXLS(distance=d, teleport_state=state).build()
    raise ValueError(f"Unknown protocol: {protocol!r}")


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
    "decoder", "decoder_time_limit", "on_decode_failure",
    "shots", "errors", "logical_error_rate", "seconds",
]
_S_GATE_COLS = [
    "experiment", "code", "method", "state_prep", "d", "rounds", "p",
    "decoder", "decoder_time_limit", "on_decode_failure",
    "shots", "errors", "logical_error_rate", "seconds",
]
_DISTILL_COLS = [
    "experiment", "d", "rounds", "p_injected", "noise_mode", "p", "p_in",
    "decoder", "decoder_time_limit", "on_decode_failure",
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "seconds",
]
_BELL_RESULT_KEYS = frozenset({"shots", "errors", "logical_error_rate", "seconds"})
_S_GATE_RESULT_KEYS = _BELL_RESULT_KEYS
_DISTILL_RESULT_KEYS = frozenset({
    "p_in", "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "seconds",
})
_RESULT_METADATA_DEFAULTS = {
    "decoder_time_limit": 0.0,
    "on_decode_failure": "error",
}


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _ck_key(row: dict, result_keys: frozenset) -> tuple:
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in result_keys
    )


def _ensure_result_schema(path: Path, columns: list[str]) -> None:
    """Upgrade pre-MLE benchmark CSVs before checkpointing or appending."""
    if not path.exists():
        return
    import pandas as pd

    df = pd.read_csv(path)
    unknown = set(df.columns) - set(columns)
    if unknown:
        raise ValueError(
            f"Cannot migrate {path}: unknown result columns {sorted(unknown)}"
        )
    changed = False
    for column, default in _RESULT_METADATA_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
            changed = True
    if changed or list(df.columns) != columns:
        df.reindex(columns=columns).to_csv(path, index=False)


def _load_done(path: Path, result_keys: frozenset, columns: list[str]) -> set:
    if not path.exists():
        return set()
    _ensure_result_schema(path, columns)
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


def _checked_decoder_config(
    name: str,
    mle_time_limit: float = 0.0,
    on_decode_failure: str = "error",
) -> DecoderConfig:
    cfg = _decoder_config(name, mle_time_limit, on_decode_failure)
    if cfg.backend != "gpu":
        return cfg

    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Decoder {name!r} requires an NVIDIA GPU, but `nvidia-smi -L` "
            "did not complete on this machine."
        ) from exc

    if probe.returncode != 0 or "GPU" not in probe.stdout:
        detail = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(
            f"Decoder {name!r} requires an NVIDIA GPU, but none is visible. "
            f"`nvidia-smi -L` returned: {detail or '<empty>'}"
        )
    return cfg


# ── Bell teleportation ─────────────────────────────────────────────────────────

_BELL_DEFAULT_DECODER = {
    "tg":    "cpu_bposd",
    "ls_zz": "pymatching",
    "ls_xx": "pymatching",
}


def _run_bell_tele(args, output_path: Path) -> None:
    done = _load_done(output_path, _BELL_RESULT_KEYS, _BELL_COLS)
    protocols = args.protocols if args.protocols else list(_BELL_DEFAULT_DECODER)

    pipeline_cache: dict = {}

    for protocol, d, state, p in product(protocols, args.distances, args.states, args.p_values):
        decoder_name = args.decoder or _BELL_DEFAULT_DECODER[protocol]

        row_proto = {
            "experiment": "bell_tele",
            "protocol": protocol,
            "state": state,
            "routing_mult": 1,
            "d": d,
            "rounds": f"pre={d} mid=1 post=1" if protocol == "tg" else f"pre={d} ls={d}",
            "p": p,
            "decoder": decoder_name,
            "decoder_time_limit": (
                args.mle_time_limit if decoder_name == "mle-ilp" else 0.0
            ),
            "on_decode_failure": args.on_decode_failure,
        }
        if _ck_key(row_proto, _BELL_RESULT_KEYS) in done:
            print(f"  SKIP {protocol} {state} d={d} p={p:.0e}")
            continue

        print(f"  [{protocol}] state={state} d={d} p={p:.0e} decoder={decoder_name}")
        with contextlib.redirect_stdout(io.StringIO()):
            circuit = _build_bell_circuit(protocol, d, state)
        noisy = _inject_bell(circuit, p)

        if decoder_name not in pipeline_cache:
            pipeline_cache[decoder_name] = SimulationPipeline(
                decoder_config=_checked_decoder_config(
                    decoder_name,
                    args.mle_time_limit,
                    args.on_decode_failure,
                ),
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                print_progress=args.progress,
                progress_interval_sec=args.progress_interval,
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


# TODO: routing experiment (LER vs routing distance / chain length).
# Requires a multi-hop LS builder (chain of N patches connected by N-1 couplers).
# Not yet implemented — omitted from ALL_EXPERIMENTS until protocol is ready.


def _inject_bell(circuit, p: float):
    noise = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    injector = NoiseInjector.from_circuit_level(noise, list(range(circuit.num_qubits)))
    return injector.inject_noise(circuit)


# ── S-gate teleportation ──────────────────────────────────────────────────────

_S_GATE_DEFAULT_DECODER = {
    "ZZ": "pymatching",
    "cnot_trans": "cpu_bposd",
}


def _valid_s_gate_combos(args):
    codes = args.codes or ["unrotated_sc"]
    methods = args.s_gate_methods or ["ZZ", "cnot_trans"]
    preps = args.state_preps or ["logical_gate"]

    for code, method, state_prep in product(codes, methods, preps):
        yield code, method, state_prep


def _s_gate_rounds(d: int, method: str) -> tuple[int, int]:
    rounds_prep = d
    rounds_gate = d if method == "ZZ" else 1
    return rounds_prep, rounds_gate


def _run_s_gate_tele(args, output_path: Path) -> None:
    done = _load_done(output_path, _S_GATE_RESULT_KEYS, _S_GATE_COLS)
    pipeline_cache: dict = {}

    for code, method, state_prep in _valid_s_gate_combos(args):
        for d, p in product(args.distances, args.p_values):
            decoder_name = args.decoder or _S_GATE_DEFAULT_DECODER[method]
            rounds_prep, rounds_gate = _s_gate_rounds(d, method)
            row_proto = {
                "experiment": "s_gate_tele",
                "code": code,
                "method": method,
                "state_prep": state_prep,
                "d": d,
                "rounds": f"prep={rounds_prep} gate={rounds_gate}",
                "p": p,
                "decoder": decoder_name,
                "decoder_time_limit": (
                    args.mle_time_limit if decoder_name == "mle-ilp" else 0.0
                ),
                "on_decode_failure": args.on_decode_failure,
            }
            if _ck_key(row_proto, _S_GATE_RESULT_KEYS) in done:
                print(f"  SKIP {code} {method} {state_prep} d={d} p={p:.0e}")
                continue

            print(
                f"  [{method}] code={code} prep={state_prep} "
                f"d={d} p={p:.0e} decoder={decoder_name}"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                circuit = SGateTeleportExperiment(
                    distance=d,
                    code=code,
                    method=method,
                    state_prep=state_prep,
                    rounds_prep=rounds_prep,
                    rounds_gate=rounds_gate,
                ).build()
            noisy = _inject_bell(circuit, p)

            if decoder_name not in pipeline_cache:
                pipeline_cache[decoder_name] = SimulationPipeline(
                    decoder_config=_checked_decoder_config(
                        decoder_name,
                        args.mle_time_limit,
                        args.on_decode_failure,
                    ),
                    max_shots=args.max_shots,
                    max_errors=args.max_errors,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    print_progress=args.progress,
                    progress_interval_sec=args.progress_interval,
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
            _append_row(output_path, row, _S_GATE_COLS)
            print(
                f"    LER={stats.logical_error_rate:.2e}  "
                f"({stats.errors}/{stats.shots:,})  {elapsed:.1f}s"
            )


# ── Distillation ──────────────────────────────────────────────────────────────

def _run_distillation(args, which: str, output_path: Path) -> None:
    done = _load_done(output_path, _DISTILL_RESULT_KEYS, _DISTILL_COLS)

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
        build_kwargs = {"rounds_gate": 1}
        obs_target   = ["W0"]

    noise_modes = args.noise_mode or ["injection"]
    p_injected_list = args.p_injected or [1e-3, 5e-3, 2e-2]
    p_list = args.p_values if args.p_values else [1e-3]
    decoder_name = args.decoder or (
        "cpu_bposd" if which == "tg" else "pymatching"
    )
    decoder_cfg = _checked_decoder_config(
        decoder_name,
        args.mle_time_limit,
        args.on_decode_failure,
    )

    if which == "ls" and decoder_cfg.backend != "cpu":
        raise ValueError("LS distillation currently expects a CPU decoder; use `pymatching`.")

    for d in args.distances:
        rounds_init = d
        print(f"\n  Building d={d}, rounds_init={rounds_init}")
        with contextlib.redirect_stdout(io.StringIO()):
            circuit, info, system = build_fn(d, rounds_init, **build_kwargs)

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
                    "rounds": rounds_init,
                    "p_injected": p_inj,
                    "noise_mode": mode,
                    "p": p,
                    "decoder": decoder_name,
                    "decoder_time_limit": (
                        args.mle_time_limit if decoder_name == "mle-ilp" else 0.0
                    ),
                    "on_decode_failure": args.on_decode_failure,
                }
                if _ck_key(row_proto, _DISTILL_RESULT_KEYS) in done:
                    print(f"  SKIP d={d} mode={mode} p={p:.0e} p_inj={p_inj:.0e}")
                    continue

                # Calibrate p_in (injection and both modes only)
                if mode in ("injection", "both"):
                    p_bg = p if mode == "both" else 0.0
                    calibration_max_shots = max(1, args.max_shots // 10)
                    calibration_kwargs = {
                        "p_injected": p_inj,
                        "p_background": p_bg,
                        "max_shots": calibration_max_shots,
                        "max_errors": 50,
                        "batch_size": min(args.batch_size, calibration_max_shots),
                    }
                    if which == "tg":
                        calibration_kwargs.update({
                            "decoder_name": decoder_cfg.name,
                            "backend": decoder_cfg.backend,
                            "decoder_params": decoder_cfg.params,
                            "on_decode_failure": decoder_cfg.on_decode_failure,
                        })
                    p_in = p_in_fn(d, rounds_init, **calibration_kwargs)
                else:
                    p_in = float("nan")

                label = f"p={p:.0e} p_inj={p_inj:.0e}" if mode != "full" else f"p={p:.0e}"
                print(f"\n  [distill_{which}] d={d} mode={mode} {label}  p_in={p_in:.3e}")

                if which == "ls":
                    stats = _run_ls_sim(
                        circuit, magic_qubits, p, p_inj, mode,
                        ps_obs, target_obs, decoder_cfg.name,
                        args.max_shots, args.max_errors,
                        batch_size=min(args.batch_size, args.max_shots),
                        num_workers=args.num_workers,
                        data_indices=magic_data,
                        decoder_params=decoder_cfg.params,
                        on_decode_failure=decoder_cfg.on_decode_failure,
                    )
                else:
                    stats = _run_tg_sim(
                        circuit, magic_qubits, p, p_inj, mode,
                        T, ps_obs, target_obs, decoder_cfg.name,
                        args.max_shots, args.max_errors,
                        num_workers=args.num_workers,
                        backend=decoder_cfg.backend,
                        batch_size=min(args.batch_size, args.max_shots),
                        decoder_params=decoder_cfg.params,
                        on_decode_failure=decoder_cfg.on_decode_failure,
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

def _decoder_config(
    name: str,
    mle_time_limit: float = 0.0,
    on_decode_failure: str = "error",
) -> DecoderConfig:
    if name == "pymatching":
        return DecoderConfig(
            "pymatching", backend="cpu", on_decode_failure=on_decode_failure
        )
    if name in ("bposd", "cpu_bposd"):
        return DecoderConfig(
            "bposd", backend="cpu", params={
                "max_iterations": 1000, "osd_order": 10,
                "bp_method": "min_sum", "ms_scaling_factor": 0,
                "osd_method": "osd_cs",
            },
            on_decode_failure=on_decode_failure,
        )
    if name in ("gpu_bposd", "nv-qldpc-decoder"):
        return DecoderConfig(
            "nv-qldpc-decoder",
            backend="gpu",
            params={
                "max_iterations": 1000,
                "osd_order": 10,
                "bp_method": "min_sum",
                "ms_scaling_factor": 0,
                "osd_method": "osd_cs",
                "use_osd": True,
            },
            on_decode_failure=on_decode_failure,
        )
    if name == "mwpf":
        return DecoderConfig(
            "mwpf", backend="cpu", params={"cluster_node_limit": 50},
            on_decode_failure=on_decode_failure,
        )
    if name == "mle-ilp":
        return DecoderConfig(
            "mle-ilp", backend="cpu", params={"time_limit": mle_time_limit},
            on_decode_failure=on_decode_failure,
        )
    raise ValueError(
        f"Unknown decoder: {name!r}. "
        "Choose: pymatching, mwpf, cpu_bposd, gpu_bposd, mle-ilp"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = ["bell_tele", "s_gate_tele", "distill_ls", "distill_tg"]


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
        help="Physical error rates (default: 5e-4 1e-3 2e-3 5e-3)",
    )
    ap.add_argument(
        "--states", nargs="+", choices=["X", "Z"], default=["X", "Z"],
        help="States for bell_tele (default: X Z)",
    )
    ap.add_argument(
        "--protocols", nargs="+",
        choices=["tg", "ls_zz", "ls_xx"],
        default=None,
        help="Protocols for bell_tele (default: tg ls_zz ls_xx)",
    )
    ap.add_argument(
        "--codes", nargs="+",
        choices=["unrotated_sc"],
        default=None,
        help="Codes for s_gate_tele (default: unrotated_sc)",
    )
    ap.add_argument(
        "--s-gate-methods", nargs="+",
        choices=["ZZ", "cnot_trans"],
        default=None,
        help="Methods for s_gate_tele (default: ZZ cnot_trans)",
    )
    ap.add_argument(
        "--state-preps", nargs="+",
        choices=["logical_gate", "inject"],
        default=None,
        help="Resource-state preparation modes for s_gate_tele (default: logical_gate)",
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
        choices=["pymatching", "mwpf", "bposd", "cpu_bposd", "gpu_bposd", "nv-qldpc-decoder", "mle-ilp"],
        help="Override decoder for all experiments (default: per-experiment default)",
    )
    ap.add_argument(
        "--mle-time-limit", type=float, default=0.0,
        help="Soft seconds/shot limit for mle-ilp; 0 is exact/unlimited (default: 0)",
    )
    ap.add_argument(
        "--on-decode-failure", choices=["error", "discard", "ignore"],
        default="error",
        help="Policy for decoder timeouts/failures (default: error)",
    )
    ap.add_argument("--max-shots",   type=int, default=1_000_000_000)
    ap.add_argument("--max-errors",  type=int, default=100)
    ap.add_argument("--batch-size",  type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--progress", action="store_true",
        help="Print periodic SimulationPipeline progress for long runs",
    )
    ap.add_argument(
        "--progress-interval", type=float, default=30.0,
        help="Seconds between progress updates when --progress is set",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="Quick mode: d=[3], 2 p-values, max_shots=100k, max_errors=20",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory for results CSVs (default: benchmarks/logical_circuits/results/)",
    )
    args = ap.parse_args()

    if not np.isfinite(args.mle_time_limit) or args.mle_time_limit < 0:
        ap.error("--mle-time-limit must be finite and non-negative")
    if args.mle_time_limit > 0 and args.decoder != "mle-ilp":
        ap.error("--mle-time-limit is only valid with --decoder mle-ilp")

    if args.quick:
        args.distances  = [3]
        args.p_values   = [1e-3, 5e-3]
        args.max_shots  = 100_000
        args.max_errors = 20
        args.batch_size = min(args.batch_size, 10_000)
        if args.p_injected is None:
            args.p_injected = [5e-3, 2e-2]
        if args.noise_mode is None:
            args.noise_mode = ["injection"]

    experiments = ALL_EXPERIMENTS if "all" in args.experiment else args.experiment

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bell_csv    = out_dir / "bell_tele_results.csv"
    s_gate_csv  = out_dir / "s_gate_tele_results.csv"
    distill_ls_csv = out_dir / "distill_ls_results.csv"
    distill_tg_csv = out_dir / "distill_tg_results.csv"

    print("=" * 60)
    print("Logical Circuits Benchmark")
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
        elif exp == "s_gate_tele":
            _run_s_gate_tele(args, s_gate_csv)
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
