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
import json
import io
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root

from benchmarks._config import (
    add_run_options, benchmark_decoder, benchmark_noise, checkpoint_key, checkpoint_rows,
    decoder_metadata, validate_run_options,
)
from lightstim.config.results import RESULT_COLUMNS, append_result_row, normalize_result_row, upgrade_csv
from lightstim.config.runner import run_config_cli
from lightstim.noise.injector import NoiseInjector
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# Bell teleportation builders — direct protocol imports
from lightstim.protocols.bell_teleportation import (
    BellTeleportTG,
    BellTeleportZZLS,
    BellTeleportXXLS,
)
from lightstim.protocols.gate_teleport import SGateTeleportExperiment


def _build_bell_circuit(protocol: str, d: int, state: str, **noise) -> "stim.Circuit":
    if protocol == "tg":
        return BellTeleportTG(distance=d, teleport_state=state, **noise).build()
    if protocol == "ls_zz":
        return BellTeleportZZLS(distance=d, teleport_state=state, **noise).build()
    if protocol == "ls_xx":
        return BellTeleportXXLS(distance=d, teleport_state=state, **noise).build()
    raise ValueError(f"Unknown protocol: {protocol!r}")


# Distillation builders
from lightstim.protocols.ls_distillation import (
    analyze_observables as _analyze_ls_distill,
    build_distillation_circuit as _build_ls_distill,
    estimate_p_in as _estimate_p_in_ls,
    run_simulation as _run_ls_sim,
    _LS_MAGIC_NAMES,
)
from lightstim.protocols.tg_distillation import (
    analyze_observables as _analyze_tg_distill,
    build_distillation_circuit as _build_tg_distill,
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
_LEGACY_INPUT_FIELDS = set(_BELL_COLS + _S_GATE_COLS + _DISTILL_COLS)
_RUN_FIELDS = {"backend", "noise_model", "noise_custom_params", "seed", "decoder_params", "config_sha256"}
_BELL_COLS, _S_GATE_COLS, _DISTILL_COLS = [
    list(dict.fromkeys([*RESULT_COLUMNS, *cols]))
    for cols in (_BELL_COLS, _S_GATE_COLS, _DISTILL_COLS)
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
    fields = (_LEGACY_INPUT_FIELDS - result_keys - {"p_in", "post_selected_shots", "post_selection_rate"}) | _RUN_FIELDS
    defaults = {"backend": "cpu", "noise_model": "circuit_level", "noise_custom_params": "{}", "seed": 0}
    # Older CSVs carry the CLI nickname. Reconstruct the historical preset
    # for their checkpoint, including the selected CPU/GPU implementation.
    row = dict(row)
    name = row.get("decoder")
    if name and (row.get("decoder_params") is None or isinstance(row.get("decoder_params"), float)):
        cfg = benchmark_decoder(name, mle_time_limit=row.get("decoder_time_limit", 0.0),
                                on_decode_failure=row.get("on_decode_failure", "error"))
        row.update(decoder=cfg.name, **decoder_metadata(cfg))
    return checkpoint_key(row, fields, defaults)


def _ensure_result_schema(path: Path, columns: list[str]) -> None:
    """Upgrade pre-MLE benchmark CSVs before checkpointing or appending."""
    upgrade_csv(path, columns, lambda row: _RESULT_METADATA_DEFAULTS)


def _load_done(path: Path, result_keys: frozenset, columns: list[str]) -> set:
    if not path.exists() or not path.stat().st_size:
        return set()
    _ensure_result_schema(path, columns)
    return {_ck_key(r, result_keys) for r in checkpoint_rows(path)}


def _append_row(path: Path, row: dict, cols: list) -> None:
    row = normalize_result_row(row)
    coordinates = {key: row[key] for key in (
        "experiment", "protocol", "code", "method", "state_prep", "basis", "distance",
        "p", "noise_mode", "p_injected",
    ) if row.get(key) is not None}
    row["sweep_coordinates"] = json.dumps(coordinates, sort_keys=True)
    append_result_row(path, {**dict.fromkeys(cols), **row})


def _checked_decoder_config(
    name: str,
    mle_time_limit: float = 0.0,
    on_decode_failure: str = "error",
    backend: str | None = None,
) -> DecoderConfig:
    cfg = _decoder_config(name, mle_time_limit, on_decode_failure, backend)
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
    "tg":    "bposd",
    "ls_zz": "pymatching",
    "ls_xx": "pymatching",
}


def _run_metadata(args, decoder, p, *, distill_mode=None):
    return {**decoder_metadata(decoder), "decoder": decoder.name,
            **dict.fromkeys(("p_1q", "p_2q", "p_meas", "p_reset"), p),
            "p_idle": 0.0 if distill_mode and args.noise_model == "circuit_level" else p,
            "mode": "hybrid" if distill_mode else "decode",
            "seed": args.seed, "noise_model": args.noise_model,
            "noise_custom_params": json.dumps(args.noise_custom_params, sort_keys=True)}


def _run_bell_tele(args, output_path: Path) -> None:
    done = _load_done(output_path, _BELL_RESULT_KEYS, _BELL_COLS)
    protocols = args.protocols if args.protocols else list(_BELL_DEFAULT_DECODER)

    pipeline_cache: dict = {}

    for protocol, d, state, p in product(protocols, args.distances, args.states, args.p_values):
        decoder_name = args.decoder or _BELL_DEFAULT_DECODER[protocol]

        decoder_cfg = _decoder_config(decoder_name, args.mle_time_limit, args.on_decode_failure, args.backend)
        row_proto = {
            "experiment": "bell_tele",
            "protocol": protocol,
            "state": state,
            "routing_mult": 1,
            "d": d,
            "rounds": f"pre={d} mid=1 post=1" if protocol == "tg" else f"pre={d} ls={d}",
            "p": p,
            **_run_metadata(args, decoder_cfg, p),
        }
        if _ck_key(row_proto, _BELL_RESULT_KEYS) in done:
            print(f"  SKIP {protocol} {state} d={d} p={p:.0e}")
            continue

        print(f"  [{protocol}] state={state} d={d} p={p:.0e} decoder={decoder_name}")
        with contextlib.redirect_stdout(io.StringIO()):
            noise = benchmark_noise(p, args.noise_model, args.noise_custom_params)
            opts = {} if args.noise_model == "circuit_level" else dict(noise_params=noise, noise_model=args.noise_model)
            circuit = _build_bell_circuit(protocol, d, state, **opts)
            noisy = _inject_bell(circuit, noise) if not opts else circuit

        if decoder_name not in pipeline_cache:
            pipeline_cache[decoder_name] = SimulationPipeline(
                decoder_config=_checked_decoder_config(
                    decoder_name,
                    args.mle_time_limit,
                    args.on_decode_failure,
                    args.backend,
                ),
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
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
            "accepted": stats.post_selected_shots,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": round(elapsed, 2),
        }
        _append_row(output_path, row, _BELL_COLS)
        print(f"    LER={stats.logical_error_rate:.2e}  ({stats.errors}/{stats.shots:,})  {elapsed:.1f}s")


# TODO: routing experiment (LER vs routing distance / chain length).
# Requires a multi-hop LS builder (chain of N patches connected by N-1 couplers).
# Not yet implemented — omitted from ALL_EXPERIMENTS until protocol is ready.


def _inject_bell(circuit, noise):
    injector = NoiseInjector.from_circuit_level(noise, list(range(circuit.num_qubits)))
    return injector.inject_noise(circuit)


# ── S-gate teleportation ──────────────────────────────────────────────────────

_S_GATE_DEFAULT_DECODER = {
    "ZZ": "pymatching",
    "cnot_trans": "bposd",
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
            decoder_cfg = _decoder_config(decoder_name, args.mle_time_limit, args.on_decode_failure, args.backend)
            row_proto = {
                "experiment": "s_gate_tele",
                "code": code,
                "method": method,
                "state_prep": state_prep,
                "d": d,
                "rounds": f"prep={rounds_prep} gate={rounds_gate}",
                "p": p,
                **_run_metadata(args, decoder_cfg, p),
            }
            if _ck_key(row_proto, _S_GATE_RESULT_KEYS) in done:
                print(f"  SKIP {code} {method} {state_prep} d={d} p={p:.0e}")
                continue

            print(
                f"  [{method}] code={code} prep={state_prep} "
                f"d={d} p={p:.0e} decoder={decoder_name}"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                noise = benchmark_noise(p, args.noise_model, args.noise_custom_params)
                opts = {} if args.noise_model == "circuit_level" else dict(noise_params=noise, noise_model=args.noise_model)
                circuit = SGateTeleportExperiment(
                    distance=d,
                    code=code,
                    method=method,
                    state_prep=state_prep,
                    rounds_prep=rounds_prep,
                    rounds_gate=rounds_gate, **opts,
                ).build()
            noisy = _inject_bell(circuit, noise) if not opts else circuit

            if decoder_name not in pipeline_cache:
                pipeline_cache[decoder_name] = SimulationPipeline(
                    decoder_config=_checked_decoder_config(
                        decoder_name,
                        args.mle_time_limit,
                        args.on_decode_failure,
                        args.backend,
                    ),
                    max_shots=args.max_shots,
                    max_errors=args.max_errors,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=args.seed,
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
                "accepted": stats.post_selected_shots,
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
        analyze_fn   = _analyze_ls_distill
    else:
        build_fn    = _build_tg_distill
        p_in_fn     = _estimate_p_in_tg
        magic_names = _TG_MAGIC_NAMES
        build_kwargs = {"rounds_gate": 1}
        analyze_fn   = _analyze_tg_distill

    noise_modes = args.noise_mode or ["injection"]
    p_injected_list = args.p_injected or [1e-3, 5e-3, 2e-2]
    p_list = args.p_values if args.p_values else [1e-3]
    decoder_name = args.decoder or (
        "bposd" if which == "tg" else "pymatching"
    )
    decoder_cfg = _checked_decoder_config(
        decoder_name,
        args.mle_time_limit,
        args.on_decode_failure,
        args.backend,
    )

    for d in args.distances:
        rounds_init = d
        print(f"\n  Building d={d}, rounds_init={rounds_init}")
        with contextlib.redirect_stdout(io.StringIO()):
            circuit, info, system = build_fn(d, rounds_init, **build_kwargs)

        T, target_obs, ps_obs, _, _ = analyze_fn(circuit, system)
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
                    **_run_metadata(args, decoder_cfg, p, distill_mode=mode),
                    "postselection_region": "corrected_observables" if which == "ls" else "observables",
                }
                if _ck_key(row_proto, _DISTILL_RESULT_KEYS) in done:
                    print(f"  SKIP d={d} mode={mode} p={p:.0e} p_inj={p_inj:.0e}")
                    continue

                background = None if mode == "injection" else benchmark_noise(
                    p, args.noise_model, args.noise_custom_params,
                    p_idle=0.0 if args.noise_model == "circuit_level" else None,
                )
                # Calibrate p_in (injection and both modes only)
                if mode in ("injection", "both"):
                    p_bg = p if mode == "both" else 0.0
                    calibration_max_shots = max(1, args.max_shots // 10)
                    calibration_kwargs = {
                        "p_injected": p_inj,
                        "p_background": p_bg,
                        "noise_model": args.noise_model, "background_noise": background, "seed": args.seed,
                        "max_shots": calibration_max_shots,
                        "max_errors": 50,
                        "batch_size": min(args.batch_size, calibration_max_shots),
                        "decoder_name": decoder_cfg.name, "backend": decoder_cfg.backend,
                        "decoder_params": decoder_cfg.params,
                        "on_decode_failure": decoder_cfg.on_decode_failure,
                    }
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
                        data_indices=magic_data, backend=decoder_cfg.backend,
                        decoder_params=decoder_cfg.params,
                        on_decode_failure=decoder_cfg.on_decode_failure,
                        noise_model=args.noise_model, background_noise=background, seed=args.seed,
                        background_data_indices=system.data_indices,
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
                        noise_model=args.noise_model, background_noise=background, seed=args.seed,
                        background_data_indices=system.data_indices,
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
    backend: str | None = None,
) -> DecoderConfig:
    return benchmark_decoder(name, backend, mle_time_limit=mle_time_limit, on_decode_failure=on_decode_failure)


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
        "--states", "--basis", nargs="+", choices=["X", "Z"], default=["X", "Z"],
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
        choices=[
            "pymatching", "mwpf", "bposd", "cpu_bposd", "gpu_bposd",
            "nv-qldpc-decoder", "mle-ilp", "ionq-beam-search",
        ],
        help="Override decoder for all experiments (default: per-experiment default)",
    )
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
    add_run_options(ap)
    if run_config_cli(ap):
        return
    args = ap.parse_args()
    validate_run_options(ap, args)

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
