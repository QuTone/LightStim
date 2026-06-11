"""
General memory experiment runner for LightStim.

Sweeps any combination of QEC codes × distances × error rates.
Results are saved to CSV with per-task checkpointing (append-on-complete).

Supported codes
---------------
Topological (require --distances):
    rotated_sc, unrotated_sc, toric, color, xzzx_sc
BB codes (distance fixed by code, --distances ignored):
    bb_72_12_6, bb_108_8_10, bb_144_12_12, bb_288_12_18

Decoders
--------
    pymatching   CPU MWPM        (default for surface codes)
    mwpf         CPU MWPF        (general purpose)
    cpu_bposd    CPU BP+OSD      (good for QLDPC codes, requires stimbposd)
    gpu_bposd    GPU BP+OSD      (recommended for BB codes, requires CUDA)

CSV output schema (keys / data)
---------------------------------
    code, distance, p, basis, rounds, decoder_name
    shots, errors, logical_error_rate, seconds, n_data, n_total, k

Usage
-----
    # Surface code family, 3 distances:
    venv/bin/python benchmarks/memory/run_memory.py \\
        --codes rotated_sc unrotated_sc toric \\
        --distances 3 5 7 \\
        --p-values 1e-3 5e-3 1e-2 \\
        --decoder pymatching --num-workers 8

    # BB codes on GPU:
    venv/bin/python benchmarks/memory/run_memory.py \\
        --codes bb_72_12_6 bb_144_12_12 \\
        --p-values 1e-3 3e-3 1e-2 \\
        --decoder gpu_bposd

    # Color code with MWPF, save to custom path:
    venv/bin/python benchmarks/memory/run_memory.py \\
        --codes color --distances 3 5 7 \\
        --p-values 1e-3 5e-3 1e-2 \\
        --decoder mwpf \\
        --output benchmarks/memory/results/color_mwpf.csv
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root → lightstim importable

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.toric import ToricCode, ToricCodeExtractionBlock
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.xzzx import (
    XZZXSurfaceCode, XZZXSurfaceCodeExtractionBlock, xzzx_memory_basis,
)
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline

# ── Code registry ─────────────────────────────────────────────────────────────

_BB_CONFIGS = {
    "bb_72_12_6":   {"l": 6,  "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 6},
    "bb_90_8_10":   {"l": 15, "m": 3,  "A": [[9,0],[0,1],[0,2]], "B": [[0,0],[2,0],[7,0]], "d": 10},
    "bb_108_8_10":  {"l": 9,  "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 10},
    "bb_144_12_12": {"l": 12, "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 12},
    "bb_288_12_18": {"l": 12, "m": 12, "A": [[3,0],[0,2],[0,7]], "B": [[0,3],[1,0],[2,0]], "d": 18},  # needs logical_presets entry
}

_TOPO_CODES = {"rotated_sc", "unrotated_sc", "toric", "color", "xzzx_sc"}
_BB_CODES   = set(_BB_CONFIGS)
ALL_CODES   = sorted(_TOPO_CODES | _BB_CODES)


def _make_code(code_name: str, distance: int):
    if code_name == "rotated_sc":
        return RotatedSurfaceCode(distance=distance), RotatedSurfaceCodeExtractionBlock
    if code_name == "unrotated_sc":
        return UnrotatedSurfaceCode(distance=distance), UnrotatedSurfaceCodeExtractionBlock
    if code_name == "toric":
        return ToricCode(distance=distance), ToricCodeExtractionBlock
    if code_name == "color":
        return ColorCode(distance=distance), ColorCodeExtractionBlock
    if code_name == "xzzx_sc":
        return XZZXSurfaceCode(distance=distance), XZZXSurfaceCodeExtractionBlock
    if code_name in _BB_CONFIGS:
        cfg = _BB_CONFIGS[code_name]
        return BBCode(l=cfg["l"], m=cfg["m"], A=cfg["A"], B=cfg["B"]), BBCodeExtractionBlock
    raise ValueError(f"Unknown code: {code_name!r}. Available: {ALL_CODES}")


def _decoder_config(name: str) -> DecoderConfig:
    if name == "pymatching":
        return DecoderConfig(name="pymatching", backend="cpu")
    if name == "mwpf":
        return DecoderConfig(name="mwpf", backend="cpu", params={"cluster_node_limit": 50})
    if name == "cpu_bposd":
        return DecoderConfig(name="bposd", backend="cpu", params={
            "max_iterations": 1000, "osd_order": 10,
            "bp_method": "min_sum", "ms_scaling_factor": 0,
            "osd_method": "osd_cs",
        })
    if name == "gpu_bposd":
        return DecoderConfig(
            name="nv-qldpc-decoder", backend="gpu",
            params={
                "max_iterations": 1000, "osd_order": 10,
                "bp_method": "min_sum", "ms_scaling_factor": 0,
                "osd_method": "osd_cs", "use_osd": True,
            },
        )
    raise ValueError(f"Unknown decoder: {name!r}. Choose: pymatching, mwpf, cpu_bposd, gpu_bposd")


# ── Circuit builder ───────────────────────────────────────────────────────────

def build_circuit(code_name: str, distance: int, p: float,
                  basis: str = "Z", rounds: int = None,
                  noise_model: str = "circuit_level"):
    """Return (circuit, n_data, n_total, k) for a noisy memory experiment."""
    code, block_cls = _make_code(code_name, distance)
    system = QECSystem()
    system.add_patch(code, name=code_name)
    noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    r = rounds if rounds is not None else distance
    # XZZX checks mix X and Z, so the memory needs a per-qubit checkerboard of
    # init/readout bases; a uniform basis yields a degenerate (distance-1) circuit.
    basis_map = xzzx_memory_basis(system, basis) if code_name == "xzzx_sc" else None
    with contextlib.redirect_stdout(io.StringIO()):
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=block_cls,
            rounds=r,
            noise_params=noise,
            noise_model=noise_model,
            basis=basis,
            data_basis_map=basis_map,
        )
        circuit = exp.build()
    n_data  = len(code.data_indices)
    n_total = circuit.num_qubits
    k       = getattr(code, "num_logicals", 1)
    return circuit, n_data, n_total, k


# ── Checkpointing ─────────────────────────────────────────────────────────────

_RESULT_COLS = frozenset({"shots", "errors", "logical_error_rate",
                           "seconds", "n_data", "n_total", "k"})


def _task_key(row: dict) -> tuple:
    """Stable key from input-only columns (used to skip completed tasks)."""
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in _RESULT_COLS
    )


def _load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {_task_key(r) for r in df.to_dict("records")}


# ── Main runner ───────────────────────────────────────────────────────────────

def run(tasks: list[dict], decoder_cfg: DecoderConfig,
        max_shots: int, max_errors: int,
        num_workers: int, batch_size: int,
        output_path: Path) -> None:
    """
    Run all tasks, skipping any already present in output_path (checkpoint resume).

    Each task dict must have: code, distance, p, basis, rounds, decoder_name.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = _load_done_keys(output_path)
    if done_keys:
        print(f"Checkpoint: {len(done_keys)} task(s) already done, skipping.")

    pending = [t for t in tasks if _task_key(t) not in done_keys]
    n_skip  = len(tasks) - len(pending)
    print(f"Tasks: {len(pending)} to run" + (f", {n_skip} skipped" if n_skip else "") + "\n")

    is_gpu = (decoder_cfg.backend == "gpu")
    pipeline = SimulationPipeline(
        decoder_config=decoder_cfg,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=100_000 if is_gpu else batch_size,
        num_workers=1 if is_gpu else num_workers,
        print_progress=True,
        progress_interval_sec=30.0,
    )

    for i, task in enumerate(pending):
        label = (f"[{i+1}/{len(pending)}] {task['code']} "
                 f"d={task['distance']} p={task['p']:.2e} basis={task['basis']}")
        print(label, flush=True)

        t0 = time.perf_counter()
        circuit, n_data, n_total, k = build_circuit(
            task["code"], task["distance"], task["p"],
            task["basis"], task["rounds"], task["noise_model"],
        )
        stats   = pipeline.run(circuit, task)
        elapsed = time.perf_counter() - t0

        row = {
            **task,
            "shots":               stats.shots,
            "errors":              stats.errors,
            "logical_error_rate":  stats.logical_error_rate,
            "seconds":             elapsed,
            "n_data":              n_data,
            "n_total":             n_total,
            "k":                   k,
        }
        pd.DataFrame([row]).to_csv(
            output_path, mode="a", header=not output_path.exists(), index=False,
        )
        print(f"  LER={stats.logical_error_rate:.2e} | "
              f"errors={stats.errors} | shots={stats.shots:,} | {elapsed:.1f}s\n")

    print(f"Done. Results → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--codes", nargs="+", required=True,
                    metavar="CODE",
                    help=f"QEC code(s) to benchmark. Built-in: {', '.join(ALL_CODES)}")
    ap.add_argument("--distances", nargs="+", type=int, default=None,
                    help="Distances to sweep (required for topological codes; "
                         "BB codes use their built-in distance)")
    ap.add_argument("--p-values", nargs="+", type=float,
                    default=np.logspace(-3, -1.5, 6).tolist(),
                    help="Physical error rate values (default: 6 log-spaced points)")
    ap.add_argument("--basis", nargs="+", choices=["Z", "X"], default=["Z"],
                    help="Logical basis to run (default: Z). Use --basis Z X to run both.")
    ap.add_argument("--rounds", type=int, default=None,
                    help="SE rounds per cycle (default: distance)")
    ap.add_argument("--decoder", choices=["pymatching", "mwpf", "cpu_bposd", "gpu_bposd"],
                    default="pymatching",
                    help="Decoder (default: pymatching)")
    ap.add_argument("--max-shots",   type=int, default=1_000_000)
    ap.add_argument("--max-errors",  type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size",  type=int, default=1_000)
    ap.add_argument("--noise-model",
                    choices=["circuit_level", "phenomenological", "code_capacity"],
                    default="circuit_level",
                    help="Noise model (default: circuit_level)")
    ap.add_argument("--output", default=None,
                    help="Output CSV path (auto-computed as results/<codes>_<decoder>.csv if omitted)")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: d=3,5 and 2 p-values (1e-3, 5e-3)")
    args = ap.parse_args()

    if args.quick:
        args.codes = args.codes if args.codes else ["rotated_sc"]
        if not args.distances:
            args.distances = [3, 5]
        args.p_values = [1e-3, 5e-3]
        args.max_shots = 100_000
        args.max_errors = 50

    # Validate distances for topological codes
    topo = [c for c in args.codes if c in _TOPO_CODES]
    if topo and not args.distances:
        ap.error(f"--distances is required for topological codes: {topo}")

    # Build task list
    tasks = []
    for code in args.codes:
        if code in _BB_CONFIGS:
            distances = [_BB_CONFIGS[code]["d"]]
        else:
            distances = args.distances
        for d in distances:
            r = args.rounds if args.rounds is not None else d
            for p in args.p_values:
                for basis in args.basis:
                    tasks.append({
                        "code": code, "distance": d, "p": p,
                        "basis": basis, "rounds": r,
                        "noise_model": args.noise_model,
                        "decoder_name": args.decoder,
                    })

    # Default output path
    if args.output is None:
        tag = "_".join(args.codes[:2]) + ("_etc" if len(args.codes) > 2 else "")
        output = SCRIPT_DIR / "results" / f"{tag}_{args.decoder}.csv"
    else:
        output = Path(args.output)

    print(f"Output:      {output}")
    print(f"Tasks:       {len(tasks)} total")
    print(f"Decoder:     {args.decoder} | noise_model={args.noise_model}")
    print(f"max_shots:   {args.max_shots:.0e} | max_errors={args.max_errors}\n")

    run(tasks, _decoder_config(args.decoder),
        args.max_shots, args.max_errors,
        args.num_workers, args.batch_size,
        output)


if __name__ == "__main__":
    main()
