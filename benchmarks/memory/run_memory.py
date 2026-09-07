"""
General memory experiment runner for LightStim.

Sweeps any combination of QEC codes × distances × error rates.
Results are saved to CSV with per-task checkpointing (append-on-complete).

Supported codes
---------------
Topological (require --distances):
    rotated_sc, rotated_sc_defect, unrotated_sc, toric, color, xzzx_sc
Subsystem (require --distances):
    bacon_shor (dedicated four-layer XZ extraction)
    subsystem_surface (planar triangle gauges, dedicated eight-layer XZ extraction)
BB codes (distance fixed by code, --distances ignored):
    bb_72_12_6, bb_108_8_10, bb_144_12_12, bb_288_12_18
HGP codes (distance fixed by code, --distances ignored):
    hgp_13_1_3, hgp_18_2_3, hgp_225_9_4

Decoders
--------
    pymatching   CPU MWPM        (default for surface codes)
    mwpf         CPU MWPF        (general purpose)
    cpu_bposd    CPU BP+OSD      (good for QLDPC codes, requires stimbposd)
    gpu_bposd    GPU BP+OSD      (recommended for BB/HGP codes, requires CUDA)

CSV output schema (keys / data)
---------------------------------
    code, distance, p, basis, rounds, se_circuit, decoder_name
    layout, block_class
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

    # HGP product-coloration memory on GPU:
    venv/bin/python benchmarks/memory/run_memory.py \\
        --codes hgp_13_1_3 hgp_18_2_3 hgp_225_9_4 \\
        --p-values 1e-3 2e-3 3e-3 \\
        --basis Z X \\
        --decoder gpu_bposd

    # Color code with MWPF, save to custom path:
    venv/bin/python benchmarks/memory/run_memory.py \\
        --codes color --distances 3 5 7 \\
        --color-se-circuits space_multiplexing bell_multiplexing \\
        --p-values 1e-3 5e-3 1e-2 \\
        --decoder mwpf \\
        --output benchmarks/memory/results/color_mwpf.csv
"""
import argparse
import contextlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import stim

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root → lightstim importable

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.protocols.rotated_surface_defect import (
    RotatedSurfaceDefectMemoryExperiment,
)
from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from lightstim.qec_code.subsystem_surface import (
    SubsystemSurfaceCode, SubsystemSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.HGP import (
    HGPProductColorationExtractionBlock,
    hgp_13_1_3,
    hgp_18_2_3,
    hgp_225_9_4,
)
from lightstim.qec_code.color_code import (
    ColorCode,
    ColorCodeBellFlaggingBlock,
    ColorCodeBellMultiplexingBlock,
    ColorCodeMiddleOutBlock,
    ColorCodeSpaceMultiplexingBlock,
    ColorCodeTimeMultiplexingBlock,
)
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

_HGP_CONFIGS = {
    "hgp_13_1_3": {
        "factory": hgp_13_1_3,
        "d": 3,
        "se_circuit": "product_coloration",
    },
    "hgp_18_2_3": {
        "factory": hgp_18_2_3,
        "d": 3,
        "se_circuit": "product_coloration",
    },
    "hgp_225_9_4": {
        "factory": hgp_225_9_4,
        "d": 4,
        "se_circuit": "product_coloration",
    },
}

_TOPO_CODES = {
    "rotated_sc",
    "rotated_sc_defect",
    "unrotated_sc",
    "toric",
    "color",
    "xzzx_sc",
}
_BB_CODES   = set(_BB_CONFIGS)
_HGP_CODES  = set(_HGP_CONFIGS)
_DISTANCE_CODES = _TOPO_CODES | {"bacon_shor", "subsystem_surface"}
ALL_CODES   = sorted(_DISTANCE_CODES | _BB_CODES | _HGP_CODES)

DEFAULT_COLOR_SE_CIRCUIT = "space_multiplexing"


@dataclass(frozen=True)
class ColorSECircuitSpec:
    layout: str
    block_class: type
    supported_bases: frozenset[str] = frozenset({"X", "Y", "Z"})


COLOR_SE_CIRCUITS = {
    "space_multiplexing": ColorSECircuitSpec(
        layout="superdense",
        block_class=ColorCodeSpaceMultiplexingBlock,
    ),
    "bell_multiplexing": ColorSECircuitSpec(
        layout="superdense",
        block_class=ColorCodeBellMultiplexingBlock,
    ),
    "bell_flagging": ColorSECircuitSpec(
        layout="superdense",
        block_class=ColorCodeBellFlaggingBlock,
    ),
    "time_multiplexing": ColorSECircuitSpec(
        layout="triangular",
        block_class=ColorCodeTimeMultiplexingBlock,
    ),
    "middle_out": ColorSECircuitSpec(
        layout="rectangle",
        block_class=ColorCodeMiddleOutBlock,
        supported_bases=frozenset({"X", "Z"}),
    ),
}


def _color_se_spec(se_circuit: str | None) -> ColorSECircuitSpec:
    name = DEFAULT_COLOR_SE_CIRCUIT if se_circuit in (None, "default") else se_circuit
    try:
        return COLOR_SE_CIRCUITS[name]
    except KeyError as ex:
        choices = ", ".join(COLOR_SE_CIRCUITS)
        raise ValueError(
            f"Unknown Color Code SE circuit: {name!r}. Available: {choices}"
        ) from ex


def _make_code(code_name: str, distance: int, se_circuit: str | None = None):
    if code_name == "subsystem_surface":
        if se_circuit not in (None, "default", "dedicated"):
            raise ValueError(f"Unknown subsystem surface SE circuit: {se_circuit!r}")
        return SubsystemSurfaceCode(distance=distance), SubsystemSurfaceCodeExtractionBlock
    if code_name == "bacon_shor":
        if se_circuit not in (None, "default", "dedicated"):
            raise ValueError(f"Unknown Bacon-Shor SE circuit: {se_circuit!r}")
        return BaconShorCode(distance=distance), BaconShorCodeExtractionBlock
    if code_name in {"rotated_sc", "rotated_sc_defect"}:
        return RotatedSurfaceCode(distance=distance), RotatedSurfaceCodeExtractionBlock
    if code_name == "unrotated_sc":
        return UnrotatedSurfaceCode(distance=distance), UnrotatedSurfaceCodeExtractionBlock
    if code_name == "toric":
        return ToricCode(distance=distance), ToricCodeExtractionBlock
    if code_name == "color":
        spec = _color_se_spec(se_circuit)
        return ColorCode(distance=distance, layout=spec.layout), spec.block_class
    if code_name == "xzzx_sc":
        return XZZXSurfaceCode(distance=distance), XZZXSurfaceCodeExtractionBlock
    if code_name in _BB_CONFIGS:
        cfg = _BB_CONFIGS[code_name]
        return BBCode(l=cfg["l"], m=cfg["m"], A=cfg["A"], B=cfg["B"]), BBCodeExtractionBlock
    if code_name in _HGP_CONFIGS:
        cfg = _HGP_CONFIGS[code_name]
        selected_se_circuit = (
            cfg["se_circuit"] if se_circuit in (None, "default") else se_circuit
        )
        if selected_se_circuit != cfg["se_circuit"]:
            raise ValueError(
                f"Unknown HGP SE circuit: {selected_se_circuit!r}. "
                f"Available for {code_name}: {cfg['se_circuit']}"
            )
        return cfg["factory"](), HGPProductColorationExtractionBlock
    raise ValueError(f"Unknown code: {code_name!r}. Available: {ALL_CODES}")


def _decoder_config(
    name: str,
    osd_order: int = 10,
    max_iterations: int = 1000,
    mle_time_limit: float = 0.0,
    on_decode_failure: str = "error",
) -> DecoderConfig:
    if name == "pymatching":
        return DecoderConfig(
            name="pymatching", backend="cpu",
            on_decode_failure=on_decode_failure,
        )
    if name == "mwpf":
        return DecoderConfig(
            name="mwpf", backend="cpu", params={"cluster_node_limit": 50},
            on_decode_failure=on_decode_failure,
        )
    if name == "cpu_bposd":
        return DecoderConfig(
            name="bposd", backend="cpu", params={
                "max_iterations": max_iterations, "osd_order": osd_order,
                "bp_method": "min_sum", "ms_scaling_factor": 0,
                "osd_method": "osd_cs",
            },
            on_decode_failure=on_decode_failure,
        )
    if name == "gpu_bposd":
        return DecoderConfig(
            name="nv-qldpc-decoder", backend="gpu",
            params={
                "max_iterations": max_iterations, "osd_order": osd_order,
                "bp_method": "min_sum", "ms_scaling_factor": 0,
                "osd_method": "osd_cs", "use_osd": True,
            },
            on_decode_failure=on_decode_failure,
        )
    if name == "mle-ilp":
        return DecoderConfig(
            name="mle-ilp", backend="cpu",
            params={"time_limit": mle_time_limit},
            on_decode_failure=on_decode_failure,
        )
    raise ValueError(
        f"Unknown decoder: {name!r}. Choose: pymatching, mwpf, cpu_bposd, "
        "gpu_bposd, mle-ilp"
    )


# ── Circuit builder ───────────────────────────────────────────────────────────

def _select_memory_detectors(circuit: stim.Circuit, basis: str) -> stim.Circuit:
    """Select existing pure-basis M/MX detectors in the Bacon-Shor memory.

    This loses complementary syndrome information. Every physical operation,
    noise channel, measurement and observable is retained. The tracker has
    already generated the full detector set; this helper constructs none.
    """
    selected = stim.Circuit()
    record_bases = []
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            targets = instruction.targets_copy()
            if not all(t.is_measurement_record_target for t in targets):
                raise ValueError("Expected measurement-record detector targets.")
            if all(record_bases[len(record_bases) + t.value] == basis for t in targets):
                selected.append(instruction)
        else:
            selected.append(instruction)
            one = stim.Circuit()
            one.append(instruction)
            if one.num_measurements:
                if instruction.name not in {"M", "MX"}:
                    raise ValueError(f"Unsupported memory measurement: {instruction.name}")
                record_bases.extend(["X" if instruction.name == "MX" else "Z"] * one.num_measurements)
    dem = selected.detector_error_model(decompose_errors=False).flattened()
    if any(sum(t.is_relative_detector_id() for t in op.targets_copy()) > 2
           for op in dem if op.type == "error"):
        raise ValueError("Selected memory DEM is not graphlike; MWPM is unsupported.")
    return selected


def build_circuit(
    code_name: str,
    distance: int,
    p: float,
    basis: str = "Z",
    rounds: int | None = None,
    noise_model: str = "circuit_level",
    se_circuit: str | None = None,
    *,
    p_idle: float | None = None,
    p_1q: float | None = None,
    detector_basis: str = "all",
):
    """Return (circuit, n_data, n_total, k) for a noisy memory experiment.

    All noise rates default to p. Optional idle / one-qubit overrides also
    support the saved no-idle Bacon-Shor baseline. By default retain all
    detectors; Bacon-Shor MWPM callers select detector_basis=basis (X or Z).
    """
    if detector_basis != "all" and (
        code_name != "bacon_shor" or detector_basis not in {"X", "Z"}
        or detector_basis != basis
    ):
        raise ValueError("detector_basis selection requires matching X/Z Bacon-Shor memory.")
    for name, rate in (("p", p), ("p_idle", p_idle), ("p_1q", p_1q)):
        if rate is not None and (not np.isfinite(rate) or not 0 <= rate <= 1):
            raise ValueError(f"{name} must be finite and between 0 and 1.")
    code, block_cls = _make_code(code_name, distance, se_circuit)
    noise = NoiseConfig(p_idle=p if p_idle is None else p_idle,
                        p_1q=p if p_1q is None else p_1q,
                        p_2q=p, p_meas=p, p_reset=p)
    r = rounds if rounds is not None else distance

    with contextlib.redirect_stdout(io.StringIO()):
        if code_name == "rotated_sc_defect":
            exp = RotatedSurfaceDefectMemoryExperiment(
                distance=distance,
                memory_basis=basis,
                pre_defect_rounds=r,
                noise_params=noise,
                noise_model=noise_model,
            )
        elif code_name == "color":
            spec = _color_se_spec(se_circuit)
            if basis not in spec.supported_bases:
                supported = ", ".join(sorted(spec.supported_bases))
                raise ValueError(
                    f"{se_circuit or DEFAULT_COLOR_SE_CIRCUIT} supports "
                    f"basis {supported}, not {basis}."
                )
            exp = MemoryExperiment(
                qec_patch=code,
                extraction_block_class=block_cls,
                rounds=r,
                noise_params=noise,
                noise_model=noise_model,
                basis=basis,
            )
        else:
            system = QECSystem()
            system.add_patch(code, name=code_name)
            # XZZX checks mix X and Z, so the memory needs a per-qubit
            # checkerboard of init/readout bases.
            basis_map = (
                xzzx_memory_basis(system, basis) if code_name == "xzzx_sc" else None
            )
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
    if detector_basis != "all":
        circuit = _select_memory_detectors(circuit, detector_basis)
    n_data  = len(code.data_indices)
    n_total = circuit.num_qubits
    k       = getattr(code, "num_logicals", 1)
    return circuit, n_data, n_total, k


# ── Checkpointing ─────────────────────────────────────────────────────────────

_RESULT_COLS = frozenset({
    "shots",
    "errors",
    "logical_error_rate",
    "seconds",
    "n_data",
    "n_total",
    "k",
    "layout",
    "block_class",
})

_RESULT_COLUMNS = [
    "code", "distance", "p", "basis", "rounds", "se_circuit",
    "p_idle", "p_1q", "p_idle_mode", "p_1q_mode", "detector_basis",
    "noise_model", "decoder_name", "decoder_time_limit",
    "on_decode_failure", "layout", "block_class", "shots", "errors",
    "logical_error_rate", "seconds", "n_data", "n_total", "k",
]
_RESULT_METADATA_DEFAULTS = {
    "decoder_time_limit": 0.0,
    "on_decode_failure": "error",
    "detector_basis": "all",
    "p_idle_mode": "sweep",
    "p_1q_mode": "sweep",
}


def _ensure_result_schema(path: Path) -> None:
    """Upgrade older benchmark CSVs before checkpointing or appending."""
    if not path.exists():
        return
    df = pd.read_csv(path)
    unknown = set(df.columns) - set(_RESULT_COLUMNS)
    if unknown:
        raise ValueError(
            f"Cannot migrate {path}: unknown result columns {sorted(unknown)}"
        )
    changed = False
    for column in ("p_idle", "p_1q"):
        if column not in df.columns:
            df[column] = df["p"]
            changed = True
    for column, default in _RESULT_METADATA_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
            changed = True
    if changed or list(df.columns) != _RESULT_COLUMNS:
        df.reindex(columns=_RESULT_COLUMNS).to_csv(path, index=False)


def _task_key(row: dict) -> tuple:
    """Stable key from input-only columns (used to skip completed tasks)."""
    row = {**_RESULT_METADATA_DEFAULTS,
           **{f"{name}_mode": "fixed" if name in row else "sweep" for name in ("p_idle", "p_1q")},
           "p_idle": row.get("p"),
           "p_1q": row.get("p"), **row}
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in _RESULT_COLS
    )


def _load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    _ensure_result_schema(path)
    df = pd.read_csv(path)
    return {_task_key(r) for r in df.to_dict("records")}


# ── Main runner ───────────────────────────────────────────────────────────────

def run(tasks: list[dict], decoder_cfg: DecoderConfig,
        max_shots: int, max_errors: int,
        num_workers: int, batch_size: int,
        output_path: Path) -> None:
    """
    Run all tasks, skipping any already present in output_path (checkpoint resume).

    Each task dict must have: code, distance, p, basis, rounds, se_circuit,
    decoder_name.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        {**_RESULT_METADATA_DEFAULTS, "p_idle": t["p"], "p_1q": t["p"],
         **{f"{name}_mode": "fixed" if name in t else "sweep" for name in ("p_idle", "p_1q")},
         "detector_basis": (
             t["basis"] if t["code"] == "bacon_shor" and decoder_cfg.name == "pymatching"
             else "all"
         ), **t}
        for t in tasks
    ]

    done_keys = _load_done_keys(output_path)
    if done_keys:
        print(f"Checkpoint: {len(done_keys)} task(s) already done, skipping.")

    pending = [t for t in tasks if _task_key(t) not in done_keys]
    n_skip  = len(tasks) - len(pending)
    print(f"Tasks: {len(pending)} to run" + (f", {n_skip} skipped" if n_skip else "") + "\n")

    pipeline = SimulationPipeline(
        decoder_config=decoder_cfg,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=1 if decoder_cfg.backend == "gpu" else num_workers,
        print_progress=True,
        progress_interval_sec=30.0,
    )

    for i, task in enumerate(pending):
        se_label = (
            f" se={task['se_circuit']}" if task["code"] == "color" else ""
        )
        label = (f"[{i+1}/{len(pending)}] {task['code']} "
                 f"d={task['distance']} p={task['p']:.2e} "
                 f"basis={task['basis']}{se_label}")
        print(label, flush=True)

        t0 = time.perf_counter()
        circuit, n_data, n_total, k = build_circuit(
            code_name=task["code"],
            distance=task["distance"],
            p=task["p"],
            basis=task["basis"],
            rounds=task["rounds"],
            noise_model=task["noise_model"],
            se_circuit=task["se_circuit"],
            p_idle=task["p_idle"],
            p_1q=task["p_1q"],
            detector_basis=task["detector_basis"],
        )
        stats   = pipeline.run(circuit, task)
        elapsed = time.perf_counter() - t0

        if task["code"] == "color":
            color_spec = _color_se_spec(task["se_circuit"])
            layout = color_spec.layout
            block_class = color_spec.block_class.__name__
        elif task["code"] == "rotated_sc_defect":
            layout = "center_data_defect"
            block_class = RotatedSurfaceDefectMemoryExperiment.__name__
        elif task["code"] in _HGP_CODES:
            layout = "canonical_interleaved_product"
            block_class = HGPProductColorationExtractionBlock.__name__
        elif task["code"] == "bacon_shor":
            layout = "square_edge_ancillas"
            block_class = BaconShorCodeExtractionBlock.__name__
        elif task["code"] == "subsystem_surface":
            layout = "planar_triangle_hypotenuse_ancillas"
            block_class = SubsystemSurfaceCodeExtractionBlock.__name__
        else:
            layout = "code_default"
            block_class = "code_default"

        row = {
            **task,
            "layout":               layout,
            "block_class":          block_class,
            "shots":               stats.shots,
            "errors":              stats.errors,
            "logical_error_rate":  stats.logical_error_rate,
            "seconds":             elapsed,
            "n_data":              n_data,
            "n_total":             n_total,
            "k":                   k,
        }
        pd.DataFrame([row], columns=_RESULT_COLUMNS).to_csv(
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
                    help="Distances to sweep (required for topological / subsystem codes; "
                         "BB/HGP codes use their built-in distance)")
    ap.add_argument("--p-values", nargs="+", type=float,
                    default=np.logspace(-3, -1.5, 6).tolist(),
                    help="Physical error rate values (default: 6 log-spaced points)")
    ap.add_argument("--p-idle", type=float, default=None,
                    help="Override idle noise rate (default: each swept p)")
    ap.add_argument("--p-1q", type=float, default=None,
                    help="Override one-qubit gate noise rate (default: each swept p)")
    ap.add_argument("--basis", nargs="+", choices=["Z", "X", "Y"], default=["Z"],
                    help="Logical basis to run (default: Z). Color Code supports "
                         "X/Y/Z except middle_out, which supports X/Z.")
    ap.add_argument(
        "--color-se-circuits",
        nargs="+",
        choices=list(COLOR_SE_CIRCUITS),
        default=[DEFAULT_COLOR_SE_CIRCUIT],
        help="Color Code syndrome-extraction circuits to benchmark "
             f"(default: {DEFAULT_COLOR_SE_CIRCUIT})",
    )
    ap.add_argument("--rounds", type=int, default=None,
                    help="Complete SE cycles per memory shot (default: distance)")
    ap.add_argument("--decoder", choices=["pymatching", "mwpf", "cpu_bposd", "gpu_bposd", "mle-ilp"],
                    default="pymatching",
                    help="Decoder (default: pymatching)")
    ap.add_argument("--osd-order", type=int, default=10,
                    help="OSD order for cpu_bposd/gpu_bposd decoders (default: 10)")
    ap.add_argument("--max-iterations", type=int, default=1000,
                    help="BP iteration limit for cpu_bposd/gpu_bposd decoders (default: 1000)")
    ap.add_argument(
        "--mle-time-limit", type=float, default=0.0,
        help="Soft seconds/shot limit for mle-ilp; 0 is exact/unlimited (default: 0)",
    )
    ap.add_argument(
        "--on-decode-failure", choices=["error", "discard", "ignore"],
        default="error",
        help="Policy for decoder timeouts/failures (default: error)",
    )
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

    unknown_codes = sorted(set(args.codes) - set(ALL_CODES))
    if unknown_codes:
        ap.error(f"Unknown code(s): {unknown_codes}. Available: {ALL_CODES}")
    if not 1 <= args.num_workers <= 48:
        ap.error("--num-workers must be between 1 and 48")
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    for name, rates in (("--p-values", args.p_values), ("--p-idle", [args.p_idle]),
                        ("--p-1q", [args.p_1q])):
        if any(rate is not None and (not np.isfinite(rate) or not 0 <= rate <= 1)
               for rate in rates):
            ap.error(f"{name} must be finite and between 0 and 1")
    if not np.isfinite(args.mle_time_limit) or args.mle_time_limit < 0:
        ap.error("--mle-time-limit must be finite and non-negative")
    if args.mle_time_limit > 0 and args.decoder != "mle-ilp":
        ap.error("--mle-time-limit is only valid with --decoder mle-ilp")

    if args.quick:
        args.codes = args.codes if args.codes else ["rotated_sc"]
        if not args.distances:
            args.distances = [3, 5]
        args.p_values = [1e-3, 5e-3]
        args.max_shots = 100_000
        args.max_errors = 50

    variable_distance = [c for c in args.codes if c in _DISTANCE_CODES]
    if variable_distance and not args.distances:
        ap.error(f"--distances is required for these codes: {variable_distance}")
    if "Y" in args.basis and any(code != "color" for code in args.codes):
        ap.error("--basis Y is currently supported here only for Color Code")
    if (
        "color" in args.codes
        and "Y" in args.basis
        and any(
            "Y" not in COLOR_SE_CIRCUITS[name].supported_bases
            for name in args.color_se_circuits
        )
    ):
        ap.error("middle_out supports only X/Z memory experiments")

    # Build task list
    tasks = []
    for code in args.codes:
        if code in _BB_CONFIGS:
            distances = [_BB_CONFIGS[code]["d"]]
        elif code in _HGP_CONFIGS:
            distances = [_HGP_CONFIGS[code]["d"]]
        else:
            distances = args.distances
        if code == "color":
            se_circuits = args.color_se_circuits
        elif code in _HGP_CONFIGS:
            se_circuits = [_HGP_CONFIGS[code]["se_circuit"]]
        elif code == "rotated_sc_defect":
            se_circuits = ["alternating_defect_gauges"]
        elif code in {"bacon_shor", "subsystem_surface"}:
            se_circuits = ["dedicated"]
        else:
            se_circuits = ["default"]
        for d in distances:
            r = args.rounds if args.rounds is not None else d
            for se_circuit in se_circuits:
                for p in args.p_values:
                    for basis in args.basis:
                        tasks.append({
                            "code": code,
                            "distance": d,
                            "p": p,
                            "p_idle": p if args.p_idle is None else args.p_idle,
                            "p_1q": p if args.p_1q is None else args.p_1q,
                            "p_idle_mode": "sweep" if args.p_idle is None else "fixed",
                            "p_1q_mode": "sweep" if args.p_1q is None else "fixed",
                            "basis": basis,
                            "rounds": r,
                            "se_circuit": se_circuit,
                            "noise_model": args.noise_model,
                            "decoder_name": args.decoder,
                            "decoder_time_limit": (
                                args.mle_time_limit
                                if args.decoder == "mle-ilp" else 0.0
                            ),
                            "on_decode_failure": args.on_decode_failure,
                        })

    # Default output path
    if args.output is None:
        tag = "_".join(args.codes[:2]) + ("_etc" if len(args.codes) > 2 else "")
        output = SCRIPT_DIR / "results" / f"{tag}_{args.decoder}.csv"
    else:
        output = Path(args.output)

    print(f"Output:      {output}")
    print(f"Tasks:       {len(tasks)} total")
    print(f"Decoder:     {args.decoder} | noise_model={args.noise_model} | "
          f"osd_order={args.osd_order} | max_iterations={args.max_iterations}")
    if args.decoder == "mle-ilp":
        limit = "unlimited" if args.mle_time_limit == 0 else f"{args.mle_time_limit:g}s/shot"
        print(f"MLE budget:  {limit} | failure_policy={args.on_decode_failure}")
    if "color" in args.codes:
        print(f"Color SE:    {', '.join(args.color_se_circuits)}")
    print(f"batch_size:  {args.batch_size}")
    effective_workers = 1 if args.decoder == "gpu_bposd" else args.num_workers
    print(f"workers:     {effective_workers}")
    print(f"max_shots:   {args.max_shots:.0e} | max_errors={args.max_errors}\n")

    run(tasks, _decoder_config(
            args.decoder,
            args.osd_order,
            args.max_iterations,
            args.mle_time_limit,
            args.on_decode_failure,
        ),
        args.max_shots, args.max_errors,
        args.num_workers, args.batch_size,
        output)


if __name__ == "__main__":
    main()
