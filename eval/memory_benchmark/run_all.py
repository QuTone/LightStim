"""
Memory Experiment Benchmark Suite — Paper Figures 1-4.

Sweeps code families × distances × physical error rates.
Outputs CSV results + generates plots.

Usage:
    # Full run (all figures, GPU for BB codes):
    /home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all

    # Specific figure:
    /home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --figure 1

    # Quick test (fewer shots):
    /home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --quick
"""

import argparse
import io
import contextlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
from src.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
from src.qec_code.surface_code.toric import ToricCode, ToricCodeExtractionBlock
from src.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
from src.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from src.ir.qec_system import QECSystem
from src.noise.config import NoiseConfig
from experiments.memory import MemoryExperiment
from src.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


BB_CONFIGS = {
    "bb_72_12_6":   {"l": 6,  "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 6},
    "bb_90_8_10":   {"l": 15, "m": 3,  "A": [[9,0],[0,1],[0,2]], "B": [[0,9],[1,0],[2,0]], "d": 10},
    "bb_108_8_10":  {"l": 9,  "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 10},
    "bb_144_12_12": {"l": 12, "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 12},
    "bb_288_12_18": {"l": 12, "m": 12, "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 18},
}


# ── Code Factory ─────────────────────────────────────────────────────
def make_code(code_name, distance, **kwargs):
    """Create a QEC code instance by name."""
    if code_name == "rotated_sc":
        code = RotatedSurfaceCode(distance=distance)
        return code, RotatedSurfaceCodeExtractionBlock
    elif code_name == "unrotated_sc":
        code = UnrotatedSurfaceCode(distance=distance)
        return code, UnrotatedSurfaceCodeExtractionBlock
    elif code_name == "toric":
        code = ToricCode(distance=distance)
        return code, ToricCodeExtractionBlock
    elif code_name == "color":
        code = ColorCode(distance=distance)
        return code, ColorCodeExtractionBlock
    elif code_name.startswith("bb_"):
        cfg = BB_CONFIGS[code_name]
        code = BBCode(l=cfg["l"], m=cfg["m"], A=cfg["A"], B=cfg["B"])
        return code, BBCodeExtractionBlock
    else:
        raise ValueError(f"Unknown code: {code_name}")


def build_circuit(code_name, distance, p, basis="Z", rounds=None, se_block_kwargs=None):
    """Build a noisy memory experiment circuit. Returns (circuit, n_data, n_total, k)."""
    code, block_class = make_code(code_name, distance)
    system = QECSystem()
    system.add_patch(code, name=code_name)

    noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    r = rounds if rounds is not None else distance

    with contextlib.redirect_stdout(io.StringIO()):
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=block_class,
            rounds=r,
            noise_params=noise,
            noise_model="circuit_level",
            basis=basis,
            se_block_kwargs=se_block_kwargs,
        )
        circuit = exp.build()

    n_data = len(code.data_indices)
    n_total = circuit.num_qubits  # includes data + syndrome qubits
    k = code.num_logicals if hasattr(code, 'num_logicals') else 1
    return circuit, n_data, n_total, k


# ── Figure Configs ───────────────────────────────────────────────────

ERROR_RATES_FIG1 = [1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 1.2e-2, 1.5e-2]
ERROR_RATES_FIG2 = [1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3]  # high→low: fast points first
ERROR_RATES_FIG4 = [5e-3, 2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
ERROR_RATES_QUICK = [1e-3, 5e-3]

# ── Experiment Plan ───────────────────────────────────────────────
#
# Fig 1: Surface Code Family — LER vs PER
#   Codes: Rotated SC, Unrotated SC, Toric  ×  d=3,5,7
#   p: [1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 1.2e-2, 1.5e-2]
#   Decoder: PyMatching (CPU, 32 workers)
#   Points: 3 codes × 3 distances × 7 p = 63
#   max_shots=1e9, max_errors=200
#
# Fig 2: BB Code Family — LER vs PER
#   Codes: [[72,12,6]], [[108,8,10]], [[144,12,12]]  × GPU BP+OSD + MWPF = 6 lines
#   p: [1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3]  (high→low: fast points first)
#   Decoder: GPU BP+OSD (max_iter=1000) + MWPF (c=50)
#   Points: 3 codes × 2 decoders × 6 p = 36
#   max_shots=1e8, max_errors=100
#
# Fig 3: Qubit Efficiency — LER/k vs N_total/k at p=1e-3
#   All codes from Fig1 + Fig2 (GPU BPOSD data) + Color Code d=3,5,7 (MWPF c=50)
#   X-axis: circuit.num_qubits / k (Physical Qubits per Logical Qubit)
#   Points: 12 topological + 3 BB = 15 data points
#   max_shots=1M, max_errors=200
#
# Fig 4: Scheduling Impact — LER vs PER (Rotated SC only, standard orientation)
#   Schedules: perpendicular (FT) vs parallel (non-FT, hook error)
#   d=3,5,7  ×  p: [5e-3, 2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]
#   Decoder: PyMatching (CPU, 32 workers)
#   Points: 2 schedules × 3 distances × 7 p = 42
#   max_shots=1e8, max_errors=100
# ──────────────────────────────────────────────────────────────────

# Figure 1: Surface Code Family — LER vs PER
FIG1_CODES = [
    ("rotated_sc",   [3, 5, 7]),
    ("unrotated_sc", [3, 5, 7]),
    ("toric",        [3, 5, 7]),
]

# Figure 2: BB codes — LER vs PER (each code × 2 decoders)
FIG2_CODES = ["bb_72_12_6", "bb_108_8_10", "bb_144_12_12"]

FIG2_DECODERS = [
    ("gpu_bposd", DecoderConfig(
        name="nv-qldpc-decoder",
        backend="gpu",
        params={
            "max_iterations": 1000,
            "osd_order": 10,
            "bp_method": "min_sum",
            "ms_scaling_factor": 0,
            "osd_method": "osd_cs",
            "use_osd": True,
        },
    )),
    ("mwpf", DecoderConfig(
        name="mwpf",
        backend="cpu",
        params={"cluster_node_limit": 50},
    )),
]

# Figure 3: Color code extra data (GPU BP+OSD, same as Fig 2 decoder)
FIG3_EXTRA_CODES = [
    ("color", [3, 5, 7]),
]

# Figure 4: Scheduling comparison (rotated SC only)
FIG4_DISTANCES = [3, 5, 7]


# ── Decoder Selection ────────────────────────────────────────────────
def get_decoder_config(code_name):
    """Choose decoder by code family."""
    if code_name.startswith("bb_"):
        return DecoderConfig(
            name="nv-qldpc-decoder",
            backend="gpu",
            params={
                "max_iterations": 1000,
                "osd_order": 10,
                "bp_method": "min_sum",
                "ms_scaling_factor": 0,
                "osd_method": "osd_cs",
                "use_osd": True,
            },
        )
    elif code_name == "color":
        return DecoderConfig(name="mwpf", backend="cpu", params={"cluster_node_limit": 50})
    else:
        return DecoderConfig(name="pymatching", backend="cpu")


# ── Run Experiments ──────────────────────────────────────────────────
def run_figure1(error_rates, max_shots, max_errors, num_workers):
    """Figure 1: Surface Code Family (Rotated, Unrotated, Toric) × d=3,5,7."""
    print("=" * 60)
    print("FIGURE 1: Surface Code Family — LER vs PER")
    print("=" * 60)

    tasks = []
    for code_name, distances in FIG1_CODES:
        decoder = get_decoder_config(code_name)
        for d in distances:
            for p in error_rates:
                circuit, n_data, n_total, k = build_circuit(code_name, d, p)
                meta = {"code": code_name, "distance": d, "p": p,
                        "n_data": n_data, "n_total": n_total, "k": k, "figure": 1}
                tasks.append((circuit, meta, decoder))

    return _run_tasks(tasks, max_shots, max_errors, num_workers,
                      checkpoint_path=OUTPUT_DIR / "fig1_surface_codes.csv")


def run_figure2(error_rates, max_shots, max_errors, num_workers,
                codes=None, decoder_filter=None, checkpoint_path=None):
    """Figure 2: BB codes × GPU BP+OSD + MWPF (6 lines).

    Args:
        codes: list of code names to run (default: FIG2_CODES)
        decoder_filter: 'gpu_bposd', 'mwpf', or None (both)
    """
    print("=" * 60, flush=True)
    print("FIGURE 2: BB Codes — LER vs PER", flush=True)
    print("=" * 60, flush=True)

    active_codes = codes if codes is not None else FIG2_CODES
    active_decoders = [(lbl, dec) for lbl, dec in FIG2_DECODERS
                       if decoder_filter is None or lbl == decoder_filter]

    tasks = []
    for decoder_label, decoder in active_decoders:
        for code_name in active_codes:
            for p in error_rates:
                d = BB_CONFIGS[code_name]["d"]
                circuit, n_data, n_total, k = build_circuit(code_name, 0, p, rounds=d)
                meta = {"code": code_name, "p": p, "n_data": n_data, "n_total": n_total,
                        "k": k, "figure": 2, "decoder_label": decoder_label}
                tasks.append((circuit, meta, decoder))

    return _run_tasks(tasks, max_shots, max_errors, num_workers,
                      checkpoint_path=checkpoint_path)


def run_figure4(error_rates, max_shots, max_errors, num_workers):
    """Figure 4: Perpendicular vs Parallel scheduling on rotated SC."""
    print("=" * 60)
    print("FIGURE 4: Scheduling Comparison — Rotated SC")
    print("=" * 60)

    tasks = []
    decoder = get_decoder_config("rotated_sc")
    for sched in ["perpendicular", "parallel"]:
        for d in FIG4_DISTANCES:
            for p in error_rates:
                circuit, n_data, n_total, k = build_circuit(
                    "rotated_sc", d, p,
                    se_block_kwargs={"scheduling": sched},
                )
                meta = {"code": "rotated_sc", "distance": d, "p": p,
                        "scheduling": sched, "n_data": n_data, "n_total": n_total,
                        "k": k, "figure": 4}
                tasks.append((circuit, meta, decoder))

    return _run_tasks(tasks, max_shots, max_errors, num_workers,
                      checkpoint_path=OUTPUT_DIR / "fig4_scheduling.csv")


_CK_RESULT_COLS = frozenset({
    "shots", "post_selected_shots", "post_selection_rate",
    "errors", "logical_error_rate", "seconds", "decoder",
    "n_data", "n_total",  # derived from circuit, not true inputs
})


def _ck_key(d: dict) -> tuple:
    """Stable checkpoint key: input fields only, floats normalized to 6-digit sci notation."""
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(d.items()) if k not in _CK_RESULT_COLS
    )


def _run_tasks(task_list, max_shots, max_errors, num_workers, checkpoint_path=None):
    """Run a list of (circuit, metadata, decoder_config) tuples with per-task checkpointing."""
    # Load existing checkpoint so already-done tasks are skipped on resume
    existing_records = []
    done_keys = set()
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        if cp.exists():
            df_ck = pd.read_csv(cp)
            existing_records = df_ck.to_dict("records")
            for rec in existing_records:
                done_keys.add(_ck_key(rec))
            print(f"  Checkpoint: {len(done_keys)} tasks already done, skipping.")

    pending = [(c, m, d) for c, m, d in task_list if _ck_key(m) not in done_keys]
    n_skip = len(task_list) - len(pending)
    if n_skip:
        print(f"  Skipping {n_skip} completed tasks, {len(pending)} remaining.")

    new_records = []
    for i, (circuit, meta, decoder) in enumerate(pending):
        code_label = meta.get("code", "?")
        d_label = meta.get("distance", "?")
        p_label = meta.get("p", "?")
        sched = meta.get("scheduling", "")
        dec_label = meta.get("decoder_label", "")
        label = f"{code_label} d={d_label} p={p_label}"
        if sched:
            label += f" [{sched}]"
        if dec_label:
            label += f" [{dec_label}]"
        print(f"\n[{i+1}/{len(pending)}] {label}", flush=True)

        if decoder.backend == "gpu":
            effective_workers = 1
            effective_batch = 100_000
        else:
            effective_workers = num_workers
            effective_batch = 1_000
        pipeline = SimulationPipeline(
            decoder_config=decoder,
            max_shots=max_shots,
            max_errors=max_errors,
            batch_size=effective_batch,
            num_workers=effective_workers,
            print_progress=True,
        )
        stats = pipeline.run(circuit, meta)
        row = {
            **meta,
            "shots": stats.shots,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": stats.seconds,
            "decoder": stats.decoder,
        }
        new_records.append(row)

        # Persist immediately — a kill/OOM never loses this result
        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            pd.DataFrame([row]).to_csv(cp, mode="a", header=not cp.exists(), index=False)

        print(f"  → LER={stats.logical_error_rate:.2e} ({stats.errors} errors, {stats.shots:,} shots, {stats.seconds:.1f}s)", flush=True)

    all_records = existing_records + new_records
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# ── Plotting ─────────────────────────────────────────────────────────

# Distance-based color palette (high-contrast qualitative)
PALETTE_DIST = {3: "#a63603", 5: "#1b9e77", 7: "#7570b3", 9: "#d95f02"}

# BB code colors (for Fig 2, one color per code)
BB_COLORS = {
    "bb_72_12_6":   "#a63603",
    "bb_108_8_10":  "#1b9e77",
    "bb_144_12_12": "#7570b3",
}

# Paper-quality style — bold fonts, matching fig4_scheduling_v2.png
PAPER_RC = {
    "font.family": "sans-serif",
    "font.weight": "bold",
    "font.size": 14,
    "axes.labelsize": 17,
    "axes.titlesize": 20,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 9,
}

CODE_LINESTYLES = {
    "rotated_sc":    "-",
    "unrotated_sc":  "--",
    "toric":         ":",
    "color":         "-.",
}

CODE_MARKERS = {
    "rotated_sc":    "o",
    "unrotated_sc":  "s",
    "toric":         "^",
    "color":         "D",
}

CODE_LABELS = {
    "rotated_sc":    "Rotated SC",
    "unrotated_sc":  "Unrotated SC",
    "toric":         "Toric",
    "color":         "Color (6-6-6)",
    "bb_72_12_6":    "[[72,12,6]]",
    "bb_108_8_10":   "[[108,8,10]]",
    "bb_144_12_12":  "[[144,12,12]]",
    "bb_288_12_18":  "[[288,12,18]]",
}

DECODER_LINESTYLES = {"gpu_bposd": "-", "mwpf": "--"}
DECODER_MARKERS    = {"gpu_bposd": "o", "mwpf": "X"}


def _apply_paper_style():
    plt.rcParams.update(PAPER_RC)


def _bold_ticks(ax):
    """Make tick labels bold (supplement rcParams font.weight for ticks)."""
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def plot_figure1(df, save_path):
    """Figure 1: Surface Code Family — color = distance, linestyle = code."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.9, 4.2))

    for code_name in ["rotated_sc", "unrotated_sc", "toric"]:
        df_code = df[df["code"] == code_name]
        ls = CODE_LINESTYLES.get(code_name, "-")
        marker = CODE_MARKERS.get(code_name, "o")
        label_base = CODE_LABELS.get(code_name, code_name)

        for d in sorted(df_code["distance"].dropna().unique()):
            df_d = df_code[df_code["distance"] == d].sort_values("p")
            color = PALETTE_DIST.get(int(d), "gray")
            ax.loglog(
                df_d["p"], df_d["logical_error_rate"] / df_d["k"],
                marker=marker, color=color, linestyle=ls,
                markeredgecolor="none",
                label=f"{label_base} d={int(d)}",
            )

    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate per Logical Qubit")
    ax.set_title("Surface Code Family: LER vs PER", pad=10)
    ax.legend(fontsize=9, ncol=2, loc="lower right", frameon=True)
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close(fig)


def plot_figure2(df, save_path):
    """Figure 2: BB Codes — color = code, linestyle/marker = decoder."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.9, 4.2))

    for code_name in FIG2_CODES:
        df_code = df[df["code"] == code_name]
        color = BB_COLORS.get(code_name, "gray")
        label_base = CODE_LABELS.get(code_name, code_name)

        for decoder_label in ["gpu_bposd", "mwpf"]:
            df_d = df_code[df_code["decoder_label"] == decoder_label].sort_values("p")
            if df_d.empty:
                continue
            ls = DECODER_LINESTYLES[decoder_label]
            marker = DECODER_MARKERS[decoder_label]
            dec_str = "GPU BP+OSD" if decoder_label == "gpu_bposd" else "MWPF"
            ax.loglog(
                df_d["p"], df_d["logical_error_rate"] / df_d["k"],
                marker=marker, color=color, linestyle=ls,
                markeredgecolor="none",
                label=f"{label_base} ({dec_str})",
            )

    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate per Logical Qubit")
    ax.set_title("BB Codes: LER vs PER", pad=10)
    ax.legend(fontsize=9, ncol=1, loc="lower right", frameon=True)
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close(fig)


def plot_figure3(df, save_path):
    """Figure 3: LER/k vs Physical Qubits per Logical Qubit (N_total/k) at fixed p."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.9, 4.2))

    for code_name in df["code"].unique():
        ls = CODE_LINESTYLES.get(code_name, "-")
        marker = CODE_MARKERS.get(code_name, "x")
        label = CODE_LABELS.get(code_name, code_name)
        # For BB codes, use gpu_bposd data only
        if code_name.startswith("bb_") and "decoder_label" in df.columns:
            df_c = df[(df["code"] == code_name) & (df["decoder_label"] == "gpu_bposd")].copy()
        else:
            df_c = df[df["code"] == code_name].copy()
        df_c = df_c.sort_values("n_total")

        n_per_k = df_c["n_total"] / df_c["k"]
        ler_per_k = df_c["logical_error_rate"] / df_c["k"]

        # Color: use distance if available, else code-specific color
        if "distance" in df_c.columns and df_c["distance"].notna().any():
            for d in sorted(df_c["distance"].dropna().unique()):
                df_dd = df_c[df_c["distance"] == d]
                color = PALETTE_DIST.get(int(d), "gray")
                ax.semilogy(
                    df_dd["n_total"] / df_dd["k"], df_dd["logical_error_rate"] / df_dd["k"],
                    marker=marker, color=color, linestyle=ls,
                    markeredgecolor="none",
                    label=f"{label} d={int(d)}",
                )
        else:
            color = BB_COLORS.get(code_name, "gray")
            ax.semilogy(
                n_per_k, ler_per_k,
                marker=marker, color=color, linestyle=ls,
                markeredgecolor="none",
                label=label,
            )

    ax.set_xlabel("Physical Qubits per Logical Qubit ($N_{\\mathrm{total}}/k$)")
    ax.set_ylabel("LER per Logical Qubit")
    ax.set_title("Qubit Efficiency at $p = 10^{-3}$", pad=10)
    ax.legend(fontsize=9, frameon=True, ncol=2)
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close(fig)


def plot_figure4(df, save_path):
    """Scheduling comparison plot — color = distance, linestyle = schedule."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.9, 4.2))

    for sched in ["perpendicular", "parallel"]:
        ls = "-" if sched == "perpendicular" else "--"
        marker = "o" if sched == "perpendicular" else "x"
        label_prefix = "FT (perp.)" if sched == "perpendicular" else "Non-FT (par.)"
        df_s = df[df["scheduling"] == sched]

        for d in sorted(df_s["distance"].unique()):
            df_d = df_s[df_s["distance"] == d].sort_values("p")
            color = PALETTE_DIST.get(int(d), "gray")
            ax.loglog(
                df_d["p"], df_d["logical_error_rate"],
                marker=marker, color=color,
                linestyle=ls, markeredgecolor="none",
                label=f"{label_prefix} d={int(d)}",
            )

    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title("CNOT Scheduling: Hook Error Impact", pad=10)
    ax.legend(fontsize=9, ncol=2, frameon=True)
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Memory Experiment Benchmark Suite")
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only a specific figure (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer error rates, fewer shots")
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--fig2-codes", type=str, default=None,
                        help="Comma-separated BB code names for fig2, e.g. bb_72_12_6,bb_108_8_10")
    parser.add_argument("--fig2-decoder", type=str, default=None, choices=["gpu_bposd", "mwpf"],
                        help="Run only one decoder type for fig2")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU device ID for GPU backend (sets CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_errors = 20 if args.quick else args.max_errors

    er1 = ERROR_RATES_QUICK if args.quick else ERROR_RATES_FIG1
    er2 = ERROR_RATES_QUICK if args.quick else ERROR_RATES_FIG2
    er4 = ERROR_RATES_QUICK if args.quick else ERROR_RATES_FIG4

    # Per-figure max_shots as specified in setup.md
    ms1 = 100_000 if args.quick else int(1e9)
    ms2 = 100_000 if args.quick else int(1e8)  # [[144,12,12]] at p=1e-3 needs ~42M shots
    ms3 = 100_000 if args.quick else int(1e6)
    ms4 = 100_000 if args.quick else int(1e8)

    all_dfs = []

    # Figure 1: Surface Code Family
    if args.figure in (None, 1):
        df1 = run_figure1(er1, ms1, max_errors, args.num_workers)
        df1.to_csv(OUTPUT_DIR / "fig1_surface_codes.csv", index=False)
        plot_figure1(df1, OUTPUT_DIR / "fig1_surface_codes.png")
        all_dfs.append(df1)

    # Figure 2: BB Codes (GPU + MWPF)
    if args.figure in (None, 2):
        me2 = 20 if args.quick else 100
        fig2_codes = args.fig2_codes.split(",") if args.fig2_codes else None
        # Compute suffix first so checkpoint_path matches the final CSV
        suffix = ""
        if args.fig2_codes:
            suffix += "_" + args.fig2_codes.replace(",", "_")
        if args.fig2_decoder:
            suffix += "_" + args.fig2_decoder
        csv_path = OUTPUT_DIR / f"fig2_bb_codes{suffix}.csv"
        df2 = run_figure2(er2, ms2, me2, args.num_workers,
                          codes=fig2_codes, decoder_filter=args.fig2_decoder,
                          checkpoint_path=csv_path)
        df2.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        # Only plot if we have the full dataset
        if not suffix:
            plot_figure2(df2, OUTPUT_DIR / "fig2_bb_codes.png")
        all_dfs.append(df2)

    # Figure 4: Scheduling (before 3, since 3 reuses data)
    if args.figure in (None, 4):
        me4 = 20 if args.quick else 100
        df4 = run_figure4(er4, ms4, me4, args.num_workers)
        df4.to_csv(OUTPUT_DIR / "fig4_scheduling.csv", index=False)
        plot_figure4(df4, OUTPUT_DIR / "fig4_scheduling.png")
        all_dfs.append(df4)

    # Figure 3: Qubit Efficiency (uses p=1e-3 from fig1 + fig2 + color code extra)
    if args.figure in (None, 3):
        # Run color code extra points for fig3 (MWPF c=50)
        print("=" * 60)
        print("FIGURE 3: Extra data — Color Code at p=1e-3 (MWPF)")
        print("=" * 60)
        color_tasks = []
        for code_name, distances in FIG3_EXTRA_CODES:
            decoder = get_decoder_config(code_name)
            for d in distances:
                circuit, n_data, n_total, k = build_circuit(code_name, d, 1e-3)
                meta = {"code": code_name, "distance": d, "p": 1e-3,
                        "n_data": n_data, "n_total": n_total, "k": k, "figure": 3}
                color_tasks.append((circuit, meta, decoder))
        df_color = _run_tasks(color_tasks, ms3, max_errors, args.num_workers,
                              checkpoint_path=OUTPUT_DIR / "fig3_color_extra.csv")
        df_color.to_csv(OUTPUT_DIR / "fig3_color_extra.csv", index=False)

        # Combine all p=1e-3 data
        if all_dfs:
            df_all = pd.concat(all_dfs + [df_color], ignore_index=True)
        else:
            csvs = list(OUTPUT_DIR.glob("fig*.csv"))
            df_all = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

        df3 = df_all[df_all["p"].between(9e-4, 1.1e-3)].copy()
        # De-duplicate; for BB codes use gpu_bposd only
        df3 = df3.drop_duplicates(subset=["code", "distance", "n_total", "decoder_label"],
                                   keep="first")
        if not df3.empty:
            df3.to_csv(OUTPUT_DIR / "fig3_efficiency.csv", index=False)
            plot_figure3(df3, OUTPUT_DIR / "fig3_efficiency.png")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"Results in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
