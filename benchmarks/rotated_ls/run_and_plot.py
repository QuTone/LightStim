"""
Rotated surface code lattice surgery — LER vs p for the §7.2 joint Pauli measurement.

Sweeps the two-patch rotated LS merge (X̄₁X̄₂ and Z̄₁Z̄₂) over distance × physical error
rate, decodes with PyMatching, and plots LER vs p (log-log), one curve per distance.

Usage:
    PYTHONPATH=. python3 benchmarks/rotated_ls/run_and_plot.py
"""
import io
import contextlib
import argparse
from pathlib import Path

import numpy as np
import pymatching
import pandas as pd

from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock, RotatedTwoPatchCoupler)
from lightstim.noise.config import NoiseConfig

OUT = Path(__file__).resolve().parent / "results"

# interaction_type -> (offset, init1, init2)  (complementary bases avoid the joint-logical degeneracy)
# Rotated convention (verified via Stim peek): ZZ = vertical stack, XX = side-by-side.
MERGES = {
    "XX": (lambda d: (2 * d + 2, 0), "Z", "X"),
    "ZZ": (lambda d: (0, 2 * d + 2), "X", "Z"),
}


def build_circuit(d, p, interaction_type):
    offset_fn, i1, i2 = MERGES[interaction_type]
    noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
    exp = TwoPatchLSExperiment(
        patch1_config={"distance": d}, patch2_config={"distance": d},
        offset=offset_fn(d), interaction_type=interaction_type,
        coupler_protocol=RotatedTwoPatchCoupler(),
        code_patch_class=RotatedSurfaceCode,
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        initial_state_patch1=i1, initial_state_patch2=i2,
        measure_state_patch1=i1, measure_state_patch2=i2,
        rounds=d, noise_params=noise, noise_model="circuit_level",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def estimate_ler(d, p, interaction_type, max_shots, max_errors, batch=20000):
    circ = build_circuit(d, p, interaction_type)
    dem = circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler()
    shots = errors = 0
    while shots < max_shots and errors < max_errors:
        n = min(batch, max_shots - shots)
        dets, obs = sampler.sample(n, separate_observables=True)
        pred = matcher.decode_batch(dets)
        errors += int(np.sum(np.any(pred != obs, axis=1)))
        shots += n
    return shots, errors, (errors / shots if shots else float("nan"))


def run(distances, p_values, max_shots, max_errors):
    rows = []
    for interaction in ("XX", "ZZ"):
        for d in distances:
            for p in p_values:
                shots, errors, ler = estimate_ler(d, p, interaction, max_shots, max_errors)
                rows.append(dict(interaction=interaction, d=d, p=p, shots=shots,
                                 errors=errors, logical_error_rate=ler))
                print(f"  {interaction} d={d} p={p:.1e}: LER={ler:.3e}  ({errors} err / {shots} shots)")
    return pd.DataFrame(rows)


def plot(df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE
    MARKERS = {3: "o", 5: "s", 7: "^"}

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)
    titles = {"XX": r"Rotated LS  $\bar X_1\bar X_2$", "ZZ": r"Rotated LS  $\bar Z_1\bar Z_2$"}
    for ax, interaction in zip(axes, ("XX", "ZZ")):
        sub = df[df["interaction"] == interaction]
        proxy = []
        for d in sorted(sub["d"].unique()):
            dd = sub[sub["d"] == d].sort_values("p")
            dd = dd[dd["logical_error_rate"] > 0]
            color = PALETTE_DISTANCE.get(int(d), "gray")
            marker = MARKERS.get(int(d), "o")
            ax.loglog(dd["p"], dd["logical_error_rate"], marker=marker, color=color,
                      ls="-", markeredgecolor="none")
            proxy.append(mlines.Line2D([], [], color=color, ls="-", marker=marker,
                                       markersize=8, markeredgecolor="none", label=f"$d = {d}$"))
        ax.set_xlabel("Physical Error Rate $p$")
        ax.set_title(titles[interaction])
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
        bold_ticks(ax)
        leg = ax.legend(handles=proxy, title="Distance", loc="lower right", frameon=True, framealpha=0.85)
        leg.get_title().set_fontweight("bold")
    axes[0].set_ylabel("Logical Error Rate (LER)")
    fig.suptitle("Rotated Surface Code — Lattice Surgery Joint Measurement (§7.2)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--p-values", type=float, nargs="+",
                    default=[1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2])
    ap.add_argument("--max-shots", type=float, default=1e6)
    ap.add_argument("--max-errors", type=int, default=100)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = run(args.distances, args.p_values, int(args.max_shots), args.max_errors)
    df.to_csv(OUT / "rotated_ls_results.csv", index=False)
    plot(df, OUT / "rotated_ls_ler.png")
