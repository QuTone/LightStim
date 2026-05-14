"""
Y-state injection benchmark with Y-specific shrink readout.

Compares two readout modes:
- unencode: measure all data qubits in Y basis (`logical_unencode`)
- shrink_single: `logical_shrink` then single-qubit corner MY readout

Outputs are written under tests/state_Y_injection/.
"""

import io
import os
import sys
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatching

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightstim.protocols.state_injection import StateInjectionExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend.post_select import (
    get_post_select_detector_indices,
    apply_post_selection,
)


READOUT_MODES = ["unencode", "shrink_single"]
DISTANCES = [3, 5, 7]
ROUNDS_LIST = [2, 3]
P2_LIST = [1e-4, 5e-4, 1e-3]

BATCH = 20_000
MAX_SHOTS = 300_000
N_ERR_STOP = 80


def build_circuit(distance: int, rounds: int, p2: float, readout_mode: str):
    noise = NoiseConfig(p_2q=p2)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = StateInjectionExperiment(
            distance=distance,
            rounds=rounds,
            injection_protocol="corner",
            inject_state="Y",
            y_readout_mode=readout_mode,
            noise_params=noise,
            noise_model="circuit_level",
            if_detector=True,
        )
        circ = exp.build()
    return circ


def run_one_config(distance: int, rounds: int, p2: float, readout_mode: str):
    circ = build_circuit(distance, rounds, p2, readout_mode)
    ps_idx = get_post_select_detector_indices(circ)
    sampler = circ.compile_detector_sampler()

    matcher = None
    try:
        dem = circ.detector_error_model()
        matcher = pymatching.Matching.from_detector_error_model(dem)
    except Exception:
        matcher = None

    total_shots = 0
    kept = 0
    post_err = 0
    no_post_err = 0
    qec_err = 0

    while total_shots < MAX_SHOTS and post_err < N_ERR_STOP:
        det_b, obs_b = sampler.sample(shots=BATCH, separate_observables=True)
        obs_col = obs_b[:, 0].astype(np.uint8)

        total_shots += BATCH
        no_post_err += int(obs_col.sum())

        if matcher is not None:
            preds = matcher.decode_batch(det_b).astype(np.uint8).flatten()
            qec_err += int((preds != obs_col).sum())

        det_f, obs_f = apply_post_selection(det_b, obs_b, ps_idx)
        kept += int(det_f.shape[0])
        if obs_f.shape[0] > 0:
            post_err += int(obs_f[:, 0].sum())

    return {
        "distance": distance,
        "rounds": rounds,
        "p2": p2,
        "readout_mode": readout_mode,
        "n_ops": len(circ),
        "n_detectors": circ.num_detectors,
        "n_observables": circ.num_observables,
        "total_shots": total_shots,
        "kept": kept,
        "post_rate": kept / total_shots if total_shots > 0 else np.nan,
        "ler_no_post": no_post_err / total_shots if total_shots > 0 else np.nan,
        "ler_post": post_err / kept if kept > 0 else np.nan,
        "ler_qec": qec_err / total_shots if matcher is not None and total_shots > 0 else np.nan,
    }


def save_diagrams():
    circ = build_circuit(distance=3, rounds=2, p2=1e-3, readout_mode="shrink_single")
    (SCRIPT_DIR / "state_Y_shrink_single_d3_r2.stim").write_text(str(circ), encoding="utf-8")

    diagram_text = str(circ.diagram("timeline-text"))
    (SCRIPT_DIR / "state_Y_shrink_single_d3_r2_timeline.txt").write_text(diagram_text, encoding="utf-8")

    try:
        diagram_svg = str(circ.diagram("timeline-svg"))
        (SCRIPT_DIR / "state_Y_shrink_single_d3_r2_timeline.svg").write_text(diagram_svg, encoding="utf-8")
    except Exception:
        pass


def plot_metric(df: pd.DataFrame, metric: str, title: str, out_name: str):
    distances = sorted(df["distance"].unique())
    rounds_list = sorted(df["rounds"].unique())

    for rounds in rounds_list:
        fig, axes = plt.subplots(1, len(distances), figsize=(5 * len(distances), 4), sharey=True)
        if len(distances) == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = df[(df["rounds"] == rounds) & (df["distance"] == d)]
            for mode in READOUT_MODES:
                mode_df = sub[sub["readout_mode"] == mode].sort_values("p2")
                ax.plot(mode_df["p2"], mode_df[metric], marker="o", label=mode)

            ax.set_xscale("log")
            if metric.startswith("ler_"):
                ax.set_yscale("log")
            ax.set_xlabel("p2")
            ax.set_title(f"d={d}, rounds={rounds}")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel(metric)
        axes[-1].legend(loc="best")
        fig.suptitle(title + f" (rounds={rounds})")
        fig.tight_layout()
        fig.savefig(SCRIPT_DIR / f"{out_name}_r{rounds}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_summary(df: pd.DataFrame):
    lines = []
    lines.append("# Y Injection Summary")
    lines.append("")
    lines.append("Comparison between `unencode` and `shrink_single` readout modes.")
    lines.append("")

    for metric in ["ler_post", "ler_no_post", "ler_qec", "post_rate"]:
        lines.append(f"## {metric}")
        pivot = (
            df.pivot_table(
                index=["distance", "rounds", "p2"],
                columns="readout_mode",
                values=metric,
            )
            .reset_index()
            .sort_values(["distance", "rounds", "p2"])
        )
        if "unencode" in pivot.columns and "shrink_single" in pivot.columns:
            ratio = pivot["shrink_single"] / pivot["unencode"]
            mean_ratio = np.nanmean(ratio.values.astype(float))
            lines.append(f"- mean(shrink_single / unencode): {mean_ratio:.4f}")
        lines.append("")

    (SCRIPT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    save_diagrams()

    rows = []
    total = len(DISTANCES) * len(ROUNDS_LIST) * len(P2_LIST) * len(READOUT_MODES)
    done = 0

    for d in DISTANCES:
        for rounds in ROUNDS_LIST:
            for p2 in P2_LIST:
                for mode in READOUT_MODES:
                    done += 1
                    print(f"[{done:>2}/{total}] d={d} r={rounds} p2={p2:.4g} mode={mode}")
                    rows.append(run_one_config(d, rounds, p2, mode))

    df = pd.DataFrame(rows).sort_values(
        ["distance", "rounds", "p2", "readout_mode"], ignore_index=True
    )
    df.to_csv(SCRIPT_DIR / "state_Y_injection_results.csv", index=False)

    wide = df.pivot_table(
        index=["distance", "rounds", "p2"],
        columns="readout_mode",
        values=["ler_post", "ler_no_post", "ler_qec", "post_rate"],
    )
    wide.columns = [f"{m}_{mode}" for m, mode in wide.columns]
    wide = wide.reset_index().sort_values(["distance", "rounds", "p2"], ignore_index=True)
    wide.to_csv(SCRIPT_DIR / "state_Y_injection_wide.csv", index=False)

    plot_metric(
        df,
        metric="ler_qec",
        title="Y injection: decoded LER",
        out_name="state_Y_injection_ler_qec",
    )
    plot_metric(
        df,
        metric="ler_post",
        title="Y injection: post-selected LER",
        out_name="state_Y_injection_ler_post",
    )
    plot_metric(
        df,
        metric="post_rate",
        title="Y injection: post-selection rate",
        out_name="state_Y_injection_post_rate",
    )
    write_summary(df)
    print(f"\nSaved outputs to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
