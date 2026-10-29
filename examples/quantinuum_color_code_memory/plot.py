"""Plot the saved six-state average with LightStim's shared paper style."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from lightstim.plot.styles import PALETTE, apply_paper_style, bold_ticks

from .circuit import NOISE_MODELS


def exact_interval(errors, shots, alpha=0.05):
    return (0 if errors == 0 else beta.ppf(alpha / 2, errors, shots - errors + 1),
            1 if errors == shots else beta.ppf(1 - alpha / 2, errors + 1, shots - errors))


def summarize(cases):
    expected = {(basis, sign) for basis in "XYZ" for sign in (1, -1)}
    if len(cases) != 6 or {(c["basis"], c["sign"]) for c in cases} != expected:
        raise ValueError("A six-state average requires exactly one result for each signed Pauli state")
    if {c["rounds"] for c in cases} != {20}:
        raise ValueError("The hardware comparison uses 20 memory rounds")
    # Simultaneous exact intervals remain valid even when runs share sampler
    # seeds. Do not pool shots and assume six independent random streams.
    intervals = np.array([exact_interval(c["errors"], c["shots"], 0.05 / 6)
                          for c in cases]) / 20.5
    rate = np.mean([c["errors"] / c["shots"] for c in cases]) / 20.5
    return float(rate), intervals.mean(axis=0)


def plot_comparison(report):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    points = [summarize([c for c in report["cases"] if c["noise_model"] == model])
              for model in NOISE_MODELS]
    # Table 1: 178 failures in 15,840 d=3 shots, normalized by r + 1/2.
    points.append((178 / 15840 / 20.5, np.array(exact_interval(178, 15840)) / 20.5))
    for x, ((rate, bounds), color, marker) in enumerate(zip(
            points, (PALETTE[0], PALETTE[1], PALETTE[7]), ("o", "s", "D"))):
        ax.errorbar(x, rate * 1e4, yerr=np.array([[rate - bounds[0]], [bounds[1] - rate]]) * 1e4,
                    fmt=marker, color=color, capsize=6)
    ax.set_xticks(range(3), ["LightStim\nCircuit level", "LightStim\nCircuit level + idling",
                             "Quantinuum\nHelios-1 (paper)"])
    ax.set_ylabel(r"Logical error / round ($\times 10^{-4}$)")
    ax.set_title("Color-code memory | d = 3, 20 rounds")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    bold_ticks(ax)
    fig.text(0.5, 0.015, "Six-state average; 95% intervals. Simulation: p = 0.001, uncalibrated Pauli noise.",
             ha="center", fontsize=10)
    fig.tight_layout(pad=0.4, rect=(0, 0.06, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/examples/quantinuum_color_code_memory/comparison.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_comparison(json.loads(args.results.read_text()))
    figure.savefig(args.output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
