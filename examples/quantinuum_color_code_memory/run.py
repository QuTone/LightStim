"""Simulate the six Pauli inputs and plot the comparison using LightStim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline

from .circuit import NOISE_MODELS, build_circuit, noise_config


DECODER_PARAMS = dict(det_beam=20, beam_climbing=True, no_revisit_dets=True,
                      merge_errors=True, pqlimit=1_000_000, num_det_orders=21,
                      seed=2384753)


def run_case(*, basis="Z", sign=1, rounds=20, noise_model="circuit_level",
             p=0.001, shots=1000, workers=1, batch_size=1000):
    noise = noise_config(noise_model, p)
    circuit = build_circuit(rounds=rounds, basis=basis, sign=sign,
                            noise=noise, noise_model=noise_model)
    stats = SimulationPipeline(
        decoder_config=DecoderConfig("tesseract", params=DECODER_PARAMS,
                                     on_decode_failure="error"),
        max_shots=shots, max_errors=shots + 1, batch_size=min(batch_size, shots),
        num_workers=workers, print_progress=False,
    ).run(circuit)
    if stats.shots != shots or stats.post_selected_shots != shots:
        raise RuntimeError("Expected all requested shots, without postselection")
    return dict(
        basis=basis, sign=sign, rounds=rounds, noise_model=noise_model,
        shots=stats.shots, errors=stats.errors, accepted=stats.post_selected_shots,
        circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
    )


def exact_interval(errors, shots, alpha=0.05):
    return (0 if errors == 0 else beta.ppf(alpha / 2, errors, shots - errors + 1),
            1 if errors == shots else beta.ppf(1 - alpha / 2, errors + 1, shots - errors))


def summarize(cases):
    expected = {(basis, sign) for basis in "XYZ" for sign in (1, -1)}
    if len(cases) != 6 or {(c["basis"], c["sign"]) for c in cases} != expected:
        raise ValueError("A six-state average requires exactly one result for each signed Pauli state")
    if {c["rounds"] for c in cases} != {20}:
        raise ValueError("The hardware comparison uses 20 memory rounds")
    # Simultaneous bounds remain valid when cases share sampler seeds.
    intervals = np.array([exact_interval(c["errors"], c["shots"], 0.05 / 6)
                          for c in cases]) / 20.5
    rate = np.mean([c["errors"] / c["shots"] for c in cases]) / 20.5
    return float(rate), intervals.mean(axis=0)


def plot_comparison(report):
    import matplotlib.pyplot as plt
    from lightstim.plot.styles import PALETTE, apply_paper_style, bold_ticks

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    points = [summarize([c for c in report["cases"] if c["noise_model"] == model])
              for model in NOISE_MODELS]
    # Paper Table 1: 178 failures in 15,840 d=3 shots; normalization r + 1/2.
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
    parser.add_argument("--shots", type=int, default=1000, help="shots per signed Pauli state")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("build/examples/quantinuum_color_code_memory/results.json"))
    args = parser.parse_args()
    if args.shots < 1 or args.workers < 1:
        parser.error("shots and workers must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        paper="https://arxiv.org/abs/2610.24077v1", distance=3, rounds=20,
        noise={model: asdict(noise_config(model)) for model in NOISE_MODELS},
        decoder=dict(name="tesseract", params=DECODER_PARAMS),
        sampling=dict(workers=args.workers, batch_size=min(1000, args.shots),
                      source="physical circuit", seed=0 if args.workers == 1 else "process-dependent"),
        versions={name: version(name) for name in ("stim", "numpy", "tesseract-decoder")},
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("circuit.py", "run.py")},
        cases=[],
    )
    for model in NOISE_MODELS:
        for basis in "XYZ":
            for sign in (1, -1):
                case = run_case(basis=basis, sign=sign, noise_model=model,
                                shots=args.shots, workers=args.workers)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{model} {basis}{sign:+}: {case['errors']}/{case['shots']}", flush=True)
    figure = plot_comparison(report)
    figure.savefig(args.output.with_name("comparison.png"), dpi=200)


if __name__ == "__main__":
    main()
