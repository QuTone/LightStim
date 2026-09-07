"""Build notebook figures and distance audits from the unified memory runner CSV.

Run from the repository root:
    python -m playground.subsystem.bacon_shor_memory.make_assets --input benchmarks/memory/results/bacon_shor_pymatching.csv

This script does not run decoding. See README.md for the shared benchmark CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import stim
from scipy.stats import beta

from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from benchmarks.memory.run_memory import build_circuit
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def plot_schedule(patch, block):
    """Draw the actual four CNOT layers, using the block's qubit pairs."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), layout="constrained")
    colors = {"X": "#bf611d", "Z": "#2176ae"}
    coords = patch.qubit_coords
    for i, (basis, layer) in enumerate(zip("XXZZ", block.x_layers + block.z_layers)):
        ax = axes[i]
        for gauge in patch.gauges:
            for data in gauge["data_indices"]:
                ancilla = gauge["syn_idx"]
                ax.plot(*zip(coords[ancilla], coords[data]), color="#e1e5e9", zorder=0)
        for ancilla, data in layer:
            control, target = (ancilla, data) if basis == "X" else (data, ancilla)
            ax.annotate("", xy=coords[target], xytext=coords[control],
                        arrowprops=dict(arrowstyle="->", color=colors[basis], lw=2.7,
                                        shrinkA=10, shrinkB=10))
        for qubit, xy in coords.items():
            data = qubit in patch.data_indices
            gauge_basis = "X" if qubit in patch.syndrome_indices_x else "Z"
            ax.scatter(*xy, s=235, marker="o" if data else "s",
                       color="#273644" if data else colors[gauge_basis], zorder=3)
            ax.text(*xy, str(qubit), ha="center", va="center", color="white", fontsize=9)
        ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5), aspect="equal",
               title=f"{basis} gauge: CNOT layer {i % 2 + 1}")
        ax.axis("off")
    fig.suptitle("Bacon–Shor d=3: dedicated X then Z extraction\n"
                 "Circles: data; squares: ancillas; arrows: control → target")
    return fig


def _without_detectors(circuit):
    out = stim.Circuit()
    for op in circuit.flattened():
        if op.name != "DETECTOR":
            out.append(op)
    return out


def check_memory(full, selected, d):
    """Check projection and bound the full circuit's fault distance from both sides."""
    assert _without_detectors(full) == _without_detectors(selected)
    full_dem = full.detector_error_model(decompose_errors=False).flattened()
    dem = selected.detector_error_model(decompose_errors=False).flattened()
    for error in dem:
        if error.type == "error":
            targets = error.targets_copy()
            assert not any(t.is_separator() for t in targets)
            assert sum(t.is_relative_detector_id() for t in targets) <= 2
    lower = len(dem.shortest_graphlike_error(ignore_ungraphlike_errors=False))
    # A graphlike subset of the full model supplies an upper-bound witness,
    # not a lower bound. Verify it also has physical circuit realizations.
    witness = full_dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    detectors, observables = set(), set()
    for error in witness:
        for target in error.targets_copy():
            if target.is_relative_detector_id():
                detectors.symmetric_difference_update({target.val})
            elif target.is_logical_observable_id():
                observables.symmetric_difference_update({target.val})
            else:
                raise AssertionError("Unexpected witness target")
    assert not detectors and observables == {0}
    locations = full.explain_detector_error_model_errors(
        dem_filter=witness, reduce_to_one_representative_error=True)
    assert len(locations) == len(witness) == lower == d
    assert all(location.circuit_error_locations for location in locations)
    ideal = full.without_noise()
    assert not ideal.compile_detector_sampler(seed=71).sample(256, append_observables=True).any()
    return dict(lower_bound=lower, upper_bound=len(witness)), dem


def make_assets(input_csv: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    p = .001
    data = pd.read_csv(input_csv)
    if (len(data) != 4 or set(data.distance) != {3, 5, 7, 9}
        or not (data.code == "bacon_shor").all()
        or not (data.basis == "Z").all() or not (data.detector_basis == "Z").all()
        or not (data.noise_model == "circuit_level").all()
        or not (data.decoder_name == "pymatching").all()
        or not (data.se_circuit == "dedicated").all()
        or not (data.p == p).all() or not (data.p_idle == 0).all()
        or not (data.p_1q == 0).all() or not (data.rounds == data.distance).all()):
        raise ValueError("Use the four-distance unified-runner baseline documented in README.md.")
    rows = []
    for d in (3, 5, 7, 9):
        full, *_ = build_circuit("bacon_shor", d, p, p_idle=0, p_1q=0)
        selected, *_ = build_circuit("bacon_shor", d, p, p_idle=0, p_1q=0, detector_basis="Z")
        bounds, dem = check_memory(full, selected, d)
        full.to_file(output / f"d{d}_full.stim")
        selected.to_file(output / f"d{d}_mwpm.stim")
        dem.to_file(output / f"d{d}_mwpm.dem")
        result = data[data.distance == d].iloc[0]
        shots, errors = int(result.shots), int(result.errors)
        assert shots > 0 and 0 <= errors <= shots
        assert np.isclose(result.logical_error_rate, errors / shots)
        ci95 = [float(beta.ppf(.025, errors, shots - errors + 1)) if errors else 0.,
                float(beta.ppf(.975, errors + 1, shots - errors)) if errors < shots else 1.]
        rows.append(dict(d=d, rounds=d, p=p, shots=shots, errors=errors,
                         ler=errors / shots, ci95=ci95,
                         full_detectors=full.num_detectors, selected_detectors=selected.num_detectors,
                         circuit_distance=bounds,
                         circuit_sha256=hashlib.sha256(str(full).encode()).hexdigest(),
                         selected_circuit_sha256=hashlib.sha256(str(selected).encode()).hexdigest()))
        print(f"d={d}: {errors}/{shots} = {errors / shots:.6g}; distance={bounds}", flush=True)
    sources = [Path(__file__).relative_to(ROOT),
               Path("benchmarks/memory/run_memory.py"),
               Path("lightstim/qec_code/bacon_shor/SE_block.py"),
               Path("lightstim/qec_code/bacon_shor/code_patch.py"),
               Path("lightstim/protocols/memory.py"),
               Path("lightstim/simulation/decoder_backend/pipeline.py")]
    summary = dict(
        noise=dict(p_2q=p, p_reset=p, p_meas=p, p_idle=0., p_1q=0.,
                   model="circuit_level", final_data_readout="noisy"),
        decoder="pymatching", backend="cpu", detector_basis="Z", memory_basis="Z",
        metric="logical failure per complete memory shot", postselection=False,
        benchmark_csv=str(input_csv.relative_to(ROOT)),
        benchmark_csv_sha256=hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        sampling=dict(seed=0, batch_size=10000, workers=1, stopping="fixed shots",
                      source="README reproduction command; these settings are not encoded in the input CSV"),
        versions={name: importlib.metadata.version(name) for name in ("stim", "pymatching", "numpy", "scipy")},
        python=platform.python_version(),
        source_sha256={str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
        results=rows,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    patch = BaconShorCode(distance=3)
    system = QECSystem()
    system.add_patch(patch, name="bs")
    fig = plot_schedule(patch, BaconShorCodeExtractionBlock(system))
    fig.savefig(output / "dedicated_se.png", dpi=160)
    plt.close(fig)
    small = MemoryExperiment(qec_patch=patch, rounds=2, basis="Z").build()
    (output / "memory_d3.svg").write_text(str(small.diagram("detslice-with-ops-svg")))
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    y = np.array([row["ler"] for row in rows])
    ci = np.array([row["ci95"] for row in rows])
    ax.errorbar([row["d"] for row in rows], y, yerr=[y - ci[:, 0], ci[:, 1] - y],
                fmt="o-", capsize=4, color="#2176ae")
    ax.set(yscale="log", xticks=[3, 5, 7, 9], xlabel="d (= number of XZ pairs)",
           ylabel="Z-memory logical error rate per shot",
           title="Dedicated Bacon–Shor / CPU MWPM / p = 0.001\n"
                 "Noisy final readout; 95% exact binomial intervals")
    ax.grid(alpha=.2, which="both")
    fig.savefig(output / "mwpm.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV produced by benchmarks/memory/run_memory.py")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/native_mwpm")
    args = parser.parse_args()
    make_assets(args.input.resolve(), args.output)
