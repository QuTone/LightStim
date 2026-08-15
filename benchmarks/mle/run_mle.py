"""Reproduce the core exact-MLE optimization benchmark.

The optimized ACG-ALP/RPC path and a direct SciPy MILP decode the same fixed
syndromes. The output records objective agreement, parity validity, timing,
and package versions as JSON so results can be archived and compared.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import stim
from scipy.optimize import LinearConstraint, milp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices
from lightstim.simulation.decoder_backend.registry import get_decoder


PRESETS = {
    "surface-d5": {"kind": "surface", "distance": 5, "rounds": 5,
                   "p": 1e-3, "shots": 20, "seed": 11},
    "bb-72-r2": {"kind": "bb", "rounds": 2, "p": 1e-3,
                  "shots": 5, "seed": 7},
}


def _surface_dem(distance: int, rounds: int, p: float) -> stim.DetectorErrorModel:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )
    return circuit.detector_error_model(
        decompose_errors=False, flatten_loops=True
    )


def _bb_dem(rounds: int, p: float) -> stim.DetectorErrorModel:
    code = BBCode(
        l=6, m=6,
        A=[[3, 0], [0, 1], [0, 2]],
        B=[[0, 3], [1, 0], [2, 0]],
    )
    system = QECSystem()
    system.add_patch(code, name="bb")
    circuit = MemoryExperiment(
        qec_system=system,
        extraction_block_class=BBCodeExtractionBlock,
        rounds=rounds,
        noise_params=NoiseConfig(
            p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p
        ),
        noise_model="circuit_level",
        basis="Z",
    ).build()
    return circuit.detector_error_model(
        decompose_errors=False, flatten_loops=True
    )


def _build_dem(config: dict) -> stim.DetectorErrorModel:
    if config["kind"] == "surface":
        return _surface_dem(
            config["distance"], config["rounds"], config["p"]
        )
    return _bb_dem(config["rounds"], config["p"])


def _direct_milp(decoder, syndrome: np.ndarray) -> tuple[np.ndarray, float]:
    sf = syndrome.astype(float)
    result = milp(
        c=decoder._c,
        constraints=LinearConstraint(decoder._A, sf, sf),
        integrality=decoder._integrality,
        bounds=decoder._bounds,
    )
    if not result.success:
        raise RuntimeError(f"direct MILP failed: {result.message}")
    correction = np.round(result.x[:decoder._n]).astype(np.uint8)
    return correction, float(result.fun)


def run_preset(name: str, shots: int | None = None) -> dict:
    """Run one deterministic preset and return its JSON-ready result."""
    config = dict(PRESETS[name])
    shot_count = config["shots"] if shots is None else shots
    if shot_count < 1:
        raise ValueError("shots must be positive")
    config["shots"] = shot_count

    dem = _build_dem(config)
    H, _, priors = dem_to_matrices(dem, sparse=True, merge_duplicates=True)
    detectors, _, _ = dem.compile_sampler(seed=config["seed"]).sample(
        shots=shot_count
    )
    syndromes = detectors.astype(np.uint8)

    decoder = get_decoder("mle-ilp", backend="cpu")
    decoder.setup(H=H, priors=priors)

    optimized_costs = []
    started = time.perf_counter()
    for syndrome in syndromes:
        correction, ok = decoder.decode_single(syndrome)
        if not ok:
            raise RuntimeError("unlimited optimized MLE unexpectedly failed")
        if not np.array_equal((H @ correction) % 2, syndrome):
            raise RuntimeError("optimized MLE returned an invalid correction")
        optimized_costs.append(float(decoder._w @ correction))
    optimized_seconds = time.perf_counter() - started

    direct_costs = []
    started = time.perf_counter()
    for syndrome in syndromes:
        correction, cost = _direct_milp(decoder, syndrome)
        if not np.array_equal((H @ correction) % 2, syndrome):
            raise RuntimeError("direct MILP returned an invalid correction")
        direct_costs.append(cost)
    direct_seconds = time.perf_counter() - started

    optimized_costs = np.asarray(optimized_costs)
    direct_costs = np.asarray(direct_costs)
    max_cost_delta = float(np.max(np.abs(optimized_costs - direct_costs)))
    return {
        "preset": name,
        "config": config,
        "shots": shot_count,
        "num_detectors": int(H.shape[0]),
        "num_error_mechanisms": int(H.shape[1]),
        "parity_valid": True,
        "objective_agreement": bool(max_cost_delta <= 1e-6),
        "max_objective_delta": max_cost_delta,
        "optimized_seconds": optimized_seconds,
        "direct_milp_seconds": direct_seconds,
        "optimized_ms_per_shot": 1e3 * optimized_seconds / shot_count,
        "direct_milp_ms_per_shot": 1e3 * direct_seconds / shot_count,
        "speedup": direct_seconds / optimized_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", nargs="+", choices=sorted(PRESETS),
        default=list(PRESETS), help="Preset(s) to run (default: all)",
    )
    parser.add_argument(
        "--shots", type=int, default=None,
        help="Override the shot count in every selected preset",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run two surface-d5 shots for a fast correctness smoke test",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    presets = ["surface-d5"] if args.quick else args.preset
    shots = 2 if args.quick else args.shots
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "stim": stim.__version__,
        },
        "results": [run_preset(name, shots=shots) for name in presets],
    }
    if not all(result["objective_agreement"] for result in report["results"]):
        raise RuntimeError("optimized and direct MILP objectives disagree")

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
