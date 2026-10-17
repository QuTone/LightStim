#!/usr/bin/env python3
"""Compare native Pauli algebra with the current tracker's reference path.

The inverse-tableau acceleration follows Hou Yuewen's LightStim PR #110;
both modes here retain the merged tracker's role and logical-history semantics.

Run from any directory, for example::

    python /path/to/LightStim/benchmarks/tracker/benchmark_pauli_backend.py \
        --distances 3 7 11 --trials 3 --validate --output /tmp/pauli-backend.json

Each timed trial constructs a fresh unrotated memory experiment and builds its
circuit with rounds equal to distance. The decomposition cache is cleared
before every trial, including the first; there is no unreported warm-up build.
Reference and native trials run sequentially. Serialization, exact circuit
comparison, and optional DEM/noiseless checks are outside the timed interval.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import stim

from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.utils import linear_algebra, pauli_backend


BACKEND_ENV = "LIGHTSTIM_PAULI_BACKEND"
MODES = ("reference", "native")


def positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_trial(distance):
    pauli_backend._clear_basis_cache()
    # Capture protocol diagnostic output without putting it in the JSON stream.
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        experiment = MemoryExperiment(
            qec_patch=UnrotatedSurfaceCode(distance=distance),
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            rounds=distance,
            noise_params=None,
        )
        circuit = experiment.build()
        seconds = time.perf_counter() - start
    return circuit, {
        "seconds": seconds,
        "cold_basis_cache": True,
        "basis_cache_entries_after": len(pauli_backend._basis_cache),
        "basis_cache_bytes_after": int(pauli_backend._basis_cache_bytes),
    }


def validate_circuit(circuit):
    circuit.detector_error_model()
    detectors, observables = circuit.compile_detector_sampler(seed=17).sample(
        64, separate_observables=True)
    if detectors.any() or observables.any():
        raise AssertionError("Noiseless sampling produced a detector or observable flip")
    return {"dem_valid": True, "noiseless_shots": 64, "noiseless_clean": True}


def benchmark_distance(distance, trials, validate):
    reference_bytes = None
    circuit = None
    mode_results = {}
    for mode in MODES:
        os.environ[BACKEND_ENV] = mode
        measured_trials = []
        for trial in range(trials):
            circuit, measurement = build_trial(distance)
            serialized = str(circuit).encode("utf-8")
            if reference_bytes is None:
                reference_bytes = serialized
            elif serialized != reference_bytes:
                raise AssertionError(
                    f"Circuit differs from reference at d={distance}, "
                    f"mode={mode}, trial={trial + 1}")
            measurement["trial"] = trial + 1
            measured_trials.append(measurement)
            print(
                f"d={distance} {mode} trial {trial + 1}/{trials}: "
                f"{measurement['seconds']:.6f} s",
                file=sys.stderr,
                flush=True,
            )
        mode_results[mode] = {
            "median_seconds": statistics.median(
                trial["seconds"] for trial in measured_trials),
            "trials": measured_trials,
        }

    result = {
        "distance": distance,
        "rounds": distance,
        "modes": mode_results,
        "speedup_reference_over_native": (
            mode_results["reference"]["median_seconds"]
            / mode_results["native"]["median_seconds"]
        ),
        "byte_identical_circuits": True,
        "circuit_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "counts": {
            "qubits": circuit.num_qubits,
            "measurements": circuit.num_measurements,
            "detectors": circuit.num_detectors,
            "observables": circuit.num_observables,
        },
        "validation": validate_circuit(circuit) if validate else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances", type=positive_integer, nargs="+", default=[3, 7, 11])
    parser.add_argument("--trials", type=positive_integer, default=3)
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    parser.add_argument(
        "--validate", action="store_true",
        help="check DEM construction and 64 noiseless shots once per distance")
    args = parser.parse_args()

    original_mode = os.environ.get(BACKEND_ENV)
    try:
        # Fail before collecting any timings if native execution is unavailable.
        os.environ[BACKEND_ENV] = "native"
        pauli_backend._use_native()
        report = {
            "benchmark": "unrotated_memory_pauli_backend",
            "repo_root": str(REPO_ROOT),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "stim": stim.__version__,
                "lightstim_distribution": package_version("lightstim"),
            },
            "platform": platform.platform(),
            "native_pauli_available": pauli_backend._NATIVE_AVAILABLE,
            "cpp_rref_available": linear_algebra._CPP_AVAILABLE,
            "trial_count_per_mode": args.trials,
            "mode_order": list(MODES),
            "timing_scope": "experiment construction and circuit build",
            "cache_policy": "clear decomposition cache before every trial",
            "results": [
                benchmark_distance(distance, args.trials, args.validate)
                for distance in args.distances
            ],
        }
    finally:
        pauli_backend._clear_basis_cache()
        if original_mode is None:
            os.environ.pop(BACKEND_ENV, None)
        else:
            os.environ[BACKEND_ENV] = original_mode

    output = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
