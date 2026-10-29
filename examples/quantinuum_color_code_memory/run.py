"""Run the six Pauli inputs through the standard LightStim simulation pipeline."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

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
    start = time.perf_counter()
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
        noise={key: noise.get(key) for key in ("p_1q", "p_2q", "p_meas", "p_reset", "p_idle")},
        shots=stats.shots, errors=stats.errors, accepted=stats.post_selected_shots,
        normalized_rate=stats.errors / stats.shots / (rounds + 0.5),
        decoder="tesseract", decoder_params=DECODER_PARAMS,
        workers=workers, batch_size=min(batch_size, shots),
        sampling="physical circuit; pipeline seed 0" if workers == 1
                 else "physical circuit; pipeline process-dependent seeds",
        seconds=time.perf_counter() - start,
        circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
        num_detectors=circuit.num_detectors, num_observables=circuit.num_observables,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000, help="shots per signed Pauli state")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=NOISE_MODELS, default=list(NOISE_MODELS))
    parser.add_argument("--output", type=Path,
                        default=Path("build/examples/quantinuum_color_code_memory/results.json"))
    args = parser.parse_args()
    if args.shots < 1 or args.workers < 1:
        parser.error("shots and workers must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        paper="https://arxiv.org/abs/2610.24077v1", distance=3, rounds=20,
        scope="Published circuit reconstruction under specified Pauli noise, not the Helios emulator",
        versions={name: version(name) for name in ("stim", "numpy", "tesseract-decoder")},
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("circuit.py", "run.py")},
        cases=[],
    )
    for model in args.models:
        for basis in "XYZ":
            for sign in (1, -1):
                case = run_case(basis=basis, sign=sign, noise_model=model,
                                shots=args.shots, workers=args.workers)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{model} {basis}{sign:+}: {case['errors']}/{case['shots']}, "
                      f"rate={case['normalized_rate']:.3g}", flush=True)


if __name__ == "__main__":
    main()
