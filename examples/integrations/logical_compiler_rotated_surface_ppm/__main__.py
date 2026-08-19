import argparse
import json
from pathlib import Path

from lightstim.noise.config import NoiseConfig

from .adapter import (
    DEFAULT_PROGRAM,
    CompilationResult,
    compile_program,
    load_program,
)

DEFAULT_OUTPUT_DIR = Path("build/examples/logical_compiler_rotated_surface_ppm")


def _evaluation_circuit(result: CompilationResult, physical_error_rate: float):
    if not 0 <= physical_error_rate <= 1:
        raise ValueError("physical_error_rate must be in [0, 1]")
    if physical_error_rate == 0:
        return result.circuit
    noise = NoiseConfig(
        p_1q=physical_error_rate,
        p_2q=physical_error_rate,
        p_meas=physical_error_rate,
        p_reset=physical_error_rate,
        p_idle=physical_error_rate,
    )
    return result.experiment.builder.build_noisy_circuit(
        noise_params=noise,
        noise_model="circuit_level",
    )


def build_manifest(
    result: CompilationResult,
    physical_error_rate: float = 0,
) -> dict:
    circuit = _evaluation_circuit(result, physical_error_rate)
    detectors, observables = result.circuit.compile_detector_sampler(
        seed=0).sample(32, separate_observables=True)
    operations = []
    for index, plan in enumerate(result.experiment.plans):
        certificate = plan.certificate
        records = result.experiment.ppm_outcomes[index].records_post_split
        operations.append({
            "kind": plan.kind,
            "schedule": plan.schedule,
            "certificate_ok": None if certificate is None else certificate.ok,
            "exact_product": (
                None if certificate is None
                else certificate.measures_exactly_the_product
            ),
            "result_records": None if records is None else list(records),
        })
    return {
        "execution": {
            "distance": result.experiment.patches[0].distance,
            "rounds": result.experiment.rounds,
            "physical_error_rate": physical_error_rate,
        },
        "operations": operations,
        "circuit": {
            "num_qubits": circuit.num_qubits,
            "num_ticks": circuit.num_ticks,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
        },
        "noiseless_check": not detectors.any() and not observables.any(),
    }


def write_artifacts(
    result: CompilationResult,
    output_dir: Path,
    physical_error_rate: float = 1e-3,
) -> tuple[dict[str, Path], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "circuit": output_dir / "circuit.stim",
        "ideal_circuit": output_dir / "circuit_ideal.stim",
        "dem": output_dir / "circuit.dem",
        "manifest": output_dir / "manifest.json",
    }
    circuit = _evaluation_circuit(result, physical_error_rate)
    dem = circuit.detector_error_model()
    paths["circuit"].write_text(f"{circuit}", encoding="utf-8")
    paths["ideal_circuit"].write_text(f"{result.circuit}", encoding="utf-8")
    paths["dem"].write_text(f"{dem}", encoding="utf-8")

    manifest = build_manifest(result, physical_error_rate)
    manifest["dem"] = {
        "num_errors": dem.num_errors,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a placed logical PPM program with LightStim.")
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    args = parser.parse_args()

    result = compile_program(
        load_program(args.program),
        distance=args.distance,
        rounds=args.rounds,
    )
    paths, manifest = write_artifacts(
        result, args.output_dir, args.physical_error_rate)

    print(f"program: {args.program.name}")
    for index, (operation, plan) in enumerate(zip(
            manifest["operations"], result.experiment.plans)):
        target = " ".join(
            f"{pauli}_{patch}" for patch, pauli in plan.request.targets)
        print(
            f"  ppm[{index}]: M({target}) -> "
            f"{operation['kind']}, {operation['schedule']}, "
            f"certificate={'PASS' if operation['certificate_ok'] else 'N/A'}"
        )
    circuit = manifest["circuit"]
    print(
        f"circuit: {circuit['num_qubits']} qubits, "
        f"{circuit['num_ticks']} ticks, "
        f"{circuit['num_detectors']} detectors, "
        f"{circuit['num_observables']} observables"
    )
    print(
        "noiseless validation: "
        f"{'PASS' if manifest['noiseless_check'] else 'FAIL'}")
    print(f"artifacts: {args.output_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name}: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
