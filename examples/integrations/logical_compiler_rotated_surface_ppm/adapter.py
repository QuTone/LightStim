"""Minimal adapter from a placed logical PPM program to LightStim."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import stim

from lightstim.protocols.rotated_surface_ppm import (
    RotatedSurfacePPMExperiment,
    RotatedSurfacePPMStep,
)
from lightstim.qec_code.surface_code.rotated.ppm import (
    RotatedSurfacePatchPlacement,
    origin_of,
)

DEFAULT_PROGRAM = Path(__file__).resolve().parent / "sequential_ppm.json"


@dataclass(frozen=True)
class CompilationResult:
    """Compiled physical circuit and its inspectable LightStim experiment."""

    circuit: stim.Circuit
    experiment: RotatedSurfacePPMExperiment


def load_program(path: Path = DEFAULT_PROGRAM) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_program(
    program: Mapping[str, Any],
    *,
    distance: int = 3,
    rounds: int | None = None,
) -> CompilationResult:
    """Compile a placed PPM program at one uniform distance and round count."""
    patch_specs = program["patches"]
    operations = program["operations"]
    patch_fields = {"cell", "orientation", "stabilizer_frame"}
    if any(
        not {"cell", "orientation"} <= set(spec)
        or set(spec) - patch_fields
        for spec in patch_specs.values()
    ):
        raise ValueError(
            "each patch requires cell and orientation; stabilizer_frame is optional"
        )
    bad_frames = {
        name: spec.get("stabilizer_frame")
        for name, spec in patch_specs.items()
        if spec.get("stabilizer_frame", "standard")
        not in {"standard", "conjugate"}
    }
    if bad_frames:
        raise ValueError(
            f"stabilizer_frame must be standard or conjugate; got {bad_frames}"
        )
    ppm_fields = {"ppm", "route"}
    if (len(operations) < 3
            or set(operations[0]) != {"prepare"}
            or set(operations[-1]) != {"measure"}
            or any("ppm" not in operation or set(operation) - ppm_fields
                   for operation in operations[1:-1])):
        raise ValueError(
            "operations must be: one prepare, one or more PPMs, one measure")
    initial_states = {
        patch: basis.upper() for patch, basis in operations[0]["prepare"].items()
    }
    final_measure_states = {
        patch: basis.upper() for patch, basis in operations[-1]["measure"].items()
    }
    if (set(initial_states) != set(patch_specs)
            or set(final_measure_states) != set(patch_specs)):
        raise ValueError("prepare and measure operations must cover every patch")

    placements = [
        RotatedSurfacePatchPlacement(
            name=name,
            origin=origin_of(*spec["cell"], distance, seam=True),
            distance=distance,
            orientation=spec["orientation"],
        )
        for name, spec in patch_specs.items()
    ]
    ppm_operations = operations[1:-1]
    steps = [
        RotatedSurfacePPMStep(
            targets=tuple(
                (patch, pauli.upper()) for patch, pauli in operation["ppm"]),
            route=tuple(tuple(cell) for cell in operation["route"]),
        )
        for operation in ppm_operations
    ]
    colour_swapped = frozenset(
        name
        for name, spec in patch_specs.items()
        if spec.get("stabilizer_frame", "standard") == "conjugate"
    )

    rounds = distance if rounds is None else rounds
    experiment = RotatedSurfacePPMExperiment(
        placements,
        steps,
        initial_states=initial_states,
        final_measure_states=final_measure_states,
        rounds=rounds,
        colour_swapped=colour_swapped,
    )
    circuit = experiment.build()

    return CompilationResult(
        circuit=circuit,
        experiment=experiment,
    )
