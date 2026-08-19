import json
from dataclasses import fields

import pymatching
import pytest

from examples.integrations.logical_compiler_rotated_surface_ppm import (
    DEFAULT_PROGRAM,
    compile_program,
    load_program,
)
from examples.integrations.logical_compiler_rotated_surface_ppm.__main__ import (
    build_manifest,
    write_artifacts,
)
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated.ppm import UnsupportedPauliError


def test_sequential_program_compiles_deterministically():
    program = load_program(DEFAULT_PROGRAM)
    first = compile_program(program)
    second = compile_program(program)

    assert str(first.circuit) == str(second.circuit)
    assert first.circuit.detector_error_model().num_errors == 0
    assert [field.name for field in fields(first)] == ["circuit", "experiment"]
    manifest = build_manifest(first)
    assert manifest == build_manifest(second)
    assert manifest["noiseless_check"]
    for operation in manifest["operations"]:
        assert operation["certificate_ok"]
        assert operation["exact_product"]
        assert operation["result_records"] is not None


def test_program_schema_uses_explicit_stabilizer_frames():
    program = load_program(DEFAULT_PROGRAM)
    program["patches"]["A"]["colour_swapped"] = True

    with pytest.raises(ValueError, match="stabilizer_frame is optional"):
        compile_program(program)

    program = load_program(DEFAULT_PROGRAM)
    program["patches"]["A"]["stabilizer_frame"] = "flipped-ish"
    with pytest.raises(ValueError, match="standard or conjugate"):
        compile_program(program)

    program = load_program(DEFAULT_PROGRAM)
    program["operations"][1]["id"] = "source-op-1"
    with pytest.raises(ValueError, match="one or more PPMs"):
        compile_program(program)


def test_y_target_is_rejected_explicitly():
    program = load_program(DEFAULT_PROGRAM)
    program["operations"][1]["ppm"][0][1] = "Y"

    with pytest.raises(UnsupportedPauliError, match="twist defect or Y-wall"):
        compile_program(program)


def test_example_writes_circuit_dem_and_manifest(tmp_path):
    result = compile_program(load_program(DEFAULT_PROGRAM))
    paths, _ = write_artifacts(result, tmp_path)

    assert paths["circuit"].read_text().startswith("QUBIT_COORDS")
    assert paths["ideal_circuit"].read_text().startswith("QUBIT_COORDS")
    assert paths["ideal_circuit"].read_text() != paths["circuit"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["dem"]["num_errors"] > 0


def test_weight3_program_is_one_true_multi_patch_measurement():
    path = DEFAULT_PROGRAM.with_name("weight3_ppm.json")
    result = compile_program(load_program(path))
    plan = result.experiment.plans[0]
    operation = build_manifest(result)["operations"][0]

    assert plan.request.targets == (
        ("q1", "Z"),
        ("q2", "Z"),
        ("q3", "Z"),
    )
    assert operation["schedule"] == "diagonal"
    assert operation["exact_product"]
    assert plan.certificate.items["no_subjoint"]


def test_checkerboard_3x3_program_compiles_three_distant_ppms():
    path = DEFAULT_PROGRAM.with_name("checkerboard_3x3_ppm.json")
    program = load_program(path)
    result = compile_program(program, rounds=1)
    plans = result.experiment.plans

    assert len(result.experiment.patches) == 9
    assert result.experiment.rounds == 1
    assert result.experiment.colour_swapped == frozenset({"q20"})
    assert [len(plan.request.targets) for plan in plans] == [2, 2, 3]
    assert [tuple(pauli for _, pauli in plan.request.targets) for plan in plans] == [
        ("Z", "Z"),
        ("X", "X"),
        ("X", "Z", "X"),
    ]
    assert [len(plan.request.route) for plan in plans] == [5, 3, 5]
    assert all(plan.kind == "corridor" for plan in plans)
    assert all(plan.schedule == "diagonal" for plan in plans)
    assert not any(plan.has_stretched_checks for plan in plans)
    assert all(plan.certificate.measures_exactly_the_product for plan in plans)

    cells = {
        name: tuple(spec["cell"])
        for name, spec in program["patches"].items()
    }
    route_sets = []
    for operation in program["operations"][1:-1]:
        target_cells = [cells[name] for name, _ in operation["ppm"]]
        assert all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) > 2
            for index, a in enumerate(target_cells)
            for b in target_cells[index + 1:]
        )
        route = {tuple(cell) for cell in operation["route"]}
        assert all(
            sum(
                abs(target[0] - cell[0]) + abs(target[1] - cell[1]) == 1
                for cell in route
            ) == 1
            for target in target_cells
        )
        route_sets.append(route)
    assert all(
        first.isdisjoint(second)
        for index, first in enumerate(route_sets)
        for second in route_sets[index + 1:]
    )

    detectors, observables = result.circuit.compile_detector_sampler(
        seed=0
    ).sample(32, separate_observables=True)
    assert not detectors.any()
    assert not observables.any()
    assert result.circuit.detector_error_model().num_errors == 0

    noise = NoiseConfig(
        p_1q=1e-3,
        p_2q=1e-3,
        p_meas=1e-3,
        p_reset=1e-3,
        p_idle=1e-3,
    )
    noisy = result.experiment.builder.build_noisy_circuit(
        noise_params=noise,
        noise_model="circuit_level",
    )
    dem = noisy.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0
    component_arity = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        for target in instruction.targets_copy():
            if target.is_separator():
                assert component_arity <= 2
                component_arity = 0
            elif target.is_relative_detector_id():
                component_arity += 1
        assert component_arity <= 2
        component_arity = 0

    matcher = pymatching.Matching.from_detector_error_model(
        dem,
        enable_correlations=True,
    )
    detectors, observables = noisy.compile_detector_sampler(seed=1).sample(
        16,
        separate_observables=True,
    )
    predictions = matcher.decode_batch(detectors, enable_correlations=True)
    assert predictions.shape == observables.shape
    assert len(dem.shortest_graphlike_error()) == 3
