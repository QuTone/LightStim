"""Rotated surface-code data-defect protocol tests."""

import numpy as np
import pymatching
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.rotated_surface_defect import (
    RotatedSurfaceDefectMemoryExperiment,
)
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode


def _uniform_noise(p: float = 1e-3) -> NoiseConfig:
    return NoiseConfig(
        p_idle=p,
        p_1q=p,
        p_2q=p,
        p_meas=p,
        p_reset=p,
    )


def _qubit_targets(instruction) -> set[int]:
    return {
        target.value
        for target in instruction.targets_copy()
        if target.is_qubit_target
    }


def test_disabling_data_qubit_preserves_canonical_stabilizers():
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="patch")
    defect_qubit = system.index_map[(3, 3)]
    canonical_supports = [
        tuple(stabilizer["data_indices"])
        for stabilizer in system.stabilizers
    ]

    affected = system.disable_data_qubits({defect_qubit})

    assert len(affected) == 4
    assert [
        tuple(stabilizer["data_indices"])
        for stabilizer in system.stabilizers
    ] == canonical_supports
    assert all(
        defect_qubit not in system.effective_stabilizer(uid)["data_indices"]
        for uid in affected
    )
    assert {
        len(system.effective_stabilizer(uid)["data_indices"])
        for uid in affected
    } == {3}


@pytest.mark.parametrize(
    ("memory_basis", "defect_basis"),
    [("Z", "Z"), ("Z", "X"), ("X", "Z"), ("X", "X")],
)
def test_defect_protocol_is_deterministic(memory_basis, defect_basis):
    experiment = RotatedSurfaceDefectMemoryExperiment(
        distance=3,
        memory_basis=memory_basis,
        defect_measure_basis=defect_basis,
    )
    circuit = experiment.build()
    detections, observables = circuit.compile_detector_sampler().sample(
        shots=100,
        separate_observables=True,
    )

    assert not np.any(detections)
    assert not np.any(observables)
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1
    assert experiment.gauge_schedule == ("Z", "X", "Z")
    circuit.detector_error_model(decompose_errors=True)


@pytest.mark.parametrize(("distance", "expected_distance"), [(3, 2), (5, 4)])
def test_noisy_defect_protocol_is_graphlike_and_decodable(
    distance,
    expected_distance,
):
    circuit = RotatedSurfaceDefectMemoryExperiment(
        distance=distance,
        noise_params=_uniform_noise(),
    ).build()
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    detections, observables = circuit.compile_detector_sampler().sample(
        shots=32,
        separate_observables=True,
    )
    predictions = matching.decode_batch(detections)

    assert predictions.shape == observables.shape
    assert len(circuit.shortest_graphlike_error()) == expected_distance


def test_defect_qubit_is_unused_after_mid_circuit_readout():
    experiment = RotatedSurfaceDefectMemoryExperiment(
        distance=3,
        noise_params=_uniform_noise(),
    )
    circuit = experiment.build().flattened()
    defect_qubit = experiment.defect_qubit
    measured = False
    saw_pre_defect_idle_noise = False

    for instruction in circuit:
        targets = _qubit_targets(instruction)
        if (
            not measured
            and instruction.name == "DEPOLARIZE1"
            and defect_qubit in targets
        ):
            saw_pre_defect_idle_noise = True
        if (
            instruction.name in {"M", "MX"}
            and defect_qubit in targets
        ):
            measured = True
            continue
        if not measured:
            continue
        assert not (
            instruction.name in {"CX", "CY", "CZ", "CNOT"}
            and defect_qubit in targets
        )
        assert not (
            instruction.name
            in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}
            and defect_qubit in targets
        )

    assert measured
    assert saw_pre_defect_idle_noise


def test_explicit_gauge_schedule_and_coordinate_validation():
    experiment = RotatedSurfaceDefectMemoryExperiment(
        distance=3,
        post_defect_schedule=("X", "X", "Z"),
    )
    experiment.build()
    assert experiment.gauge_schedule == ("X", "X", "Z")

    with pytest.raises(ValueError, match="not a data-qubit coordinate"):
        RotatedSurfaceDefectMemoryExperiment(
            distance=3,
            defect_coord=(2, 2),
        ).build()

    with pytest.raises(ValueError, match="must not be empty"):
        RotatedSurfaceDefectMemoryExperiment(
            distance=3,
            post_defect_schedule=(),
        )
