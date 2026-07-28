"""Dynamical logical S/S† on the rotated surface code.

The protocol is the mid-cycle fold-transversal construction from
arXiv:2412.01391.  These tests cover both its physical schedule and the
Builder/Tracker integration added for retained-data Color Code circuits.
"""

import pathlib
import sys

import numpy as np
import pytest
import stim

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import assert_dem_valid, assert_noiseless, assert_valid_circuit
from benchmarks.logical_ops.run_logical_ops import (
    _decoder_config,
    build_tasks,
)
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.rotated_logical_s import (
    build_rotated_s_two_way_circuit,
    build_rotated_s_y_injection_circuit,
)
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)
from lightstim.qec_code.surface_code.rotated.operation import (
    _build_half_cycle_fold_s,
    _get_half_cycle_fold_yx_parts,
    _insert_fold_after_second_cnot_layer,
)


def _make_system(distance=3, offset=(0.0, 0.0)):
    system = QECSystem()
    patch = system.add_patch(
        RotatedSurfaceCode(distance=distance),
        name="patch",
        offset=offset,
    )
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    op_set = RotatedSurfaceCodeLogicalOpSet(
        extraction_block_class=RotatedSurfaceCodeExtractionBlock
    )
    return system, patch, tracker, builder, op_set


def _logical_vector(patch, pauli_type, num_qubits):
    vector = np.zeros(2 * num_qubits, dtype=np.uint8)
    logical = next(
        logical
        for logical in patch.logical_ops
        if logical["type"] == pauli_type
    )
    for qubit, pauli in logical["pauli"].items():
        if pauli in ("X", "Y"):
            vector[qubit] = 1
        if pauli in ("Z", "Y"):
            vector[num_qubits + qubit] = 1
    return vector


def _symplectic_product(lhs, rhs, num_qubits):
    return int(
        (
            lhs[:num_qubits] @ rhs[num_qubits:]
            + lhs[num_qubits:] @ rhs[:num_qubits]
        )
        % 2
    )


def _observable_reference_parity(circuit):
    """Return raw noiseless parity of observable zero's annotations."""
    measurement_count = 0
    observable_measurements = []
    for instruction in circuit.flattened():
        if instruction.name == "OBSERVABLE_INCLUDE":
            if instruction.gate_args_copy() == [0.0]:
                observable_measurements.extend(
                    measurement_count + target.value
                    for target in instruction.targets_copy()
                    if target.is_measurement_record_target
                )
            continue

        single_instruction = stim.Circuit()
        single_instruction.append(
            instruction.name,
            instruction.targets_copy(),
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
        measurement_count += single_instruction.num_measurements

    reference = circuit.reference_sample()
    return int(reference[observable_measurements].sum() % 2)


def _detector_measurement_qubits(circuit):
    measurement_qubits = []
    detector_qubits = []
    measurement_gates = {
        "M",
        "MZ",
        "MX",
        "MY",
        "MR",
        "MRZ",
        "MRX",
        "MRY",
    }

    for instruction in circuit.flattened():
        if instruction.name in measurement_gates:
            measurement_qubits.extend(
                target.value
                for target in instruction.targets_copy()
                if target.is_qubit_target
            )
        elif instruction.name == "DETECTOR":
            detector_qubits.append({
                measurement_qubits[len(measurement_qubits) + target.value]
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            })

    return detector_qubits


@pytest.mark.smoke
@pytest.mark.parametrize("offset", [(0.0, 0.0), (10.0, 20.0)])
def test_half_cycle_fold_uses_data_and_syndrome_qubits(offset):
    system, patch, _, _, _ = _make_system(distance=3, offset=offset)
    diagonal_data, diagonal_syndromes, mirror_pairs = (
        _get_half_cycle_fold_yx_parts(system, patch)
    )
    dx, dy = offset

    assert {
        system.qubit_coords[qubit]
        for qubit in diagonal_data
    } == {
        (1 + dx, 1 + dy),
        (3 + dx, 3 + dy),
        (5 + dx, 5 + dy),
    }
    assert {
        system.qubit_coords[qubit]
        for qubit in diagonal_syndromes
    } == {
        (2 + dx, 2 + dy),
        (4 + dx, 4 + dy),
    }
    assert any(
        a in system.active_syndrome_indices
        or b in system.active_syndrome_indices
        for a, b in mirror_pairs
    )

    fold = _build_half_cycle_fold_s(system, patch, inverse=False)
    assert [instruction.name for instruction in fold] == [
        "CZ",
        "S",
        "S_DAG",
    ]


@pytest.mark.smoke
def test_fold_is_inserted_between_cnot_layers_two_and_three():
    system, patch, _, _, _ = _make_system(distance=3)
    ordinary = RotatedSurfaceCodeExtractionBlock(system).circuit
    fold = _build_half_cycle_fold_s(system, patch, inverse=False)
    dynamical = _insert_fold_after_second_cnot_layer(ordinary, fold)

    cnot_layers = 0
    cnot_layers_at_fold = []
    for instruction in dynamical:
        if instruction.name in ("CX", "CNOT"):
            cnot_layers += 1
        elif instruction.name in ("CZ", "S", "S_DAG"):
            cnot_layers_at_fold.append(cnot_layers)

    assert cnot_layers == 4
    assert cnot_layers_at_fold == [2, 2, 2]


@pytest.mark.smoke
@pytest.mark.parametrize("distance", [3, 5])
def test_tracker_accepts_mid_cycle_cz_and_tracks_logical_s(distance):
    system, patch, tracker, builder, op_set = _make_system(distance=distance)
    builder.write_coordinates()
    builder.initialize(
        {qubit: "X" for qubit in patch.data_indices},
        system.num_qubits,
    )
    ordinary = RotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(ordinary.circuit, rounds=2)

    op_set.fold_transversal_s(builder, patch)

    num_qubits = system.num_qubits
    logical_x = _logical_vector(patch, "X", num_qubits)
    logical_z = _logical_vector(patch, "Z", num_qubits)
    tracked_logical = tracker.logicals.matrix[0]

    # A logical Y representative anti-commutes with both canonical X_L and Z_L.
    assert _symplectic_product(
        tracked_logical, logical_x, num_qubits
    ) == 1
    assert _symplectic_product(
        tracked_logical, logical_z, num_qubits
    ) == 1

    # The following ordinary round must also be processed correctly: its X
    # detectors inherit the X/Z record mixing created by the S-SE round.
    builder.apply_syndrome_extraction(ordinary.circuit, rounds=1)
    op_set.fold_transversal_s_dag(builder, patch)
    builder.apply_data_readout(
        {qubit: "X" for qubit in patch.data_indices}
    )

    assert_valid_circuit(builder.circuit)
    assert_noiseless(builder.circuit)
    assert_dem_valid(builder.circuit)
    assert _observable_reference_parity(builder.circuit) == 0
    assert any(
        qubits & set(system.active_syndrome_indices_x)
        and qubits & set(system.active_syndrome_indices_z)
        for qubits in _detector_measurement_qubits(builder.circuit)
    )


@pytest.mark.smoke
def test_s_squared_has_the_logical_z_phase():
    system, patch, _, builder, op_set = _make_system(distance=3)
    builder.initialize(
        {qubit: "X" for qubit in patch.data_indices},
        system.num_qubits,
    )
    ordinary = RotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(ordinary.circuit, rounds=2)

    op_set.fold_transversal_s(builder, patch)
    op_set.fold_transversal_s(builder, patch)
    builder.apply_data_readout(
        {qubit: "X" for qubit in patch.data_indices}
    )

    # S^2|+> = Z|+> = |->. The tracker is intentionally phase-free, while
    # Stim's raw reference sample retains this deterministic logical sign.
    assert _observable_reference_parity(builder.circuit) == 1
    assert_noiseless(builder.circuit)


@pytest.mark.smoke
def test_noisy_dynamical_s_round_builds_hypergraph_dem():
    system, patch, _, builder, op_set = _make_system(distance=3)
    builder.initialize(
        {qubit: "X" for qubit in patch.data_indices},
        system.num_qubits,
    )
    ordinary = RotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(ordinary.circuit, rounds=3)
    op_set.fold_transversal_s(builder, patch)
    builder.apply_syndrome_extraction(ordinary.circuit, rounds=2)
    op_set.fold_transversal_s_dag(builder, patch)
    builder.apply_data_readout(
        {qubit: "X" for qubit in patch.data_indices}
    )

    noise = NoiseConfig(
        p_1q=1e-3,
        p_2q=1e-3,
        p_meas=1e-3,
        p_reset=1e-3,
        p_idle=1e-3,
    )
    noisy = builder.build_noisy_circuit(
        noise,
        noise_model="circuit_level",
    )
    dem = noisy.detector_error_model()

    assert dem.num_detectors == noisy.num_detectors
    assert dem.num_observables == noisy.num_observables == 1
    assert "error(" in str(dem)
    assert "DEPOLARIZE2" in str(noisy)


@pytest.mark.smoke
@pytest.mark.parametrize("distance", [3, 5])
def test_rotated_s_two_way_experiment_is_noiseless(distance):
    circuit = build_rotated_s_two_way_circuit(
        distance=distance,
        rounds=2,
    )

    assert_valid_circuit(circuit)
    assert_noiseless(circuit)
    assert circuit.detector_error_model().num_observables == 1
    assert _observable_reference_parity(circuit) == 0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("gate", "expected_reference_parity"),
    [("S_DAG", 0), ("S", 1)],
)
def test_noiseless_y_injection_has_deterministic_x_result(
    gate,
    expected_reference_parity,
):
    circuit = build_rotated_s_y_injection_circuit(
        distance=3,
        gate=gate,
        padding_rounds=2,
    )

    assert_valid_circuit(circuit)
    assert_noiseless(circuit)
    assert _observable_reference_parity(circuit) == expected_reference_parity
    assert all(
        instruction.name != "Z"
        for instruction in circuit.flattened()
    )


@pytest.mark.smoke
def test_y_injection_noise_is_confined_to_target_s_round():
    p = 1e-3
    circuit = build_rotated_s_y_injection_circuit(
        distance=3,
        gate="S_DAG",
        padding_rounds=2,
        noise_params=NoiseConfig(
            p_1q=p,
            p_2q=p,
            p_meas=p,
            p_reset=p,
            p_idle=p,
        ),
    )

    # The target dynamical S-SE round retains one SE_start idle marker.  Every
    # injection/padding round uses a noiseless marker and therefore receives no
    # idle noise.
    assert sum(
        instruction.name == "TICK" and instruction.tag == "SE_start"
        for instruction in circuit.flattened()
    ) == 1
    assert sum(
        instruction.name == "DEPOLARIZE1"
        for instruction in circuit.flattened()
    ) > 0
    assert sum(
        instruction.name == "DEPOLARIZE2"
        for instruction in circuit.flattened()
    ) > 0
    assert circuit.detector_error_model().num_observables == 1


@pytest.mark.smoke
def test_logical_ops_runner_builds_all_rotated_s_experiments():
    tasks = build_tasks(
        "S_rotated",
        distances=[3],
        p_values=[1e-3],
        rounds=2,
    )

    assert len(tasks) == 3
    assert {
        metadata["sub_experiment"]
        for _, metadata in tasks
    } == {
        "S_then_S_DAG",
        "S_DAG_plusY_to_X",
        "S_plusY_to_minusX",
    }
    assert {
        metadata["noisy_gate_count"]
        for _, metadata in tasks
    } == {1, 2}
    assert all(metadata["gate"] == "S_rotated" for _, metadata in tasks)
    assert all(circuit.num_observables == 1 for circuit, _ in tasks)


@pytest.mark.smoke
def test_logical_ops_runner_keeps_historical_bposd_settings():
    expected = {
        "max_iterations": 1000,
        "osd_order": 10,
        "bp_method": "min_sum",
        "ms_scaling_factor": 0,
        "osd_method": "osd_cs",
    }
    cpu = _decoder_config("cpu_bposd")
    gpu = _decoder_config("gpu_bposd")

    assert cpu.params == expected
    assert gpu.params == {**expected, "use_osd": True}
