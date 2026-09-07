"""Bacon-Shor center/gauge algebra and protected-memory regressions."""

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock
from lightstim.qec_code.repetition import RepetitionCode
from lightstim.utils.linear_algebra import row_echelon


def _symplectic(records, n):
    matrix = np.zeros((len(records), 2 * n), dtype=np.uint8)
    for i, record in enumerate(records):
        for qubit, basis in record["pauli"].items():
            matrix[i, qubit] = basis in {"X", "Y"}
            matrix[i, n + qubit] = basis in {"Z", "Y"}
    return matrix


def _rank(matrix):
    return row_echelon(matrix.copy())[1]


def _commutators(left, right):
    n = left.shape[1] // 2
    return (left[:, :n] @ right[:, n:].T + left[:, n:] @ right[:, :n].T) % 2


@pytest.mark.parametrize("distance", [2, 3, 4])
def test_bacon_shor_declares_center_gauges_and_bare_logicals(distance):
    patch = BaconShorCode(distance=distance)
    n = patch.num_qubits
    gauges = _symplectic(patch.gauges, n)
    center = _symplectic(patch.stabilizers, n)
    logicals = _symplectic(patch.logical_ops, n)

    assert len(patch.data_indices) == distance**2
    assert len(patch.gauges) == 2 * distance * (distance - 1)
    assert len(patch.syndrome_indices) == len(patch.gauges)
    assert all(len(g["pauli"]) == 2 for g in patch.gauges)
    assert all(s["syn_idx"] is None for s in patch.stabilizers)
    assert _rank(gauges) == 2 * distance * (distance - 1)
    assert _rank(center) == 2 * (distance - 1)
    assert _rank(np.vstack([gauges, center])) == _rank(gauges)
    assert not _commutators(center, gauges).any()
    assert _rank(_commutators(gauges, gauges)) == 2 * (distance - 1)**2
    assert not _commutators(logicals, gauges).any()
    assert _commutators(logicals, logicals).tolist() == [[0, 1], [1, 0]]
    assert patch.num_logicals == 1
    assert all(len(op["pauli"]) == distance for op in patch.logical_ops)


@pytest.mark.parametrize("distance", [None, 0, 1, -3, 2.5, True, "3"])
def test_bacon_shor_requires_integer_distance_at_least_two(distance):
    with pytest.raises(ValueError, match="integer >= 2"):
        BaconShorCode(distance=distance)


@pytest.mark.parametrize("distance", [2, 3])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("order", [("X", "Z"), ("Z", "X")])
def test_bacon_shor_memory_infers_each_gauge_phase(distance, basis, order):
    experiment = MemoryExperiment(
        qec_patch=BaconShorCode(distance=distance),
        basis=basis,
        rounds=3,
        se_block_kwargs={"basis_order": order},
    )
    circuit = experiment.build()
    detectors, observables = circuit.compile_detector_sampler(seed=3).sample(
        256, separate_observables=True,
    )
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1
    assert not detectors.any()
    assert not observables.any()
    circuit.detector_error_model()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_bacon_shor_noisy_memory_has_detector_error_model(basis):
    circuit = MemoryExperiment(
        qec_patch=BaconShorCode(distance=3),
        basis=basis,
        rounds=3,
        noise_params=NoiseConfig(
            p_1q=0.001, p_2q=0.001, p_meas=0.001, p_reset=0.001, p_idle=0.001,
        ),
    ).build()
    dem = circuit.detector_error_model()
    assert dem.num_errors > 0
    assert dem.num_observables == 1
    assert dem.num_detectors == circuit.num_detectors


def test_bacon_shor_repeated_single_basis_phases_keep_static_declaration():
    system = QECSystem()
    system.add_patch(BaconShorCode(distance=3), name="bs")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)
    active_center = set(system.active_stabilizer_indices)
    center_supports = [dict(s["pauli"]) for s in system.stabilizers]

    for order in [("X", "Z"), ("Z",), ("Z",), ("X",), ("Z",)]:
        block = GenericCSSGaugeExtractionBlock(system, basis_order=order)
        builder.apply_syndrome_extraction(
            block.circuit, rounds=1, measurement_blocks=block.measurement_blocks,
        )
        assert tracker.logicals.count == 1
        assert tracker.expected_num_logicals == 1
        assert system.active_stabilizer_indices == active_center
        assert [dict(s["pauli"]) for s in system.stabilizers] == center_supports

    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    samples = builder.circuit.compile_detector_sampler(seed=4).sample(256, append_observables=True)
    assert not samples.any()
    builder.circuit.detector_error_model()


def _physical_operations(circuit):
    physical = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}:
            physical.append(instruction)
    return physical


def _affine_annotation_rows(circuit):
    """Record parities augmented by their noiseless reference values."""
    reference = circuit.reference_sample().astype(np.uint8)
    detectors = []
    observable = np.zeros(circuit.num_measurements, dtype=np.uint8)
    count = 0
    for instruction in circuit.flattened():
        if instruction.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            row = np.zeros(circuit.num_measurements, dtype=np.uint8)
            for target in instruction.targets_copy():
                assert target.is_measurement_record_target
                row[count + target.value] ^= 1
            if instruction.name == "DETECTOR":
                detectors.append(np.append(row, (row @ reference) % 2))
            else:
                assert instruction.gate_args_copy() == [0.0]
                observable ^= row
        else:
            single = stim.Circuit()
            single.append(instruction)
            count += single.num_measurements
    assert count == circuit.num_measurements
    return np.asarray(detectors, dtype=np.uint8), np.append(observable, (observable @ reference) % 2)


@pytest.mark.parametrize("distance", [2, 3])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("order", [("X", "Z"), ("Z", "X")])
def test_bacon_shor_compressed_memory_preserves_all_record_relations(distance, basis, order):
    rounds = 7
    compressed = MemoryExperiment(
        qec_patch=BaconShorCode(distance=distance),
        basis=basis,
        rounds=rounds,
        se_block_kwargs={"basis_order": order},
    ).build()

    system = QECSystem()
    system.add_patch(BaconShorCode(distance=distance), name="explicit")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: basis for q in system.data_indices}, system.num_qubits)
    block = GenericCSSGaugeExtractionBlock(system, basis_order=order)
    for _ in range(rounds):
        builder.apply_syndrome_extraction(
            block.circuit, rounds=1, measurement_blocks=block.measurement_blocks,
        )
    builder.apply_data_readout({q: basis for q in system.data_indices})
    explicit = builder.circuit

    assert any(
        isinstance(instruction, stim.CircuitRepeatBlock)
        and instruction.repeat_count == rounds - 2
        for instruction in compressed
    )
    assert _physical_operations(compressed) == _physical_operations(explicit)
    compressed_detectors, compressed_logical = _affine_annotation_rows(compressed)
    explicit_detectors, explicit_logical = _affine_annotation_rows(explicit)

    # Equal affine rowspaces mean identical detectability for every possible
    # record fault, even when the emitted detector generators differ.
    joint_rank = _rank(np.vstack([compressed_detectors, explicit_detectors]))
    assert joint_rank == _rank(compressed_detectors) == _rank(explicit_detectors)
    # Logical representatives may differ by detector products. Their normalized
    # parity must agree on every syndrome-free record, including logical faults.
    difference = compressed_logical ^ explicit_logical
    assert _rank(np.vstack([compressed_detectors, difference])) == joint_rank


def test_bacon_shor_and_ordinary_patch_keep_two_protected_logicals():
    system = QECSystem()
    system.add_patch(RepetitionCode(distance=3), name="ordinary")
    system.add_patch(BaconShorCode(distance=2), name="subsystem", offset=(10, 0))
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)

    # Measure only the ordinary patch's checks, using its registered global
    # supports. The Bacon-Shor centers have no direct measurement ancilla.
    ordinary_checks = [s for s in system.active_stabilizers if s["patch_name"] == "ordinary"]
    ordinary = stim.Circuit()
    ordinary.append("R", [s["syn_idx"] for s in ordinary_checks])
    ordinary.append("TICK", tag="SE_start")
    for neighbor in range(2):
        for check in ordinary_checks:
            ordinary.append("CX", [sorted(check["data_indices"])[neighbor], check["syn_idx"]])
        ordinary.append("TICK")
    ordinary.append("M", [s["syn_idx"] for s in ordinary_checks])
    builder.apply_syndrome_extraction(ordinary, rounds=1)

    gauges = GenericCSSGaugeExtractionBlock(system)
    builder.apply_syndrome_extraction(
        gauges.circuit, rounds=3, measurement_blocks=gauges.measurement_blocks,
    )
    assert tracker.logicals.count == 2
    assert tracker.expected_num_logicals == 2
    assert _rank(tracker.stabilizers.matrix) == 5  # two ordinary + three gauge constraints

    # Both logical-Z state constraints remain in the full conditioned span.
    full_state = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    logical_z = _symplectic([op for op in system.logical_ops if op["type"] == "Z"], system.num_qubits)
    assert _rank(np.vstack([full_state, logical_z])) == _rank(full_state)

    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    circuit = builder.circuit
    assert circuit.num_observables == 2
    detectors, observables = circuit.compile_detector_sampler(seed=5).sample(256, separate_observables=True)
    assert not detectors.any()
    assert not observables.any()
    circuit.detector_error_model()
