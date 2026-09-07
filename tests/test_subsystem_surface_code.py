"""Planar triangle-code algebra, physical extraction, and native memory checks."""

from itertools import combinations

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock
from lightstim.qec_code.subsystem_surface import (
    SubsystemSurfaceCode,
    SubsystemSurfaceCodeExtractionBlock,
)
from lightstim.utils.linear_algebra import row_echelon


def _matrix(records, n):
    result = np.zeros((len(records), 2 * n), dtype=np.uint8)
    for row, record in enumerate(records):
        for q, basis in record["pauli"].items():
            result[row, q] = basis in {"X", "Y"}
            result[row, n + q] = basis in {"Z", "Y"}
    return result


def _rank(matrix):
    return row_echelon(matrix.copy())[1]


def _commutes(left, right):
    n = left.shape[1] // 2
    return not ((left[:, :n] @ right[:, n:].T + left[:, n:] @ right[:, :n].T) % 2).any()


def _pauli(record, n):
    result = stim.PauliString(n)
    for q, basis in record["pauli"].items():
        result[q] = basis
    return result


def _system(d):
    system = QECSystem()
    system.add_patch(SubsystemSurfaceCode(distance=d), name="ssc")
    return system


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_planar_algebra_and_local_layout(d):
    patch = SubsystemSurfaceCode(distance=d)
    n = patch.num_qubits
    s, g, logicals = [_matrix(records, n) for records in (patch.stabilizers, patch.gauges, patch.logical_ops)]
    assert len(patch.data_indices) == 3 * d**2 - 2 * d
    assert len(patch.syndrome_indices) == len(patch.gauges) == 4 * d * (d - 1)
    assert len(patch.stabilizers) == _rank(s) == 2 * d**2 - 2
    assert _rank(g) == 4 * d * (d - 1)
    assert _rank(np.vstack([g, s])) == _rank(g)
    assert _commutes(s, g) and _commutes(logicals, g)
    assert not _commutes(logicals[:1], logicals[1:])
    assert (_rank(g) - _rank(s)) // 2 == patch.num_gauge_qubits == (d - 1)**2
    assert len(patch.data_indices) - _rank(s) - patch.num_gauge_qubits == patch.num_logicals == 1
    assert sum(len(r["pauli"]) == 6 for r in patch.stabilizers) == 2 * (d - 1)**2
    assert sum(len(r["pauli"]) == 2 for r in patch.stabilizers) == 4 * (d - 1)
    assert all(r["syn_idx"] is None for r in patch.stabilizers)
    assert all(len(r["pauli"]) == 2 * d - 1 for r in patch.logical_ops)
    for gauge in patch.gauges:
        a = gauge["syn_coord"]
        assert len(gauge["pauli"]) in {2, 3}
        for q in gauge["data_indices"]:
            x, y = patch.qubit_coords[q]
            assert (x - a[0])**2 + (y - a[1])**2 == 2
    assert not {(4*i+3, 4*j+3) for i in range(d-1) for j in range(d-1)} & set(patch.index_map)


@pytest.mark.parametrize("d", [2, 3])
def test_minimum_dressed_distance_independent_of_parent_label(d):
    patch = SubsystemSurfaceCode(distance=d)
    for basis, coords in (("X", [(3, 4*i+1) for i in range(d)]),
                          ("Z", [(4*i+1, 3) for i in range(d)])):
        checks = [s for s in patch.stabilizers if s["type"] != basis]
        conjugate = next(op for op in patch.logical_ops if op["type"] != basis)
        witness = {patch.index_map[c] for c in coords}
        assert all(len(witness & set(s["pauli"])) % 2 == 0 for s in checks)
        assert len(witness & set(conjugate["pauli"])) % 2 == 1
        # Any mixed CSS logical has a nontrivial pure-X or pure-Z component.
        # Exhaust the smaller pure-sector supports to exclude distance d-1.
        for weight in range(1, d):
            for support in combinations(patch.data_indices, weight):
                support = set(support)
                assert (len(support & set(conjugate["pauli"])) % 2 == 0
                        or any(len(support & set(s["pauli"])) % 2 for s in checks))


@pytest.mark.parametrize("distance", [None, 0, 1, -3, 2.5, True, "3"])
def test_invalid_distance(distance):
    with pytest.raises(ValueError, match="integer >= 2"):
        SubsystemSurfaceCode(distance=distance)


@pytest.mark.parametrize("order", [("X", "Z"), ("Z", "X"), ("Z", "Z", "X")])
def test_signed_physical_measurements_match_generic_extraction(order):
    system = _system(3)
    dedicated = SubsystemSurfaceCodeExtractionBlock(system, basis_order=order)
    generic = GenericCSSGaugeExtractionBlock(system, basis_order=order)
    assert dedicated.depth_x == dedicated.depth_z == 4
    assert dedicated.cnot_depth == 4 * len(order)
    assert generic.depth_x == generic.depth_z == 3
    for basis, physical in zip(order, dedicated.measurement_blocks):
        gauges = sorted((g for g in system.active_gauges if g["type"] == basis), key=lambda g: g["syn_idx"])
        for measurement, gauge in enumerate(gauges):
            assert physical.has_flow(stim.Flow(input=_pauli(gauge, system.num_qubits), measurements=[measurement]), unsigned=False)
        for center in system.active_stabilizers:
            p = _pauli(center, system.num_qubits)
            assert physical.has_flow(stim.Flow(input=p, output=p), unsigned=False)
    for left, right in ((dedicated.circuit, generic.circuit), (generic.circuit, dedicated.circuit)):
        assert all(right.has_flow(f, unsigned=False) for f in left.flow_generators())


def test_schedule_supports_transformed_placement_and_rejects_missing_interactions():
    system = _system(2)
    patch = SubsystemSurfaceCode(distance=3)
    patch.transpose_coords()
    patch.rotate_coords(np.pi / 4, center=(0, 0))
    system.add_patch(patch, name="placed", offset=(20.125, 30.25))
    system.active_gauge_indices = {i for i, g in enumerate(system.gauges) if g["patch_name"] == "placed"}
    block = SubsystemSurfaceCodeExtractionBlock(system)
    local = SubsystemSurfaceCodeExtractionBlock(_system(3))
    mapping = system.local_to_global_map["placed"]
    for global_layers, local_layers in ((block.x_layers, local.x_layers), (block.z_layers, local.z_layers)):
        assert global_layers == [[(mapping[a], mapping[q]) for a, q in layer] for layer in local_layers]
        for layer in global_layers:
            endpoints = [q for pair in layer for q in pair]
            assert len(endpoints) == len(set(endpoints))
    gauge = system.active_gauges[0]
    gauge["pauli"] = {q: gauge["type"] for q in sorted(system.data_indices)[:2]}
    gauge["data_indices"] = list(gauge["pauli"])
    with pytest.raises(ValueError, match="outside its diagonal neighbors"):
        SubsystemSurfaceCodeExtractionBlock(system)


@pytest.mark.parametrize("d,rounds", [(2, 1), (3, 2), (3, 7), (5, 3)])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_noiseless_memory(d, rounds, basis):
    experiment = MemoryExperiment(qec_patch=SubsystemSurfaceCode(distance=d), basis=basis, rounds=rounds)
    assert experiment.block_class is SubsystemSurfaceCodeExtractionBlock
    circuit = experiment.build()
    assert circuit.num_qubits == 7 * d**2 - 6 * d
    assert circuit.num_detectors > 0 and circuit.num_observables == 1
    samples = circuit.compile_detector_sampler(seed=3).sample(256, append_observables=True)
    assert not samples.any()
    circuit.detector_error_model()
    if rounds == 7:
        assert any(isinstance(op, stim.CircuitRepeatBlock) for op in circuit)


def test_gauge_phase_changes_preserve_the_static_center():
    system = _system(3)
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)
    active_center = set(system.active_stabilizer_indices)
    center_supports = [dict(s["pauli"]) for s in system.stabilizers]
    for basis in ("X", "Z", "Z", "X", "Z"):
        block = SubsystemSurfaceCodeExtractionBlock(system, basis_order=(basis,))
        builder.apply_syndrome_extraction(block.circuit, rounds=1, measurement_blocks=block.measurement_blocks)
        assert tracker.logicals.count == tracker.expected_num_logicals == 1
        assert system.active_stabilizer_indices == active_center
        assert [s["pauli"] for s in system.stabilizers] == center_supports
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    assert not builder.circuit.compile_detector_sampler(seed=4).sample(128, append_observables=True).any()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_noise_model_produces_a_dem_without_decoding(basis):
    circuit = MemoryExperiment(
        qec_patch=SubsystemSurfaceCode(distance=3), rounds=3, basis=basis,
        noise_params=NoiseConfig(p_idle=0.001, p_1q=0.001, p_2q=0.001, p_meas=0.001, p_reset=0.001),
    ).build()
    dem = circuit.detector_error_model()
    assert dem.num_errors > 0 and dem.num_observables == 1
    assert dem.num_detectors == circuit.num_detectors
