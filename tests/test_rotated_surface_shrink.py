"""Contraction preserves an encoded channel, including measured Pauli frames."""

import math
import pickle

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic


def _setup(distance=5, basis="X", *, offset=(0, 0), reference=False, source_rounds=1):
    system = QECSystem()
    ref = system.add_patch(RotatedSurfaceCode(distance=distance), name="ref", offset=(-30, 0)) if reference else None
    source = system.add_patch(RotatedSurfaceCode(distance=distance), name="patch", offset=offset)
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    system.register_builder(builder)
    system.register_tracker(tracker)
    executor = LogicalExecutor(builder)
    executor.register_op_set(RotatedSurfaceCode, RotatedSurfaceCodeLogicalOpSet(RotatedSurfaceCodeExtractionBlock))
    builder.write_coordinates()
    init = {q: ("Z" if ref else basis) for q in source.data_indices}
    if ref:
        init.update({q: "X" for q in ref.data_indices})
    builder.initialize(init, system.num_qubits)
    if source_rounds:
        builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=source_rounds)
    if ref:
        executor.apply_logical_operation("transversal_cnot", [ref, source])
    return system, source, ref, builder, executor


def _logical(patch, basis, n):
    operator = stim.PauliString(n)
    for q, p in next(op for op in patch.logical_ops if op["type"] == basis)["pauli"].items():
        operator[q] = p
    return operator


def _ideal(builder):
    circuit = builder.to_stim_circuit()
    circuit.detector_error_model(decompose_errors=True)
    detections, observables = circuit.compile_detector_sampler(seed=19).sample(128, separate_observables=True)
    assert not detections.any()
    assert not observables.any()
    return circuit


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distances", [(5, 3), (7, 3), (7, 5)])
@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_shrink_retains_ids_closes_detectors_and_emits_only_requested_se(basis, distances, rounds):
    d, target = distances
    system, source, _, builder, executor = _setup(d, basis, offset=(12.25, -8.5))
    tracker = builder.tracker
    old_coords = dict(system.qubit_coords)
    old_n = system.num_qubits
    first_instruction = len(builder.circuit)
    first_measurement = builder.circuit.num_measurements
    result = executor.apply_logical_operation("shrink", [source], target_distance=target, rounds=rounds)
    assert builder.system is system and builder.tracker is tracker
    assert result.distance_x == result.distance_z == target
    assert source.distance_x == d
    assert result is not source
    assert set(system.patches) == {"patch"} and not system.coupler_patches
    assert result.qubit_coords.items() <= old_coords.items()
    assert system.qubit_coords == old_coords and system.num_qubits == tracker.num_qubits == old_n
    assert system.data_indices == result.data_indices
    assert system.syndrome_indices == result.syndrome_indices
    assert system.active_qubit_indices == result.data_indices
    assert set(system.index_to_owner_map) == set(result.qubit_coords)
    assert set(system.coord_to_owner_map) == set(result.qubit_coords.values())
    assert system.num_logicals == tracker.expected_num_logicals == tracker.logicals.count == 1
    assert builder.circuit.num_observables == 0
    delta = builder.circuit[first_instruction:].flattened()
    assert sum(i.name == "CX" for i in delta) == 4 * rounds
    assert builder.circuit.num_measurements - first_measurement == d*d-target*target + rounds*(target*target-1)
    for instruction in delta:
        if instruction.name in {"R", "RX", "RY"}:
            assert not {t.value for t in instruction.targets_copy()} & source.data_indices
    builder.apply_data_readout({q: basis for q in result.data_indices})
    assert _ideal(builder).num_observables == 1


@pytest.mark.parametrize("distances", [(5, 3), (7, 3), (7, 5)])
@pytest.mark.parametrize("rounds", [0, 1])
def test_bell_channel_is_preserved_with_the_measured_logical_frame(distances, rounds):
    d, target = distances
    system, source, ref, builder, executor = _setup(d, reference=True, offset=(2.25, 4.5))
    first_record = builder.circuit.num_measurements
    spectator = pickle.dumps((system.patches["ref"], system.local_to_global_map["ref"],
                              [s for s in system.stabilizers if s.get("patch_name") == "ref"]))
    result = executor.apply_logical_operation("shrink", [source], target_distance=target, rounds=rounds)
    assert spectator == pickle.dumps((system.patches["ref"], system.local_to_global_map["ref"],
                                     [s for s in system.stabilizers if s.get("patch_name") == "ref"]))
    ox = min(x for x, y in source.data_coords)
    oy = min(y for x, y in source.data_coords)
    removed = sorted(source.data_indices-result.data_indices)
    mx = [q for q in removed if source.qubit_coords[q][1]-oy >= source.qubit_coords[q][0]-ox]
    mz = [q for q in removed if q not in mx]
    rec = {q: first_record+i for i, q in enumerate(mx+mz)}
    seen = set()
    for seed in range(32):
        sim = stim.TableauSimulator(seed=seed)
        sim.do_circuit(builder.circuit)
        record = sim.current_measurement_record()
        px = sum(record[rec[q]] for q in mx if source.qubit_coords[q][0] == ox) % 2
        pz = sum(record[rec[q]] for q in mz if source.qubit_coords[q][1] == oy) % 2
        seen.add((px, pz))
        for basis, parity in (("X", px), ("Z", pz)):
            left, right = _logical(ref, basis, system.num_qubits), _logical(result, basis, system.num_qubits)
            assert sim.peek_observable_expectation(left*right) == (-1)**parity
            assert sim.peek_observable_expectation(right) == 0
        # Testing the explicit frame correction proves coherence, rather
        # than just preparing two separately known basis states.
        if px:
            sim.do_pauli_string(_logical(result, "Z", system.num_qubits))
        if pz:
            sim.do_pauli_string(_logical(result, "X", system.num_qubits))
        for basis in ("X", "Z"):
            assert sim.peek_observable_expectation(
                _logical(ref, basis, system.num_qubits)*_logical(result, basis, system.num_qubits)
            ) == 1
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
    builder.apply_data_readout({q: "X" for q in ref.data_indices | result.data_indices})
    _ideal(builder)


def test_shortened_check_records_contain_old_syndrome_and_removed_data():
    system, source, _, builder, executor = _setup()
    old_records = {check["syn_coord"]: set(builder.tracker.stabilizers.records[i])
                   for i, check in enumerate(system.active_stabilizers)}
    first_record = builder.circuit.num_measurements
    result = executor.apply_logical_operation("shrink", [source], target_distance=3)
    removed = sorted(source.data_indices-result.data_indices)
    mx = [q for q in removed if source.qubit_coords[q][1] >= source.qubit_coords[q][0]]
    mz = [q for q in removed if q not in mx]
    rec = {q: first_record+i for i, q in enumerate(mx+mz)}
    basis = stabilizers_to_symplectic(system, result.stabilizers, system.num_qubits)
    coeffs, dependent, _ = solve_linear_decomposition(builder.tracker.stabilizers.matrix, basis, reduce_weight=False)
    assert dependent.all()
    for target, coefficients in zip(result.stabilizers, coeffs):
        previous = next(s for s in source.stabilizers if s["syn_coord"] == target["syn_coord"])
        expected = old_records[target["syn_coord"]] ^ {rec[q] for q in set(previous["pauli"])-set(target["pauli"])}
        actual = set()
        for i in np.flatnonzero(coefficients):
            actual.symmetric_difference_update(builder.tracker.stabilizers.records[i])
        assert actual == expected


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_repeated_shrink_and_regrowth_reuse_resources_and_keep_noise_history(basis):
    system, patch, _, builder, executor = _setup(7, basis)
    old_coords = dict(system.qubit_coords)
    for op, distance, rounds in [("shrink", 5, 0), ("shrink", 3, 1), ("enlarge", 7, 1), ("shrink", 3, 0)]:
        patch = executor.apply_logical_operation(op, [patch], target_distance=distance, rounds=rounds)
        assert system.qubit_coords == old_coords
        assert system.num_qubits == len(old_coords)
        assert set(system.index_to_owner_map) == set(patch.qubit_coords)
    builder.apply_data_readout({q: basis for q in patch.data_indices})
    _ideal(builder)
    noisy = builder.build_noisy_circuit(NoiseConfig(p_1q=.001, p_2q=.001, p_meas=.001, p_reset=.001, p_idle=.001))
    assert noisy.detector_error_model(decompose_errors=True).num_errors > 0


def test_idle_noise_covers_source_before_shrink_but_only_live_data_after():
    _, source, _, builder, executor = _setup()
    result = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=1)
    builder.apply_data_readout({q: "X" for q in result.data_indices})
    noisy = builder.build_noisy_circuit(NoiseConfig(p_idle=.001), noise_model="code_capacity")
    idle_sets = [{t.value for t in i.targets_copy()} for i in noisy.flattened() if i.name == "DEPOLARIZE1"]
    assert idle_sets == [source.data_indices, result.data_indices]


@pytest.mark.parametrize("active", [False, True])
def test_source_boundary_dependency_prevents_shrinking_even_if_coupler_is_inactive(active):
    from lightstim.qec_code.surface_code.rotated.enlargement import _EnlargementCoupler
    system, source, _, builder, executor = _setup()
    system.register_coupler(_EnlargementCoupler(), ["patch"], name="pending_growth",
                            target_patch=RotatedSurfaceCode(distance=7))
    if active:
        builder.activate_coupler("pending_growth")
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError):
        executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert pickle.dumps((system, source, builder)) == before


def test_released_region_can_host_a_fresh_patch_with_transversal_cnot():
    system, source, _, builder, executor = _setup(7)
    original_coords = dict(system.index_map)
    target = executor.apply_logical_operation("shrink", [source], target_distance=3)
    fresh = system.add_patch(RotatedSurfaceCode(distance=3), name="fresh", offset=(6, 6))
    for q, xy in fresh.qubit_coords.items():
        if xy in original_coords:
            assert q == original_coords[xy]
    builder.initialize({q: "Z" for q in fresh.data_indices}, system.num_qubits)
    builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1)
    executor.apply_logical_operation("transversal_cnot", [target, fresh])
    builder.apply_data_readout({q: "Z" for q in target.data_indices | fresh.data_indices})
    _ideal(builder)


def test_released_region_reregistration_restores_check_ownership_for_ppm():
    from lightstim.qec_code.surface_code.rotated.ppm import (
        RotatedSurfacePatchPlacement,
        RotatedSurfacePPMRequest,
        lower_ppm,
    )

    system, source, _, builder, executor = _setup(7)
    target = executor.apply_logical_operation("shrink", [source], target_distance=3)
    retired_uids = source._registered_stabilizer_uids - target._registered_stabilizer_uids
    fresh = system.add_patch(RotatedSurfaceCode(distance=3), name="fresh", offset=(8, 0))
    builder.initialize({q: "Z" for q in fresh.data_indices}, system.num_qubits)
    builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1)

    assert not fresh._registered_stabilizer_uids & retired_uids
    assert all(system.stabilizers[uid]["patch_name"] == "fresh"
               for uid in fresh._registered_stabilizer_uids)
    placements = [
        RotatedSurfacePatchPlacement("patch", (1, 1), 3, "X_vertical"),
        RotatedSurfacePatchPlacement("fresh", (9, 1), 3, "X_vertical"),
    ]
    request = RotatedSurfacePPMRequest(
        targets=(("patch", "X"), ("fresh", "X")), route=(),
    )
    plan = lower_ppm(placements, request, system=system)
    assert plan.kind == "corridor"
    assert plan.certificate.measures_exactly_the_product


@pytest.mark.parametrize("rounds", [0, 1])
def test_legacy_z_only_readout_requires_a_current_se_cache(rounds):
    system, source, _, builder, executor = _setup(basis="Z", source_rounds=0)
    builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1, z_only=True)
    result = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=rounds)
    if rounds == 0:
        before = str(builder.circuit)
        with pytest.raises(ValueError, match="current target-code SE"):
            builder.apply_data_readout({q: "Z" for q in result.data_indices}, z_only=True)
        assert str(builder.circuit) == before
    builder.apply_data_readout({q: "Z" for q in result.data_indices}, z_only=bool(rounds))
    _ideal(builder)


def test_target_only_se_invalidates_legacy_z_only_cache_with_spectators():
    system, source, _, builder, executor = _setup(basis="Z", source_rounds=0)
    spectator = system.add_patch(RotatedSurfaceCode(distance=3), name="spectator", offset=(20, 0))
    builder.initialize({q: "Z" for q in spectator.data_indices}, system.num_qubits)
    builder.apply_syndrome_extraction(
        RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1, z_only=True,
    )
    target = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=1)
    measurements = {q: "Z" for q in target.data_indices | spectator.data_indices}
    before = pickle.dumps((system, builder))
    with pytest.raises(ValueError, match="current target-code SE"):
        builder.apply_data_readout(measurements, z_only=True)
    assert pickle.dumps((system, builder)) == before

    builder.apply_syndrome_extraction(
        RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1, z_only=True,
    )
    builder.apply_data_readout(measurements, z_only=True)
    _ideal(builder)


@pytest.mark.parametrize("kwargs", [
    {"target_distance": 5}, {"target_distance": 7}, {"target_distance": 1},
    {"target_distance": 2}, {"target_distance": 3.0}, {"target_distance": True},
    {"target_distance": 3, "rounds": -1}, {"target_distance": 3, "rounds": True},
    {"target_distance": 3, "direction": "+x,+y"},
])
def test_invalid_shrink_parameters_leave_circuit_and_state_unchanged(kwargs):
    system, source, _, builder, executor = _setup()
    before = pickle.dumps((system, source, builder))
    with pytest.raises((ValueError, TypeError)):
        executor.apply_logical_operation("shrink", [source], **kwargs)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("failure", ["rebase_stabilizers_onto_code_basis", "apply_syndrome_extraction"])
def test_late_failure_is_atomic_after_measurement_and_metadata_update(monkeypatch, failure):
    system, source, _, builder, executor = _setup()
    before = pickle.dumps((system, source, builder))
    tracker = builder.tracker
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate contraction failure")
    monkeypatch.setattr(SyndromeTracker if failure.startswith("rebase") else CircuitBuilder, failure, fail)
    with pytest.raises(RuntimeError, match="deliberate contraction failure"):
        executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=1)
    assert builder.system is system and builder.tracker is tracker
    assert pickle.dumps((system, source, builder)) == before


def test_unencoded_and_stale_sources_are_rejected():
    system, source, _, builder, executor = _setup(source_rounds=0)
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError, match="not established"):
        executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert pickle.dumps((system, source, builder)) == before
    builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1)
    executor.apply_logical_operation("shrink", [source], target_distance=3)
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError, match="global patch"):
        executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_coherent_encoding_can_shrink_without_any_syndrome_extraction(basis):
    system, patch, _, builder, executor = _setup(source_rounds=0, basis="Z")
    data = sorted(patch.data_indices)
    positions = {q: i for i, q in enumerate(data)}
    logical = next(op for op in patch.logical_ops if op["type"] == basis)
    stabilizers = []
    for record in patch.stabilizers + [logical]:
        pauli = stim.PauliString(len(data))
        for q, axis in record["pauli"].items():
            pauli[positions[q]] = axis
        stabilizers.append(pauli)
    encoder = stim.Circuit()
    for instruction in stim.Tableau.from_stabilizers(stabilizers).to_circuit():
        encoder.append(instruction.name, [data[t.value] for t in instruction.targets_copy()], instruction.gate_args_copy())
    builder.apply_unitary_block(encoder)
    assert builder.circuit.num_measurements == 0
    target = executor.apply_logical_operation("shrink", [patch], target_distance=3)
    assert builder.circuit.num_measurements == 16
    builder.apply_data_readout({q: basis for q in target.data_indices})
    _ideal(builder)


def test_y_injection_shrink_and_native_mid_round_s_dagger():
    system = QECSystem()
    patch = system.add_patch(RotatedSurfaceCode(distance=5), name="patch")
    tracker = SyndromeTracker(system.num_qubits, 1)
    builder = CircuitBuilder(tracker, system)
    system.register_builder(builder)
    system.register_tracker(tracker)
    executor = LogicalExecutor(builder)
    executor.register_op_set(RotatedSurfaceCode, RotatedSurfaceCodeLogicalOpSet(RotatedSurfaceCodeExtractionBlock))
    executor.apply_logical_operation("state_injection", [patch], inject_state="Y", rounds=1, post_select_coords=set())
    target = executor.apply_logical_operation("shrink", [patch], target_distance=3)
    executor.apply_logical_operation("fold_transversal_s_dag", [target])
    builder.apply_data_readout({q: "X" for q in target.data_indices})
    _ideal(builder)


@pytest.mark.parametrize("condition", ["transpose", "rotation", "defect", "inactive", "unregistered"])
def test_unsupported_source_is_rejected_atomically(condition):
    system, patch, _, builder, executor = _setup()
    if condition == "transpose":
        patch.transpose_coords()
    elif condition == "rotation":
        patch.rotate_coords(math.pi/2)
    elif condition == "defect":
        system.disable_data_qubits({min(patch.data_indices)})
    elif condition == "inactive":
        system.active_stabilizer_indices.discard(min(patch._registered_stabilizer_uids))
    else:
        patch = RotatedSurfaceCode(distance=5)
    before = pickle.dumps((system, patch, builder))
    with pytest.raises((ValueError, TypeError)):
        executor.apply_logical_operation("shrink", [patch], target_distance=3)
    assert pickle.dumps((system, patch, builder)) == before
