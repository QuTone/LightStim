"""Physical-channel and lifecycle regressions for retained-data enlargement."""

import math
import pickle
from collections import Counter

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


def _setup(*, local=None, offset=(0, 0), reference=False):
    system = QECSystem()
    other = None
    if reference:
        other = system.add_patch(RotatedSurfaceCode(distance=3), name="reference")
    patch = system.add_patch(
        local if local is not None else RotatedSurfaceCode(distance=3),
        name="patch",
        offset=offset,
    )
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    system.register_tracker(tracker)
    system.register_builder(builder)
    executor = LogicalExecutor(builder)
    executor.register_op_set(
        RotatedSurfaceCode,
        RotatedSurfaceCodeLogicalOpSet(RotatedSurfaceCodeExtractionBlock),
    )
    builder.write_coordinates()
    return system, patch, other, builder, executor


def _prepare(builder, patch, basis):
    builder.initialize(
        {q: basis for q in patch.data_indices}, builder.system.num_qubits
    )


def _ordinary_round(builder):
    builder.apply_syndrome_extraction(
        RotatedSurfaceCodeExtractionBlock(builder.system).circuit, rounds=1
    )


def _instructions(circuit):
    # Define-by-run inserts coordinates at the front of the circuit.
    return [i for i in circuit.flattened() if i.name != "QUBIT_COORDS"]


def _targets(instruction):
    return {t.value for t in instruction.targets_copy() if t.is_qubit_target}


def _logical(patch, basis, n):
    result = stim.PauliString(n)
    record = next(op for op in patch.logical_ops if op["type"] == basis)
    for q, pauli in record["pauli"].items():
        result[q] = pauli
    return result


def _assert_ideal(circuit):
    # Do not allow gauge detectors or ignore decomposition failures.
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circuit.num_detectors > 0
    assert dem.num_observables == circuit.num_observables > 0
    detections, observables = circuit.compile_detector_sampler(seed=572).sample(
        shots=128, separate_observables=True
    )
    assert not np.any(detections)
    assert not np.any(observables)


def _snapshot(system, patch, builder):
    # Include tracker rows/records, registry and ownership maps, circuit,
    # define-by-run hooks, and the caller's source handle in the atomicity check.
    return pickle.dumps((system, patch, builder), protocol=5)


@pytest.mark.parametrize(
    ("basis", "rounds", "target_distance"),
    [("X", 1, 5), ("Z", 3, 5), ("X", 2, 7)],
)
def test_enlarge_preserves_data_and_uses_requested_four_cnot_rounds(
    basis, rounds, target_distance
):
    system, source, _, builder, executor = _setup(offset=(12.25, -8.5))
    _prepare(builder, source, basis)
    _ordinary_round(builder)
    old_data = set(source.data_indices)
    old_coords = dict(source.qubit_coords)
    prefix = _instructions(builder.circuit)

    enlarged = executor.apply_logical_operation(
        "enlarge", [source], target_distance=target_distance, rounds=rounds
    )

    assert isinstance(enlarged, RotatedSurfaceCode)
    assert enlarged is not source
    assert enlarged.distance_x == enlarged.distance_z == target_distance
    assert source.distance_x == source.distance_z == 3
    assert old_data <= enlarged.data_indices
    assert all(enlarged.qubit_coords[q] == coord for q, coord in old_coords.items())
    expected = RotatedSurfaceCode(distance=target_distance)
    expected.shift_coords(12.25, -8.5)
    assert set(enlarged.qubit_coords.values()) == set(expected.qubit_coords.values())
    assert enlarged.index_map == {coord: q for q, coord in enlarged.qubit_coords.items()}
    assert set(system.patches) == {"patch"}
    assert system.patches["patch"][0].distance_x == target_distance
    assert not system.coupler_patches
    assert not system.paused_stabilizer_indices
    assert system.num_logicals == builder.tracker.expected_num_logicals == 1
    assert builder.tracker.num_qubits == system.num_qubits

    instructions = _instructions(builder.circuit)
    assert instructions[:len(prefix)] == prefix
    delta = instructions[len(prefix):]
    resets = Counter(
        q
        for instruction in delta
        if instruction.name in {"R", "RX", "RY", "MR", "MRX", "MRY"}
        for q in _targets(instruction)
        if q in enlarged.data_indices
    )
    assert set(resets) == enlarged.data_indices - old_data
    assert set(resets.values()) == {1}
    assert sum(i.name == "CX" for i in delta) == 4 * rounds
    measured = Counter(
        q for instruction in delta if instruction.name == "M"
        for q in _targets(instruction)
    )
    assert measured == Counter({q: rounds for q in enlarged.syndrome_indices})
    for instruction in delta:
        if instruction.name == "CX":
            targets = instruction.targets_copy()
            assert len(_targets(instruction)) == len(targets), "a CNOT layer reuses a qubit"

    builder.apply_data_readout({q: basis for q in enlarged.data_indices})
    _assert_ideal(builder.to_stim_circuit())


def test_enlarge_preserves_bell_channel_and_leaves_other_patch_untouched():
    # Constructor shift plus placement offset intentionally differs from the
    # historical .shift metadata. Geometry must come from actual coordinates.
    local = RotatedSurfaceCode(distance=3, shift=(2.25, -4.5))
    system, source, reference, builder, executor = _setup(
        local=local, offset=(20.5, 30.25), reference=True
    )
    assert any(a != b for a, b in system.local_to_global_map["patch"].items())
    _prepare(builder, reference, "X")
    _prepare(builder, source, "Z")
    _ordinary_round(builder)
    executor.apply_logical_operation("transversal_cnot", [reference, source])
    prefix = _instructions(builder.circuit)
    other_registry = pickle.dumps((
        system.patches["reference"],
        system.local_to_global_map["reference"],
        [s for s in system.stabilizers if s.get("patch_name") == "reference"],
        [op for op in system.logical_ops if op.get("patch_name") == "reference"],
    ))
    other_active = set(system.active_stabilizer_indices) & reference._registered_stabilizer_uids

    enlarged = executor.apply_logical_operation(
        "enlarge", [source], target_distance=5, rounds=2
    )

    assert pickle.dumps((
        system.patches["reference"],
        system.local_to_global_map["reference"],
        [s for s in system.stabilizers if s.get("patch_name") == "reference"],
        [op for op in system.logical_ops if op.get("patch_name") == "reference"],
    )) == other_registry
    assert set(system.active_stabilizer_indices) & reference._registered_stabilizer_uids == other_active
    assert all(
        system.index_to_owner_map[q] == "reference"
        and system.qubit_coords[q] == coord
        for q, coord in reference.qubit_coords.items()
    )
    assert system.num_logicals == builder.tracker.expected_num_logicals == 2
    delta = _instructions(builder.circuit)[len(prefix):]
    assert all(not (_targets(i) & set(reference.qubit_coords)) for i in delta)

    # A Bell reference checks the full logical identity channel, including
    # coherence, without relying on the tracker's choice of logical rows.
    for seed in range(8):
        simulator = stim.TableauSimulator(seed=seed)
        simulator.do_circuit(builder.circuit)
        for basis in ("X", "Z"):
            output = _logical(enlarged, basis, system.num_qubits)
            ref = _logical(reference, basis, system.num_qubits)
            assert simulator.peek_observable_expectation(ref * output) == 1
            assert simulator.peek_observable_expectation(output) == 0
            assert simulator.peek_observable_expectation(ref) == 0

    before = _snapshot(system, source, builder)
    with pytest.raises(ValueError):
        executor.apply_logical_operation("enlarge", [source], target_distance=7)
    assert _snapshot(system, source, builder) == before


@pytest.mark.parametrize("grown_is_control", [True, False])
def test_enlarged_patch_can_form_bell_pair_with_fresh_patch(grown_is_control):
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, "X" if grown_is_control else "Z")
    _ordinary_round(builder)
    grown = executor.apply_logical_operation("enlarge", [source], target_distance=5)
    fresh = system.add_patch(RotatedSurfaceCode(distance=5), name="fresh", offset=(20, 0))
    _prepare(builder, fresh, "Z" if grown_is_control else "X")
    _ordinary_round(builder)
    patches = [grown, fresh] if grown_is_control else [fresh, grown]
    executor.apply_logical_operation("transversal_cnot", patches)

    # A deterministic detector model alone does not catch incorrect data
    # correspondence: it can silently describe the wrong logical operation.
    for seed in range(4):
        simulator = stim.TableauSimulator(seed=seed)
        simulator.do_circuit(builder.circuit)
        for basis in ("X", "Z"):
            assert simulator.peek_observable_expectation(
                _logical(grown, basis, system.num_qubits)
                * _logical(fresh, basis, system.num_qubits)
            ) == 1
        for patch in (grown, fresh):
            for record in patch.stabilizers:
                operator = stim.PauliString(system.num_qubits)
                for q, pauli in record["pauli"].items():
                    operator[q] = pauli
                assert abs(simulator.peek_observable_expectation(operator)) == 1


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_repeated_enlargement_uses_returned_global_patch(basis):
    system, patch, _, builder, executor = _setup()
    _prepare(builder, patch, basis)
    _ordinary_round(builder)
    retained = dict(patch.qubit_coords)
    for distance in (5, 7):
        patch = executor.apply_logical_operation(
            "enlarge", [patch], target_distance=distance, rounds=1
        )
        assert all(patch.qubit_coords[q] == coord for q, coord in retained.items())
        retained = dict(patch.qubit_coords)
        assert len(patch.data_indices) == distance ** 2
        assert set(system.patches) == {"patch"}
        assert system.num_logicals == builder.tracker.expected_num_logicals == 1
    _ordinary_round(builder)
    builder.apply_data_readout({q: basis for q in patch.data_indices})
    _assert_ideal(builder.to_stim_circuit())


def test_y_injection_enlarge_native_s_dagger_pipeline():
    system, patch, _, builder, executor = _setup()
    executor.apply_logical_operation(
        "state_injection", [patch], inject_state="Y", rounds=1,
        post_select_coords=set(),
    )
    patch = executor.apply_logical_operation("enlarge", [patch], target_distance=5)
    executor.apply_logical_operation("fold_transversal_s_dag", [patch])
    builder.apply_data_readout({q: "X" for q in patch.data_indices})
    circuit = builder.to_stim_circuit()
    assert any(i.name == "S_DAG" for i in circuit.flattened())
    _assert_ideal(circuit)


def test_enlargement_supports_circuit_level_noise_without_changing_clean_output():
    system, patch, _, builder, executor = _setup()
    _prepare(builder, patch, "X")
    _ordinary_round(builder)
    patch = executor.apply_logical_operation(
        "enlarge", [patch], target_distance=5, rounds=2
    )
    builder.apply_data_readout({q: "X" for q in patch.data_indices})
    clean = builder.to_stim_circuit().copy()
    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                    p_reset=1e-3, p_idle=1e-3),
        noise_model="circuit_level",
    )

    assert builder.circuit == clean
    assert all(i.tag != "noiseless" for i in clean.flattened())
    assert "DEPOLARIZE1" in str(noisy)
    assert "DEPOLARIZE2" in str(noisy)
    dem = noisy.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == clean.num_detectors
    assert dem.num_observables == clean.num_observables
    assert "error(" in str(dem)
    _assert_ideal(noisy.without_noise())


def test_enlargement_refreshes_legacy_z_only_readout_offsets():
    system, patch, _, builder, executor = _setup()
    _prepare(builder, patch, "Z")
    builder.apply_syndrome_extraction(
        RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1, z_only=True,
    )
    patch = executor.apply_logical_operation(
        "enlarge", [patch], target_distance=5, rounds=2,
    )
    builder.apply_data_readout({q: "Z" for q in patch.data_indices}, z_only=True)
    _assert_ideal(builder.to_stim_circuit())
    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_2q=.001, p_meas=.001), noise_model="circuit_level",
    )
    assert noisy.detector_error_model(decompose_errors=True).num_errors > 0


@pytest.mark.parametrize("failed_method", ["apply_syndrome_extraction", "combine_patches"])
def test_late_enlargement_failure_does_not_leak_staged_state(monkeypatch, failed_method):
    system, patch, _, builder, executor = _setup()
    _prepare(builder, patch, "Z")
    _ordinary_round(builder)
    tracker = builder.tracker
    before = _snapshot(system, patch, builder)
    calls = []

    def fail_after_allocation(staged, *args, **kwargs):
        assert staged is not builder
        assert staged.system.num_qubits > system.num_qubits
        assert staged.system.coupler_patches
        assert staged.tracker.num_qubits == staged.system.num_qubits
        calls.append(failed_method)
        raise RuntimeError("deliberate staged failure")

    monkeypatch.setattr(CircuitBuilder, failed_method, fail_after_allocation)
    with pytest.raises(RuntimeError, match="deliberate staged failure"):
        executor.apply_logical_operation("enlarge", [patch], target_distance=5)

    assert calls == [failed_method]
    assert builder.system is system
    assert builder.tracker is tracker
    assert _snapshot(system, patch, builder) == before


def test_unencoded_source_is_rejected_without_creating_code_constraints():
    system, patch, _, builder, executor = _setup()
    _prepare(builder, patch, "Z")
    before = _snapshot(system, patch, builder)
    with pytest.raises(ValueError, match="checks must be established"):
        executor.apply_logical_operation("enlarge", [patch], target_distance=5)
    assert _snapshot(system, patch, builder) == before


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_coherently_encoded_source_needs_no_extra_extraction_or_tracker_edit(basis):
    system, patch, _, builder, executor = _setup()
    data = sorted(patch.data_indices)
    local_index = {q: i for i, q in enumerate(data)}
    logical = next(op for op in patch.logical_ops if op["type"] == basis)
    stabilizers = []
    for record in patch.stabilizers + [logical]:
        pauli = stim.PauliString(len(data))
        for qubit, axis in record["pauli"].items():
            pauli[local_index[qubit]] = axis
        stabilizers.append(pauli)
    encoder = stim.Tableau.from_stabilizers(stabilizers).to_circuit()
    physical_encoder = stim.Circuit()
    for instruction in encoder:
        physical_encoder.append(
            instruction.name,
            [data[target.value] for target in instruction.targets_copy()],
            instruction.gate_args_copy(),
        )
    _prepare(builder, patch, "Z")
    builder.apply_unitary_block(physical_encoder)
    assert builder.circuit.num_measurements == 0
    assert builder.tracker.logicals.count == 0

    patch = executor.apply_logical_operation("enlarge", [patch], target_distance=5)

    assert builder.circuit.num_measurements == len(patch.syndrome_indices)
    assert builder.tracker.logicals.count == builder.tracker.expected_num_logicals == 1
    simulator = stim.TableauSimulator(seed=119)
    simulator.do_circuit(builder.circuit)
    assert simulator.peek_observable_expectation(
        _logical(patch, basis, system.num_qubits)
    ) == 1
    builder.apply_data_readout({q: basis for q in patch.data_indices})
    _assert_ideal(builder.to_stim_circuit())


@pytest.mark.parametrize("kwargs", [
    {"target_distance": 1},
    {"target_distance": 3},
    {"target_distance": 4},
    {"target_distance": True},
    {"target_distance": 5.0},
    {"target_distance": 5, "rounds": 0},
    {"target_distance": 5, "rounds": -1},
    {"target_distance": 5, "rounds": True},
    {"target_distance": 5, "rounds": 1.5},
    {"target_distance": 5, "direction": "+x"},
    {"target_distance": 5, "direction": "-x,-y"},
])
def test_invalid_enlarge_arguments_do_not_mutate_any_state(kwargs):
    system, patch, _, builder, executor = _setup()
    _prepare(builder, patch, "Z")
    before = _snapshot(system, patch, builder)
    with pytest.raises((TypeError, ValueError)):
        executor.apply_logical_operation("enlarge", [patch], **kwargs)
    assert _snapshot(system, patch, builder) == before


@pytest.mark.parametrize("geometry", ["distance_one", "rectangle", "transpose", "rotation", "defect", "unregistered"])
def test_unsupported_source_geometry_is_rejected_before_mutation(geometry):
    local = RotatedSurfaceCode(distance=3)
    if geometry == "distance_one":
        local = RotatedSurfaceCode(distance=1)
    elif geometry == "rectangle":
        local = RotatedSurfaceCode(distance_x=3, distance_z=5)
    elif geometry == "transpose":
        local.transpose_coords()
    elif geometry == "rotation":
        local.rotate_coords(math.pi / 2)
    system, patch, _, builder, executor = _setup(local=local)
    _prepare(builder, patch, "Z")
    if geometry == "defect":
        system.disable_data_qubits({min(patch.data_indices)})
    elif geometry == "unregistered":
        patch = local
    before = _snapshot(system, patch, builder)
    with pytest.raises((TypeError, ValueError)):
        executor.apply_logical_operation("enlarge", [patch], target_distance=7)
    assert _snapshot(system, patch, builder) == before


@pytest.mark.parametrize("other_active", [False, True], ids=["dormant", "active"])
def test_registered_patch_collisions_are_atomic_even_when_other_is_dormant(other_active):
    system, patch, _, builder, executor = _setup()
    other = system.add_patch(
        RotatedSurfaceCode(distance=3), offset=(8, 0),
        name="obstacle", is_active=other_active,
    )
    _prepare(builder, patch, "Z")
    if other_active:
        _prepare(builder, other, "Z")
    before = _snapshot(system, patch, builder)
    with pytest.raises(ValueError):
        executor.apply_logical_operation("enlarge", [patch], target_distance=5)
    assert _snapshot(system, patch, builder) == before
