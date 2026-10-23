"""Unrotated contraction preserves a channel and its measured Pauli frame."""

import copy
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
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic


class DoubleSpacingSurfaceCode(UnrotatedSurfaceCode):
    """The geometric contract used by the carving output family."""

    def build(self):
        super().build()
        self.qubit_coords = {q: (2*x, 2*y) for q, (x, y) in self.qubit_coords.items()}
        self.index_map = {xy: q for q, xy in self.qubit_coords.items()}
        self._rebuild_grid_map()
        for check in self.stabilizers:
            check["syn_coord"] = self.qubit_coords[check["syn_idx"]]

    def transform_vector(self, vector):
        x, y = super().transform_vector(vector)
        return 2*x, 2*y


def _setup(distance=5, basis="X", *, scaled=False, transposed=False,
           reference=False, source_rounds=1):
    system = QECSystem()
    ref = system.add_patch(UnrotatedSurfaceCode(distance=distance), name="ref",
                           offset=(-40, 0)) if reference else None
    cls = DoubleSpacingSurfaceCode if scaled else UnrotatedSurfaceCode
    local = cls(distance=distance)
    if transposed:
        local.transpose_coords()
    source = system.add_patch(local, name="patch", offset=(12.25, -8.5))
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    system.register_builder(builder)
    system.register_tracker(tracker)
    executor = LogicalExecutor(builder)
    for patch_type in {UnrotatedSurfaceCode, cls}:
        executor.register_op_set(patch_type, UnrotatedSurfaceCodeLogicalOpSet(
            UnrotatedSurfaceCodeExtractionBlock))
    builder.write_coordinates()
    init = {q: "Z" if ref else basis for q in source.data_indices}
    if ref:
        init.update({q: "X" for q in ref.data_indices})
    builder.initialize(init, system.num_qubits)
    if source_rounds:
        _ordinary_round(builder, rounds=source_rounds)
    if ref:
        # Physical IDs preserve native local site correspondence through a
        # transpose or scale, unlike a comparison of global coordinates.
        mapping = system.local_to_global_map
        pairs = [q for local_q in sorted(local.data_indices)
                 for q in (mapping["ref"][local_q], mapping["patch"][local_q])]
        unitary = stim.Circuit()
        unitary.append("CX", pairs)
        builder.apply_unitary_block(unitary)
    return system, source, ref, builder, executor


def _ordinary_round(builder, *, rounds=1, z_only=False):
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(builder.system, scheduling="4tick").circuit,
        rounds=rounds, z_only=z_only,
    )


def _pauli(record, n):
    operator = stim.PauliString(n)
    for q, axis in record["pauli"].items():
        operator[q] = axis
    return operator


def _logical(patch, basis, n):
    if basis == "Y":
        return 1j * _logical(patch, "X", n) * _logical(patch, "Z", n)
    return _pauli(next(op for op in patch.logical_ops if op["type"] == basis), n)


def _native_coords(patch):
    origin = (min(x for x, _ in patch.data_coords), min(y for _, y in patch.data_coords))
    axes = np.column_stack((patch.transform_vector((1, 0)),
                            patch.transform_vector((0, 1))))
    return {q: np.linalg.solve(axes, np.array(xy)-origin)
            for q, xy in patch.qubit_coords.items()}


def _removed_records(source, target, first_record):
    coords = _native_coords(source)
    removed = sorted(source.data_indices-target.data_indices)
    mx = [q for q in removed if coords[q][1] >= coords[q][0]]
    mz = [q for q in removed if q not in mx]
    records = {q: first_record+i for i, q in enumerate(mx+mz)}
    return coords, mx, mz, records


def _ideal(builder):
    circuit = builder.to_stim_circuit()
    circuit.detector_error_model(decompose_errors=True)
    detections, observables = circuit.compile_detector_sampler(seed=19).sample(
        64, separate_observables=True)
    assert not detections.any()
    assert not observables.any()
    return circuit


@pytest.mark.parametrize("scaled,transposed", [(False, False), (False, True),
                                               (True, False), (True, True)])
@pytest.mark.parametrize("distances", [(5, 3), (7, 3), (7, 5)])
def test_exact_signed_shrink_flows_and_native_measurement_bases(scaled, transposed, distances):
    """State-independent flows prove the channel, including its random frame."""
    distance, target_distance = distances
    system, source, _, builder, executor = _setup(
        distance, scaled=scaled, transposed=transposed)
    prefix = len(builder.circuit)
    target = executor.apply_logical_operation(
        "shrink", [source], target_distance=target_distance)
    suffix = builder.circuit[prefix:].flattened()
    coords, mx, mz, records = _removed_records(source, target, 0)
    measured = {q.value: "X" if i.name == "MX" else "Z"
                for i in suffix if i.name in {"MX", "M"}
                for q in i.targets_copy()}
    assert measured == {q: "X" if coords[q][1] >= coords[q][0] else "Z"
                        for q in source.data_indices-target.data_indices}
    assert suffix.num_measurements == len(mx)+len(mz)
    assert not any(i.name in {"R", "RX", "RY", "CX", "MPP"} for i in suffix)
    assert all(i.tag != "post-select" for i in suffix)
    assert type(target) is type(source)
    assert target.is_transposed == transposed

    for basis in ("X", "Z"):
        old = _logical(source, basis, system.num_qubits)
        new = _logical(target, basis, system.num_qubits)
        lost = [records[q] for q in source.data_indices-target.data_indices if old[q]]
        assert suffix.has_flow(stim.Flow(input=old, output=new, measurements=lost))
        assert not suffix.has_flow(stim.Flow(input=old, output=new))
    for check in target.stabilizers:
        old = next(s for s in source.stabilizers if s["syn_coord"] == check["syn_coord"])
        lost = set(old["pauli"])-set(check["pauli"])
        assert all(measured[q] == old["pauli"][q] for q in lost)
        assert suffix.has_flow(stim.Flow(
            input=_pauli(old, system.num_qubits),
            output=_pauli(check, system.num_qubits),
            measurements=[records[q] for q in sorted(lost)],
        ))


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_shrink_retains_ids_closes_detectors_and_counts_native_se(basis, rounds):
    system, source, _, builder, executor = _setup(basis=basis)
    tracker = builder.tracker
    old_coords = dict(system.qubit_coords)
    old_n = system.num_qubits
    prefix = len(builder.circuit)
    first_record = builder.circuit.num_measurements
    target = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=rounds)
    assert builder.system is system and builder.tracker is tracker
    assert target.distance_x == target.distance_z == 3
    assert source.distance_x == 5 and target is not source
    assert set(system.patches) == {"patch"} and not system.coupler_patches
    assert target.qubit_coords.items() <= old_coords.items()
    assert system.qubit_coords == old_coords and system.num_qubits == tracker.num_qubits == old_n
    assert system.data_indices == system.active_qubit_indices == target.data_indices
    assert system.syndrome_indices == target.syndrome_indices
    assert set(system.index_to_owner_map) == set(target.qubit_coords)
    assert set(system.coord_to_owner_map) == set(target.qubit_coords.values())
    assert system.num_logicals == tracker.expected_num_logicals == tracker.logicals.count == 1
    assert builder.circuit.num_observables == 0
    delta = builder.circuit[prefix:].flattened()
    assert sum(i.name == "CX" for i in delta) == 4*rounds
    assert builder.circuit.num_measurements-first_record == 28 + 12*rounds
    for instruction in delta:
        if instruction.name in {"R", "RX", "RY"}:
            assert not {t.value for t in instruction.targets_copy()} & source.data_indices
        if instruction.name == "CX":
            ids = [t.value for t in instruction.targets_copy()]
            assert len(ids) == len(set(ids))
    # A subsequent ordinary round must discover only the retained code.
    _ordinary_round(builder)
    builder.apply_data_readout({q: basis for q in target.data_indices})
    assert _ideal(builder).num_observables == 1


@pytest.mark.parametrize("scaled,transposed,distances,rounds", [
    (False, False, (5, 3), 0), (False, True, (7, 3), 1),
    (True, False, (7, 5), 0), (True, True, (5, 3), 2),
])
def test_bell_spectator_and_both_measured_logical_byproducts(scaled, transposed, distances, rounds):
    distance, target_distance = distances
    system, source, ref, builder, executor = _setup(
        distance, scaled=scaled, transposed=transposed, reference=True)
    first_record, prefix = builder.circuit.num_measurements, len(builder.circuit)
    spectator = copy.deepcopy((system.patches["ref"][0].__dict__, system.patches["ref"][1],
                               system.local_to_global_map["ref"],
                               [s for s in system.stabilizers if s.get("patch_name") == "ref"]))
    target = executor.apply_logical_operation(
        "shrink", [source], target_distance=target_distance, rounds=rounds)
    assert spectator == (system.patches["ref"][0].__dict__, system.patches["ref"][1],
                         system.local_to_global_map["ref"],
                         [s for s in system.stabilizers if s.get("patch_name") == "ref"])
    assert all(not {t.value for t in i.targets_copy() if t.is_qubit_target} & set(ref.qubit_coords)
               for i in builder.circuit[prefix:].flattened())
    coords, mx, mz, records = _removed_records(source, target, first_record)
    seen = set()
    for seed in range(24):
        sim = stim.TableauSimulator(seed=seed)
        sim.do_circuit(builder.circuit)
        record = sim.current_measurement_record()
        px = sum(record[records[q]] for q in mx if coords[q][0] == 0) % 2
        pz = sum(record[records[q]] for q in mz if coords[q][1] == 0) % 2
        seen.add((px, pz))
        for basis, parity in (("X", px), ("Z", pz)):
            left, right = _logical(ref, basis, system.num_qubits), _logical(target, basis, system.num_qubits)
            assert sim.peek_observable_expectation(left*right) == (-1)**parity
            assert sim.peek_observable_expectation(right) == 0
        if px:
            sim.do_pauli_string(_logical(target, "Z", system.num_qubits))
        if pz:
            sim.do_pauli_string(_logical(target, "X", system.num_qubits))
        for basis in ("X", "Z"):
            assert sim.peek_observable_expectation(
                _logical(ref, basis, system.num_qubits)*_logical(target, basis, system.num_qubits)) == 1
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
    builder.apply_data_readout({q: "X" for q in ref.data_indices | target.data_indices})
    _ideal(builder)


def test_target_check_records_are_old_syndromes_xor_removed_data():
    system, source, _, builder, executor = _setup(scaled=True, transposed=True)
    old_records = {s["syn_coord"]: set(builder.tracker.stabilizers.records[i])
                   for i, s in enumerate(system.active_stabilizers)}
    first_record = builder.circuit.num_measurements
    target = executor.apply_logical_operation("shrink", [source], target_distance=3)
    _, _, _, records = _removed_records(source, target, first_record)
    basis = stabilizers_to_symplectic(system, target.stabilizers, system.num_qubits)
    coeffs, dependent, _ = solve_linear_decomposition(
        builder.tracker.stabilizers.matrix, basis, reduce_weight=False)
    assert dependent.all()
    for check, coefficients in zip(target.stabilizers, coeffs):
        old = next(s for s in source.stabilizers if s["syn_coord"] == check["syn_coord"])
        expected = old_records[check["syn_coord"]] ^ {
            records[q] for q in set(old["pauli"])-set(check["pauli"])}
        actual = set()
        for i in np.flatnonzero(coefficients):
            actual.symmetric_difference_update(builder.tracker.stabilizers.records[i])
        assert actual == expected


@pytest.mark.parametrize("basis,scaled,transposed", [
    ("X", False, False), ("Z", True, True), ("Y", False, False), ("Y", True, True),
])
def test_coherent_input_including_y_needs_no_source_or_target_se(basis, scaled, transposed):
    system, source, _, builder, executor = _setup(
        basis="Z", scaled=scaled, transposed=transposed, source_rounds=0)
    data = sorted(source.data_indices)
    full_stabilizers = [_pauli(c, system.num_qubits) for c in source.stabilizers]
    full_stabilizers.append(_logical(source, basis, system.num_qubits))
    stabilizers = []
    for full in full_stabilizers:
        compressed = stim.PauliString([full[q] for q in data])
        compressed.sign = full.sign
        stabilizers.append(compressed)
    physical = stim.Circuit()
    for i in stim.Tableau.from_stabilizers(stabilizers).to_circuit():
        physical.append(i.name, [data[t.value] for t in i.targets_copy()], i.gate_args_copy())
    builder.apply_unitary_block(physical)
    assert builder.circuit.num_measurements == 0
    target = executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert builder.circuit.num_measurements == 28
    if basis == "Y":
        executor.apply_logical_operation("fold_transversal_s_dag", [target])
    builder.apply_data_readout({q: "X" if basis == "Y" else basis for q in target.data_indices})
    _ideal(builder)


@pytest.mark.parametrize("basis,scaled,transposed", [("X", False, False), ("Z", True, True)])
def test_repeated_shrink_regrowth_reuses_physical_ids_and_preserves_history(basis, scaled, transposed):
    system, patch, _, builder, executor = _setup(7, basis, scaled=scaled, transposed=transposed)
    old_coords = dict(system.qubit_coords)
    for op, distance, rounds in [("shrink", 5, 0), ("shrink", 3, 1),
                                  ("enlarge", 7, 1), ("shrink", 3, 0)]:
        patch = executor.apply_logical_operation(op, [patch], target_distance=distance, rounds=rounds)
        assert system.qubit_coords == old_coords
        assert system.num_qubits == len(old_coords)
        assert set(system.index_to_owner_map) == set(patch.qubit_coords)
    builder.apply_data_readout({q: basis for q in patch.data_indices})
    _ideal(builder)
    noisy = builder.build_noisy_circuit(NoiseConfig(p_1q=.001, p_2q=.001, p_meas=.001,
                                                   p_reset=.001, p_idle=.001))
    assert noisy.detector_error_model(decompose_errors=True).num_errors > 0


def test_released_region_can_host_a_fresh_patch_and_transversal_cnot():
    system, source, _, builder, executor = _setup(7)
    old_coords = dict(system.index_map)
    target = executor.apply_logical_operation("shrink", [source], target_distance=3)
    retired_uids = source._registered_stabilizer_uids-target._registered_stabilizer_uids
    fresh = system.add_patch(UnrotatedSurfaceCode(distance=3), name="fresh", offset=(20.25, -.5))
    assert all(q == old_coords[xy] for q, xy in fresh.qubit_coords.items())
    assert not fresh._registered_stabilizer_uids & retired_uids
    assert all(system.stabilizers[uid]["patch_name"] == "fresh"
               for uid in fresh._registered_stabilizer_uids)
    builder.initialize({q: "Z" for q in fresh.data_indices}, system.num_qubits)
    _ordinary_round(builder)
    executor.apply_logical_operation("transversal_cnot", [target, fresh])
    builder.apply_data_readout({q: "Z" for q in target.data_indices | fresh.data_indices})
    _ideal(builder)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_noise_preserves_detectors_and_has_distance_three(basis):
    _, source, _, builder, executor = _setup(basis=basis, source_rounds=2)
    target = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=2)
    builder.apply_data_readout({q: basis for q in target.data_indices})
    clean = _ideal(builder).copy()
    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_1q=.001, p_2q=.001, p_meas=.001, p_reset=.001, p_idle=.001),
        noise_model="circuit_level")
    assert builder.circuit == clean
    assert noisy.num_detectors == clean.num_detectors
    assert noisy.num_observables == clean.num_observables == 1
    dem = noisy.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0
    assert len(dem.shortest_graphlike_error()) == 3


def test_idle_noise_uses_source_then_only_retained_data():
    _, source, _, builder, executor = _setup()
    target = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=1)
    builder.apply_data_readout({q: "X" for q in target.data_indices})
    noisy = builder.build_noisy_circuit(NoiseConfig(p_idle=.001), noise_model="code_capacity")
    idle = [{t.value for t in i.targets_copy()} for i in noisy.flattened() if i.name == "DEPOLARIZE1"]
    assert idle == [source.data_indices, target.data_indices]


@pytest.mark.parametrize("rounds", [0, 1])
def test_legacy_z_only_readout_requires_current_target_round(rounds):
    system, source, _, builder, executor = _setup(basis="Z", source_rounds=0)
    _ordinary_round(builder, z_only=True)
    target = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=rounds)
    if not rounds:
        before = str(builder.circuit)
        with pytest.raises(ValueError, match="current target-code SE"):
            builder.apply_data_readout({q: "Z" for q in target.data_indices}, z_only=True)
        assert str(builder.circuit) == before
    builder.apply_data_readout({q: "Z" for q in target.data_indices}, z_only=bool(rounds))
    _ideal(builder)


def test_target_only_se_keeps_spectator_legacy_cache_invalid():
    system, source, _, builder, executor = _setup(basis="Z", source_rounds=0)
    spectator = system.add_patch(UnrotatedSurfaceCode(distance=3), name="spectator", offset=(-20, 0))
    builder.initialize({q: "Z" for q in spectator.data_indices}, system.num_qubits)
    _ordinary_round(builder, z_only=True)
    target = executor.apply_logical_operation("shrink", [source], target_distance=3, rounds=1)
    measurements = {q: "Z" for q in target.data_indices | spectator.data_indices}
    before = pickle.dumps((system, builder))
    with pytest.raises(ValueError, match="current target-code SE"):
        builder.apply_data_readout(measurements, z_only=True)
    assert pickle.dumps((system, builder)) == before
    _ordinary_round(builder, z_only=True)
    builder.apply_data_readout(measurements, z_only=True)
    _ideal(builder)


@pytest.mark.parametrize("kwargs", [
    {"target_distance": 5}, {"target_distance": 7}, {"target_distance": 1},
    {"target_distance": 2}, {"target_distance": 3.0}, {"target_distance": True},
    {"target_distance": 3, "rounds": -1}, {"target_distance": 3, "rounds": True},
    {"target_distance": 3, "direction": "+x,+y"},
])
def test_invalid_parameters_are_atomic(kwargs):
    system, source, _, builder, executor = _setup()
    before = pickle.dumps((system, source, builder))
    with pytest.raises((ValueError, TypeError)):
        executor.apply_logical_operation("shrink", [source], **kwargs)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("condition", ["rotation", "defect", "inactive", "unregistered", "geometry"])
def test_unsupported_sources_are_atomic(condition):
    system, source, _, builder, executor = _setup()
    if condition == "rotation":
        source.rotate_coords(math.pi/2)
    elif condition == "defect":
        system.disable_data_qubits({min(source.data_indices)})
    elif condition == "inactive":
        system.active_stabilizer_indices.discard(min(source._registered_stabilizer_uids))
    elif condition == "unregistered":
        source = UnrotatedSurfaceCode(distance=5)
    else:
        source.qubit_coords[min(source.data_indices)] = (100, 100)
    before = pickle.dumps((system, source, builder))
    with pytest.raises((ValueError, TypeError)):
        executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("active", [False, True])
def test_boundary_dependencies_prevent_shrinking_even_for_inactive_couplers(active):
    from lightstim.qec_code.surface_code.unrotated.enlargement import _EnlargementCoupler
    system, source, _, builder, executor = _setup()
    local_target = UnrotatedSurfaceCode(distance=7)
    local_target.shift_coords(12.25, -8.5)
    system.register_coupler(_EnlargementCoupler(), ["patch"], name="pending_growth",
                            target_patch=local_target)
    if active:
        builder.activate_coupler("pending_growth")
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError):
        executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("failure", ["rebase_stabilizers_onto_code_basis", "apply_syndrome_extraction"])
def test_late_failure_after_measurement_and_adoption_is_atomic(monkeypatch, failure):
    system, source, _, builder, executor = _setup()
    before = pickle.dumps((system, source, builder))
    tracker = builder.tracker
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate contraction failure")
    monkeypatch.setattr(SyndromeTracker if failure.startswith("rebase") else CircuitBuilder,
                        failure, fail)
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
    _ordinary_round(builder)
    executor.apply_logical_operation("shrink", [source], target_distance=3)
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError, match="global patch"):
        executor.apply_logical_operation("shrink", [source], target_distance=3)
    assert pickle.dumps((system, source, builder)) == before
