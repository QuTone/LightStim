"""Corner enlargement preserves unrotated/scaled logical states and resources."""

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
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)


class DoubleSpacingSurfaceCode(UnrotatedSurfaceCode):
    """Exercise the same geometric contract as the carving output subclass."""

    def build(self):
        super().build()
        self.qubit_coords = {q: (2 * x, 2 * y) for q, (x, y) in self.qubit_coords.items()}
        self.index_map = {xy: q for q, xy in self.qubit_coords.items()}
        self._rebuild_grid_map()
        for check in self.stabilizers:
            check["syn_coord"] = self.qubit_coords[check["syn_idx"]]

    def transform_vector(self, vector):
        x, y = super().transform_vector(vector)
        return 2 * x, 2 * y


def _setup(*, scaled=False, transposed=False, reference=False, distance=3):
    system = QECSystem()
    ref = system.add_patch(UnrotatedSurfaceCode(distance=3), name="ref") if reference else None
    cls = DoubleSpacingSurfaceCode if scaled else UnrotatedSurfaceCode
    local = cls(distance=distance)
    if transposed:
        local.transpose_coords()
    patch = system.add_patch(local, name="code", offset=(30.25, -10.5))
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    system.register_builder(builder)
    system.register_tracker(tracker)
    executor = LogicalExecutor(builder)
    executor.register_op_set(cls, UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock))
    if cls is not UnrotatedSurfaceCode:
        executor.register_op_set(UnrotatedSurfaceCode, UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock))
    builder.write_coordinates()
    return system, patch, ref, builder, executor


def _logical(patch, basis, n):
    if basis == "Y":
        return 1j * _logical(patch, "X", n) * _logical(patch, "Z", n)
    result = stim.PauliString(n)
    for q, axis in next(op for op in patch.logical_ops if op["type"] == basis)["pauli"].items():
        result[q] = axis
    return result


def _prepare(builder, patch, basis):
    builder.initialize({q: basis for q in patch.data_indices}, builder.system.num_qubits)


def _ordinary_round(builder):
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(builder.system, scheduling="4tick").circuit,
        rounds=1,
    )


def _assert_ideal(circuit):
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circuit.num_detectors > 0
    dets, obs = circuit.compile_detector_sampler(seed=8).sample(64, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


@pytest.mark.parametrize("scaled,transposed", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("basis,rounds,distance", [("X", 1, 5), ("Z", 2, 5), ("X", 1, 7)])
def test_enlarge_retains_data_and_uses_four_cnot_layers(scaled, transposed, basis, rounds, distance):
    system, source, _, builder, executor = _setup(scaled=scaled, transposed=transposed)
    _prepare(builder, source, basis)
    _ordinary_round(builder)
    old_coords = dict(source.qubit_coords)
    prefix = [i for i in builder.circuit.flattened() if i.name != "QUBIT_COORDS"]
    target = executor.apply_logical_operation("enlarge", [source], target_distance=distance, rounds=rounds)
    assert type(target) is type(source)
    assert target.is_transposed == transposed
    assert target.distance_x == target.distance_z == distance
    assert len(target.data_indices) == distance**2 + (distance - 1)**2
    assert len(target.stabilizers) == len(target.data_indices) - 1
    assert all(target.qubit_coords[q] == xy for q, xy in old_coords.items())
    assert source.data_indices <= target.data_indices
    assert system.data_indices.isdisjoint(system.syndrome_indices)
    assert system.num_logicals == builder.tracker.expected_num_logicals == 1
    assert set(system.patches) == {"code"}
    assert not system.coupler_patches
    assert not system.paused_stabilizer_indices
    instructions = [i for i in builder.circuit.flattened() if i.name != "QUBIT_COORDS"]
    assert instructions[:len(prefix)] == prefix
    delta = instructions[len(prefix):]
    resets = Counter(t.value for i in delta if i.name in {"R", "RX"}
                     for t in i.targets_copy() if t.value in target.data_indices)
    assert resets == Counter({q: 1 for q in target.data_indices - source.data_indices})
    assert sum(i.name == "CX" for i in delta) == 4 * rounds
    for i in delta:
        if i.name == "CX":
            ids = [t.value for t in i.targets_copy()]
            assert len(ids) == len(set(ids))
    assert sum(len(i.targets_copy()) for i in delta if i.name == "M") == rounds * len(target.stabilizers)
    assert all(i.tag != "post-select" for i in delta)
    builder.apply_data_readout({q: basis for q in target.data_indices})
    _assert_ideal(builder.to_stim_circuit())


@pytest.mark.parametrize("transposed", [False, True])
def test_enlarge_preserves_bell_channel_and_does_not_touch_spectator(transposed):
    system, source, ref, builder, executor = _setup(transposed=transposed, reference=True)
    _prepare(builder, source, "Z")
    _prepare(builder, ref, "X")
    _ordinary_round(builder)
    # Pair sites by inverse orientation, independent of exact-type executor
    # pair matching. This prepares the same encoded Bell channel in both frames.
    source_local = system.patches["code"][0]
    mapping = system.local_to_global_map
    pairs = [q for local_q in sorted(source_local.data_indices)
             for q in (mapping["ref"][local_q], mapping["code"][local_q])]
    unitary = stim.Circuit()
    unitary.append("CX", pairs)
    builder.apply_unitary_block(unitary)
    prefix = [i for i in builder.circuit.flattened() if i.name != "QUBIT_COORDS"]
    saved = pickle.dumps((system.patches["ref"], system.local_to_global_map["ref"]))
    target = executor.apply_logical_operation("enlarge", [source], target_distance=5, rounds=2)
    assert pickle.dumps((system.patches["ref"], system.local_to_global_map["ref"])) == saved
    delta = [i for i in builder.circuit.flattened() if i.name != "QUBIT_COORDS"][len(prefix):]
    assert all(not {t.value for t in i.targets_copy() if t.is_qubit_target} & set(ref.qubit_coords) for i in delta)
    for seed in range(8):
        simulator = stim.TableauSimulator(seed=seed)
        simulator.do_circuit(builder.circuit)
        for basis in ("X", "Z"):
            assert simulator.peek_observable_expectation(
                _logical(ref, basis, system.num_qubits) * _logical(target, basis, system.num_qubits)
            ) == 1
            assert simulator.peek_observable_expectation(_logical(target, basis, system.num_qubits)) == 0


@pytest.mark.parametrize("scaled,transposed", [(False, False), (True, True)])
def test_y_channel_followed_by_fold_s_dagger_and_x_readout(scaled, transposed):
    system, source, _, builder, executor = _setup(scaled=scaled, transposed=transposed)
    data = sorted(source.data_indices)
    local = {q: i for i, q in enumerate(data)}
    stabilizers = []
    for check in source.stabilizers:
        op = stim.PauliString(len(data))
        for q, axis in check["pauli"].items():
            op[local[q]] = axis
        stabilizers.append(op)
    logical_y = _logical(source, "Y", system.num_qubits)
    logical = stim.PauliString([logical_y[q] for q in data])
    logical.sign = logical_y.sign
    encoder = stim.Tableau.from_stabilizers(stabilizers + [logical]).to_circuit()
    physical = stim.Circuit()
    for i in encoder:
        physical.append(i.name, [data[t.value] for t in i.targets_copy()], i.gate_args_copy())
    _prepare(builder, source, "Z")
    builder.apply_unitary_block(physical)
    target = executor.apply_logical_operation("enlarge", [source], target_distance=5)
    simulator = stim.TableauSimulator(seed=5)
    simulator.do_circuit(builder.circuit)
    assert simulator.peek_observable_expectation(_logical(target, "Y", system.num_qubits)) == 1
    executor.apply_logical_operation("fold_transversal_s_dag", [target])
    builder.apply_data_readout({q: "X" for q in target.data_indices})
    _assert_ideal(builder.to_stim_circuit())


@pytest.mark.parametrize("kwargs", [
    {"target_distance": 3}, {"target_distance": 4}, {"target_distance": True},
    {"target_distance": 5, "rounds": 0}, {"target_distance": 5, "rounds": True},
    {"target_distance": 5, "direction": "-x,-y"},
])
def test_invalid_options_are_atomic(kwargs):
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, "Z")
    before = pickle.dumps((system, source, builder))
    with pytest.raises((ValueError, TypeError)):
        executor.apply_logical_operation("enlarge", [source], **kwargs)
    assert pickle.dumps((system, source, builder)) == before


def test_late_failure_is_atomic(monkeypatch):
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, "Z")
    _ordinary_round(builder)
    before = pickle.dumps((system, source, builder))
    def fail(staged, *args, **kwargs):
        assert staged is not builder
        assert staged.system.coupler_patches
        raise RuntimeError("deliberate adoption failure")
    monkeypatch.setattr(CircuitBuilder, "combine_patches", fail)
    with pytest.raises(RuntimeError, match="deliberate adoption failure"):
        executor.apply_logical_operation("enlarge", [source], target_distance=5)
    assert pickle.dumps((system, source, builder)) == before


def test_repeated_enlargement_preserves_registration():
    system, source, _, builder, executor = _setup(scaled=True, transposed=True)
    _prepare(builder, source, "X")
    _ordinary_round(builder)
    for distance in (5, 7):
        source = executor.apply_logical_operation("enlarge", [source], target_distance=distance)
        assert source.distance_x == source.distance_z == distance
        assert set(system.patches) == {"code"}
    _ordinary_round(builder)
    builder.apply_data_readout({q: "X" for q in source.data_indices})
    _assert_ideal(builder.to_stim_circuit())


def test_unprepared_code_is_rejected_without_seeding_constraints():
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, "Z")
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError, match="checks must be established"):
        executor.apply_logical_operation("enlarge", [source], target_distance=5)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("active", [False, True])
def test_growth_rejects_registered_patch_overlap(active):
    system, source, _, builder, executor = _setup()
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="obstacle", offset=(36.25, -10.5), is_active=active)
    _prepare(builder, source, "Z")
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError, match="overlaps registered patch"):
        executor.apply_logical_operation("enlarge", [source], target_distance=5)
    assert pickle.dumps((system, source, builder)) == before


def test_retired_source_handle_is_rejected():
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, "X")
    _ordinary_round(builder)
    executor.apply_logical_operation("enlarge", [source], target_distance=5)
    before = pickle.dumps((system, source, builder))
    with pytest.raises(ValueError):
        executor.apply_logical_operation("enlarge", [source], target_distance=7)
    assert pickle.dumps((system, source, builder)) == before


@pytest.mark.parametrize("scaled,transposed", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("distance,target_distance", [(3, 5), (3, 7), (5, 7)])
def test_exact_growth_flows_and_local_initialization(scaled, transposed, distance, target_distance):
    """Verify the channel, not just the noiseless readout of one input state.

    Signed Stim flows prove that both logical generators survive growth with
    no outcome-dependent byproduct. Together these preserve an arbitrary
    logical state, including one entangled with an untouched reference.
    Target checks flow to their actual syndrome records; random new checks
    are not silently assumed to have positive eigenvalue.
    """
    system, source, _, builder, executor = _setup(
        scaled=scaled, transposed=transposed, distance=distance,
    )
    _prepare(builder, source, "X")
    _ordinary_round(builder)
    prefix = [i for i in builder.circuit.flattened() if i.name != "QUBIT_COORDS"]
    target = executor.apply_logical_operation(
        "enlarge", [source], target_distance=target_distance, rounds=2,
    )
    suffix = stim.Circuit()
    for instruction in [i for i in builder.circuit.flattened()
                        if i.name != "QUBIT_COORDS"][len(prefix):]:
        suffix.append(instruction)

    ox = min(x for x, _ in target.data_coords)
    oy = min(y for _, y in target.data_coords)
    axes = np.column_stack((target.transform_vector((1, 0)),
                            target.transform_vector((0, 1))))
    reset_axes = {
        q.value: "X" if instruction.name == "RX" else "Z"
        for instruction in suffix if instruction.name in {"RX", "R"}
        for q in instruction.targets_copy() if q.value in target.data_indices
    }
    assert set(reset_axes) == target.data_indices - source.data_indices
    for q, basis in reset_axes.items():
        x, y = np.linalg.solve(axes, np.array(target.qubit_coords[q]) - (ox, oy))
        assert basis == ("X" if y >= x else "Z")

    for basis in ("X", "Z"):
        assert suffix.has_flow(stim.Flow(
            input=_logical(source, basis, system.num_qubits),
            output=_logical(target, basis, system.num_qubits),
        ))

    final_measurement = next(i for i in reversed(list(suffix)) if i.name == "M")
    measured = [q.value for q in final_measurement.targets_copy()]
    first_record = suffix.num_measurements - len(measured)
    for check in target.stabilizers:
        pauli = stim.PauliString(system.num_qubits)
        for q, axis in check["pauli"].items():
            pauli[q] = axis
        assert suffix.has_flow(stim.Flow(
            output=pauli,
            measurements=[first_record + measured.index(check["syn_idx"])],
        ))


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_noise_injection_keeps_detectors_and_input_distance(basis):
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, basis)
    _ordinary_round(builder)
    target = executor.apply_logical_operation("enlarge", [source], target_distance=5, rounds=2)
    builder.apply_data_readout({q: basis for q in target.data_indices})
    clean = builder.to_stim_circuit().copy()
    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_1q=.001, p_2q=.001, p_reset=.001, p_meas=.001, p_idle=.001),
        noise_model="circuit_level",
    )
    assert builder.circuit == clean
    assert noisy.num_detectors == clean.num_detectors
    assert noisy.num_observables == clean.num_observables == 1
    dem = noisy.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0
    # The complete experiment includes distance-3 preparation before growth.
    # Enlargement cannot remove a logical fault already present at its input.
    assert len(dem.shortest_graphlike_error()) == 3
    _assert_ideal(noisy.without_noise())


def test_enlargement_refreshes_legacy_z_only_readout_offsets():
    system, source, _, builder, executor = _setup()
    _prepare(builder, source, "Z")
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1, z_only=True,
    )
    target = executor.apply_logical_operation("enlarge", [source], target_distance=5, rounds=2)
    builder.apply_data_readout({q: "Z" for q in target.data_indices}, z_only=True)
    _assert_ideal(builder.to_stim_circuit())
    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_2q=.001, p_meas=.001), noise_model="circuit_level",
    )
    assert noisy.detector_error_model(decompose_errors=True).num_errors > 0
