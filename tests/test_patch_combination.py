"""Patch adoption preserves physical state and fails before committing."""

from copy import deepcopy
import pickle

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.coupler import LogicalCouplerProtocol
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code._operation_utils import require_global_patch
from lightstim.utils.linear_algebra import solve_linear_decomposition


class _Repetition(QECPatch):
    def _process_params(self):
        self.length = self.params["length"]
        self.origin = self.params.get("origin", 0)

    def build(self):
        for i in range(self.length):
            self.add_qubit(self.origin + 2 * i, 0, "data")
        for i in range(self.length - 1):
            self.add_qubit(self.origin + 2 * i + 1, 0, "syndrome_z")
            self.create_stim_stabilizer(
                {(self.origin + 2 * i, 0): "Z", (self.origin + 2 * i + 2, 0): "Z"},
                (self.origin + 2 * i + 1, 0), "Z",
            )
        self.create_stim_logical({(self.origin, 0): "Z"}, "Z")
        self.create_stim_logical({(self.origin + 2 * i, 0): "X"
                                  for i in range(self.length)}, "X")
        self.num_logicals = 1


class _Extension(LogicalCouplerProtocol):
    EXPECTED_PATCH_COUNT = 1

    def _build_coupler_geometry(self, coupler_patch, patches, **params):
        for x in (6, 8):
            coupler_patch.add_qubit(x, 0, "data")
        for x in (5, 7):
            coupler_patch.add_qubit(x, 0, "syndrome_z")
            coupler_patch.stabilizers.append({
                "pauli": {(x - 1, 0): "Z", (x + 1, 0): "Z"},
                "type": "Z", "syn_coord": (x, 0),
            })
        if params.get("replace_boundary"):
            coupler_patch.conflicting_stabilizer_coords = {(3, 0)}
            coupler_patch.stabilizers.append({
                "pauli": {(0, 0): "Z", (4, 0): "Z"},
                "type": "Z", "syn_coord": (3, 0),
            })


def _extract(builder):
    chunk = stim.Circuit()
    syndromes = builder.system.active_syndrome_indices
    chunk.append("R", syndromes)
    for check in builder.system.active_stabilizers:
        syn = check["syn_idx"]
        for data in check["data_indices"]:
            chunk.append("CX", [data, syn])
    chunk.append("M", syndromes)
    builder.apply_syndrome_extraction(chunk, rounds=1)


def _fixture(*, measured=True, replace_boundary=False):
    system = QECSystem()
    spectator = system.add_patch(_Repetition(length=3, origin=100), name="spectator")
    original = system.add_patch(_Repetition(length=3), name="data")
    system.register_coupler(_Extension(), ["data"], name="extension",
                            replace_boundary=replace_boundary)
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in spectator.data_indices | original.data_indices}, system.num_qubits)
    _extract(builder)
    extension = system.coupler_patches["extension"]
    mapping = system.local_to_global_map["extension"]
    builder.initialize({mapping[q]: "X" for q in extension.data_indices}, system.num_qubits)
    builder.activate_coupler("extension")
    if measured:
        _extract(builder)
    return builder, original, spectator


def _snapshot(builder):
    return pickle.dumps((builder.system.__dict__, builder.tracker.__dict__, str(builder.circuit)))


def _parities(tracker, operators):
    matrix = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    records = tracker.stabilizers.records + tracker.logicals.records
    coeffs, known, _ = solve_linear_decomposition(matrix, operators, reduce_weight=False)
    assert np.all(known)
    result = []
    for row in coeffs:
        parity = set()
        for i in np.flatnonzero(row):
            parity.symmetric_difference_update(records[i])
        result.append(parity)
    return result


def test_combination_preserves_ids_state_records_spectator_and_circuit():
    builder, original, spectator = _fixture()
    system, tracker = builder.system, builder.tracker
    circuit = builder.circuit.copy()
    before = deepcopy(tracker.__dict__)
    active = set(system.active_stabilizer_indices)
    coords, signatures = dict(system.qubit_coords), dict(system._stabilizer_signatures)
    old_spectator = system.patches["spectator"]
    old_num_qubits, old_num_logicals = system.num_qubits, system.num_logicals
    target = _Repetition(length=5)
    target_before = deepcopy(target.__dict__)

    output = builder.combine_patches(["data", "extension"], target, name="data")

    assert system.num_qubits == old_num_qubits
    assert system.num_logicals == old_num_logicals
    assert system.qubit_coords == coords
    assert system._stabilizer_signatures == signatures
    assert system.active_stabilizer_indices == active
    assert builder.circuit == circuit
    assert system.patches["spectator"] is old_spectator
    require_global_patch(builder, spectator)
    require_global_patch(builder, output)
    with pytest.raises(ValueError, match="global patch"):
        require_global_patch(builder, original)
    assert system.coupler_patches == {}
    assert system.paused_stabilizer_indices == {}
    assert "extension" not in system.local_to_global_map
    assert target.__dict__ == target_before
    assert any(q != g for q, g in system.local_to_global_map["data"].items())
    assert output.data_indices == {system.index_map[(x, 0)] for x in range(0, 10, 2)}
    assert system.patches["data"][0].data_indices == target.data_indices
    assert all(system.index_to_owner_map[q] == "data" for q in output.qubit_coords)
    assert pickle.dumps(tracker.__dict__) == pickle.dumps(before)

    _extract(builder)
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    samples, observables = builder.to_stim_circuit().compile_detector_sampler(seed=5).sample(
        20, separate_observables=True,
    )
    assert not np.any(samples)
    assert not np.any(observables)


def test_safe_reclassification_preserves_known_operator_parities():
    builder, _, _ = _fixture()
    tracker = builder.tracker
    # An established row was placed in the logical table by an upstream
    # extraction path. Adoption may reclassify it, but cannot measure it.
    old = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    parities = _parities(tracker, old)
    tracker.logicals.matrix = np.vstack([tracker.logicals.matrix, tracker.stabilizers.matrix[:1]])
    tracker.logicals.records += tracker.stabilizers.records[:1]
    tracker.stabilizers.matrix = tracker.stabilizers.matrix[1:]
    tracker.stabilizers.records = tracker.stabilizers.records[1:]
    tracker._classified_stabilizer_rows = set(range(tracker.stabilizers.count))
    before_circuit = builder.circuit.copy()
    before_measurements = tracker.total_measurements

    builder.combine_patches(["data", "extension"], _Repetition(length=5), name="grown")

    assert _parities(tracker, old) == parities
    tracker.validate_logical_count()
    assert tracker.total_measurements == before_measurements
    assert builder.circuit == before_circuit
    assert set(builder.system.patches) == {"spectator", "grown"}


def test_only_active_checks_are_adopted_and_paused_checks_stay_retired():
    builder, original, _ = _fixture(replace_boundary=True)
    system = builder.system
    retired = set(system.paused_stabilizer_indices["extension"])
    assert len(retired) == 1
    old_check_records = deepcopy([system.stabilizers[u] for u in sorted(retired)])
    old_check_count = len(system.stabilizers)
    target = _Repetition(length=5)
    target.stabilizers[1]["pauli"] = {0: "Z", 2: "Z"}
    target.stabilizers[1]["data_indices"] = [0, 2]

    output = builder.combine_patches(["data", "extension"], target, name="data")

    assert not retired & output._registered_stabilizer_uids
    assert not retired & system.active_stabilizer_indices
    assert len(system.stabilizers) == old_check_count
    assert [system.stabilizers[u] for u in sorted(retired)] == old_check_records
    assert system.paused_stabilizer_indices == {}
    assert output._registered_stabilizer_uids != original._registered_stabilizer_uids


@pytest.mark.parametrize("fault", ["geometry", "checks", "logical", "inactive", "absorbed", "external"])
def test_rejection_is_atomic(fault):
    builder, _, _ = _fixture()
    target = _Repetition(length=5)
    if fault == "geometry":
        target.shift_coords(1, 0)
    elif fault == "checks":
        target.stabilizers[0]["pauli"] = {0: "Z"}
        target.stabilizers[0]["data_indices"] = [0]
    elif fault == "logical":
        target.logical_ops[1] = deepcopy(target.logical_ops[0])
        target.logical_ops[1]["type"] = "X"
    elif fault == "inactive":
        builder.deactivate_coupler("extension")
    elif fault == "absorbed":
        builder.tracker.record_absorbed_op(builder.tracker.logicals.matrix[0])
    elif fault == "external":
        other = deepcopy(builder.system.coupler_patches["extension"])
        builder.system.patches["other"] = (other, (0, 0))
        builder.system.coupler_patches["other"] = other
        builder.system.local_to_global_map["other"] = {}
    before = _snapshot(builder)

    with pytest.raises((ValueError, RuntimeError)):
        builder.combine_patches(["data", "extension"], target, name="grown")

    assert _snapshot(builder) == before


def test_unmeasured_target_cannot_be_established_by_combination():
    builder, _, _ = _fixture(measured=False)
    before = _snapshot(builder)
    with pytest.raises(ValueError, match="not established"):
        builder.combine_patches(["data", "extension"], _Repetition(length=5), name="grown")
    assert _snapshot(builder) == before
