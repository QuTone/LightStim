"""Native Pauli annotations must survive default record-history export."""

import pytest
import stim

from lightstim.ir.logical_history import LogicalHistoryRelation
from lightstim.ir.tracker import SyndromeTracker


def _tracker(circuit):
    tracker = SyndromeTracker(circuit.num_qubits)
    tracker.total_measurements = circuit.num_measurements
    tracker.logical_history = [LogicalHistoryRelation((0,), 0, (0,))]
    return tracker


@pytest.mark.parametrize("annotation", [
    "OBSERVABLE_INCLUDE(0) Z1",
    "OBSERVABLE_INCLUDE(0) rec[-1] Z1",
    "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(0) Z1",
])
def test_default_export_preserves_native_pauli_observable(annotation):
    circuit = stim.Circuit("R 0 1\nX_ERROR(0.1) 0\nM 0\n" + annotation)
    tracker = _tracker(circuit)
    original = circuit.copy()
    circuit.detector_error_model()
    assert tracker.append_logical_history_observables(circuit) == [1]
    assert circuit[:len(original)] == original
    assert circuit.num_observables == 2
    assert circuit[-1].targets_copy() == [stim.target_rec(-1)]
    assert tracker.append_logical_history_observables(circuit) == []
    dem = circuit.detector_error_model()
    # The new target still sees the physical X fault; preserving native
    # Pauli syntax must not suppress a record-history outcome.
    assert any(inst.type == "error" and stim.target_logical_observable_id(1)
               in inst.targets_copy() for inst in dem)


def test_record_basis_audit_rejects_pauli_annotations_without_mutation():
    circuit = stim.Circuit("R 0 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1] Z1")
    tracker = _tracker(circuit)
    original = circuit.copy()
    with pytest.raises(ValueError, match="independent_only=True requires record-only"):
        tracker.append_logical_history_observables(circuit, independent_only=True)
    assert circuit == original
    assert tracker.total_observables == 0
    # The failure of the optional record-space audit does not poison default export.
    assert tracker.append_logical_history_observables(circuit) == [1]


def test_pauli_contribution_to_history_id_is_not_silently_dropped():
    circuit = stim.Circuit("R 0 1\nM 0\n"
                           "OBSERVABLE_INCLUDE[logical-history](0) rec[-1]\n"
                           "OBSERVABLE_INCLUDE(0) Z1")
    tracker = _tracker(circuit)
    original = circuit.copy()
    with pytest.raises(ValueError, match="Tagged logical-history observables"):
        tracker.append_logical_history_observables(circuit)
    assert circuit == original
    assert tracker.total_observables == 0
