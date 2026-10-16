"""Projection/reset preserve selected subspaces and unknown eigenvalues."""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.tableau import PauliTableau
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.subsystem_algebra import independent_row_indices
from tests.test_tracker_v2_readout import _rows, _assert_complete


def test_v2_repeated_check_preserves_its_local_record_representative():
    tracker = SyndromeTracker(2, logical_tracking="slabs")
    full = PauliTableau(2)
    full.add_stabilizers(_rows("ZI", "IZ", "ZZ", n=2), [[0], [1], [2]])
    kind, records, logical_indices = tracker._v2_commuting_relation(
        full, 3, 0, _rows("ZZ", n=2)[0], _rows("ZZ", n=2), 3, 0)
    # All three S rows are valid and bare. Replacing the last row by ZI*IZ
    # would give m0+m1+m3 instead of the adjacent same-check detector m2+m3.
    assert (kind, records, logical_indices) == ("detector", [2, 3], ())
    physical = stim.Circuit("R 0 1\nM 0 1\nMPP Z0*Z1\nMPP Z0*Z1")
    assert physical.has_flow(stim.Flow(measurements=records), unsigned=False)


def test_v2_exact_bare_precedes_repeated_measurement_representative():
    tracker = SyndromeTracker(2, logical_tracking="slabs")
    full = PauliTableau(2)
    full.add_stabilizers(_rows("ZZ", n=2), [[0]])
    result = tracker._v2_commuting_relation(
        full, 1, 0, _rows("ZZ", n=2)[0], _rows("ZZ", "ZZ", n=2), 1, 1)
    assert result == ("detector", [0, 2], ())


def test_v2_repeated_measurement_is_a_detector_after_a_logical_pivot():
    tracker = SyndromeTracker(2, logical_tracking="slabs")
    full = PauliTableau(2)
    # The first random ZZ measurement replaced an L row, so its fresh value
    # is still in the L portion until C refresh. It is already repeat-check
    # information within this B step.
    full.add_stabilizers(_rows("ZZ", n=2), [[0]])
    result = tracker._v2_commuting_relation(
        full, 0, 1, _rows("ZZ", n=2)[0], _rows("ZZ", "ZZ", n=2), 0, 1)
    assert result == ("detector", [0, 1], ())


@pytest.mark.parametrize("unknown_bare", [False, True])
def test_v2_exact_match_does_not_turn_protected_or_unknown_S_into_a_detector(unknown_bare):
    tracker = SyndromeTracker(2, logical_tracking="slabs")
    full = PauliTableau(2)
    full.add_stabilizers(_rows("ZZ", n=2), [[-1] if unknown_bare else [0]])
    if unknown_bare:
        # The known value comes from L, not from the matching unknown S row.
        full.add_stabilizers(_rows("ZZ", n=2), [[]])
    else:
        tracker._dressed_stabilizer_rows = {0}
    kind, records, _ = tracker._v2_commuting_relation(
        full, 1, int(unknown_bare), _rows("ZZ", n=2)[0], _rows("ZZ", n=2), 1, 0)
    assert kind == "logical"
    assert records == ([1] if unknown_bare else [0, 1])


def _selected_matrix(tracker):
    return tracker.stabilizers.matrix[sorted(tracker.post_select_row_indices)]


def _mid(tracker, circuit, paulis, qubits, bases, *, reset_paulis=None, unitary=None):
    tracker.process_mid_measurement(
        circuit=circuit,
        forward_symplectic_matrix=(np.eye(2*tracker.num_qubits, dtype=np.uint8)
                                  if unitary is None else tracker.get_forward_symplectic_matrix(
                                      unitary, tracker.num_qubits)),
        back_propagated_paulis=_rows(*paulis, n=tracker.num_qubits),
        reset_paulis=reset_paulis,
        measurement_qubit_indices=qubits, measurement_bases=bases,
        measurement_coords=[(q, 0) for q in qubits],
        discarded_measurement_qubit_indices=set() if reset_paulis is None else set(qubits))


@pytest.mark.parametrize("entry", ["process_resets", "process_initialization", "reset_records_for_qubits"])
def test_v2_reset_unknown_products_never_create_a_known_eigenvalue(entry):
    tracker = SyndromeTracker(2)
    tracker.stabilizers.add_stabilizers(_rows("XX", "XI", n=2), [[-1], [-1]])
    if entry == "reset_records_for_qubits":
        tracker.reset_records_for_qubits([0])
        tracker.process_initialization(_rows("ZI", n=2))
    else:
        getattr(tracker, entry)(_rows("ZI", n=2))
    residual = [records for row, records in zip(tracker.stabilizers.matrix, tracker.stabilizers.records)
                if np.array_equal(row, _rows("IX", n=2)[0])]
    assert residual and all(-1 in records for records in residual)
    circuit = stim.Circuit("RX 0\nR 1 0\nMX 1")
    tracker.process_data_measurement(circuit, _rows("IX", n=2), {0: (0, 0), 1: (1, 0)})
    assert circuit.num_detectors == circuit.num_observables == 0
    _assert_complete(tracker, circuit)


@pytest.mark.parametrize("disposable", [False, True])
def test_v2_mid_refresh_does_not_independently_postselect_factors(disposable):
    n = 4 if disposable else 2
    tracker = SyndromeTracker(n)
    tracker.stabilizers.add_stabilizers(_rows("ZZII" if disposable else "ZZ", n=n))
    tracker.logicals.add_stabilizers(_rows("ZIII" if disposable else "ZI", n=n))
    tracker.post_select_row_indices = {0}
    circuit = stim.Circuit("R 0 1")
    if disposable:
        unitary = stim.Circuit("CX 0 2 1 3")
        circuit += stim.Circuit("R 2 3") + unitary + stim.Circuit("M 2 3")
        _mid(tracker, circuit, ("ZIII", "IZII"), [2, 3], ["Z", "Z"],
             reset_paulis=_rows("IIZI", "IIIZ", n=4), unitary=unitary)
    else:
        circuit.append("M", [0, 1])
        _mid(tracker, circuit, ("ZI", "IZ"), [0, 1], ["Z", "Z"])
    selected = _selected_matrix(tracker)
    assert len(independent_row_indices(selected)) == 1
    target = _rows("ZZII" if disposable else "ZZ", n=n)
    _, known, _ = solve_linear_decomposition(selected, target, reduce_weight=False)
    assert known.all()
    assert all(tracker.stabilizers.records[i] == [] for i in tracker.post_select_row_indices)

    # XX commutes with the selected ZZ parity. Independently tagging each Z
    # factor would falsely reject this paired fault after the mid measurement.
    circuit.append("X_ERROR", [0, 1], 1)
    circuit.append("M", [0, 1])
    tracker.process_data_measurement(circuit,
        _rows("ZIII", "IZII", n=n) if disposable else _rows("ZI", "IZ", n=n),
        {q: (q, 0) for q in range(n)})
    detector_indices, cursor = [], 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            if instruction.tag == "post-select":
                detector_indices.append(cursor)
            cursor += 1
    assert detector_indices
    detection_events, _ = circuit.compile_detector_sampler(seed=17).sample(8, separate_observables=True)
    assert not detection_events[:, detector_indices].any()
    assert detection_events.any()


@pytest.mark.parametrize("entry", ["process_resets", "process_initialization"])
def test_v2_reset_does_not_postselect_a_product_with_an_unselected_pivot(entry):
    tracker = SyndromeTracker(3)
    tracker.stabilizers.add_stabilizers(_rows("ZIZ", "IZZ", n=3))
    tracker.logicals.add_stabilizers(_rows("XXX", n=3))
    tracker.post_select_row_indices = {1}
    getattr(tracker, entry)(_rows("IIZ", n=3))
    assert not tracker.post_select_row_indices
    assert any(np.array_equal(row, _rows("ZZI", n=3)[0]) for row in tracker.stabilizers.matrix)
    circuit = stim.Circuit("R 0 1 2\nH 0\nCX 0 1 0 2\nR 2\nM 0 1")
    tracker.process_data_measurement(circuit, _rows("ZII", "IZI", n=3),
                                     {q: (q, 0) for q in range(3)})
    assert not any(instruction.name == "DETECTOR" and instruction.tag == "post-select"
                   for instruction in circuit)
    _assert_complete(tracker, circuit)


def test_v2_mid_projection_only_retains_selected_commuting_subspace():
    tracker = SyndromeTracker(3)
    tracker.stabilizers.add_stabilizers(_rows("ZIZ", "IZZ", n=3))
    tracker.logicals.add_stabilizers(_rows("XXX", n=3))
    tracker.post_select_row_indices = {1}
    circuit = stim.Circuit("R 0 1 2\nH 0\nCX 0 1 0 2\nMX 2")
    _mid(tracker, circuit, ("IIX",), [2], ["X"])
    assert not tracker.post_select_row_indices
    circuit.append("M", [0, 1])
    tracker.process_data_measurement(circuit, _rows("ZII", "IZI", n=3),
                                     {q: (q, 0) for q in range(3)})
    assert not any(instruction.name == "DETECTOR" and instruction.tag == "post-select"
                   for instruction in circuit)
    _assert_complete(tracker, circuit)


def test_v2_explicit_absorption_updates_the_actual_dressed_role():
    tracker = SyndromeTracker(1)
    tracker.logicals.add_stabilizers(_rows("Z", n=1), [[0]])
    tracker.total_measurements = 1
    assert tracker.record_absorbed_op(_rows("Z", n=1)[0])
    assert tracker._dressed_stabilizer_rows
    assert tracker.absorbed_ops.count == 1
    assert tracker.absorbed_ops.records == [[0]]
    before = tracker.absorbed_ops.matrix.copy()
    tracker.validate_logical_count()
    np.testing.assert_array_equal(tracker.absorbed_ops.matrix, before)
    assert tracker.absorbed_ops.records == [[0]]
    circuit = stim.Circuit("RX 0\nM 0 0")
    tracker.process_data_measurement(circuit, _rows("Z", n=1), {0: (0, 0)})
    assert circuit.num_observables == 1
    _assert_complete(tracker, circuit)
