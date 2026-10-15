"""Readout closes protected combinations before reducing surviving support.

The small physical circuits isolate row transformations rather than relying on
the tracker to define the expected measurement relations or their signs.
"""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _full_record_parities,
)


def _rows(*paulis, n):
    return np.asarray([
        np.concatenate(stim.PauliString(pauli).to_numpy()) for pauli in paulis
    ], dtype=np.uint8).reshape(len(paulis), 2 * n)


def _rank(matrix):
    return len(_basis([
        sum(int(bit) << column for column, bit in enumerate(row)) for row in matrix
    ]))


def _read(tracker, circuit, *qubits, active=()):
    n = tracker.num_qubits
    circuit.append("M", list(qubits))
    measured = []
    for qubit in qubits:
        pauli = ["I"] * n
        pauli[qubit] = "Z"
        measured.append("".join(pauli))
    tracker.process_data_measurement(
        circuit=circuit, final_paulis=_rows(*measured, n=n),
        idx_to_coord_map={q: (q, 0) for q in range(n)},
        active_stabilizers=_rows(*active, n=n))


def _mid(tracker, circuit, pauli):
    n = tracker.num_qubits
    targets = []
    for qubit, letter in enumerate(pauli):
        if letter != "I":
            if targets:
                targets.append(stim.target_combiner())
            targets.append(stim.target_z(qubit))
    circuit.append("MPP", targets)
    before = len(tracker.logical_history)
    tracker.process_mid_measurement(
        circuit=circuit,
        forward_symplectic_matrix=np.eye(2 * n, dtype=np.uint8),
        back_propagated_paulis=_rows(pauli, n=n), reset_paulis=None,
        measurement_qubit_indices=[next(q for q, letter in enumerate(pauli)
                                        if letter != "I")],
        measurement_bases=["Z"], measurement_coords=[(0, 0)],
        discarded_measurement_qubit_indices=set())
    # The builder normally appends logical_history during finalization. Here
    # append each new relation once so that intermediate completeness is tested.
    for relation in tracker.logical_history[before:]:
        circuit.append("OBSERVABLE_INCLUDE", [
            stim.target_rec(record - tracker.total_measurements)
            for record in relation.records], tracker.allocate_observable())
    return len(tracker.logical_history) - before


def _assert_state_and_relations(tracker, circuit, *, observables, detectors):
    physical, detector_rows, observable_rows = _annotations(circuit)
    emitted = detector_rows + observable_rows
    complete = _full_record_parities(physical)
    assert circuit.num_observables == observables
    assert circuit.num_detectors == detectors
    assert len(_basis(emitted)) == len(_basis(complete))
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in emitted
    ], unsigned=False)
    circuit.detector_error_model()
    n = tracker.num_qubits
    for tableau in (tracker.stabilizers, tracker.logicals, tracker.absorbed_logicals):
        assert len(tableau.records) == tableau.count
        for row, records in zip(tableau.matrix, tableau.records):
            pauli = stim.PauliString.from_numpy(xs=row[:n].astype(bool),
                                               zs=row[n:].astype(bool))
            assert physical.has_flow(stim.Flow(output=pauli, measurements=records),
                                     unsigned=False)
    protected = np.vstack([tracker.logicals.matrix, tracker.absorbed_logicals.matrix])
    assert _rank(protected) == tracker.logicals.count + tracker.absorbed_logicals.count
    tracker.validate_logical_count(context="protected partial-readout closure")


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("intervening_mid", [False, True])
def test_partial_readout_closes_combination_of_live_logicals(order, intervening_mid):
    """L0*L1=Z0 must close now, before the two residual Z1 rows coincide."""
    tracker = SyndromeTracker(2, expected_num_logicals=2, logical_tracking="slabs")
    logicals = _rows("ZZ", "IZ", n=2)
    tracker.logicals.add_stabilizers(logicals[list(order)], [[], []])
    circuit = stim.Circuit("R 0 1")

    _read(tracker, circuit, 0)
    _assert_state_and_relations(tracker, circuit, observables=1, detectors=0)
    assert tracker.logicals.count == 1
    assert tracker.absorbed_logicals.count == 0
    assert tracker.expected_num_logicals == 1

    if intervening_mid:
        assert _mid(tracker, circuit, "IZ") == 1
        _assert_state_and_relations(tracker, circuit, observables=2, detectors=0)
    _read(tracker, circuit, 1)
    _assert_state_and_relations(tracker, circuit,
                               observables=2 + int(intervening_mid), detectors=0)
    assert tracker.expected_num_logicals == 0


@pytest.mark.parametrize("stab_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("absorbed_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("intervening_mid", [False, True])
def test_partial_readout_closes_absorbed_combination_before_retaining_quotient(
        stab_order, absorbed_order, intervening_mid):
    """A=[Z0Z1,Z0], read Z1: emit A0*A1; keep one Z0 direction."""
    tracker = SyndromeTracker(2, expected_num_logicals=2, logical_tracking="slabs")
    bank = _rows("ZZ", "ZI", n=2)
    tracker.stabilizers.add_stabilizers(bank[list(stab_order)], [[], []])
    tracker.absorbed_logicals.add_stabilizers(bank[list(absorbed_order)], [[], []])
    circuit = stim.Circuit("R 0 1")

    _read(tracker, circuit, 1, active=("ZI",))
    _assert_state_and_relations(tracker, circuit, observables=1, detectors=0)
    assert tracker.logicals.count == 0
    assert tracker.absorbed_logicals.count == 1
    assert tracker.expected_num_logicals == 1

    if intervening_mid:
        assert _mid(tracker, circuit, "ZI") == 0
        _assert_state_and_relations(tracker, circuit, observables=1, detectors=1)
    _read(tracker, circuit, 0)
    _assert_state_and_relations(tracker, circuit, observables=2,
                               detectors=int(intervening_mid))
    assert tracker.expected_num_logicals == 0


@pytest.mark.parametrize("stab_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("absorbed_order", [(0, 1), (1, 0)])
def test_retained_absorbed_combination_rebases_S_to_active_check(stab_order, absorbed_order):
    """Keeping A0*A1 as active ZZ must leave ZZ representable by S alone."""
    tracker = SyndromeTracker(3, expected_num_logicals=3, logical_tracking="slabs")
    bank = _rows("ZII", "IZI", n=3)
    tracker.stabilizers.add_stabilizers(bank[list(stab_order)], [[], []])
    tracker.logicals.add_stabilizers(_rows("IIZ", n=3), [[]])
    tracker.absorbed_logicals.add_stabilizers(bank[list(absorbed_order)], [[], []])
    circuit = stim.Circuit("R 0 1 2")

    _read(tracker, circuit, 2, active=("ZZI",))
    _assert_state_and_relations(tracker, circuit, observables=1, detectors=0)
    assert tracker.logicals.count == tracker.absorbed_logicals.count == 1
    assert _rank(np.vstack([tracker.stabilizers.matrix, _rows("ZZI", n=3)])) == _rank(
        tracker.stabilizers.matrix)
    assert _mid(tracker, circuit, "ZZI") == 0
    _assert_state_and_relations(tracker, circuit, observables=1, detectors=1)
    _read(tracker, circuit, 0, 1)
    _assert_state_and_relations(tracker, circuit, observables=3, detectors=1)
    assert tracker.expected_num_logicals == 0


@pytest.mark.parametrize("spectator_first", [False, True])
def test_absorbed_release_preserves_explicit_postselected_closure(spectator_first):
    """An explicitly selected S parity remains selected when A also releases it."""
    tracker = SyndromeTracker(2, expected_num_logicals=2, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(_rows("ZI", n=2), [[]])
    tracker.post_select_row_indices = {0}
    tracker.logicals.add_stabilizers(_rows("IZ", n=2), [[]])
    tracker.absorbed_logicals.add_stabilizers(_rows("ZI", n=2), [[]])
    circuit = stim.Circuit("R 0 1")
    if spectator_first:
        _read(tracker, circuit, 1)
        assert tracker.post_select_row_indices
        _read(tracker, circuit, 0)
    else:
        _read(tracker, circuit, 0, 1)
    _assert_state_and_relations(tracker, circuit, observables=2, detectors=1)
    selected = [instruction for instruction in circuit
                if instruction.name == "DETECTOR" and instruction.tag == "post-select"]
    assert len(selected) == 1
    assert tracker.post_select_row_indices == set()
    # Inserting a physical bit flip before either readout must reject on the
    # postselected parity, even though that parity is also a logical output.
    faulty = stim.Circuit("R 0 1\nX_ERROR(1) 0") + circuit[1:]
    detector_events, _ = faulty.compile_detector_sampler(seed=7).sample(
        8, separate_observables=True)
    assert detector_events.all()
