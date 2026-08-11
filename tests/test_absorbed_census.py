"""Absorbed-DOF census: one ledger, derived count, logical equivalence.

absorbed_ops is the single census ledger — every path that absorbs a
logical DOF records the OPERATOR; the count is always derived as
rank([stabilizers; absorbed]) - rank(stabilizers).  There is deliberately
no separately maintained integer (the old `_absorbed_logical_dofs` counter
demanded perfect increment/decrement pairing from every path and drifted
silently when one forgot).  Geometry-independent, like
test_tracker_closure_contract.
"""
import numpy as np
import pytest

from lightstim.ir.tracker import SyndromeTracker


def _z_row(n, qubits):
    row = np.zeros(2 * n, dtype=np.uint8)
    for q in qubits:
        row[n + q] = 1
    return row


def test_census_dedups_representatives_at_insertion():
    # Reviewer-mandated regression: two representatives differing by a
    # stabilizer are ONE logical relation, not two.  The dedup happens at
    # insertion: the second representative reduces to zero against the
    # bank + ledger and is skipped.
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]
    rep_a = _z_row(n, [2, 3])
    rep_b = (_z_row(n, [2, 3]) + _z_row(n, [0, 1])) % 2  # rep_a * stabilizer

    assert tracker.record_absorbed_op(rep_a) is True
    assert tracker.record_absorbed_op(rep_b) is False
    assert tracker.absorbed_ops.count == 1
    assert tracker.num_absorbed_dof() == 1


def test_group_member_relation_is_not_banked():
    # An operator already expressed by the current stabilizer rows holds
    # no NEW logical DOF at insertion time and must not enter the ledger.
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]

    assert tracker.record_absorbed_op(_z_row(n, [0, 1])) is False
    assert tracker.num_absorbed_dof() == 0


def test_banked_relation_survives_its_own_closure_row():
    # After a merge, the joint's CLOSURE row (same relation, carrying the
    # merge records) legitimately enters the stabilizer bank.  The banked
    # DOF must keep counting — the census is the ledger's own rank, never
    # quotiented by the bank (regression for the measurement-block round
    # over post-PPM state, where the quotient version tripped the alarm).
    n = 4
    tracker = SyndromeTracker(n, 1)
    assert tracker.record_absorbed_op(_z_row(n, [0, 1])) is True
    # the closure row appears in the bank AFTER the relation was banked
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[3]]

    assert tracker.num_absorbed_dof() == 1
    tracker.validate_logical_count(context="closure row coexists")


def test_alarm_fires_when_a_relation_is_lost():
    n = 4
    tracker = SyndromeTracker(n, 1)  # budget: one logical DOF somewhere
    tracker.absorbed_ops.matrix = _z_row(n, [2, 3]).reshape(1, -1)
    tracker.absorbed_ops.records = [[]]
    tracker.validate_logical_count(context="healthy state")  # no raise

    # A path that loses the relation without accounting for it must trip
    # the alarm at the next validation — the ledger IS the count, so the
    # two can no longer drift apart silently.
    tracker.absorbed_ops.matrix = np.zeros((0, 2 * n), dtype=np.uint8)
    tracker.absorbed_ops.records = []
    with pytest.raises(RuntimeError, match="absorbed logical DOFs"):
        tracker.validate_logical_count(context="after losing the relation")


def test_block_absorb_records_the_operator():
    n = 4
    tracker = SyndromeTracker(n, 0)
    consumed = _z_row(n, [1, 2])
    tracker._gauge_logical_vectors = [np.array([1], dtype=np.uint8)]

    tracker._record_measurement_logical_effects(
        set(), old_logicals_current_frame=consumed.reshape(1, -1))

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == consumed).all()
    assert tracker.num_absorbed_dof() == 1


def test_block_absorb_skips_surviving_logicals():
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker._gauge_logical_vectors = [np.array([1], dtype=np.uint8)]

    tracker._record_measurement_logical_effects(
        {0}, old_logicals_current_frame=_z_row(n, [1, 2]).reshape(1, -1))

    assert tracker.absorbed_ops.count == 0
    assert tracker.num_absorbed_dof() == 0


def test_block_absorb_requires_the_frame():
    tracker = SyndromeTracker(4, 0)
    tracker._gauge_logical_vectors = [np.array([1], dtype=np.uint8)]

    with pytest.raises(ValueError, match="cannot be recorded"):
        tracker._record_measurement_logical_effects(set())


def test_reset_folds_relation_off_reset_qubits_mod_group():
    # A relation is only defined mod the stabilizer group: when qubit 0 is
    # re-initialized, Z0Z1 survives as Z1Z2 via the group element Z0Z2.
    n = 4
    tracker = SyndromeTracker(n, 1)
    tracker.stabilizers.matrix = _z_row(n, [0, 2]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]
    tracker.absorbed_ops.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[]]

    tracker.reset_records_for_qubits([0])

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [1, 2])).all()
    assert tracker.num_absorbed_dof() == 1
    tracker.validate_logical_count(context="after fold")  # still accounted


def test_reset_annihilates_unfoldable_relation_and_alarm_fires():
    # No group element can move Z0Z1 off qubit 0: the re-init destroys the
    # relation.  The row is dropped and the census alarm reports the loss.
    n = 4
    tracker = SyndromeTracker(n, 1)
    tracker.absorbed_ops.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[]]

    tracker.reset_records_for_qubits([0])

    assert tracker.absorbed_ops.count == 0
    with pytest.raises(RuntimeError, match="absorbed logical DOFs"):
        tracker.validate_logical_count(context="after destructive reset")


def test_reset_fold_carries_the_stabilizer_records():
    # Re-expressing a banked relation off reset qubits multiplies it by a
    # stabilizer row; the fold is only a valid re-representation if that
    # row's RECORDS fold in alongside its Pauli (adversarial review: the
    # old post-clean fold silently dropped them - a parity corruption
    # invisible to the census and to p=0 sampling).
    n = 4
    tracker = SyndromeTracker(n, 1)
    stab = _z_row(n, [0, 1])
    tracker.stabilizers.matrix = stab.reshape(1, -1)
    tracker.stabilizers.records = [[5]]
    tracker.absorbed_ops.matrix = _z_row(n, [1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[7]]

    tracker.reset_records_for_qubits([1])

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [0])).all()
    assert tracker.absorbed_ops.records[0] == [5, 7], \
        "the folded stabilizer's record must ride along"


def test_reset_fold_rejects_sentinel_poisoned_stabilizer():
    # If the only stabilizer that could carry the relation off the reset
    # qubits has already lost its records to the UNMEASURED sentinel, the
    # relation's parity is unreconstructable - fail loud, never fold
    # silently.
    from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
    n = 4
    tracker = SyndromeTracker(n, 1)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[UNMEASURED_STAB_RECORD]]
    tracker.absorbed_ops.matrix = _z_row(n, [1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[7]]

    with pytest.raises(RuntimeError, match="cannot be reconstructed"):
        tracker.reset_records_for_qubits([1])
