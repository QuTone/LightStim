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


def test_census_counts_up_to_logical_equivalence():
    # Reviewer-mandated regression: two representatives differing by a
    # stabilizer are ONE logical relation, not two.
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]
    rep_a = _z_row(n, [2, 3])
    rep_b = (_z_row(n, [2, 3]) + _z_row(n, [0, 1])) % 2  # rep_a * stabilizer
    tracker.absorbed_ops.matrix = np.vstack([rep_a, rep_b]).astype(np.uint8)
    tracker.absorbed_ops.records = [[], []]

    assert tracker.num_absorbed_dof() == 1


def test_group_member_relation_holds_no_dof():
    # A relation that has itself become a stabilizer holds no logical DOF.
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]
    tracker.absorbed_ops.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[]]

    assert tracker.num_absorbed_dof() == 0


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
