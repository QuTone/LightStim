"""Absorbed-DOF census: one ledger, derived count, logical equivalence.

absorbed_ops is the single census ledger — every path that absorbs a
logical DOF records the OPERATOR; the count is always derived as the
ledger's own GF(2) rank, with logical-equivalence deduplication at
INSERTION time (record_absorbed_op reduces against [stabilizers ∪
ledger] and skips dependent representatives; the count is deliberately
NOT quotiented by the bank — see num_absorbed_dof).  There is
deliberately no separately maintained integer (the old
`_absorbed_logical_dofs` counter demanded perfect increment/decrement
pairing from every path and drifted silently when one forgot).
Geometry-independent, like test_tracker_closure_contract.
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


def test_corridor_fold_carries_stabilizer_records():
    # Corridor readout re-expresses a banked relation off the bus via a
    # stabilizer row: the row's RECORDS must fold in alongside its Pauli
    # (same contract as the reset fold - a silent truncation corrupts the
    # banked parity invisibly to the census and to p=0 sampling).
    import stim
    n = 3
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[7]]
    tracker.absorbed_ops.matrix = _z_row(n, [1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[5]]
    fp = np.vstack([_z_row(n, [1]), _z_row(n, [2])])

    tracker.process_data_measurement(
        stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
        resolve_absorbed=False)

    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [0])).all()
    assert tracker.absorbed_ops.records == [[5, 7]], \
        "the folded stabilizer's record must ride along"


def test_corridor_residual_folds_against_readout_records():
    # Bus support the stabilizer group cannot cancel is folded against the
    # readout's own measured Paulis - and their measurement records ride
    # along.  (The old code hard-zeroed the columns and kept the truncated
    # operator with unchanged records.)
    import stim
    n = 3
    tracker = SyndromeTracker(n, 0)
    tracker.absorbed_ops.matrix = _z_row(n, [0, 2]).reshape(1, -1)
    tracker.absorbed_ops.records = [[5]]
    fp = np.vstack([_z_row(n, [0]), _z_row(n, [1])])   # bus = {0, 1}

    tracker.process_data_measurement(
        stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
        resolve_absorbed=False)

    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [2])).all()
    assert tracker.absorbed_ops.records == [[0, 5]], \
        "the readout record for the folded bus component must ride along"


def test_corridor_refuses_to_consume_bus_only_relation():
    # A banked relation supported ONLY on the measured bus would be fully
    # consumed by the corridor readout; resolve_absorbed=False promises no
    # resolution, and silently dropping the row leaks the DOF from the
    # books (the census then misfires at an unrelated later checkpoint).
    import stim
    n = 2
    tracker = SyndromeTracker(n, 1)
    tracker.absorbed_ops.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[5]]
    fp = np.vstack([_z_row(n, [0]), _z_row(n, [1])])

    with pytest.raises(RuntimeError, match="resolve_absorbed=True"):
        tracker.process_data_measurement(
            stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
            resolve_absorbed=False)


def test_corridor_refuses_undetermined_bus_support():
    # An X component on a Z-measured bus qubit anticommutes with the
    # readout: the banked parity is destroyed, not transformable.
    import stim
    n = 2
    tracker = SyndromeTracker(n, 1)
    row = np.zeros(2 * n, dtype=np.uint8)
    row[0] = 1                       # X0
    row[n + 1] = 1                   # Z1
    tracker.absorbed_ops.matrix = row.reshape(1, -1)
    tracker.absorbed_ops.records = [[5]]
    fp = _z_row(n, [0]).reshape(1, -1)

    with pytest.raises(RuntimeError, match="corrupted"):
        tracker.process_data_measurement(
            stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
            resolve_absorbed=False)


def test_half_read_remainder_carries_records():
    # A patch readout that half-reads a banked relation re-banks the
    # remainder; the remainder's records = seed records XOR the fold's
    # stabilizer/measurement records.  (The old code banked [] - the
    # relation's parity silently vanished; relay measured_absorbed would
    # then emit a wrong observable.)
    import stim
    n = 2
    tracker = SyndromeTracker(n, 2)     # standing Z1 + banked Z0Z1
    tracker.logicals.matrix = _z_row(n, [1]).reshape(1, -1)
    tracker.logicals.records = [[]]
    tracker.absorbed_ops.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.absorbed_ops.records = [[5]]
    fp = _z_row(n, [0]).reshape(1, -1)  # read q0 only

    tracker.process_data_measurement(
        stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
        resolve_absorbed=True)

    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [1])).all()
    assert tracker.absorbed_ops.records == [[0, 5]], \
        "seed record and the fold's measurement record must both survive"
    tracker.validate_logical_count(context="after half-read re-pricing")


def test_half_read_remainder_dedups_through_the_single_entrance():
    # The remainder enters the ledger through record_absorbed_op (the
    # ledger's ONLY entrance): a remainder that differs from an existing
    # banked relation by a stabilizer is the SAME relation - it must not
    # be double-banked, and the standing representative must be KEPT.
    import stim
    n = 3
    tracker = SyndromeTracker(n, 3)
    tracker.stabilizers.matrix = _z_row(n, [1, 2]).reshape(1, -1)
    tracker.stabilizers.records = [[4]]
    tracker.logicals.matrix = _z_row(n, [1]).reshape(1, -1)
    tracker.logicals.records = [[]]
    tracker.absorbed_ops.matrix = np.vstack(
        [_z_row(n, [0, 1]), _z_row(n, [2])])
    tracker.absorbed_ops.records = [[5], [6]]
    fp = _z_row(n, [0]).reshape(1, -1)  # read q0: remainder = Z1

    tracker.process_data_measurement(
        stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
        resolve_absorbed=True)

    # Z1 reduces against [Z1Z2 (stab); Z2 (banked)] -> already priced:
    # not re-banked, standing row kept, books balance.
    assert tracker.logicals.count == 1
    assert tracker.absorbed_ops.count == 1
    assert tracker.absorbed_ops.records == [[6]]
    tracker.validate_logical_count(context="after deduped re-pricing")


def test_gauge_absorb_carries_seed_records():
    # A record-pinned logical consumed by a gauge measurement keeps its
    # pin: the ledger row's records are the banked parity a later readout
    # (relay measured_absorbed) XORs with the measuring records.  (The old
    # code recorded the operator with empty records - the pin vanished.)
    n = 3
    tracker = SyndromeTracker(n, 0)
    tracker._gauge_logical_vectors = [np.array([0, 1], dtype=np.uint8)]
    old = np.vstack([_z_row(n, [0]), _z_row(n, [1])])

    tracker._record_measurement_logical_effects(
        {0}, old_logicals_current_frame=old,
        old_logicals_records=[[], [9]])

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [1])).all()
    assert tracker.absorbed_ops.records == [[9]], \
        "the consumed logical's seed records must ride into the ledger"


def test_terminal_observable_refuses_sentinel_records():
    # The stab-row branch and the relay path both refuse sentinel parities;
    # the logical-observable branch must too - silently skipping the -1
    # (the old behaviour) publishes an observable with a WRONG parity.
    import stim
    from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
    n = 2
    tracker = SyndromeTracker(n, 1)
    tracker.logicals.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.logicals.records = [[UNMEASURED_STAB_RECORD]]
    fp = np.vstack([_z_row(n, [0]), _z_row(n, [1])])

    with pytest.raises(RuntimeError, match="UNMEASURED sentinel"):
        tracker.process_data_measurement(
            stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
            resolve_absorbed=True)


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


def test_real_reset_path_rejects_banked_support():
    """The production reset path (process_resets, reached from
    CircuitBuilder.initialize and every SE round's ancilla resets) must
    fail loud when a reset touches a banked relation — the reset would
    destroy the relation's support while its parity stays banked."""
    import pytest as _pytest
    n = 4
    tracker = SyndromeTracker(n, 1)
    tracker.record_absorbed_op(_z_row(n, [0, 1]))

    reset = np.zeros((1, 2 * n), dtype=np.uint8)
    reset[0, n + 0] = 1          # reset qubit 0 in Z
    with _pytest.raises(RuntimeError, match="banked absorbed relations"):
        tracker.process_resets(reset)


def test_real_reset_path_ignores_disjoint_resets():
    # ancilla-style resets on qubits outside every banked relation pass
    # through untouched (the per-round hot path stays cheap and silent)
    n = 4
    tracker = SyndromeTracker(n, 1)
    tracker.record_absorbed_op(_z_row(n, [2, 3]))

    reset = np.zeros((2, 2 * n), dtype=np.uint8)
    reset[0, n + 0] = 1
    reset[1, n + 1] = 1
    tracker.process_resets(reset)
    assert tracker.num_absorbed_dof() == 1
