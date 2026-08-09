"""tracker.declare_logical — record-seeded logical birth (Gidney Y-init, C1).

A protocol (e.g. the in-place Y-basis initialization, arXiv:2302.07395) can
leave a would-be logical operator DETERMINED by the tracked stabilizer rows:
its sign is the XOR of the pinning rows' measurement records. LightStim's
implicit promotion only births records-less rows; declare_logical is the
explicit path: move the operator into self.logicals carrying the pinning
records, drop one pinning stabilizer row (rank preserved), budget += 1.
"""
import numpy as np
import pytest

from lightstim.ir.tracker import SyndromeTracker


def _tracker(n, rows, records):
    t = SyndromeTracker(num_qubits=n, expected_num_logicals=0)
    t.stabilizers.add_stabilizers(np.array(rows, dtype=np.uint8),
                                  [list(r) for r in records])
    return t


def _z(n, *qs):
    v = np.zeros(2 * n, dtype=np.uint8)
    for q in qs:
        v[n + q] = 1
    return v


def _x(n, *qs):
    v = np.zeros(2 * n, dtype=np.uint8)
    for q in qs:
        v[q] = 1
    return v


def test_moves_row_records_and_budget():
    # stabilizers: Z0 (pinned by record 3), Z1 (reset row, no records)
    n = 2
    t = _tracker(n, [_z(n, 0), _z(n, 1)], [[3], []])
    recs = t.declare_logical(_z(n, 0, 1))
    assert sorted(recs) == [3]
    assert t.logicals.count == 1
    assert sorted(t.logicals.records[0]) == [3]
    assert np.array_equal(t.logicals.matrix[0], _z(n, 0, 1))
    assert t.stabilizers.count == 1          # one pinning row removed
    assert t.expected_num_logicals == 1      # budget bumped at declare time
    # the operator is no longer derivable from stabilizers alone
    with pytest.raises(ValueError):
        t.declare_logical(_z(n, 0, 1))


def test_record_xor_cancels_duplicates():
    # Y_L-style: pinned by two rows sharing a record -> XOR cancels it
    n = 3
    t = _tracker(n, [_z(n, 0), _z(n, 1), _z(n, 2)], [[5, 9], [9], []])
    recs = t.declare_logical(_z(n, 0, 1, 2))
    assert sorted(recs) == [5]


def test_rejects_undetermined_operator():
    n = 2
    t = _tracker(n, [_z(n, 0), _z(n, 1)], [[], []])
    with pytest.raises(ValueError, match="not determined|span"):
        t.declare_logical(_x(n, 0))
    # tracker unchanged after the failed declare
    assert t.stabilizers.count == 2 and t.logicals.count == 0
    assert t.expected_num_logicals == 0


def test_explicit_records_override():
    n = 2
    t = _tracker(n, [_z(n, 0), _z(n, 1)], [[3], []])
    recs = t.declare_logical(_z(n, 0, 1), records=[7, 8])
    assert sorted(recs) == [7, 8]
    assert sorted(t.logicals.records[0]) == [7, 8]


def test_census_guardrail_consistent_after_declare():
    # (absorbed + logicals.count) == expected must hold right after declare
    n = 2
    t = _tracker(n, [_z(n, 0), _z(n, 1)], [[3], []])
    t.declare_logical(_z(n, 0, 1))
    absorbed = getattr(t, 'absorbed_logical_rank', 0) or 0
    assert absorbed + t.logicals.count == t.expected_num_logicals
