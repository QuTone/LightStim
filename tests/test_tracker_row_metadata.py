"""Stabilizer-row metadata stays valid across row deletion/reordering.

post_select_row_indices and stabilizer_with_logical_components hold ROW
indices into the stabilizer tableau.  Any path that deletes rows must shift
them (a stale index silently post-selects or reclassifies a DIFFERENT row),
and any path that recombines the basis — where old indices lose all meaning
— must fail loud instead.  Geometry-independent: drives SyndromeTracker
directly, like test_tracker_closure_contract.
"""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker


def _z_row(n, qubits):
    row = np.zeros(2 * n, dtype=np.uint8)
    for q in qubits:
        row[n + q] = 1
    return row


def test_terminal_readout_shifts_surviving_post_select():
    n = 4
    circuit = stim.Circuit("""
        R 0 1 2 3
        MPP Z0*Z1
        MPP Z2*Z3
        M 0 1
    """)
    tracker = SyndromeTracker(n, 0)
    tracker.total_measurements = 2
    tracker.stabilizers.matrix = np.vstack(
        [_z_row(n, [0, 1]), _z_row(n, [2, 3])])
    tracker.stabilizers.records = [[0], [1]]
    tracker.post_select_row_indices = {1}

    final_paulis = np.zeros((2, 2 * n), dtype=np.uint8)
    final_paulis[0, n + 0] = 1
    final_paulis[1, n + 1] = 1
    tracker.process_data_measurement(
        circuit, final_paulis,
        idx_to_coord_map={q: (float(q), 0.0) for q in range(n)})

    # Row 0 closed out (consumed); row 1 survived and now sits at index 0.
    assert tracker.stabilizers.count == 1
    assert tracker.post_select_row_indices == {0}


@pytest.mark.parametrize("method", ["stabilizer_canonicalization",
                                    "rebase_stabilizers_onto_code_basis"])
@pytest.mark.parametrize("pending", ["post_select_row_indices",
                                     "stabilizer_with_logical_components"])
def test_basis_recombination_rejects_pending_row_metadata(method, pending):
    tracker = SyndromeTracker(4, 0)
    tracker.stabilizers.matrix = _z_row(4, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]
    getattr(tracker, pending).add(0)

    # The guard sits at function entry, before the system is touched.
    with pytest.raises(RuntimeError, match="recombined"):
        getattr(tracker, method)(system=None)
