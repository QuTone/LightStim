"""Narrow API guards for unknown eigenvalues and destructive readout inputs."""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker


def _rows(*paulis):
    return np.asarray([np.concatenate(stim.PauliString(pauli).to_numpy())
                       for pauli in paulis], dtype=np.uint8)


def test_absorption_cannot_cancel_two_unknown_logical_values():
    tracker = SyndromeTracker(2)
    tracker.logicals.add_stabilizers(_rows('ZI', 'IZ'), [[-1], [-1]])
    with pytest.raises(ValueError, match='known constraint'):
        tracker.record_absorbed_op(_rows('ZZ')[0])
    assert tracker.stabilizers.count == tracker.absorbed_ops.count == 0
    assert tracker.logicals.records == [[-1], [-1]]


def test_absorption_does_not_reduce_a_known_operator_against_unknown_S():
    tracker = SyndromeTracker(2)
    tracker.stabilizers.add_stabilizers(_rows('ZI'), [[-1]])
    tracker.logicals.add_stabilizers(_rows('ZZ'), [[0]])
    assert tracker.record_absorbed_op(_rows('ZZ')[0])
    np.testing.assert_array_equal(tracker.absorbed_ops.matrix, _rows('ZZ'))
    assert tracker.absorbed_ops.records == [[0]]
    # The unknown first-qubit value cannot turn known ZZ into known IZ.
    assert tracker.stabilizers.records == [[-1], [0]]


@pytest.mark.parametrize('measurements', [('ZI', 'ZI'), ('ZI', 'IZ', 'ZZ')])
def test_destructive_readout_rejects_dependent_rows_without_mutation(measurements):
    tracker = SyndromeTracker(2)
    tracker.logicals.add_stabilizers(_rows('XI', 'IX'), [[], []])
    before = tracker.logicals.matrix.copy()
    circuit = stim.Circuit('RX 0 1')
    with pytest.raises(ValueError, match='independent measurement rows'):
        tracker.process_data_measurement(
            circuit, _rows(*measurements), {0: (0, 0), 1: (1, 0)})
    assert tracker.total_measurements == tracker.total_observables == 0
    np.testing.assert_array_equal(tracker.logicals.matrix, before)
    assert circuit == stim.Circuit('RX 0 1')
