"""Export contract: historical targets must preserve existing task semantics."""

import copy

import pytest
import stim

from lightstim.ir.logical_history import LogicalHistoryRelation
from lightstim.ir.tracker import SyndromeTracker


def _tracker(circuit, *supports):
    tracker = SyndromeTracker(circuit.num_qubits)
    tracker.total_measurements = circuit.num_measurements
    tracker.logical_history = [
        LogicalHistoryRelation(tuple(support), max(support, default=0), (0,))
        for support in supports
    ]
    return tracker


def test_export_uses_complete_accumulated_observable_not_its_fragments():
    circuit = stim.Circuit("R 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
                           "OBSERVABLE_INCLUDE(0) rec[-1]")
    tracker = _tracker(circuit, (0,), (1,))
    original = circuit.copy()
    # Existing OBS0 spans 11, not its two individual contributions 10 and 01.
    assert tracker.append_logical_history_observables(circuit) == [1]
    assert circuit[:len(original)] == original
    assert circuit[-1].targets_copy() == [stim.target_rec(-2)]
    assert tracker.append_logical_history_observables(circuit) == []
    assert circuit.num_observables == 2


def test_export_reduces_modulo_detectors_and_handles_repeat_offsets():
    circuit = stim.Circuit("R 0\nREPEAT 3 {\nM 0\nDETECTOR rec[-1]\n}")
    tracker = _tracker(circuit, (0,), (0, 2))
    original = circuit.copy()
    assert tracker.append_logical_history_observables(circuit) == []
    assert circuit == original


def test_export_respects_reserved_ids_and_does_not_modify_live_census():
    circuit = stim.Circuit("R 0\nM 0")
    tracker = _tracker(circuit, (0,))
    assert tracker.allocate_observable() == 0
    assert tracker.allocate_observable() == 1
    census = (tracker.expected_num_logicals, tracker.logicals.count, tracker.num_absorbed_dof())
    assert tracker.append_logical_history_observables(circuit) == [2]
    assert tracker.total_observables == 3
    assert census == (tracker.expected_num_logicals, tracker.logicals.count, tracker.num_absorbed_dof())


def test_export_accepts_odd_ideal_parity_and_uses_stim_reference_convention():
    circuit = stim.Circuit("R 0\nX 0\nM 0")
    tracker = _tracker(circuit, (0,))
    assert tracker.append_logical_history_observables(circuit) == [0]
    assert circuit.reference_sample().tolist() == [True]
    assert circuit.has_flow(stim.Flow(measurements=[0]), unsigned=True)
    assert not circuit.has_flow(stim.Flow(measurements=[0]))
    # The raw bit is one, but there is no logical ERROR relative to the
    # reference. A signless tracker must never assume ideal XOR == zero.
    dets, obs = circuit.compile_detector_sampler().sample(16, separate_observables=True)
    assert dets.shape == (16, 0)
    assert not obs.any()
    circuit.detector_error_model()


@pytest.mark.parametrize("support", [(-1,), (2,), (0, 2)])
def test_export_rejects_unknown_or_unavailable_records_atomically(support):
    circuit = stim.Circuit("R 0\nM 0")
    tracker = _tracker(circuit, (0,), support)
    original = circuit.copy()
    with pytest.raises(ValueError, match="known measurement indices"):
        tracker.append_logical_history_observables(circuit)
    assert circuit == original
    assert tracker.total_observables == 0


def test_export_rejects_nondeterministic_history_even_if_another_candidate_is_valid():
    circuit = stim.Circuit("R 0 1\nH 1\nM 0 1")
    tracker = _tracker(circuit, (0,), (1,))
    original = circuit.copy()
    with pytest.raises(ValueError, match="not deterministic"):
        tracker.append_logical_history_observables(circuit)
    assert circuit == original
    assert tracker.total_observables == 0


def test_export_requires_the_complete_record_stream():
    circuit = stim.Circuit("R 0\nM 0")
    tracker = _tracker(circuit, (0,))
    tracker.total_measurements = 2
    with pytest.raises(ValueError, match="complete circuit"):
        tracker.append_logical_history_observables(circuit)


def test_history_is_not_shared_by_tracker_copies():
    circuit = stim.Circuit("R 0\nM 0")
    tracker = _tracker(circuit, (0,))
    probe = copy.deepcopy(tracker)
    probe.logical_history.clear()
    assert len(tracker.logical_history) == 1
    with pytest.raises(AttributeError):
        tracker.logical_history[0].records = ()
