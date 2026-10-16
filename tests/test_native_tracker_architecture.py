"""The native state owns coordinate transitions, not just query acceleration."""

import copy

import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from lightstim.utils.linear_algebra import row_echelon, solve_linear_decomposition
from lightstim.utils.subsystem_algebra import combine_records


@pytest.fixture
def native():
    module = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
    if not hasattr(module.PersistentRecordState, "refresh_complement"):
        pytest.skip("Full native tracker extension has not been built")
    return module


def test_native_complement_preserves_physical_component_order(native, monkeypatch):
    matrix = np.zeros((4, 8), dtype=np.uint8)
    matrix[0, 4] = matrix[1, 6] = 1
    matrix[2, [6, 7]] = 1
    matrix[3, [4, 5]] = 1
    state = native.PersistentRecordState(matrix, [[], [], [4], [5]])
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    _, _, expected = solve_linear_decomposition(matrix[:2], matrix[2:], reduce_weight=False)
    assert expected == [1, 0]
    assert state.ordered_complement([0, 1], [2, 3]) == expected


def test_native_refresh_canonicalizes_logicals_and_keeps_source_lifetime(native, monkeypatch):
    # Multiple independent complements exercise the native selected-index
    # lifetime, and a dependent logical row exercises record-only lineage.
    matrix = np.zeros((6, 12), dtype=np.uint8)
    matrix[0, 6] = matrix[1, 8] = 1
    matrix[2, [7, 9]] = 1
    matrix[3, [7, 10]] = 1
    matrix[4] = matrix[2] ^ matrix[3]
    matrix[5] = matrix[0]
    records = [[0], [1], [-1], [4, 4, 7], [11], [0]]
    measurements = matrix[:2].copy()
    state = native.PersistentRecordState(matrix, records)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    canonical, _, transform, _ = row_echelon(matrix[2:5], reduced=True)
    expected = matrix.copy()
    expected[2:5] = canonical
    canonical_records = combine_records(transform, records[2:5])
    for row, coefficients in zip(canonical_records, transform):
        if any(r < 0 for j in np.flatnonzero(coefficients) for r in records[2 + int(j)]) and not any(r < 0 for r in row):
            row.append(-1)
    _, _, complement = solve_linear_decomposition(
        measurements, np.vstack([expected[2:5], expected[:2]]), reduce_weight=False)
    assert state.refresh_complement(measurements, 2, 3) == complement
    np.testing.assert_array_equal(state.rows(), expected)
    assert state.records()[2:5] == canonical_records
    for _ in range(20):
        clone = state.copy()
        assert clone.refresh_complement(measurements, 2, 3) == complement
        clone.refresh_sources(measurements, [[20], [21]], [2, 3, 4, 5])
        assert clone.records()[:2] == [[20], [21]]


def test_native_views_are_lazy_mutable_and_deepcopy_independent(native, monkeypatch):
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "persistent")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    tracker = SyndromeTracker(3)
    tracker.stabilizers.add_stabilizers(np.array([[0, 0, 0, 1, 0, 0]], dtype=np.uint8), [[2]])
    state = tracker._sync_record_state()
    tracker._read_record_state()
    before = state.stats.get("physical_materializations", 0)
    assert tracker.stabilizers.count == 1
    assert tracker._sync_record_state() is state
    assert state.stats.get("physical_materializations", 0) == before
    clone = copy.deepcopy(tracker)
    clone.stabilizers.records[0].append(7)
    clone.stabilizers.matrix[0, 4] = 1
    clone._sync_record_state()
    assert clone._record_state.records() == [[2, 7]]
    assert state.records() == [[2]]
    np.testing.assert_array_equal(state.rows()[0], [0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(clone._record_state.rows()[0], [0, 0, 0, 1, 1, 0])


def test_native_trace_and_insert_preserve_records_and_rank(native):
    matrix = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    state = native.PersistentRecordState(matrix, [[-1], [4], [-1]])
    kept, ns, nl, na = state.trace_qubits([0], 2, 0)
    assert (kept, ns, nl, na) == ([], 0, 0, 0)
    state.insert_sources(np.array([[0, 0, 1, 0]], dtype=np.uint8), [[]], 0)
    np.testing.assert_array_equal(state.rows(), [[0, 0, 1, 0]])
    assert state.records() == [[]]


@pytest.mark.parametrize("n", [3, 9, 63, 64, 65])
def test_measurement_batch_coordinates_follow_every_projection_rebase(native, n):
    if not hasattr(native.PersistentRecordState, "begin_measurement_batch"):
        pytest.skip("Batch query extension has not been rebuilt")
    rng = np.random.default_rng(103 + n)
    rows = np.zeros((n, 2 * n), dtype=np.uint8)
    rows[:, n:] = np.eye(n, dtype=np.uint8)
    measurements = rng.integers(0, 2, (32, 2 * n), dtype=np.uint8)
    state = native.PersistentRecordState(rows, [[i] for i in range(n)])
    state.begin_measurement_batch(measurements)
    for index, row in enumerate(measurements):
        anti = state.measurement_anticommuting(index)
        assert anti == state.anticommuting(row)
        if anti:
            state.project(row, anti[0], n + index)
        for target in (0, index, 31):
            assert state.measurement_anticommuting(target) == state.anticommuting(measurements[target])
    assert state.stats["measurement_coordinate_imports"] == len(measurements)


@pytest.mark.parametrize("mode", ["reference", "persistent"])
def test_failed_mid_measurement_keeps_preprojection_public_banks(native, monkeypatch, mode):
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    tracker = SyndromeTracker(2)
    tracker.stabilizers.add_stabilizers(np.array([[0, 0, 1, 0]], dtype=np.uint8), [[3]])
    tracker.logicals.add_stabilizers(np.array([[0, 0, 0, 1]], dtype=np.uint8), [[-1]])
    if mode == "persistent":
        tracker._sync_record_state()
        tracker._read_record_state()
    before = [(bank.matrix.copy(), copy.deepcopy(bank.records))
              for bank in (tracker.stabilizers, tracker.logicals)]
    # First row projects S; the second asks for an unbanked logical parity.
    measurements = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unmeasured logical record"):
        tracker.process_mid_measurement(
            stim.Circuit(), np.eye(4, dtype=np.uint8), measurements, None,
            [0, 1], ["X", "Z"], [(0, 0), (1, 0)], set(),
            circuit_chunk=stim.Circuit("MX 0\nM 1"))
    for bank, (matrix, records) in zip((tracker.stabilizers, tracker.logicals), before):
        np.testing.assert_array_equal(bank.matrix, matrix)
        assert bank.records == records


@pytest.mark.parametrize("mode", ["reference", "persistent"])
@pytest.mark.parametrize("mask", [None, [True, False, False, False, True],
                                   [False, True, False, False, True]])
@pytest.mark.parametrize("early_annotation_error", [False, True])
def test_batch_failure_preserves_ordered_prior_events_and_public_state(
    native, monkeypatch, mode, mask, early_annotation_error,
):
    from lightstim.ir.logical_history import LogicalHistoryRelation

    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    tracker = SyndromeTracker(4, post_select_detector_coords={(0, 0, 0), (1, 0, 0)})
    source = np.zeros((4, 8), dtype=np.uint8)
    source[:, 4:] = np.eye(4, dtype=np.uint8)
    tracker.stabilizers.add_stabilizers(source[[0, 2]], [[2, 2, 3], [7]])
    tracker.logicals.add_stabilizers(source[[1, 3]], [[4, 4, 6], [-1]])
    tracker._classified_stabilizer_rows = {0}
    tracker.total_measurements = 10
    if mode == "persistent":
        tracker._sync_record_state()
        tracker._read_record_state()
    before = [(bank.matrix.copy(), copy.deepcopy(bank.records))
              for bank in (tracker.stabilizers, tracker.logicals)]
    # Flag -> exact check -> logical history -> projection -> unknown logical.
    # A late error preserves prior emissions, but must not commit projection.
    measurements = np.zeros((5, 8), dtype=np.uint8)
    measurements[1], measurements[2], measurements[4] = source[[0, 1, 3]]
    measurements[3, 2] = 1
    coords = [(i, 0) for i in range(5)]
    if early_annotation_error:
        # The first nonsuppressed check fails before the later native error;
        # event replay must preserve that error precedence and history prefix.
        coords = []
    circuit = stim.Circuit()
    expected_error = IndexError if early_annotation_error else ValueError
    with pytest.raises(expected_error):
        tracker.process_mid_measurement(
            circuit, np.eye(8, dtype=np.uint8), measurements, None,
            [0, 0, 1, 2, 3], ["Z", "Z", "Z", "X", "Z"], coords, set(),
            no_detector_mask=mask,
            circuit_chunk=stim.Circuit("M 0 0 1\nMX 2\nM 3"))

    assert tracker.total_measurements == 15
    assert tracker._classified_stabilizer_rows == {0}
    for bank, (matrix, records) in zip((tracker.stabilizers, tracker.logicals), before):
        np.testing.assert_array_equal(bank.matrix, matrix)
        assert bank.records == records
    expected = stim.Circuit()
    if not early_annotation_error:
        if mask is None or not mask[0]:
            expected.append("DETECTOR", [stim.target_rec(-5)], [0, 0, 0], tag="post-select")
        if mask is None or not mask[1]:
            expected.append("DETECTOR", [stim.target_rec(i) for i in (-13, -12, -4)],
                            [1, 0, 0], tag="post-select")
    assert circuit == expected
    assert tracker.logical_history == ([] if early_annotation_error else [
        LogicalHistoryRelation(records=(6, 12), measurement_index=12, logical_indices=(0,))])


def test_native_batch_uses_redundant_stabilizer_before_protected_rows(native):
    rows = np.tile(np.array([[0, 1]], dtype=np.uint8), (4, 1))
    state = native.PersistentRecordState(rows, [[1], [2], [3], [4]])
    result = state.process_measurement_batch(np.array([[1, 0]], dtype=np.uint8),
                                             2, 1, [], [], 10)
    assert result == {"events": [], "classified": [0], "error": None}
    np.testing.assert_array_equal(state.rows(), [[1, 0], [0, 0], [0, 0], [0, 0]])
    assert state.records() == [[10], [1, 2], [1, 3], [1, 4]]
    assert state.stats["native_measurement_batches"] == 1


def test_native_batch_carries_absorbed_pivot_error_after_prior_flag(native):
    state = native.PersistentRecordState(np.array([[0, 1]], dtype=np.uint8), [[2]])
    result = state.process_measurement_batch(np.array([[0, 0], [1, 0]], dtype=np.uint8),
                                             0, 0, [], [], 10)
    assert result["events"] == [(0, "detector", [10], [])]
    assert result["error"] == ("RuntimeError", "L_absorbed cannot be an independent projection pivot")
    np.testing.assert_array_equal(state.rows(), [[0, 1]])
    assert state.records() == [[2]]


@pytest.mark.parametrize("n", [3, 65])
def test_native_readout_projection_batch_matches_dense_elimination(native, n):
    rows = np.zeros((n + 1, 2 * n), dtype=np.uint8)
    rows[:n, n:] = np.eye(n, dtype=np.uint8)
    rows[n] = rows[0] ^ rows[1]
    records = [[-1]] + [[i] for i in range(1, n)] + [[-1]]
    measured = np.zeros((3, 2 * n), dtype=np.uint8)
    measured[np.arange(3), [0, n - 1, 1]] = 1
    absolute = [n + 4, n + 7, n + 9]
    expected, expected_records = rows.copy(), copy.deepcopy(records)
    destroyed = set()
    for row, record in zip(measured, absolute):
        commutators = (row[:n] @ expected[:, n:].T + row[n:] @ expected[:, :n].T) % 2
        anti = [int(i) for i in np.flatnonzero(commutators) if int(i) not in destroyed]
        if not anti:
            continue
        pivot = anti[0]
        for other in anti[1:]:
            unknown = any(i < 0 for i in expected_records[other] + expected_records[pivot])
            expected[other] ^= expected[pivot]
            parity = set(expected_records[other]) ^ set(expected_records[pivot])
            if unknown:
                parity.add(-1)
            expected_records[other] = sorted(parity)
        expected[pivot] = row
        expected_records[pivot] = [record]
        destroyed.add(pivot)
    state = native.PersistentRecordState(rows, records)
    assert state.project_readout_batch(measured, absolute, n) == sorted(destroyed)
    np.testing.assert_array_equal(state.rows(), expected)
    assert state.records() == expected_records
    assert state.stats["native_readout_batches"] == 1


@pytest.mark.parametrize("mode", ["reference", "persistent"])
def test_readout_batch_failure_keeps_preprojection_banks(native, monkeypatch, mode):
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    tracker = SyndromeTracker(2)
    tracker.stabilizers.add_stabilizers(np.array([[0, 0, 1, 0]], dtype=np.uint8), [[3]])
    # Deliberately invalid witness: the second measurement must reject an
    # independent absorbed-only pivot after the first has projected S.
    tracker.absorbed_ops.add_stabilizers(np.array([[0, 0, 0, 1]], dtype=np.uint8), [[4]])
    if mode == "persistent":
        tracker._sync_record_state()
        tracker._read_record_state()
    before = [(bank.matrix.copy(), copy.deepcopy(bank.records))
              for bank in (tracker.stabilizers, tracker.logicals, tracker.absorbed_ops)]
    circuit = stim.Circuit()
    with pytest.raises(RuntimeError, match="L_absorbed supplied the only anticommuting readout pivot"):
        tracker.process_data_measurement(
            circuit, np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8),
            {0: (0, 0), 1: (1, 0)})
    assert circuit == stim.Circuit()
    assert tracker.total_measurements == 2
    for bank, (matrix, records) in zip(
        (tracker.stabilizers, tracker.logicals, tracker.absorbed_ops), before,
    ):
        np.testing.assert_array_equal(bank.matrix, matrix)
        assert bank.records == records
