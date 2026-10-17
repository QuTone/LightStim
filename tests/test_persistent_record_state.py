"""Physical coordinates and record arithmetic have an independent dense oracle."""

import copy
import pickle

import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.tableau import PauliTableau
from lightstim.utils.linear_algebra import (
    _solve_linear_decomposition_core,
    check_commutativity,
)
from tests.test_persistent_tracker_semantics import persistent_extension


pytestmark = pytest.mark.smoke


def _rows(*paulis):
    return np.asarray([np.concatenate(stim.PauliString(pauli).to_numpy())
                       for pauli in paulis], dtype=np.uint8)


def _xor_records(first, second):
    result = set(first) ^ set(second)
    if any(record < 0 for record in [*first, *second]):
        result.add(-1)
    return sorted(result)


def _random_clifford(rng, n):
    circuit = stim.Circuit()
    for _ in range(3 * n):
        circuit.append(str(rng.choice(["H", "S", "SQRT_X"])), [int(rng.integers(n))])
        if n > 1:
            circuit.append("CX", rng.choice(n, size=2, replace=False).tolist())
    return circuit


def _assert_query(engine, matrix, records, target, *, indices=None):
    if indices is None:
        anti, coefficients, dependent, parity = engine.query(target)
        expected_anti = np.flatnonzero(check_commutativity(target[None, :], matrix)[0])
        assert list(anti) == expected_anti.tolist()
        assert list(engine.anticommuting(target)) == expected_anti.tolist()
        indices = list(range(len(matrix)))
    else:
        coefficients, dependent, parity = engine.decompose(target, indices)
    expected, expected_dependent, _ = _solve_linear_decomposition_core(
        matrix[indices], target[None, :], reduce_weight=False)
    assert bool(dependent) == bool(expected_dependent[0])
    full_coefficients = np.zeros(len(matrix), dtype=np.uint8)
    full_coefficients[indices] = expected[0]
    np.testing.assert_array_equal(coefficients, full_coefficients)
    if dependent:
        expected_parity = set()
        unknown = False
        for index in np.flatnonzero(full_coefficients):
            for record in records[int(index)]:
                unknown |= record < 0
                if record in expected_parity:
                    expected_parity.remove(record)
                else:
                    expected_parity.add(record)
        if unknown and not any(record < 0 for record in expected_parity):
            expected_parity.add(-1)
        assert list(parity) == sorted(expected_parity)
        np.testing.assert_array_equal(np.asarray(coefficients) @ matrix % 2, target)
    else:
        assert list(parity) == []


@pytest.mark.parametrize("seed", range(16))
def test_persistent_random_projection_gates_and_reconciliation(persistent_extension, seed):
    rng = np.random.default_rng(seed)
    n = 1 + seed % 6
    initial = _random_clifford(rng, n)
    generators = SyndromeTracker.get_forward_symplectic_matrix(initial, n)[n:]
    combinations = rng.integers(0, 2, size=(3, n), dtype=np.uint8)
    matrix = np.vstack([generators, combinations @ generators % 2])
    records = [sorted(rng.choice(20, size=int(rng.integers(4)), replace=False).tolist())
               for _ in matrix]
    records[seed % len(records)] = [-1]
    engine = persistent_extension.PersistentRecordState(matrix, records)
    for step in range(7):
        target = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
        _assert_query(engine, matrix, records, target)
        subset = list(range(step % 3, len(matrix)))
        _assert_query(engine, matrix, records, target, indices=subset)
        anti = np.flatnonzero(check_commutativity(target[None, :], matrix)[0])
        if len(anti):
            pivot = int(anti[step % len(anti)])
            old_pivot, old_records = matrix[pivot].copy(), list(records[pivot])
            engine.project(target, pivot, 100 + step)
            for row in anti:
                if row != pivot:
                    matrix[row] ^= old_pivot
                    records[row] = _xor_records(records[row], old_records)
            matrix[pivot] = target
            records[pivot] = [100 + step]
            np.testing.assert_array_equal(engine.rows(), matrix)
            assert engine.records() == records

        gates = _random_clifford(rng, n)
        forward = SyndromeTracker.get_forward_symplectic_matrix(gates, n)
        engine.unitary(str(gates))
        matrix = (matrix @ forward % 2).astype(np.uint8)
        np.testing.assert_array_equal(engine.rows(), matrix)
        assert engine.records() == records

        # Reorder and replace the public row bank without resetting the frame.
        order = rng.permutation(len(matrix))
        matrix = matrix[order]
        records = [records[int(index)] for index in order]
        engine.replace_state(matrix, records)
        for row in matrix:
            _assert_query(engine, matrix, records, row)
    stats = engine.stats
    assert stats["constructions"] == 1
    assert stats["replacements"] == 8
    assert stats["unitaries"] > 0


def test_persistent_deepcopy_after_pending_query_is_independent(persistent_extension):
    matrix = _rows("XX", "ZZ", "YY", "II")
    records = [[0], [1], [2], [0, 1, 2]]
    engine = persistent_extension.PersistentRecordState(matrix, records)
    _assert_query(engine, matrix, records, _rows("YY")[0])
    probe = copy.deepcopy(engine)
    probe.unitary("H 0")
    transformed = matrix @ SyndromeTracker.get_forward_symplectic_matrix(stim.Circuit("H 0"), 2) % 2
    np.testing.assert_array_equal(probe.rows(), transformed)
    np.testing.assert_array_equal(engine.rows(), matrix)
    assert probe.records() == engine.records() == records
    _assert_query(engine, matrix, records, _rows("YY")[0])
    _assert_query(probe, transformed, records, transformed[2])


def test_persistent_incomplete_state_does_not_use_frame_completion(persistent_extension):
    matrix = _rows("ZII", "ZII", "III")
    records = [[0], [1], [0, 1]]
    # Start with a full state, then deliberately discard directions. The old
    # inverse frame still contains them as completion, never as known rows.
    engine = persistent_extension.PersistentRecordState(_rows("ZII", "IZI", "IIZ"),
                                                        [[0], [2], [3]])
    engine.replace_state(matrix, records)
    for target in _rows("ZII", "IZI", "IIX", "ZZI", "III"):
        _assert_query(engine, matrix, records, target)
    assert engine.records() == records
    engine.replace_state(matrix[:1], records[:1])
    for target in _rows("ZII", "IZI", "IIX", "ZZI", "III"):
        _assert_query(engine, matrix[:1], records[:1], target)


def test_persistent_unknown_projection_remains_unknown(persistent_extension):
    matrix = _rows("XX", "XI")
    records = [[-1], [-1]]
    engine = persistent_extension.PersistentRecordState(matrix, records)
    engine.project(_rows("ZI")[0], 0, [5])
    np.testing.assert_array_equal(engine.rows(), _rows("ZI", "IX"))
    assert engine.records() == [[5], [-1]]


def test_persistent_fold_retains_redundant_and_record_only_rows(persistent_extension):
    matrix = _rows("ZI", "IZ", "ZZ", "II")
    records = [[], [4], [4], [8, 9]]
    engine = persistent_extension.PersistentRecordState(matrix, records)
    # The engine's unitary entry follows the tracker's Clifford-only contract.
    engine.unitary("R 1\nCX 0 1\nX_ERROR(0.1) 0\nM 1")
    engine.fold_measured([1], ["Z"], 7)
    np.testing.assert_array_equal(engine.rows(), _rows("ZI", "ZI", "II", "II"))
    assert engine.records() == [[], [4, 7], [4, 7], [8, 9]]
    _assert_query(engine, engine.rows(), engine.records(), _rows("IX")[0])
    assert engine.stats["folds"] == 1
    assert engine.stats["fold_fastpaths"] == 1


def test_persistent_fold_outside_frame_uses_reconciliation(persistent_extension):
    engine = persistent_extension.PersistentRecordState(_rows("XX"), [[4]])
    engine.fold_measured([1], ["X"], 7)
    np.testing.assert_array_equal(engine.rows(), _rows("XI"))
    assert engine.records() == [[4, 7]]
    assert engine.stats["folds"] == 1
    assert engine.stats["fold_fastpaths"] == 0
    _assert_query(engine, engine.rows(), engine.records(), _rows("XI")[0])


def test_persistent_literal_records_survive_import_and_gates(persistent_extension):
    matrix, records = _rows("Z"), [[5, 3, 5]]
    engine = persistent_extension.PersistentRecordState(matrix, records)
    assert engine.records() == records
    engine.unitary("H 0\nH 0")
    assert engine.records() == records
    _assert_query(engine, matrix, records, matrix[0])
    assert list(engine.decompose(matrix[0], [0])[2]) == [3]


def test_persistent_projection_uses_pauli_tableau_set_record_semantics(persistent_extension):
    engine = persistent_extension.PersistentRecordState(_rows("XX", "XI"),
                                                        [[5, 5], [3, 5, 3]])
    engine.project(_rows("ZI")[0], 0, 7)
    np.testing.assert_array_equal(engine.rows(), _rows("ZI", "IX"))
    assert engine.records() == [[7], [3]]


def test_persistent_fold_normalizes_even_untouched_records(persistent_extension):
    engine = persistent_extension.PersistentRecordState(_rows("ZI", "IZ"),
                                                        [[5, 3, 5], [9, 9]])
    engine.fold_measured([1], ["Z"], 7)
    np.testing.assert_array_equal(engine.rows(), _rows("ZI", "II"))
    assert engine.records() == [[3, 5], [7, 9]]


def test_persistent_logical_relation_cancels_duplicate_record_occurrences(
    persistent_extension, monkeypatch,
):
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "reference")
    tracker = SyndromeTracker(1)
    full = PauliTableau(1)
    full.add_stabilizers(_rows("Z"), [[5, 5]])
    measurement = _rows("Z")
    engine = persistent_extension.PersistentRecordState(full.matrix, full.records)
    reference = tracker._v2_commuting_relation(full, 0, 1, measurement[0], measurement, 10, 0)
    native = tracker._v2_commuting_relation(full, 0, 1, measurement[0], measurement, 10, 0,
                                           record_state=engine)
    assert native == reference == ("logical", [10], (0,))


def test_persistent_pickle_preserves_state_without_mutating_snapshot_source(persistent_extension):
    engine = persistent_extension.PersistentRecordState(_rows("XX", "ZZ", "YY"),
                                                        [[5, 3, 5], [8], [9]])
    engine.project(_rows("ZI")[0], 0, 10)
    # Preserve a literal row expression through public reconciliation too.
    matrix, records = engine.rows(), engine.records()
    records[0] = [10, 4, 4]
    engine.replace_state(matrix, records)
    engine.query(matrix[1])  # Leave the inverse frame's query view active.
    before = dict(engine.stats)
    first, second = pickle.dumps(engine), pickle.dumps(engine)
    assert first == second
    assert engine.stats == before
    restored = pickle.loads(first)
    assert restored.stats == before
    np.testing.assert_array_equal(restored.rows(), matrix)
    assert restored.records() == records
    for row in matrix:
        _assert_query(restored, matrix, records, row)
    restored.unitary("H 1")
    transformed = matrix @ SyndromeTracker.get_forward_symplectic_matrix(stim.Circuit("H 1"), 2) % 2
    np.testing.assert_array_equal(restored.rows(), transformed)
    np.testing.assert_array_equal(engine.rows(), matrix)


def test_persistent_tracker_dictionary_can_be_pickled_for_atomicity(
    persistent_extension, monkeypatch,
):
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "persistent")
    tracker = SyndromeTracker(2)
    tracker.process_initialization(_rows("ZI", "IZ"))
    tracker.process_unitary_block(stim.Circuit("H 0"))
    first, second = pickle.dumps(tracker.__dict__), pickle.dumps(tracker.__dict__)
    assert first == second
    restored = pickle.loads(first)
    np.testing.assert_array_equal(restored["_record_state"].rows(), tracker._record_state.rows())
    assert restored["_record_state"].records() == tracker._record_state.records()


@pytest.mark.parametrize("n", [63, 64, 65, 127, 128, 129])
def test_persistent_coordinate_words_cross_boundaries(persistent_extension, n):
    matrix = np.zeros((4, 2 * n), dtype=np.uint8)
    matrix[0, n] = 1
    matrix[1, 2 * n - 1] = 1
    matrix[2] = matrix[0] ^ matrix[1]
    matrix[3, n + n // 2] = 1
    records = [[1], [2], [3], [4]]
    engine = persistent_extension.PersistentRecordState(matrix, records)
    target = np.zeros(2 * n, dtype=np.uint8)
    target[[0, n - 1]] = 1
    _assert_query(engine, matrix, records, target)
    engine.project(target, 0, 5)
    matrix[1] ^= matrix[0]
    records[1] = [1, 2]
    matrix[0] = target
    records[0] = [5]
    gates = stim.Circuit(f"H 0\nCX 0 {n - 1}\nSWAP {n // 2} {n - 1}")
    engine.unitary(str(gates))
    matrix = matrix @ SyndromeTracker.get_forward_symplectic_matrix(gates, n) % 2
    np.testing.assert_array_equal(engine.rows(), matrix)
    assert engine.records() == records
    for row in matrix:
        _assert_query(engine, matrix, records, row)
