"""Exact row-space and record-lineage contracts for tracker classification."""

import numpy as np
import pytest

from lightstim.ir.tableau import PauliTableau
from lightstim.ir.tracker_algebra import (
    Rows, classify_stabilizers, readout_closures, record_dependencies,
    split_known_rows,
)
from lightstim.utils.subsystem_algebra import (
    combine_records, independent_row_indices, reduce_modulo,
)


def rows(strings, records=None, *, width=None):
    if not strings:
        return Rows.empty(width)
    n = len(strings[0])
    matrix = np.zeros((len(strings), 2*n), dtype=np.uint8)
    for i, pauli in enumerate(strings):
        assert len(pauli) == n
        for q, letter in enumerate(pauli):
            matrix[i, q] = letter in "XY"
            matrix[i, n+q] = letter in "YZ"
    return Rows(matrix, records if records is not None else [[] for _ in strings])


def rank(matrix):
    return len(independent_row_indices(matrix))


def test_intersection_finds_bare_product_and_preserves_actual_records():
    S = rows(["ZZ", "IZ"], [[0], [1]])
    empty = Rows.empty(4)
    result = classify_stabilizers(S, rows(["ZI"]).matrix)
    assert result.bare.records == ((0, 1),)
    assert np.array_equal(result.bare.matrix, rows(["ZI"]).matrix)
    assert result.dressed_quotient_rank == 1
    # K does not carry or donate its own empty records; true S lineage wins.
    assert np.array_equal(result.bare_lineage, [[1, 1]])


def test_postselection_rows_remain_exact_and_mixed_lineage_is_not_OR_tagged():
    S = rows(["ZZ", "IZ", "ZI"], [[0], [1], [0, 1]])
    empty = Rows.empty(4)
    result = classify_stabilizers(S, rows(["ZI"]).matrix,
                                  postselected_rows=[0])
    assert result.exact_postselection_map == {0: 1}
    assert result.dressed.records[0] == (0,)
    assert result.safely_postselected_output_rows == frozenset({1})
    assert result.bare.records == ((0, 1),)


def test_dependent_selected_rows_are_retained_without_confusing_rank_and_count():
    S = rows(["ZI", "ZI", "IZ"], [[0], [2], [1]])
    empty = Rows.empty(4)
    result = classify_stabilizers(S, rows(["ZI"]).matrix,
                                  postselected_rows=[0, 1], include_dependencies=True)
    assert len(result.bare.matrix) == 2
    assert rank(result.bare.matrix) == 1
    assert result.bare.records == ((0,), (2,))
    assert result.exact_postselection_map == {0: 0, 1: 1}
    assert result.source_dependencies.records == ((0, 2),)


def test_classification_uses_only_S_records_without_modifying_source():
    S = rows(["ZZ"], [[4]])
    S_before = S.matrix.copy()
    result = classify_stabilizers(S, Rows.empty(4).matrix)
    assert result.dressed.records == ((4,),)
    assert np.array_equal(S.matrix, S_before)
    assert S.records == ((4,),)


def test_unknown_sentinel_is_rejected_and_split_without_canceling_it():
    with pytest.raises(ValueError, match="unknown sentinel"):
        rows(["Z"], [[-1]])
    tableau = PauliTableau(2)
    tableau.add_stabilizers(rows(["ZI", "IZ", "ZZ"]).matrix,
                           [[-1], [4], [-1, -1]])
    with pytest.raises(ValueError, match="unknown sentinel"):
        classify_stabilizers(tableau, rows(["ZI", "IZ"]).matrix)
    separated = split_known_rows(tableau)
    assert separated.known_indices == (1,)
    assert separated.unknown_indices == (0, 2)
    assert separated.known.records == ((4,),)
    assert tableau.records == [[-1], [4], [-1, -1]]


def test_readout_closes_products_and_prioritizes_bare_space():
    bare = rows(["ZZI", "IZZ"], [[0], [1]])
    dressed = rows(["ZII"], [[2]])
    live = Rows.empty(6)
    M = rows(["ZII", "IIZ"], [[3], [4]])
    result = readout_closures(bare, dressed, live, M)
    # Neither bare input row is individually measurable, their product is.
    assert np.array_equal(result.detectors.rows.matrix, rows(["ZIZ"]).matrix)
    assert result.detectors.parities == ((0, 1, 3, 4),)
    assert len(result.observables.rows.matrix) == 1
    assert result.observables.parities == ((2, 3),)
    assert np.array_equal(result.detectors.state_lineage, [[1, 1, 0]])


def test_readout_extracts_combination_of_live_logicals():
    empty = Rows.empty(4)
    L = rows(["XX", "XI"], [[0], [1]])
    result = readout_closures(empty, empty, L, rows(["IX"], [[2]]))
    assert not len(result.detectors.rows.matrix)
    assert result.observables.parities == ((0, 1, 2),)
    assert np.array_equal(result.observables.state_lineage, [[1, 1]])


def test_pre_readout_bare_roles_survive_until_emission():
    # Post-readout K=0 must not be applied before this closure: ZZ is a DET.
    bare = rows(["ZZ"], [[0]])
    empty = Rows.empty(4)
    result = readout_closures(bare, empty, empty,
                              rows(["ZI", "IZ"], [[1], [2]]), postselected_rows=[0])
    assert result.detectors.parities == ((0, 1, 2),)
    assert not len(result.observables.rows.matrix)
    assert result.postselection.parities == ((0, 1, 2),)


@pytest.mark.parametrize("seed", range(64))
def test_random_linear_algebra_invariants(seed):
    rng = np.random.default_rng(seed)
    width = 6
    s = rng.integers(0, 2, size=(int(rng.integers(0, 9)), width), dtype=np.uint8)
    k = rng.integers(0, 2, size=(int(rng.integers(0, 8)), width), dtype=np.uint8)
    S = Rows(s, [[i] for i in range(len(s))])
    empty = Rows.empty(width)
    selected = list(range(min(2, len(s))))
    result = classify_stabilizers(S, k, postselected_rows=selected)
    bare, dressed = result.bare, result.dressed
    assert rank(bare.matrix) == rank(s) + rank(k) - rank(np.vstack([s, k]))
    assert rank(np.vstack([bare.matrix, dressed.matrix])) == rank(s)
    assert not reduce_modulo(bare.matrix, k).any()
    for table, lineage in ((bare, result.bare_lineage), (dressed, result.dressed_lineage)):
        assert np.array_equal(lineage @ s % 2, table.matrix)
        assert table.records == tuple(tuple(parity) for parity in combine_records(lineage, S.records))
    m = rng.integers(0, 2, size=(int(rng.integers(0, 8)), width), dtype=np.uint8)
    M = Rows(m, [[100+i] for i in range(len(m))])
    closure = readout_closures(bare, dressed, empty, M)
    det = closure.detectors.rows.matrix
    obs = closure.observables.rows.matrix
    assert rank(np.vstack([det, obs])) == rank(s) + rank(m) - rank(np.vstack([s, m]))
    assert rank(det) == rank(bare.matrix) + rank(m) - rank(np.vstack([bare.matrix, m]))
    state = np.vstack([bare.matrix, dressed.matrix])
    state_records = bare.records + dressed.records
    for closed in (closure.detectors, closure.observables):
        assert np.array_equal(closed.state_lineage @ state % 2, closed.rows.matrix)
        assert np.array_equal(closed.measurement_lineage @ m % 2, closed.rows.matrix)
        prior = combine_records(closed.state_lineage, state_records)
        measured = combine_records(closed.measurement_lineage, M.records)
        expected = tuple(tuple(sorted(set(a) ^ set(b))) for a, b in zip(prior, measured))
        assert closed.parities == expected


def test_all_empty_shapes_are_well_defined():
    empty = Rows.empty(4)
    result = classify_stabilizers(empty, empty.matrix)
    assert result.bare_lineage.shape == (0, 0)
    assert result.dressed_lineage.shape == (0, 0)
    closed = readout_closures(empty, empty, empty, empty)
    assert closed.detectors.state_lineage.shape == (0, 0)
    assert closed.observables.measurement_lineage.shape == (0, 0)


def test_classification_preserves_every_local_bare_representative_and_order():
    S = rows(["ZZI", "IZZ", "ZIZ", "ZII"], [[0], [1], [2], [3]])
    K = rows(["ZZI", "IZZ"]).matrix
    result = classify_stabilizers(S, K, postselected_rows=[1])
    assert np.array_equal(result.bare.matrix, S.matrix[:3])
    assert result.bare.records == ((0,), (1,), (2,))
    assert np.array_equal(result.bare_lineage, np.eye(4, dtype=np.uint8)[:3])
    assert result.original_row_map == {0: 0, 1: 1, 2: 2, 3: 3}
    assert result.exact_postselection_map == {1: 1}
    assert len(result.bare.matrix) == 3 and rank(result.bare.matrix) == 2


def test_classification_completes_hidden_bare_product_after_original_rows():
    S = rows(["IIZ", "ZZI", "IZI"], [[0], [1], [2]])
    result = classify_stabilizers(S, rows(["IIZ", "ZII"]).matrix)
    assert np.array_equal(result.bare.matrix, rows(["IIZ", "ZII"]).matrix)
    assert result.bare.records == ((0,), (1, 2))
    assert np.array_equal(result.bare_lineage, [[1, 0, 0], [0, 1, 1]])


def test_readout_preserves_redundant_local_detectors_before_any_products():
    bare = rows(["ZZI", "IZZ", "ZIZ"], [[0], [1], [2]])
    empty = Rows.empty(6)
    M = rows(["ZII", "IZI", "IIZ"], [[3], [4], [5]])
    result = readout_closures(bare, empty, empty, M, postselected_rows=[1])
    assert np.array_equal(result.detectors.rows.matrix, bare.matrix)
    assert result.detectors.parities == ((0, 3, 4), (1, 4, 5), (2, 3, 5))
    assert np.array_equal(result.detectors.state_lineage, np.eye(3, dtype=np.uint8))
    assert result.postselection.parities == ((1, 4, 5),)


def test_readout_appends_hidden_bare_closure_after_local_closure():
    bare = rows(["IIIZ", "ZZII", "IZZI"], [[0], [1], [2]])
    empty = Rows.empty(8)
    M = rows(["ZIII", "IIZI", "IIIZ"], [[3], [4], [5]])
    result = readout_closures(bare, empty, empty, M)
    assert np.array_equal(result.detectors.rows.matrix, rows(["IIIZ", "ZIZI"]).matrix)
    assert result.detectors.parities == ((0, 5), (1, 2, 3, 4))


def test_protected_closure_prefers_original_observable_representatives():
    empty = Rows.empty(6)
    bare = rows(["ZZI"], [[0]])
    dressed = rows(["IIZ"], [[1]])
    live = rows(["ZII"], [[2]])
    M = rows(["ZII", "IZI", "IIZ"], [[3], [4], [5]])
    result = readout_closures(bare, dressed, live, M)
    assert np.array_equal(result.observables.rows.matrix, np.vstack([dressed.matrix, live.matrix]))
    assert result.observables.parities == ((1, 5), (2, 3))
    assert np.array_equal(result.observables.state_lineage, [[0, 1, 0], [0, 0, 1]])


def test_pauli_tableau_inputs_and_record_only_dependencies_are_supported():
    bare = PauliTableau(1)
    bare.add_stabilizers(rows(["Z", "Z"]).matrix, [[0], [1]])
    M = PauliTableau(1)
    M.add_stabilizers(rows(["Z"]).matrix, [[2]])
    empty = PauliTableau(1)
    result = readout_closures(bare, empty, empty, M, include_dependencies=True)
    assert result.detectors.parities == ((0, 2), (1, 2))
    assert result.source_dependencies.records == ((0, 1),)
    dependencies = record_dependencies(bare)
    assert np.array_equal(dependencies.lineage, [[1, 1]])
    assert dependencies.records == ((0, 1),)


@pytest.mark.parametrize("seed", range(32))
def test_all_bare_preserves_exact_rows_metadata_and_record_dependencies(seed, monkeypatch):
    """Containment needs no intersection solve, even with redundant selected S.

    The whole supplied S list is the required output, including Pauli-zero
    rows with nonempty records. This checks representative equality rather
    than only the weaker span equality needed by the general classifier.
    """
    rng = np.random.default_rng(seed)
    width = 2 * int(rng.integers(1, 13))
    k_count = int(rng.integers(0, 18))
    s_count = int(rng.integers(0, 25))
    K = rng.integers(0, 2, size=(k_count, width), dtype=np.uint8)
    coefficients = rng.integers(0, 2, size=(s_count, k_count), dtype=np.uint8)
    matrix = coefficients @ K % 2
    if s_count:
        matrix[-1] = 0  # Its nonempty records must survive the fast path.
    S = Rows(matrix, [[i] for i in range(s_count)])
    selected = tuple(i for i in range(s_count) if rng.integers(0, 2))
    dependencies = record_dependencies(S)

    def unnecessary_intersection(*args, **kwargs):
        pytest.fail("All-bare classification should not solve an intersection")

    monkeypatch.setattr("lightstim.ir.tracker_algebra.intersect_row_spaces",
                        unnecessary_intersection)
    result = classify_stabilizers(
        S, K, postselected_rows=selected, include_dependencies=True)

    np.testing.assert_array_equal(result.bare.matrix, S.matrix)
    assert result.bare.records == S.records
    assert not np.shares_memory(result.bare.matrix, S.matrix)
    np.testing.assert_array_equal(result.bare_lineage, np.eye(s_count, dtype=np.uint8))
    assert result.dressed.matrix.shape == (0, width)
    assert result.dressed.records == ()
    assert result.dressed_lineage.shape == (0, s_count)
    assert result.original_row_map == {i: i for i in range(s_count)}
    assert result.exact_postselection_map == {i: i for i in selected}
    assert result.safely_postselected_output_rows == frozenset(selected)
    assert result.dressed_quotient_rank == 0
    np.testing.assert_array_equal(result.source_dependencies.lineage, dependencies.lineage)
    assert result.source_dependencies.records == dependencies.records
