"""Exact native coordinate algebra against the retained #124 policy oracle."""

from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from lightstim.ir.tracker_algebra import Rows, classify_stabilizers, readout_closures
from lightstim.utils import pauli_backend as backend
from lightstim.utils.linear_algebra import kernel_gf2, solve_linear_decomposition
from lightstim.utils.subsystem_algebra import intersect_row_spaces, reduce_modulo

pytestmark = pytest.mark.skipif(backend.PackedRows is None, reason="Native packed role algebra not built")


def assert_exact(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif is_dataclass(a):
        for field in fields(a):
            assert_exact(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert list(a.items()) == list(b.items())
    else:
        assert a == b


def compare_modes(monkeypatch, function, *args, **kwargs):
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    expected = function(*args, **kwargs)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    actual = function(*args, **kwargs)
    assert_exact(expected, actual)
    return actual


@pytest.mark.parametrize("width", [0, 2, 6, 62, 64, 66, 126, 128, 130])
@pytest.mark.parametrize("seed", range(12))
def test_classification_exact_arbitrary_roles_and_source_lineages(width, seed, monkeypatch):
    rng = np.random.default_rng(seed + width * 101)
    count = int(rng.integers(0, 24))
    sources = rng.integers(0, 2, (count, width), dtype=np.uint8)
    roles = rng.integers(0, 2, (int(rng.integers(0, 24)), width), dtype=np.uint8)
    # Deliberately dependent, record-only, and exact selected rows.
    if count > 3:
        sources[-1] = 0
        sources[-2] = sources[0] ^ sources[1]
        sources[-3] = sources[0]
    source = Rows(sources, [[i, 100 + i] for i in range(count)])
    selected = [i for i in range(count) if rng.integers(0, 2)]
    compare_modes(monkeypatch, classify_stabilizers, source, roles,
                  postselected_rows=selected, include_dependencies=True)


@pytest.mark.parametrize("seed", range(80))
def test_readout_exact_arbitrary_noncommuting_subspaces(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    width = 2 * int(rng.integers(0, 15))
    values = []
    next_record = 0
    for _ in range(4):
        count = int(rng.integers(0, 14))
        matrix = rng.integers(0, 2, (count, width), dtype=np.uint8)
        if count > 2:
            matrix[-1] = matrix[0] ^ matrix[1]
        values.append(Rows(matrix, [[next_record + i] for i in range(count)]))
        next_record += count
    selected = [i for i in range(sum(v.count for v in values[:3])) if rng.integers(0, 2)]
    compare_modes(monkeypatch, readout_closures, *values,
                  postselected_rows=selected, include_dependencies=True)


@pytest.mark.parametrize("seed", range(40))
def test_coordinate_change_preserves_exact_classification_and_closure_lineage(seed, monkeypatch):
    """Inverse-tableau coordinates may change pivots but never source policy."""
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    rng = np.random.default_rng(seed)
    width = 16
    transform = np.eye(width, dtype=np.uint8)
    for _ in range(100):
        i, j = rng.choice(width, 2, replace=False)
        transform[i] ^= transform[j]
    banks = [rng.integers(0, 2, (int(rng.integers(1, 18)), width), dtype=np.uint8)
             for _ in range(4)]
    packed = [backend.PackedRows(x) for x in banks]
    changed = [backend.PackedRows(x @ transform % 2) for x in banks]
    selected = list(range(0, len(banks[0]), 2))
    before = backend._packed_native.packed_classify_stabilizers(packed[0], packed[1], selected)
    after = backend._packed_native.packed_classify_stabilizers(changed[0], changed[1], selected)
    for key in before:
        if isinstance(before[key], backend.PackedRows):
            expected = before[key].numpy()
            if key in {"bare", "dressed"}:
                expected = expected @ transform % 2
            np.testing.assert_array_equal(expected, after[key].numpy())
        else:
            assert before[key] == after[key]
    before = backend._packed_native.packed_readout_closures(*packed, selected)
    after = backend._packed_native.packed_readout_closures(*changed, selected)
    for group in before:
        for key in before[group]:
            expected = before[group][key].numpy()
            if key == "rows":
                expected = expected @ transform % 2
            np.testing.assert_array_equal(expected, after[group][key].numpy())


@pytest.mark.parametrize("shape", [(0, 0), (0, 65), (65, 0), (9, 65), (65, 9), (127, 129)])
def test_nullspace_and_intersection_exact_word_boundaries(shape, monkeypatch):
    rng = np.random.default_rng(sum(shape))
    matrix = rng.integers(0, 2, shape, dtype=np.uint8)
    compare_modes(monkeypatch, kernel_gf2, matrix)
    other = rng.integers(0, 2, (max(0, shape[0] - 3), shape[1]), dtype=np.uint8)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    expected = intersect_row_spaces(matrix, other)
    expected_residue = reduce_modulo(matrix, other)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    for a, b in zip(expected, intersect_row_spaces(matrix, other)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(expected_residue, reduce_modulo(matrix, other))


@pytest.mark.parametrize("seed", range(30))
def test_general_decomposition_noncommuting_and_block_order(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    basis = rng.integers(0, 2, (int(rng.integers(0, 24)), 24), dtype=np.uint8)
    targets = rng.integers(0, 2, (int(rng.integers(0, 24)), 24), dtype=np.uint8)
    basis[:len(basis)//2, 12:] = 0
    basis[len(basis)//2:, :12] = 0
    targets[:len(targets)//2, 12:] = 0
    targets[len(targets)//2:, :12] = 0
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    expected = solve_linear_decomposition(basis, targets, reduce_weight=False)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    actual = solve_linear_decomposition(basis, targets, reduce_weight=False)
    for a, b in zip(expected[:2], actual[:2]):
        np.testing.assert_array_equal(a, b)
    assert expected[2] == actual[2]


@pytest.mark.parametrize("prefix_count", [0, 1, 62, 63, 64, 65, 126, 127, 128, 129])
def test_extended_basis_preserves_earliest_source_coefficients_across_word_boundaries(prefix_count, monkeypatch):
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    rng = np.random.default_rng(prefix_count)
    matrix = rng.integers(0, 2, (prefix_count + 7, 146), dtype=np.uint8)
    if prefix_count:
        matrix[prefix_count] = matrix[0]
    matrix[-1] = 0
    targets = np.vstack([matrix, rng.integers(0, 2, (12, 146), dtype=np.uint8)])
    parent = backend.PackedBasis(matrix[:prefix_count])
    parent_result = parent.decompose(targets)
    extended = parent.extended(matrix[prefix_count:])
    fresh = backend.PackedBasis(matrix)
    for a, b in zip(extended.decompose(targets), fresh.decompose(targets)):
        np.testing.assert_array_equal(a, b)
    for before, after in zip(parent_result, parent.decompose(targets)):
        np.testing.assert_array_equal(before, after)
    backend._clear_basis_cache()
    assert backend._get_packed_basis(matrix[:prefix_count]).rank == parent.rank
    cached = backend._get_packed_basis(matrix)
    for a, b in zip(cached.decompose(targets), fresh.decompose(targets)):
        np.testing.assert_array_equal(a, b)
    matrix[0] ^= 1
    changed = backend._get_packed_basis(matrix)
    for a, b in zip(changed.decompose(targets), backend.PackedBasis(matrix).decompose(targets)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [1, 63, 64, 65, 127, 128, 129])
def test_persistent_batch_fold_matches_sequential_physical_set_updates(n):
    """Direct inverse rows retain duplicate-qubit and literal-record behavior."""
    rng = np.random.default_rng(n)
    axes = rng.integers(0, 2, n)
    generators = np.zeros((n, 2 * n), dtype=np.uint8)
    generators[np.arange(n), np.arange(n) + n * axes] = 1
    coefficients = rng.integers(0, 2, (n + 4, n), dtype=np.uint8)
    matrix = np.vstack([generators, coefficients @ generators % 2,
                        np.zeros((1, 2 * n), dtype=np.uint8)])
    records = [[i, i, 500 + i % n] if i % 3 else [-1, -1, i]
               for i in range(len(matrix))]
    qubits = list(rng.permutation(n)) + [0, 0]
    bases = ["Z" if axes[q] else "X" for q in qubits]
    expected = matrix.copy()
    expected_records = [set(record) for record in records]
    for j, q in enumerate(qubits):
        for r in range(len(expected)):
            if expected[r, q] or expected[r, n + q]:
                expected[r, q] = expected[r, n + q] = 0
                expected_records[r] ^= {500 + j}
    state = backend._packed_native.PersistentRecordState(matrix, records)
    state.fold_measured(qubits, bases, 500)
    np.testing.assert_array_equal(state.rows(), expected)
    assert state.records() == [sorted(record) for record in expected_records]
    assert state.stats["fold_fastpaths"] == 1


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("shape", [(0, 7), (5, 0), (7, 11)])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int64, np.bool_, np.float64])
def test_packed_commutativity_rectangular_and_dtype_contract(n, shape, dtype, monkeypatch):
    from lightstim.utils.linear_algebra import check_commutativity

    rng = np.random.default_rng(n + sum(shape))
    left = rng.integers(0, 2, (shape[0], 2 * n), dtype=np.uint8).astype(dtype)[:, ::-1]
    right = rng.integers(0, 2, (shape[1], 2 * n), dtype=np.uint8).astype(dtype)[:, ::-1]
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    expected = check_commutativity(left, right)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    actual = check_commutativity(left, right)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
