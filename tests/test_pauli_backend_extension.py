"""Growing Pauli bases reuse immutable cached prefixes without changing rows."""

import numpy as np
import pytest
import stim

from lightstim.utils import pauli_backend
from lightstim.utils.linear_algebra import _solve_linear_decomposition_core


pytestmark = pytest.mark.smoke


@pytest.fixture
def native_class(monkeypatch):
    if not pauli_backend._NATIVE_AVAILABLE:
        pytest.skip("optional native Pauli backend has not been built")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    pauli_backend._clear_basis_cache()
    yield pauli_backend._CommutingBasis
    pauli_backend._clear_basis_cache()


def pauli_rows(*strings):
    return np.asarray([
        np.concatenate(stim.PauliString(string).to_numpy()) for string in strings
    ], dtype=np.uint8)


def assert_decomposition(instance, basis, targets):
    actual = instance.decompose(targets)
    expected = _solve_linear_decomposition_core(basis, targets, reduce_weight=False)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_extension_counts_redundant_source_rows_and_keeps_parent_unchanged(native_class):
    prefix = pauli_rows("ZII", "ZII", "III")
    suffix = pauli_rows("IZI", "ZZI", "IIZ", "III", "IZI")
    targets = pauli_rows("ZZZ", "IZI", "IIZ", "XII", "III")
    parent = native_class(prefix)
    before = parent.decompose(targets)
    child = parent.extended(suffix)
    assert parent.commuting and child.commuting
    assert_decomposition(parent, prefix, targets)
    assert_decomposition(child, np.vstack([prefix, suffix]), targets)
    np.testing.assert_array_equal(child.decompose(targets)[0][0],
                                  [1, 0, 0, 1, 0, 1, 0, 0])
    after = parent.decompose(targets)
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2] == before[2]


def test_noncommuting_extension_preserves_parent_and_requests_fallback(native_class):
    prefix = pauli_rows("ZI", "ZI")
    suffix = pauli_rows("IZ", "XI")
    targets = pauli_rows("ZI", "IZ", "II")
    parent = native_class(prefix)
    child = parent.extended(suffix)
    assert parent.commuting and not child.commuting
    assert not child.extended(pauli_rows("II")).commuting
    with pytest.raises(ValueError, match="not commuting"):
        child.decompose(targets)
    assert_decomposition(parent, prefix, targets)

    cached_parent = pauli_backend._get_basis(prefix)
    assert pauli_backend._get_basis(np.vstack([prefix, suffix])) is None
    assert pauli_backend._get_basis(prefix) is cached_parent
    assert_decomposition(cached_parent, prefix, targets)


def test_one_cached_parent_can_produce_independent_branches(native_class):
    prefix = pauli_rows("ZII", "III")
    parent = native_class(prefix)
    first_suffix = pauli_rows("IZI", "ZZI")
    second_suffix = pauli_rows("IIX", "IIX")
    first = parent.extended(first_suffix)
    second = parent.extended(second_suffix)
    targets = pauli_rows("ZZI", "ZIX", "IXI", "III")
    assert_decomposition(parent, prefix, targets)
    assert_decomposition(first, np.vstack([prefix, first_suffix]), targets)
    assert_decomposition(second, np.vstack([prefix, second_suffix]), targets)


@pytest.mark.parametrize("shape", [(1, 3), (1, 6), (4,)])
def test_extension_rejects_invalid_register_without_changing_parent(native_class, shape):
    prefix = pauli_rows("ZI")
    parent = native_class(prefix)
    with pytest.raises(ValueError):
        parent.extended(np.zeros(shape, dtype=np.uint8))
    assert_decomposition(parent, prefix, pauli_rows("ZI", "IZ"))


def test_zero_width_and_empty_extensions_preserve_source_count(native_class):
    parent = native_class(np.zeros((3, 0), dtype=np.uint8))
    child = parent.extended(np.zeros((2, 0), dtype=np.uint8))
    unchanged = child.extended(np.zeros((0, 0), dtype=np.uint8))
    targets = np.zeros((4, 0), dtype=np.uint8)
    for instance, count in [(parent, 3), (child, 5), (unchanged, 5)]:
        coeffs, dependent, new = instance.decompose(targets)
        assert coeffs.shape == (4, count)
        assert not coeffs.any() and dependent.all() and new == []


@pytest.mark.parametrize("seed", range(8))
def test_incremental_random_clifford_basis_matches_fresh_construction(native_class, seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 5
    circuit = stim.Circuit()
    circuit.append("I", range(n))
    for _ in range(5 * n):
        circuit.append(str(rng.choice(["H", "S", "SQRT_X"])),
                       [int(rng.integers(n))])
        circuit.append("CX", rng.choice(n, 2, replace=False).tolist())
    tableau = circuit.to_tableau()
    generators = np.asarray([
        np.concatenate(tableau.z_output(i).to_numpy()) for i in range(n)
    ], dtype=np.uint8)
    basis = rng.integers(0, 2, size=(n + 4, n), dtype=np.uint8) @ generators % 2
    products = rng.integers(0, 2, size=(6, len(basis)), dtype=np.uint8) @ basis % 2
    targets = np.vstack([products, rng.integers(0, 2, size=(6, 2 * n), dtype=np.uint8)])
    current = native_class(basis[:0])
    for stop in range(1, len(basis) + 1):
        previous = current
        current = previous.extended(basis[stop - 1:stop])
        assert_decomposition(previous, basis[:stop - 1], targets)
        assert_decomposition(current, basis[:stop], targets)
        fresh = native_class(basis[:stop]).decompose(targets)
        actual = current.decompose(targets)
        np.testing.assert_array_equal(actual[0], fresh[0])
        np.testing.assert_array_equal(actual[1], fresh[1])
        assert actual[2] == fresh[2]


def test_cache_extends_longest_prefix_even_when_shorter_one_was_used_later(monkeypatch):
    calls = []

    class FakeBasis:
        commuting = True
        estimated_bytes = 32

        def __init__(self, rows):
            self.rows = rows.copy()
            calls.append(("construct", rows.shape))

        def extended(self, rows):
            calls.append(("extend", len(self.rows), len(rows)))
            result = object.__new__(type(self))
            result.rows = np.vstack([self.rows, rows])
            return result

    monkeypatch.setattr(pauli_backend, "_CommutingBasis", FakeBasis)
    pauli_backend._clear_basis_cache()
    basis = np.asarray([[0, 0, i % 2, (i // 2) % 2] for i in range(5)], dtype=np.uint8)
    try:
        two = pauli_backend._get_basis(basis[:2])
        four = pauli_backend._get_basis(basis[:4])
        pauli_backend._get_basis(basis[:3])
        five = pauli_backend._get_basis(basis)
        assert calls[-1] == ("extend", 4, 1)
        np.testing.assert_array_equal(two.rows, basis[:2])
        np.testing.assert_array_equal(four.rows, basis[:4])
        np.testing.assert_array_equal(five.rows, basis)
        # Matching byte prefixes from a different register cannot be reused.
        pauli_backend._get_basis(np.zeros((4, 2), dtype=np.uint8))
        assert calls[-1] == ("construct", (4, 2))
    finally:
        pauli_backend._clear_basis_cache()
