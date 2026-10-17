"""Unsigned Pauli acceleration must preserve the reference algebra exactly."""

import numpy as np
import pytest
import stim

from lightstim.utils import pauli_backend
from lightstim.utils.linear_algebra import (
    _solve_linear_decomposition_core,
    solve_linear_decomposition,
)


pytestmark = pytest.mark.smoke


@pytest.fixture
def native(monkeypatch):
    if not pauli_backend._NATIVE_AVAILABLE:
        pytest.skip("optional native Pauli backend has not been built")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    return pauli_backend


def pauli_rows(*strings):
    return np.asarray(
        [np.concatenate(stim.PauliString(s).to_numpy()) for s in strings],
        dtype=np.uint8,
    )


def clifford_matrix(rng, n):
    circuit = stim.Circuit()
    circuit.append("I", range(n))
    for _ in range(5 * n):
        circuit.append(str(rng.choice(["H", "S", "SQRT_X"])), [int(rng.integers(n))])
        if n > 1:
            circuit.append("CX", rng.choice(n, 2, replace=False).tolist())
    tableau = circuit.to_tableau()
    return np.asarray([
        np.concatenate(p.to_numpy())
        for p in [*(tableau.x_output(q) for q in range(n)),
                  *(tableau.z_output(q) for q in range(n))]
    ], dtype=np.uint8)


def assert_reference_decomposition(native, basis, targets):
    original_basis, original_targets = basis.copy(), targets.copy()
    expected = _solve_linear_decomposition_core(basis, targets, reduce_weight=False)
    actual = native.solve_commuting_decomposition(basis, targets)
    assert actual is not None, "a commuting unsigned basis must use the native path"
    coeffs, dependent, new_indices = actual
    np.testing.assert_array_equal(coeffs, expected[0])
    np.testing.assert_array_equal(dependent, expected[1])
    assert list(new_indices) == expected[2]
    np.testing.assert_array_equal(coeffs[dependent] @ basis % 2, targets[dependent])
    assert not np.any(coeffs[~dependent])
    np.testing.assert_array_equal(basis, original_basis)
    np.testing.assert_array_equal(targets, original_targets)
    return actual


@pytest.mark.parametrize("seed", range(24))
def test_random_commuting_basis_matches_reference_coefficients_and_extension(native, seed):
    rng = np.random.default_rng(seed)
    n = 1 + seed % 6
    forward = clifford_matrix(rng, n)
    # Arbitrary products of the Z images commute, even with redundant rows.
    generators = forward[n:]
    basis = rng.integers(0, 2, size=(n + 3, n), dtype=np.uint8) @ generators % 2
    products = rng.integers(0, 2, size=(5, len(basis)), dtype=np.uint8) @ basis % 2
    arbitrary = rng.integers(0, 2, size=(7, 2 * n), dtype=np.uint8)
    targets = np.vstack([products, arbitrary, arbitrary[0] ^ arbitrary[1]])
    assert_reference_decomposition(native, basis, targets)


def test_phase_inconsistent_redundant_generators_are_unsigned(native):
    # +XX * +ZZ = -YY, so these cannot all be positive stabilizers of a state.
    # Tracker row algebra ignores the phases and must still accept this basis.
    basis = pauli_rows("XX", "ZZ", "YY", "XX", "II")
    targets = pauli_rows("YY", "XX", "ZZ", "II", "XI", "IZ")
    coeffs, dependent, _ = assert_reference_decomposition(native, basis, targets)
    np.testing.assert_array_equal(coeffs[0], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(dependent, [True, True, True, True, False, False])


def test_incomplete_basis_distinguishes_commutation_from_membership(native):
    basis = pauli_rows("ZII")
    targets = pauli_rows("ZII", "IZI", "XII", "IXI", "III", "ZZI", "IZI")
    _, dependent, new_indices = assert_reference_decomposition(native, basis, targets)
    np.testing.assert_array_equal(dependent, [True, False, False, False, True, False, False])
    # Later targets outside the original span need not add a new dimension.
    assert list(new_indices) == [1, 2, 3]


def test_extension_indices_keep_global_target_order(native):
    basis = pauli_rows("ZII")
    targets = pauli_rows("IIZ", "IZI", "III", "IZZ", "XII", "XZI")
    _, dependent, new_indices = assert_reference_decomposition(native, basis, targets)
    assert list(new_indices) == [0, 1, 4]
    np.testing.assert_array_equal(dependent, [False, False, True, False, False, False])


def test_public_solver_preserves_historical_block_order(native, monkeypatch):
    basis = pauli_rows("ZII")
    targets = pauli_rows("IXI", "ZZI", "IZI")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    expected = solve_linear_decomposition(basis, targets, reduce_weight=False)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    actual = solve_linear_decomposition(basis, targets, reduce_weight=False)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2] == [1, 0]


def test_public_solver_preserves_weight_reduction_for_redundant_records(native, monkeypatch):
    basis = pauli_rows("XX", "ZZ", "YY")
    targets = pauli_rows("YY")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    expected = solve_linear_decomposition(basis, targets)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    actual = solve_linear_decomposition(basis, targets)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[0], [[0, 0, 1]])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


@pytest.mark.parametrize("basis_rows,target_rows", [(0, 0), (0, 6), (4, 0), (4, 6)])
def test_empty_and_zero_bases_preserve_shapes(native, basis_rows, target_rows):
    basis = np.zeros((basis_rows, 6), dtype=np.uint8)
    targets = np.eye(6, dtype=np.uint8)[:target_rows]
    coeffs, dependent, _ = assert_reference_decomposition(native, basis, targets)
    assert coeffs.shape == (target_rows, basis_rows)
    assert dependent.shape == (target_rows,)


def test_noncommuting_basis_requests_reference_fallback(native):
    assert native.solve_commuting_decomposition(
        pauli_rows("XI", "ZI"), pauli_rows("YI", "II")
    ) is None


def test_mutating_a_previously_noncommuting_basis_rechecks_support(native):
    basis = pauli_rows("XI", "ZI")
    targets = pauli_rows("ZI", "II")
    assert native.solve_commuting_decomposition(basis, targets) is None
    basis[0] = basis[1]
    _, dependent, _ = assert_reference_decomposition(native, basis, targets)
    assert dependent.all()


def test_ordinary_odd_width_linear_system_requests_reference_fallback(native):
    assert native.solve_commuting_decomposition(
        np.eye(3, dtype=np.uint8), np.ones((1, 3), dtype=np.uint8)
    ) is None


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int64])
def test_noncontiguous_inputs_and_integer_types(native, dtype):
    basis = pauli_rows("XXI", "ZZI", "YYI", "IIZ").astype(dtype)[:, ::-1]
    targets = pauli_rows("YYI", "XXZ", "IZI", "IIZ").astype(dtype)[:, ::-1]
    assert not basis.flags.c_contiguous
    assert_reference_decomposition(native, basis, targets)


def test_mutating_basis_and_targets_does_not_reuse_stale_decomposition(native):
    basis = pauli_rows("XI")
    targets = pauli_rows("XI", "ZI")
    first = assert_reference_decomposition(native, basis, targets)
    np.testing.assert_array_equal(first[1], [True, False])
    basis[:] = pauli_rows("ZI")
    second = assert_reference_decomposition(native, basis, targets)
    np.testing.assert_array_equal(second[1], [False, True])
    targets[:] = pauli_rows("ZI", "II")
    third = assert_reference_decomposition(native, basis, targets)
    np.testing.assert_array_equal(third[1], [True, True])


@pytest.mark.parametrize("n", [63, 64, 65, 127, 128, 129])
def test_decomposition_crosses_packed_word_boundaries(native, n):
    basis = np.zeros((4, 2 * n), dtype=np.uint8)
    basis[0, [n, 2 * n - 1]] = 1
    basis[1, [n + n // 2, 2 * n - 1]] = 1
    basis[2] = basis[0] ^ basis[1]
    targets = np.vstack([basis[2], basis[0], np.zeros(2 * n, dtype=np.uint8)])
    targets[2, n - 1] = 1
    assert_reference_decomposition(native, basis, targets)


@pytest.mark.parametrize("seed", range(12))
def test_pauli_propagation_matches_unsigned_matrix_product(native, seed):
    rng = np.random.default_rng(seed)
    n = 1 + seed % 6
    forward = clifford_matrix(rng, n)
    rows = rng.integers(0, 2, size=(11, 2 * n), dtype=np.uint8)
    original_rows, original_forward = rows.copy(), forward.copy()
    actual = native.propagate_paulis(rows, forward)
    np.testing.assert_array_equal(actual, rows @ forward % 2)
    np.testing.assert_array_equal(rows, original_rows)
    np.testing.assert_array_equal(forward, original_forward)


@pytest.mark.parametrize("n", [1, 64, 65, 128, 129])
def test_pauli_propagation_packed_boundaries_and_empty_rows(native, n):
    rng = np.random.default_rng(n)
    # An H on each qubit exchanges the full X and Z blocks.
    forward = np.roll(np.eye(2 * n, dtype=np.uint8), n, axis=1)
    rows = rng.integers(0, 2, size=(5, 2 * n), dtype=np.uint8)
    np.testing.assert_array_equal(native.propagate_paulis(rows, forward), rows @ forward % 2)
    empty = native.propagate_paulis(rows[:0], forward)
    assert empty.shape == (0, 2 * n)


def test_mutating_clifford_matrix_does_not_reuse_stale_propagation(native):
    rows = pauli_rows("X", "Y", "Z")
    forward = np.eye(2, dtype=np.uint8)
    np.testing.assert_array_equal(native.propagate_paulis(rows, forward), rows)
    forward[:] = [[0, 1], [1, 0]]
    np.testing.assert_array_equal(native.propagate_paulis(rows, forward), rows[:, ::-1])


def test_reference_mode_works_without_native_calls(monkeypatch):
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")

    def unexpected_native_call(*args):
        pytest.fail("reference mode called the optional native backend")

    monkeypatch.setattr(pauli_backend, "_get_basis", unexpected_native_call)
    monkeypatch.setattr(pauli_backend, "_propagate_paulis_native", unexpected_native_call)
    rows = pauli_rows("X", "Y", "Z")
    forward = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    assert pauli_backend.solve_commuting_decomposition(rows[:1], rows) is None
    np.testing.assert_array_equal(pauli_backend.propagate_paulis(rows, forward), rows @ forward % 2)


@pytest.mark.parametrize("mode", ["auto", "reference", "native"])
def test_missing_optional_extension_falls_back_unless_explicitly_required(monkeypatch, mode):
    monkeypatch.setattr(pauli_backend, "_NATIVE_AVAILABLE", False)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", mode)
    rows = pauli_rows("Z")
    forward = np.eye(2, dtype=np.uint8)
    if mode == "native":
        with pytest.raises(ImportError, match="optional"):
            pauli_backend.solve_commuting_decomposition(rows, rows)
        with pytest.raises(ImportError, match="optional"):
            pauli_backend.propagate_paulis(rows, forward)
    else:
        assert pauli_backend.solve_commuting_decomposition(rows, rows) is None
        np.testing.assert_array_equal(pauli_backend.propagate_paulis(rows, forward), rows)
