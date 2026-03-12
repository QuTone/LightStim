"""Tests for solve_linear_decomposition with weight reduction."""
import numpy as np
import sys
sys.path.insert(0, ".")
from src.utils.linear_algebra import solve_linear_decomposition


def _verify_decomposition(basis: np.ndarray, targets: np.ndarray, coeffs: np.ndarray, is_dependent: np.ndarray):
    """Verify coeffs @ basis = targets for dependent rows, over GF(2)."""
    for k in range(targets.shape[0]):
        if is_dependent[k]:
            reconstructed = (coeffs[k] @ basis) % 2
            assert np.array_equal(reconstructed, targets[k]), (
                f"Row {k}: coeffs @ basis != target. "
                f"Got {reconstructed}, expected {targets[k]}"
            )


def test_user_example():
    """
    User's example: basis = [x1, x2, ..., xn, x1+x2+...+xn], target = x1+x2+...+xn.
    Original gives [1,1,...,1,0], we want [0,0,...,0,1].
    """
    n = 5
    # Unit rows x1..xn
    basis = np.eye(n + 1, dtype=np.uint8)[:n]
    # Last row = sum of first n
    last_row = np.zeros(n + 1, dtype=np.uint8)
    last_row[:n] = 1
    basis = np.vstack([basis, last_row])  # (n+1, n+1)
    # Target = last row = x1+...+xn
    target = last_row.reshape(1, -1)
    coeffs, is_dep, _ = solve_linear_decomposition(basis, target, reduce_weight=True)
    assert is_dep[0], "Target should be dependent"
    # Want [0,0,...,0,1] -> weight 1
    expected = np.zeros(n + 1, dtype=np.uint8)
    expected[-1] = 1
    assert np.array_equal(coeffs[0], expected), (
        f"Expected [0,...,0,1], got {coeffs[0]}"
    )
    _verify_decomposition(basis, target, coeffs, is_dep)


def test_without_weight_reduction():
    """Same example with reduce_weight=False -> original [1,1,...,1,0]."""
    n = 5
    basis = np.eye(n + 1, dtype=np.uint8)[:n]
    last_row = np.zeros(n + 1, dtype=np.uint8)
    last_row[:n] = 1
    basis = np.vstack([basis, last_row])
    target = last_row.reshape(1, -1)
    coeffs, is_dep, _ = solve_linear_decomposition(basis, target, reduce_weight=False)
    assert is_dep[0]
    expected_old = np.ones(n + 1, dtype=np.uint8)
    expected_old[-1] = 0
    assert np.array_equal(coeffs[0], expected_old)
    _verify_decomposition(basis, target, coeffs, is_dep)


def test_independent_basis():
    """When basis is full rank, reduce_weight should not change result."""
    basis = np.eye(4, dtype=np.uint8)
    target = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    coeffs, is_dep, _ = solve_linear_decomposition(basis, target, reduce_weight=True)
    assert is_dep[0]
    expected = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert np.array_equal(coeffs[0], expected)
    _verify_decomposition(basis, target, coeffs, is_dep)


def test_multiple_targets():
    """Multiple targets: one benefits from reduction, one doesn't."""
    n = 4
    basis = np.eye(n + 1, dtype=np.uint8)[:n]
    last_row = np.zeros(n + 1, dtype=np.uint8)
    last_row[:n] = 1
    basis = np.vstack([basis, last_row])
    # Target 1: last row -> should get [0,0,0,0,1]
    # Target 2: first row -> [1,0,0,0,0]
    targets = np.vstack([last_row, np.eye(n + 1, dtype=np.uint8)[0]])
    coeffs, is_dep, _ = solve_linear_decomposition(basis, targets, reduce_weight=True)
    assert is_dep[0] and is_dep[1]
    assert coeffs[0, -1] == 1 and np.sum(coeffs[0]) == 1
    assert coeffs[1, 0] == 1 and np.sum(coeffs[1]) == 1
    _verify_decomposition(basis, targets, coeffs, is_dep)


def test_symplectic_format():
    """Test with 2n format (e.g. 2 qubits -> 4 cols)."""
    # basis: 4 rows, 4 cols (2 qubits)
    basis = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
    ], dtype=np.uint8)
    target = np.array([[1, 1, 1, 0]], dtype=np.uint8)  # row3 = row0+row1+row2
    coeffs, is_dep, _ = solve_linear_decomposition(basis, target, reduce_weight=True)
    assert is_dep[0]
    # Best: use row3 directly -> [0,0,0,1]
    assert np.sum(coeffs[0]) == 1 and coeffs[0, 3] == 1
    _verify_decomposition(basis, target, coeffs, is_dep)


def test_greedy_multi_step():
    """Null space with 2+ vectors: may need multiple greedy steps."""
    # basis: rows 0,1,2,3 and row4 = row0+row1, row5 = row2+row3
    # Target = row0+row1+row2+row3. Original might use rows 0,1,2,3 -> weight 4.
    # Better: row4+row5 = row0+row1+row2+row3 -> weight 2.
    basis = np.eye(6, dtype=np.uint8)[:4]
    row4 = (basis[0] + basis[1]) % 2
    row5 = (basis[2] + basis[3]) % 2
    basis = np.vstack([basis, row4, row5])
    target = (row4 + row5) % 2
    target = target.reshape(1, -1)
    coeffs, is_dep, _ = solve_linear_decomposition(basis, target, reduce_weight=True)
    assert is_dep[0]
    # Optimal: coeffs = [0,0,0,0,1,1] -> weight 2
    assert np.sum(coeffs[0]) == 2
    assert coeffs[0, 4] == 1 and coeffs[0, 5] == 1
    _verify_decomposition(basis, target, coeffs, is_dep)


def test_empty_basis():
    """Empty basis edge case."""
    basis = np.zeros((0, 4), dtype=np.uint8)
    targets = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    coeffs, is_dep, new_basis = solve_linear_decomposition(basis, targets, reduce_weight=True)
    assert len(new_basis) == 1
    assert not is_dep[0]


if __name__ == "__main__":
    test_user_example()
    print("test_user_example: OK")
    test_without_weight_reduction()
    print("test_without_weight_reduction: OK")
    test_independent_basis()
    print("test_independent_basis: OK")
    test_multiple_targets()
    print("test_multiple_targets: OK")
    test_symplectic_format()
    print("test_symplectic_format: OK")
    test_greedy_multi_step()
    print("test_greedy_multi_step: OK")
    test_empty_basis()
    print("test_empty_basis: OK")
    print("\nAll tests passed.")
