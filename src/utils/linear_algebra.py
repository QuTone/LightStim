import numpy as np
from typing import Tuple, List

# Linear Algebra Utils

# --- Optional C++ acceleration ---
# Try to import the bitpacked GF(2) RREF backend.
# Falls back to pure Python if not available (e.g., no C++ compiler).
# Build with: python src/utils/cpp/build.py
_CPP_AVAILABLE = False
try:
    from src.utils.cpp._gf2_rref_cpp import row_echelon as _row_echelon_cpp
    _CPP_AVAILABLE = True
except ImportError:
    try:
        # Also try relative import for when running from project root
        from .cpp._gf2_rref_cpp import row_echelon as _row_echelon_cpp
        _CPP_AVAILABLE = True
    except (ImportError, SystemError):
        _row_echelon_cpp = None


def row_echelon(mat, reduced=False):
    """GF(2) (reduced) row echelon form. Uses C++ backend if available."""
    if _CPP_AVAILABLE:
        mat_u8 = np.ascontiguousarray(mat, dtype=np.uint8)
        rref, rank, transform, pivot_cols = _row_echelon_cpp(mat_u8, reduced)
        return [np.asarray(rref), rank, np.asarray(transform), list(pivot_cols)]
    return _row_echelon_python(mat, reduced)


def _row_echelon_python(mat, reduced=False):
    r"""Converts a binary matrix to (reduced) row echelon form via Gaussian Elimination, 
    also works for rank-deficient matrix. Unlike the make_systematic method,
    no column swaps will be performed.

    Input 
    ----------
    mat : ndarry
        A binary matrix in numpy.ndarray format.
    reduced: bool
        Defaults to False. If true, the reduced row echelon form is returned. 
    
    Output
    -------
    row_ech_form: ndarray
        The row echelon form of input matrix.
    rank: int
        The rank of the matrix.
    transform: ndarray
        The transformation matrix such that (transform_matrix@matrix)=row_ech_form
    pivot_cols: list
        List of the indices of pivot num_cols found during Gaussian elimination
    """

    m, n = np.shape(mat)
    # Don't do "m<=n" check, allow over-complete matrices
    mat = np.copy(mat)
    # Convert to bool for faster arithmetics
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    # Allow all-zero column. Row operations won't induce all-zero columns, if they are not present originally.
    # The make_systematic method will swap all-zero columns with later non-all-zero columns.
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(n):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if not mat[pivot_row, col]:
            # Find a row with a 1 in this column
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if mat[swap_row_index, col]:
                # Swap rows
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                # Transformation matrix update to reflect this row swap
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if mat[pivot_row, col]: # will evaluate to True if this column is not all-zero
            if not reduced: # clean entries below the pivot 
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:           # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:    
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m: # no more rows to search
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]


def create_lambda_matrix(n: int) -> np.ndarray:
    """
    Creates the standard symplectic form Lambda for Interleaved format.
    Lambda = Block Diag([0 1; 1 0], ...)
    
    Args:
        n: Number of qubits.
    
    Returns:
        (2n, 2n) numpy array (uint8). Lambda = [0 I_n; I_n 0]
    """
    # Simply create it every time. Fast enough for construction phase.
    L = np.zeros((2*n, 2*n), dtype=np.uint8)
    idx = np.arange(n)
    L[idx, n+idx] = 1
    L[n+idx, idx] = 1

    return L

def check_commutativity(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """
    Vectorized check of commutation between two sets of Pauli strings.
    Uses the symplectic inner product: (P, Q) anti-commute iff X_P·Z_Q + Z_P·X_Q = 1 (mod 2).
    Avoids allocating a full (2N, 2N) Lambda matrix.

    Args:
        mat_a: (N_a, 2*num_qubits) matrix.
        mat_b: (N_b, 2*num_qubits) matrix.

    Returns:
        (N_a, N_b) binary matrix. [i, j] = 1 if mat_a[i] anti-commutes with mat_b[j].
    """
    if mat_a.shape[1] != mat_b.shape[1]:
        raise ValueError(f"Matrices must have the same number of columns. "
                        f"mat_a has {mat_a.shape[1]} columns, mat_b has {mat_b.shape[1]} columns")

    if mat_a.shape[1] % 2 != 0:
        raise ValueError(f"Number of columns must be even (2*num_qubits). Got {mat_a.shape[1]} columns")

    n = mat_a.shape[1] // 2
    # Symplectic inner product without allocating (2N, 2N) Lambda matrix:
    #   result = A_x @ B_z.T + A_z @ B_x.T  (mod 2)
    a_x, a_z = mat_a[:, :n], mat_a[:, n:]
    b_x, b_z = mat_b[:, :n], mat_b[:, n:]
    return (a_x @ b_z.T + a_z @ b_x.T) % 2

def kernel_gf2(M: np.ndarray) -> np.ndarray:
    """
    Right null space of M over GF(2) as rows.
    Returns (k, n) matrix whose rows span ker(M) = {x : M @ x = 0}.
    """
    null_cols = _null_space_gf2(M)  # (n, k) columns
    return null_cols.T.astype(np.uint8)  # (k, n) rows


def kernel_rows_via_transpose_gf2(M: np.ndarray) -> np.ndarray:
    """
    Right null space of M over GF(2) as rows.
    Uses row_echelon(M.T) trick (faster for fat matrices m < n).
    Same interface as kernel_gf2.
    """
    m, n = M.shape
    if m >= n:
        return kernel_gf2(M)  # use standard path when M is tall
    Mt = M.T.astype(bool)
    _, rank, transform, _ = row_echelon(Mt)
    ker = transform[rank:, :].astype(np.uint8)
    return ker


def row_basis_gf2(A: np.ndarray) -> np.ndarray:
    """
    Row basis of A over GF(2).
    Returns linearly independent rows of A.
    """
    _, _, _, pivot_cols = row_echelon(A.T)
    return A[np.array(pivot_cols)].astype(np.uint8)


def quotient_basis_gf2(kernel_rows: np.ndarray, rowspace_basis: np.ndarray) -> np.ndarray:
    """
    Vectors in kernel that are not in rowspace (kernel / rowspace).
    Used for CSS logical operators: lz in ker(Hx) but not in im(Hz.T).
    """
    if kernel_rows.shape[0] == 0:
        return np.zeros((0, kernel_rows.shape[1]), dtype=np.uint8)
    if rowspace_basis.shape[0] == 0:
        return kernel_rows.astype(np.uint8)

    log_stack = np.vstack([rowspace_basis, kernel_rows]) % 2
    _, _, _, pivots = row_echelon(log_stack.T)
    n_rowspace = rowspace_basis.shape[0]
    log_op_indices = [i for i in range(n_rowspace, log_stack.shape[0]) if i in pivots]
    if not log_op_indices:
        return np.zeros((0, kernel_rows.shape[1]), dtype=np.uint8)
    return log_stack[log_op_indices].astype(np.uint8)


def _null_space_gf2(A: np.ndarray) -> np.ndarray:
    """
    Computes the right null space of A over GF(2).
    Returns columns of N such that A @ N = 0.

    Args:
        A: (m, n) matrix over GF(2).

    Returns:
        (n, k) matrix whose columns span the null space. k = n - rank(A).
    """
    m, n = A.shape
    rref, rank, _, pivot_cols = row_echelon(A.astype(np.uint8).copy(), reduced=True)
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]

    if len(free_cols) == 0:
        return np.zeros((n, 0), dtype=np.uint8)

    null_vecs = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        # Back-substitute: for each pivot row from bottom, express pivot var in terms of rest
        # Row i has pivot at pivot_cols[i]. rref[i, :] @ x = 0 => x[pivot_cols[i]] = sum_{j!=piv} rref[i,j]*x[j]
        for i in range(rank - 1, -1, -1):
            pcol = pivot_cols[i]
            val = 0
            for j in range(n):
                if j != pcol and rref[i, j] and x[j]:
                    val ^= 1
            x[pcol] = val
        null_vecs.append(x)

    return np.column_stack(null_vecs) if null_vecs else np.zeros((n, 0), dtype=np.uint8)


def _left_null_space_gf2(B: np.ndarray) -> np.ndarray:
    """
    Left null space of B: vectors v such that v @ B = 0.
    Equivalently, right null space of B.T.
    Returns (M, k) matrix whose COLUMNS are null space basis vectors.
    """
    N = _null_space_gf2(B.T)  # (M, k), columns are (M,) vectors with v @ B = 0
    return N


def _greedy_reduce_weight(coeffs: np.ndarray, null_rows: np.ndarray) -> np.ndarray:
    """
    Greedily reduce Hamming weight of coeffs by adding null space vectors.
    Modifies coeffs in place for each row that is dependent (non-zero).
    null_rows: (k, M) matrix, each row is a left null space vector.
    """
    if null_rows.shape[0] == 0:
        return coeffs

    coeffs = coeffs.copy()
    n_rows, M = coeffs.shape

    for row_idx in range(n_rows):
        c = coeffs[row_idx].copy()
        current_weight = int(np.sum(c))
        if current_weight == 0:
            continue

        # Pool of null vectors we can still add (each helps at most once in GF(2))
        remaining = list(range(null_rows.shape[0]))
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_idx = -1
            for i in remaining:
                v = null_rows[i]
                new_c = (c + v) % 2
                new_weight = int(np.sum(new_c))
                delta = current_weight - new_weight
                if delta > best_delta:
                    best_delta = delta
                    best_idx = i

            if best_idx >= 0 and best_delta > 0:
                c = (c + null_rows[best_idx]) % 2
                current_weight -= best_delta
                remaining.remove(best_idx)
                improved = True
            else:
                break

        coeffs[row_idx] = c

    return coeffs


def _find_blocks(vectors: np.ndarray) -> List[List[int]]:
    """
    Partition vector indices into independent blocks via Union-Find on qubit support.

    Two vectors are in the same block if they share any non-zero qubit column
    (transitively). Returns a list of groups, each group is a list of row indices.

    Args:
        vectors: (K, 2N) binary matrix.

    Returns:
        List of groups (list of row indices). Single-block means no block structure.
    """
    K, cols = vectors.shape
    if K == 0:
        return []

    # Union-Find
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build column → row index mapping (which rows touch each column)
    col_to_rows = {}
    for i in range(K):
        nz_cols = np.where(vectors[i])[0]
        for c in nz_cols:
            if c in col_to_rows:
                # Union this row with the first row that touches this column
                union(i, col_to_rows[c])
            else:
                col_to_rows[c] = i

    # Group by root
    groups = {}
    for i in range(K):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


def solve_linear_decomposition(
    basis: np.ndarray,
    targets: np.ndarray,
    reduce_weight: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Solves for x in: x @ basis = targets (over GF(2)).

    Automatically detects block-diagonal structure and performs per-block RREF
    when multiple independent blocks exist (e.g., multi-patch QEC systems
    before inter-patch gates). This can provide orders-of-magnitude speedup
    for systems with many independent patches.

    Args:
        basis: (M, 2N) matrix.
        targets: (K, 2N) matrix.
        reduce_weight: If True (default), when basis has linear dependencies,
            greedily reduces Hamming weight of decomposition by adding null space vectors.

    Returns:
        coeffs: (K, M) matrix. Coefficients relative to the ORIGINAL basis.
                Only valid/meaningful if is_dependent[k] is True.
        is_dependent: (K,) boolean. True if target[k] is strictly generated by basis.
        new_basis_indices: List[int]. Indices of 'targets' (0 to K-1) that represent
                           NEW linear independent dimensions (Logical Basis).
                           These are the 'pivots' found in the target section.
    """
    n_basis = basis.shape[0] if basis.shape[0] > 0 else 0
    n_targets = targets.shape[0]

    # --- Block-diagonal detection ---
    # Stack all vectors (basis + targets) and find independent blocks.
    # If multiple blocks exist, solve each independently for a large speedup.
    if n_basis > 0 and n_targets > 0:
        all_vectors = np.vstack([basis, targets])
        blocks = _find_blocks(all_vectors)

        if len(blocks) > 1:
            # Multiple independent blocks detected — solve per-block
            coeffs = np.zeros((n_targets, n_basis), dtype=np.uint8)
            is_dependent = np.ones(n_targets, dtype=bool)
            new_basis_indices = []

            for block_indices in blocks:
                # Split block indices into basis and target indices
                b_local = [i for i in block_indices if i < n_basis]
                t_local = [i - n_basis for i in block_indices if i >= n_basis]

                if not t_local:
                    continue  # No targets in this block

                # Extract sub-matrices (only relevant columns)
                if b_local:
                    sub_basis = basis[b_local]
                else:
                    sub_basis = np.zeros((0, basis.shape[1]), dtype=basis.dtype)
                sub_targets = targets[t_local]

                # Find relevant columns (non-zero in this block)
                block_rows = sub_basis if sub_basis.shape[0] > 0 else sub_targets
                all_block = np.vstack([sub_basis, sub_targets]) if sub_basis.shape[0] > 0 else sub_targets
                active_cols = np.where(np.any(all_block, axis=0))[0]

                if len(active_cols) == 0:
                    # All-zero block — targets are trivially dependent (zero vectors)
                    continue

                # Solve the sub-problem on reduced column space
                sub_b = sub_basis[:, active_cols] if sub_basis.shape[0] > 0 else np.zeros((0, len(active_cols)), dtype=basis.dtype)
                sub_t = sub_targets[:, active_cols]

                sub_coeffs, sub_dep, sub_new = _solve_linear_decomposition_core(
                    sub_b, sub_t, reduce_weight
                )

                # Map results back to global indices
                for local_k, global_k in enumerate(t_local):
                    is_dependent[global_k] = sub_dep[local_k]
                    if sub_dep[local_k]:
                        for local_m, global_m in enumerate(b_local):
                            coeffs[global_k, global_m] = sub_coeffs[local_k, local_m]

                for local_idx in sub_new:
                    new_basis_indices.append(t_local[local_idx])

            return coeffs, is_dependent, new_basis_indices

    # --- Fallback: single block (no block structure or empty basis) ---
    return _solve_linear_decomposition_core(basis, targets, reduce_weight)


def _solve_linear_decomposition_core(
    basis: np.ndarray,
    targets: np.ndarray,
    reduce_weight: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Core RREF-based linear decomposition (single block)."""
    if basis.shape[0] == 0:
        system = targets.T
        n_basis = 0
    else:
        system = np.hstack([basis.T, targets.T])
        n_basis = basis.shape[0]

    n_targets = targets.shape[0]

    rref, rank, _, pivot_cols = row_echelon(system, reduced=True)

    coeffs = np.zeros((n_targets, n_basis), dtype=np.uint8)
    is_dependent = np.ones(n_targets, dtype=bool)
    new_basis_indices = []

    row_to_basis_map = {}
    for r, p_col in enumerate(pivot_cols):
        if p_col < n_basis:
            row_to_basis_map[r] = p_col
        else:
            target_idx = p_col - n_basis
            new_basis_indices.append(target_idx)
            is_dependent[target_idx] = False

    for k in range(n_targets):
        if not is_dependent[k]:
            continue
        col_idx = n_basis + k
        target_col_vec = rref[:, col_idx]
        nonzero_rows = np.where(target_col_vec)[0]
        for r in nonzero_rows:
            if r in row_to_basis_map:
                basis_idx = row_to_basis_map[r]
                coeffs[k, basis_idx] = 1
            else:
                is_dependent[k] = False
                coeffs[k, :] = 0
                break

    # Greedy weight reduction when basis has dependencies
    if reduce_weight and n_basis > 0 and basis.shape[0] > 0:
        null_cols = _left_null_space_gf2(basis)  # (M, k), columns are null vectors
        if null_cols.shape[1] > 0:
            null_rows = null_cols.T  # (k, M) for iteration
            coeffs = _greedy_reduce_weight(coeffs, null_rows)

    return coeffs, is_dependent, new_basis_indices