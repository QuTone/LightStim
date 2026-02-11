from typing import Tuple, Dict, List, Optional, Set
import numpy as np
import stim
from src.ir.qec_patch import QECPatch


# ---------------------------------------------------------------------------
# GF(2) Linear Algebra Helpers
# ---------------------------------------------------------------------------

def _gf2_row_echelon(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Row reduce M over GF(2) to row echelon form.
    Returns (reduced matrix, list of pivot column indices).
    """
    M = M.copy() % 2
    nrows, ncols = M.shape
    pivots = []
    row = 0
    for col in range(ncols):
        # Find pivot in this column
        found = None
        for r in range(row, nrows):
            if M[r, col] == 1:
                found = r
                break
        if found is None:
            continue
        # Swap rows
        M[[row, found]] = M[[found, row]]
        # Eliminate below
        for r in range(nrows):
            if r != row and M[r, col] == 1:
                M[r] = (M[r] + M[row]) % 2
        pivots.append(col)
        row += 1
    return M, pivots


def _gf2_rank(M: np.ndarray) -> int:
    """Compute rank of binary matrix over GF(2)."""
    _, pivots = _gf2_row_echelon(M)
    return len(pivots)


def _gf2_kernel(M: np.ndarray) -> np.ndarray:
    """
    Compute kernel (null space) of M over GF(2).
    Returns a matrix whose rows form a basis of ker(M).
    """
    M = M.copy() % 2
    nrows, ncols = M.shape
    # Augment with identity for tracking
    aug = np.hstack([M.T, np.eye(ncols, dtype=int)])  # ncols x (nrows + ncols)
    aug = aug % 2
    reduced, pivots = _gf2_row_echelon(aug)

    # Kernel vectors: rows of reduced where the left part (M.T columns) is all zero
    kernel_rows = []
    for r in range(ncols):
        if np.all(reduced[r, :nrows] == 0):
            kernel_rows.append(reduced[r, nrows:])

    if len(kernel_rows) == 0:
        return np.zeros((0, ncols), dtype=int)
    return np.array(kernel_rows, dtype=int) % 2


def _gf2_quotient_basis(kernel: np.ndarray, rowspace_generators: np.ndarray) -> np.ndarray:
    """
    Given kernel basis vectors and rowspace generators, find representatives
    of kernel / rowspace. Returns vectors in kernel that are independent of
    the rowspace generators.
    """
    if kernel.shape[0] == 0:
        return np.zeros((0, kernel.shape[1]), dtype=int)
    if rowspace_generators.shape[0] == 0:
        return kernel.copy()

    # Stack rowspace generators on top of kernel
    combined = np.vstack([rowspace_generators, kernel]) % 2
    # Row reduce
    reduced, pivots = _gf2_row_echelon(combined)

    n_rowspace = rowspace_generators.shape[0]
    rank_rowspace = _gf2_rank(rowspace_generators)

    # The new independent vectors from the kernel portion
    # are the ones that survive after accounting for rowspace
    result = []
    for r in range(reduced.shape[0]):
        if np.any(reduced[r] != 0):
            result.append(reduced[r])

    result = np.array(result, dtype=int) % 2 if result else np.zeros((0, kernel.shape[1]), dtype=int)

    # The first rank_rowspace rows span the rowspace, the rest are the quotient
    if result.shape[0] > rank_rowspace:
        return result[rank_rowspace:]
    return np.zeros((0, kernel.shape[1]), dtype=int)


# ---------------------------------------------------------------------------
# BBCode Patch
# ---------------------------------------------------------------------------

class BBCode(QECPatch):
    """
    Bivariate Bicycle (BB) Code.

    A CSS code defined by polynomial matrices A, B over the cyclic group
    algebra of Z_l x Z_m. Each stabilizer is weight-6 (3 terms from A,
    3 terms from B).

    The code parameters [[n, k, d]] are:
    - n = 2 * l * m  (number of data qubits)
    - k computed from rank of parity check matrices
    - d optionally provided as metadata

    Parameters (passed via **kwargs):
    ----------------------------------
    l : int
        Size of first cyclic group dimension.
    m : int
        Size of second cyclic group dimension.
    A : list[list[int]]
        3x2 polynomial matrix. Each row [x_exp, y_exp] is a monomial x^a * y^b.
        Example: [[3,0],[0,1],[0,2]] represents x^3 + y + y^2.
    B : list[list[int]]
        3x2 polynomial matrix, same format as A.
    shift : tuple, optional (default: (0, 0))
        Global coordinate offset.
    d : int, optional
        Code distance (metadata only, not used in construction).

    Examples:
    ---------
    # [[144,12,12]] gross code
    >>> code = BBCode(l=12, m=6, A=[[3,0],[0,1],[0,2]], B=[[0,3],[1,0],[2,0]])

    # [[72,12,6]] code
    >>> code = BBCode(l=6, m=6, A=[[3,0],[0,1],[0,2]], B=[[0,3],[1,0],[2,0]], d=6)

    # From config dict
    >>> code = BBCode.from_config({'l': 12, 'm': 6, 'A': [[3,0],[0,1],[0,2]], 'B': [[0,3],[1,0],[2,0]]})
    """

    def _process_params(self):
        self.l = self.params.get('l')
        self.m = self.params.get('m')
        self.A = self.params.get('A')
        self.B = self.params.get('B')
        self.shift = self.params.get('shift', (0, 0))
        self.code_distance = self.params.get('d', None)

        if self.l is None or self.m is None:
            raise ValueError("Both 'l' and 'm' must be provided.")
        if self.A is None or self.B is None:
            raise ValueError("Both 'A' and 'B' polynomial matrices must be provided.")
        if len(self.A) != 3 or len(self.B) != 3:
            raise ValueError("A and B must each have exactly 3 rows (3 monomials).")
        for row in self.A + self.B:
            if len(row) != 2:
                raise ValueError("Each monomial must be [x_exponent, y_exponent].")
        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two numbers.")

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        l = self.l
        m = self.m
        A = self.A
        B = self.B

        # -------------------------------------------------------------------
        # Phase 1: Geometry Registration
        # -------------------------------------------------------------------
        # Coordinate layout on 2l x 2m grid:
        #   Even rows (y=2i): X-ancilla at (2j, y), data at (2j+1, y)
        #   Odd rows (y=2i+1): data at (2j, y), Z-ancilla at (2j+1, y)
        for y in range(2 * m):
            if y % 2 == 0:  # Even rows: X-stabilizer ancillas + data
                for j in range(l):
                    self.add_qubit(2 * j, y, role='syndrome_x')
                    self.add_qubit(2 * j + 1, y, role='data')
            else:  # Odd rows: data + Z-stabilizer ancillas
                for j in range(l):
                    self.add_qubit(2 * j, y, role='data')
                    self.add_qubit(2 * j + 1, y, role='syndrome_z')

        # -------------------------------------------------------------------
        # Phase 2: Stabilizer Construction
        # -------------------------------------------------------------------
        # X-stabilizers: ancilla at (2j, 2i), 6 data qubits
        for i in range(m):
            for j in range(l):
                syn_coord = (2 * j, 2 * i)
                targets = {}
                # 3 data qubits from A (left-type: even x, odd y)
                for row in A:
                    dx, dy = row
                    data_coord = (2 * ((j + dx) % l), 2 * ((i + dy) % m) + 1)
                    targets[data_coord] = 'X'
                # 3 data qubits from B (right-type: odd x, even y)
                for row in B:
                    dx, dy = row
                    data_coord = (2 * ((j + dx) % l) + 1, 2 * ((i + dy) % m))
                    targets[data_coord] = 'X'
                self.create_stim_stabilizer(targets, syn_coord, 'X')

        # Z-stabilizers: ancilla at (2j+1, 2i+1), 6 data qubits
        for i in range(m):
            for j in range(l):
                syn_coord = (2 * j + 1, 2 * i + 1)
                targets = {}
                # 3 data qubits from -B (left-type: even x, odd y)
                for row in B:
                    dx, dy = row
                    data_coord = (2 * ((j - dx) % l), 2 * ((i - dy) % m) + 1)
                    targets[data_coord] = 'Z'
                # 3 data qubits from -A (right-type: odd x, even y)
                for row in A:
                    dx, dy = row
                    data_coord = (2 * ((j - dx) % l) + 1, 2 * ((i - dy) % m))
                    targets[data_coord] = 'Z'
                self.create_stim_stabilizer(targets, syn_coord, 'Z')

        # -------------------------------------------------------------------
        # Phase 3: Logical Operators (computed numerically)
        # -------------------------------------------------------------------
        self._build_logical_operators()

        # -------------------------------------------------------------------
        # Phase 4: Shift
        # -------------------------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def _build_polynomial_matrix(self, poly: List[List[int]]) -> np.ndarray:
        """
        Build an (l*m) x (l*m) binary matrix from a polynomial (list of monomials).
        Each monomial [a, b] contributes a cyclic shift: x^a * y^b.
        """
        l, m = self.l, self.m
        n2 = l * m
        I_l = np.eye(l, dtype=int)
        I_m = np.eye(m, dtype=int)

        result = np.zeros((n2, n2), dtype=int)
        for a, b in poly:
            # x^a: cyclic shift of I_l by a positions
            x_shift = np.roll(I_l, a, axis=1)
            # y^b: cyclic shift of I_m by b positions
            y_shift = np.roll(I_m, b, axis=1)
            # Kronecker product gives the combined shift matrix
            result = (result + np.kron(x_shift, y_shift)) % 2
        return result

    def _build_logical_operators(self):
        """Compute logical X and Z operators from parity check matrices."""
        l, m = self.l, self.m
        n2 = l * m  # half the data qubits
        n = 2 * n2  # total data qubits

        # Build check matrices
        A_mat = self._build_polynomial_matrix(self.A)
        B_mat = self._build_polynomial_matrix(self.B)

        Hx = np.hstack([A_mat, B_mat]) % 2  # shape: n2 x n
        Hz = np.hstack([B_mat.T, A_mat.T]) % 2  # shape: n2 x n

        # Compute k = n - 2 * rank(Hx) (for CSS codes with equal X/Z check counts)
        rank_Hx = _gf2_rank(Hx)
        k = n - 2 * rank_Hx
        self.num_logicals = k

        # Find logical Z: ker(Hx) / rowspace(Hz)
        ker_Hx = _gf2_kernel(Hx)
        lz_basis = _gf2_quotient_basis(ker_Hx, Hz)

        # Find logical X: ker(Hz) / rowspace(Hx)
        ker_Hz = _gf2_kernel(Hz)
        lx_basis = _gf2_quotient_basis(ker_Hz, Hx)

        # Build the data qubit index mapping: column index in H -> qubit index in patch
        # Columns 0..n2-1 correspond to "left-type" data qubits (even x, odd y)
        # Columns n2..n-1 correspond to "right-type" data qubits (odd x, even y)
        data_coords_sorted = sorted(self.data_indices)
        left_type_data = []  # (even x, odd y) data qubits
        right_type_data = []  # (odd x, even y) data qubits

        for idx in sorted(self.data_indices):
            coord = self.qubit_coords[idx]
            x, y = coord
            if int(round(x)) % 2 == 0 and int(round(y)) % 2 == 1:
                left_type_data.append(idx)
            elif int(round(x)) % 2 == 1 and int(round(y)) % 2 == 0:
                right_type_data.append(idx)

        # Sort to match the matrix column ordering (by (i,j) = (y//2, x//2))
        left_type_data.sort(key=lambda idx: (
            int(round(self.qubit_coords[idx][1])) // 2,
            int(round(self.qubit_coords[idx][0])) // 2
        ))
        right_type_data.sort(key=lambda idx: (
            int(round(self.qubit_coords[idx][1])) // 2,
            int(round(self.qubit_coords[idx][0])) // 2
        ))

        col_to_qubit_idx = left_type_data + right_type_data

        # Register logical Z operators
        for row in lz_basis:
            targets = {}
            for col in range(n):
                if row[col] == 1:
                    qubit_idx = col_to_qubit_idx[col]
                    targets[self.qubit_coords[qubit_idx]] = 'Z'
            if targets:
                self.create_stim_logical(targets, 'Z')

        # Register logical X operators
        for row in lx_basis:
            targets = {}
            for col in range(n):
                if row[col] == 1:
                    qubit_idx = col_to_qubit_idx[col]
                    targets[self.qubit_coords[qubit_idx]] = 'X'
            if targets:
                self.create_stim_logical(targets, 'X')

    def get_info(self):
        info = super().get_info()
        info.update({
            'l': self.l,
            'm': self.m,
            'A': self.A,
            'B': self.B,
            'code_distance': self.code_distance,
            'k': self.num_logicals,
            'n_data': len(self.data_indices),
            'num_x_syndromes': len(self.syndrome_indices_x),
            'num_z_syndromes': len(self.syndrome_indices_z),
            'data_coords': self.data_coords,
            'syndrome_coords_z': self.syndrome_coords_z,
            'syndrome_coords_x': self.syndrome_coords_x,
            'syndrome_coords': self.syndrome_coords,
            'stabilizers': self.stabilizers,
            'logical_ops': self.logical_ops,
            'index_map': self.index_map,
            'qubit_coords': self.qubit_coords,
            'num_logicals': self.num_logicals,
        })
        return info
