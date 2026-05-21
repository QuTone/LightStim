from typing import Tuple, Dict, List, Optional, Set
import numpy as np
import stim
from lightstim.ir.qec_patch import QECPatch
from lightstim.utils.linear_algebra import row_echelon

# ---------------------------------------------------------------------------
# Polynomial helpers for scalable logical construction
# ---------------------------------------------------------------------------

def _transpose_exponents(monomials: List[List[int]], l: int, m: int) -> List[List[int]]:
    """Invert monomial exponents: (i,j) -> (-i mod l, -j mod m)."""
    return [[(-i) % l, (-j) % m] for i, j in monomials]


def _multiply_monomial_by_polynomial(
    monomial: List[int],
    polynomial: List[List[int]],
    l: int,
    m: int,
    qubit_type: str,
) -> List[Tuple[int, int]]:
    """
    Map monomial * polynomial to 2D data qubit coordinates.
    monomial: [a, b] as x^a * y^b
    polynomial: list of [p, q] monomials
    qubit_type: 'L' (left-type: even x, odd y) or 'R' (right-type: odd x, even y)
    Returns list of (x, y) integer coordinates.
    """
    a, b = monomial
    if qubit_type == 'L':
        return [
            (2 * ((a + p) % l), 2 * ((b + q) % m) + 1)
            for p, q in polynomial
        ]
    elif qubit_type == 'R':
        return [
            (2 * ((a + p) % l) + 1, 2 * ((b + q) % m))
            for p, q in polynomial
        ]
    else:
        raise ValueError("qubit_type must be 'L' or 'R'")


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
        # Optional: polynomial params for scalable logical construction
        self.f = self.params.get('f')
        self.g = self.params.get('g')
        self.h = self.params.get('h')
        self.alpha = self.params.get('alpha')
        self.beta = self.params.get('beta')

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
        if (self.f is not None or self.g is not None or self.h is not None) and (
            self.f is None or self.g is None or self.h is None or self.alpha is None or self.beta is None
        ):
            raise ValueError("If using polynomial logicals, all of f, g, h, alpha, beta must be provided.")

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
        """Dispatch to polynomial or numerical logical construction."""
        # 1. Explicit polynomial params provided
        if self.f is not None and self.g is not None and self.h is not None:
            self._build_logical_operators_polynomial()
            return

        # 2. Check preset lookup
        from .logical_presets import get_preset
        preset = get_preset(self.l, self.m, self.A, self.B)
        if preset is not None:
            self.f, self.g, self.h = preset['f'], preset['g'], preset['h']
            self.alpha, self.beta = preset['alpha'], preset['beta']
            self._build_logical_operators_polynomial()
            return

        # 3. Fallback to numerical (feasible up to l*m ~200, ~1-5s for l*m=144)
        NUMERICAL_THRESHOLD = 200
        if self.l * self.m <= NUMERICAL_THRESHOLD:
            self._build_logical_operators_numerical()
            return

        # 4. Too large, need preset or explicit params
        raise ValueError(
            f"Logical operators for BB code with l={self.l}, m={self.m} require precomputed "
            "(f,g,h,alpha,beta). Either provide them as params, use a smaller code "
            "(l*m <= 200 for numerical fallback), or add this (l,m,A,B) to logical_presets."
        )

    def _build_logical_operators_polynomial(self):
        """Build logical X and Z from polynomial params (f,g,h,alpha,beta). O(k) in code size."""
        l, m = self.l, self.m
        f, g, h = self.f, self.g, self.h
        alpha, beta = self.alpha, self.beta

        self.num_logicals = 2 * len(alpha)  # k/2 pairs, each contributes 2 logical qubits

        for i in range(len(alpha)):
            # logical_X[2i]: alpha[i] * f on L-type only
            coords_x0 = _multiply_monomial_by_polynomial(alpha[i], f, l, m, 'L')
            targets = {coord: 'X' for coord in coords_x0}
            if targets:
                self.create_stim_logical(targets, 'X')

            # logical_Z[2i]: beta[i] * h^T on L + beta[i] * g^T on R
            hT = _transpose_exponents(h, l, m)
            gT = _transpose_exponents(g, l, m)
            coords_z0 = _multiply_monomial_by_polynomial(beta[i], hT, l, m, 'L')
            coords_z0 += _multiply_monomial_by_polynomial(beta[i], gT, l, m, 'R')
            targets = {coord: 'Z' for coord in coords_z0}
            if targets:
                self.create_stim_logical(targets, 'Z')

            # logical_Z[2i+1]: beta[i] * f^T on R-type only
            fT = _transpose_exponents(f, l, m)
            coords_z1 = _multiply_monomial_by_polynomial(beta[i], fT, l, m, 'R')
            targets = {coord: 'Z' for coord in coords_z1}
            if targets:
                self.create_stim_logical(targets, 'Z')

            # logical_X[2i+1]: alpha[i] * g on L + alpha[i] * h on R
            coords_x1 = _multiply_monomial_by_polynomial(alpha[i], g, l, m, 'L')
            coords_x1 += _multiply_monomial_by_polynomial(alpha[i], h, l, m, 'R')
            targets = {coord: 'X' for coord in coords_x1}
            if targets:
                self.create_stim_logical(targets, 'X')

    def _build_logical_operators_numerical(self):
        """Compute Z/X logical operators via GF(2) Gaussian elimination.

        Z-logicals: ker(Hx) / row(Hz).  X-logicals: ker(Hz) / row(Hx).
        """
        l, m = self.l, self.m
        n2, n = l * m, 2 * l * m

        A_mat = self._build_polynomial_matrix(self.A)
        B_mat = self._build_polynomial_matrix(self.B)

        # CSS parity check matrices (n2 × n).
        # np.kron(x_shift, y_shift) assigns column index j the qubit at
        # (x_idx = j//m, y_idx = j%m); columns j >= n2 are right-type.
        Hx = np.hstack([A_mat, B_mat]) % 2
        Hz = np.hstack([B_mat.T, A_mat.T]) % 2

        def _null_space(M):
            """Basis for ker(M) over GF(2), returned as rows."""
            _, rank, transform, _ = row_echelon(M.T.astype(np.uint8))
            return np.array(transform[rank:]) % 2

        def _row_basis(M):
            """Independent rows of M over GF(2)."""
            rref, rank, _, _ = row_echelon(M.astype(np.uint8))
            return np.array(rref[:rank]) % 2

        def _quotient(ker_vecs, im_vecs, k):
            """k representatives of ker_vecs / im_vecs over GF(2)."""
            stack = np.vstack([im_vecs, ker_vecs]) % 2
            _, _, _, pivots = row_echelon(stack.T.astype(np.uint8))
            n_im = len(im_vecs)
            reps = [i for i in range(n_im, len(stack)) if i in pivots]
            assert len(reps) == k, f"Expected {k} logicals, found {len(reps)}"
            return stack[reps]

        hz_basis = _row_basis(Hz)
        hx_basis = _row_basis(Hx)
        k = n - len(hx_basis) - len(hz_basis)
        self.num_logicals = k

        lz_basis = _quotient(_null_space(Hx), hz_basis, k)
        lx_basis = _quotient(_null_space(Hz), hx_basis, k)

        # Map polynomial column index → patch qubit index.
        # Kronecker block ordering: col j → qubit at (x_idx = j//m, y_idx = j%m).
        # Left-type qubits have even x (x_idx = x//2), right-type have odd x.
        def _poly_sorted(is_left_type: bool) -> list:
            qubits = [
                idx for idx in self.data_indices
                if (int(round(self.qubit_coords[idx][0])) % 2 == 0) == is_left_type
            ]
            return sorted(qubits, key=lambda idx:
                int(round(self.qubit_coords[idx][0])) // 2 * m
                + int(round(self.qubit_coords[idx][1])) // 2
            )

        col_to_qubit = _poly_sorted(True) + _poly_sorted(False)

        for row in lz_basis:
            targets = {self.qubit_coords[col_to_qubit[j]]: 'Z' for j in np.where(row)[0]}
            if targets:
                self.create_stim_logical(targets, 'Z')

        for row in lx_basis:
            targets = {self.qubit_coords[col_to_qubit[j]]: 'X' for j in np.where(row)[0]}
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
