"""4D Geometric Code (loop-only toric code on rotated 4D lattice).

A CSS code defined on the cellulation of a 4-torus T^4_Λ:
  - Data qubits = 2-cells (faces): 6 * Det(L)
  - X-stabilizers = 1-cells (edges): 4 * Det(L), each weight 6
  - Z-stabilizers = 3-cells (cubes): 4 * Det(L), each weight 6
  - Code parameters: [[6*Det, 6, d]]

Reference: arXiv:2506.15130
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from lightstim.ir.qec_patch import QECPatch
from lightstim.utils.linear_algebra import row_echelon
from .lattice import Lattice4D


class FourDGeoCode(QECPatch):
    """4D Geometric Code on a rotated lattice.

    Parameters (passed via **kwargs):
    ---------------------------------
    L : list[list[int]]
        4x4 Hermite Normal Form matrix defining the lattice.
    name : str, optional
        Human-readable name (default: "4DGeo").
    d : int, optional
        Code distance (metadata only).

    Examples:
    ---------
    # Det3 [[18,6,3]]
    >>> code = FourDGeoCode(L=[[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,3]])

    # Hadamard [[96,6,8]]
    >>> code = FourDGeoCode(L=[[1,1,1,1],[0,2,0,2],[0,0,2,2],[0,0,0,4]])
    """

    def _process_params(self):
        self.L_matrix = self.params.get('L')
        self.code_distance = self.params.get('d', None)

        if self.L_matrix is None:
            raise ValueError("'L' (4x4 HNF matrix) must be provided.")
        if len(self.L_matrix) != 4 or any(len(row) != 4 for row in self.L_matrix):
            raise ValueError("L must be a 4x4 matrix.")

        # Validate upper triangular
        for i in range(4):
            for j in range(i):
                if self.L_matrix[i][j] != 0:
                    raise ValueError(f"L must be upper triangular: L[{i}][{j}] = {self.L_matrix[i][j]} != 0")
            if self.L_matrix[i][i] <= 0:
                raise ValueError(f"Diagonal entries must be positive: L[{i}][{i}] = {self.L_matrix[i][i]}")
        # Validate off-diagonal < diagonal
        for i in range(4):
            for j in range(i + 1, 4):
                if not (0 <= self.L_matrix[i][j] < self.L_matrix[j][j]):
                    raise ValueError(
                        f"HNF requires 0 <= L[{i}][{j}] < L[{j}][{j}], "
                        f"got L[{i}][{j}]={self.L_matrix[i][j]}, L[{j}][{j}]={self.L_matrix[j][j]}"
                    )

    def build(self):
        lattice = Lattice4D(self.L_matrix)
        self._lattice = lattice
        D = lattice.det
        points = lattice.enumerate_points()

        # --- Phase 1: Register qubits with 2D coordinates ---
        # 14 cell types: 0-5 faces (data), 6-9 edges (X-anc), 10-13 cubes (Z-anc)
        # Layout: arrange types in a grid to keep the overall shape near-square.
        #   num_type_cols ≈ sqrt(28 / (D+1)) balances x ≈ y extent.
        num_type_cols = max(1, round((28 / (D + 1)) ** 0.5))
        col_gap = D + 1  # horizontal spacing between type blocks

        def _layout(global_type, pid):
            tc = global_type % num_type_cols
            tr = global_type // num_type_cols
            return float(tc * col_gap + pid), float(tr * 2)

        self._face_qubit_map = {}   # (face_type_idx, point) → qubit_index
        self._edge_qubit_map = {}   # (edge_dir, point) → qubit_index
        self._cube_qubit_map = {}   # (cube_miss, point) → qubit_index

        # Data qubits: 6 face types × D points
        for ft_idx in range(6):
            for p in points:
                pid = lattice.point_to_idx(p)
                x, y = _layout(ft_idx, pid)
                qidx = self.add_qubit(x, y, role='data')
                self._face_qubit_map[(ft_idx, p)] = qidx

        # X-ancilla qubits: 4 edge directions × D points
        for edir in range(4):
            for p in points:
                pid = lattice.point_to_idx(p)
                x, y = _layout(6 + edir, pid)
                qidx = self.add_qubit(x, y, role='syndrome_x')
                self._edge_qubit_map[(edir, p)] = qidx

        # Z-ancilla qubits: 4 cube missing directions × D points
        for cmiss in range(4):
            for p in points:
                pid = lattice.point_to_idx(p)
                x, y = _layout(10 + cmiss, pid)
                qidx = self.add_qubit(x, y, role='syndrome_z')
                self._cube_qubit_map[(cmiss, p)] = qidx

        # --- Phase 2: Build stabilizers ---
        # X-stabilizers from edge coboundary
        for edir in range(4):
            for p in points:
                support = lattice.x_stabilizer_support(edir, p)
                target_dict = {}
                for (ft_idx, fp) in support:
                    data_qidx = self._face_qubit_map[(ft_idx, fp)]
                    coord = self.qubit_coords[data_qidx]
                    target_dict[coord] = 'X'
                syn_qidx = self._edge_qubit_map[(edir, p)]
                syn_coord = self.qubit_coords[syn_qidx]
                self.create_stim_stabilizer(target_dict, syn_coord, type='X')

        # Z-stabilizers from cube boundary
        for cmiss in range(4):
            for p in points:
                support = lattice.z_stabilizer_support(cmiss, p)
                target_dict = {}
                for (ft_idx, fp) in support:
                    data_qidx = self._face_qubit_map[(ft_idx, fp)]
                    coord = self.qubit_coords[data_qidx]
                    target_dict[coord] = 'Z'
                syn_qidx = self._cube_qubit_map[(cmiss, p)]
                syn_coord = self.qubit_coords[syn_qidx]
                self.create_stim_stabilizer(target_dict, syn_coord, type='Z')

        # --- Phase 3: Logical operators (numerical) ---
        self._compute_logicals_numerically()

    def _compute_logicals_numerically(self):
        """Compute logical X and Z operators from parity check matrices.

        Uses RREF kernel computation following the BB code pattern.
        X-logicals from ker(Hz) / rowspace(Hx)
        Z-logicals from ker(Hx) / rowspace(Hz)
        """
        n = len(self.data_indices)  # 6 * Det
        n_x_stab = len(self.syndrome_indices_x)  # 4 * Det
        n_z_stab = len(self.syndrome_indices_z)  # 4 * Det

        # Build ordered list of data qubit indices
        data_idx_list = sorted(self.data_indices)
        data_idx_to_col = {idx: col for col, idx in enumerate(data_idx_list)}

        # Build Hx and Hz from stabilizer records
        Hx_rows = []
        Hz_rows = []
        for stab in self.stabilizers:
            row = np.zeros(n, dtype=int)
            for qidx, pauli_type in stab['pauli'].items():
                if qidx in data_idx_to_col:
                    row[data_idx_to_col[qidx]] = 1
            if stab['type'] == 'X':
                Hx_rows.append(row)
            else:
                Hz_rows.append(row)

        Hx = np.array(Hx_rows, dtype=int) if Hx_rows else np.zeros((0, n), dtype=int)
        Hz = np.array(Hz_rows, dtype=int) if Hz_rows else np.zeros((0, n), dtype=int)

        # Kernel computation via row_echelon on transpose
        def _kernel_and_basis(M):
            if M.shape[0] == 0:
                return np.eye(M.shape[1], dtype=np.uint8), np.zeros((0, M.shape[1]), dtype=np.uint8)
            Mt = M.T.astype(bool)
            _, rank, transform, pivot_cols = row_echelon(Mt)
            ker = transform[rank:, :].astype(np.uint8)
            basis = M[np.array(pivot_cols[:rank])].astype(np.uint8) if rank > 0 else np.zeros((0, M.shape[1]), dtype=np.uint8)
            return ker, basis

        def _quotient_logicals(ker_rows, im_basis):
            if ker_rows.shape[0] == 0:
                return np.zeros((0, ker_rows.shape[1]), dtype=np.uint8)
            if im_basis.shape[0] == 0:
                return ker_rows
            log_stack = np.vstack([im_basis, ker_rows]) % 2
            _, _, _, pivots = row_echelon(log_stack.T)
            n_im = im_basis.shape[0]
            log_op_indices = [i for i in range(n_im, log_stack.shape[0]) if i in pivots]
            if log_op_indices:
                return log_stack[log_op_indices]
            return np.zeros((0, ker_rows.shape[1]), dtype=np.uint8)

        hx_perp, hx_basis = _kernel_and_basis(Hx)
        hz_perp, hz_basis = _kernel_and_basis(Hz)

        k = n - hx_basis.shape[0] - hz_basis.shape[0]
        self.num_logicals = k

        # Z-logicals: in ker(Hx) but not in rowspace(Hz)
        lz_basis = _quotient_logicals(hx_perp, hz_basis)
        # X-logicals: in ker(Hz) but not in rowspace(Hx)
        lx_basis = _quotient_logicals(hz_perp, hx_basis)

        # Convert to logical operator records
        for row in lz_basis:
            targets = {}
            for col in range(n):
                if row[col] == 1:
                    qidx = data_idx_list[col]
                    targets[self.qubit_coords[qidx]] = 'Z'
            if targets:
                self.create_stim_logical(targets, 'Z')

        for row in lx_basis:
            targets = {}
            for col in range(n):
                if row[col] == 1:
                    qidx = data_idx_list[col]
                    targets[self.qubit_coords[qidx]] = 'X'
            if targets:
                self.create_stim_logical(targets, 'X')

    def get_info(self):
        info = super().get_info()
        info.update({
            'L': self.L_matrix,
            'det': self._lattice.det if hasattr(self, '_lattice') else None,
            'code_distance': self.code_distance,
            'k': self.num_logicals,
            'n_data': len(self.data_indices),
            'num_x_syndromes': len(self.syndrome_indices_x),
            'num_z_syndromes': len(self.syndrome_indices_z),
        })
        return info
