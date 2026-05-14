"""
PQRMPatch: Punctured Quantum Reed-Muller code as QECPatch.

Integrates layout, Z stabilizers, X stabilizers (for post-select), and logical Z/X.
"""

from typing import List, Tuple, Set, Dict, Optional, Any
from itertools import product
import numpy as np

from lightstim.ir.qec_patch import QECPatch


# -----------------------------------------------------------------------------
# Utils (inlined from pqrm_utils)
# -----------------------------------------------------------------------------

def _bin_wt(i: int) -> int:
    """Binary weight (number of 1s) of an integer."""
    return bin(i)[2:].count('1')


def _int2bin(i: int, m: int) -> List[int]:
    """Convert integer i to binary list of length m."""
    return [int(c) for c in bin(i)[2:].rjust(m, '0')]


def _RM_generator_matrix(
    r: int, m: int, variation: str
) -> Tuple[np.ndarray, List[tuple]]:
    """Generator matrix for Reed-Muller code. variation: 'None', 'punctured', 'shortened'."""
    n = 2 ** m
    monomials = []
    for deg in range(r + 1):
        for bits in product([0, 1], repeat=m):
            if sum(bits) == deg:
                monomials.append(bits)

    G = []
    for mono in monomials:
        row = []
        for i in range(n):
            x = [int(b) for b in bin(i)[2:].rjust(m, '0')]
            val = 1
            for a, xi in zip(mono, x):
                if a == 1:
                    val &= xi
            row.append(val)
        G.append(row)

    G = np.array(G, dtype=int)
    if variation == "None":
        G_final = G
    elif variation == "punctured":
        G_final = G[:, 1:]
    elif variation == "shortened":
        G_final = G[:, 1:][1:, :]
    else:
        raise ValueError(f"Unknown variation: {variation}")
    return G_final


# Public aliases for external use
def bin_wt(i: int) -> int:
    return _bin_wt(i)


def int2bin(i: int, m: int) -> List[int]:
    return _int2bin(i, m)


def RM_generator_matrix(r: int, m: int, variation: str) -> Tuple[np.ndarray, List]:
    """Public API. Returns (matrix, monomials)."""
    G = _RM_generator_matrix(r, m, variation)
    return G, []  # monomials omitted for shortened


# -----------------------------------------------------------------------------
# Boundary Z-stabilizer config: syn_coord -> region_key (R1/R2/R3/B1..B4)
# Data deltas (which data qubits each boundary stab connects to) come from
# pqrm_se_config.get_boundary_data_deltas(), derived from TICK_DELTA_BOUNDARY.
# -----------------------------------------------------------------------------

# (x_range, y_range) -> list of (syn_coord, region_key)
_SYNDROME_BOUND_CONFIG = {
    (4, 4): [((3, 7), "B1"), ((7, 3), "R1")],
    (4, 8): [
        ((3, 15), "B1"), ((7, 3), "R1"), ((7, 5), "R2"), ((7, 9), "R3"), ((7, 11), "R1")
    ],
    (8, 8): [
        ((3, 15), "B1"), ((5, 15), "B2"), ((9, 15), "B3"), ((11, 15), "B1"),
        ((15, 3), "R1"), ((15, 5), "R2"), ((15, 9), "R3"), ((15, 11), "R1"),
    ],
}

# Supported PQRM parameters only: (rx, rz, m)
LOG_PQRM_LEN_DICT = {
    (1, 2, 4): 3,
    (1, 3, 5): 7,
    (1, 4, 6): 7,
}

PQRM_SUPPORTED_PARAMS = frozenset(LOG_PQRM_LEN_DICT.keys())


# -----------------------------------------------------------------------------
# PQRMPatch
# -----------------------------------------------------------------------------

class PQRMPatch(QECPatch):
    """
    Punctured Quantum Reed-Muller code as QECPatch.

    Parameters (via **kwargs):
        rx, rz: PQRM parameters
        m: dimension (code size 2^m - 1 if punctured)
        punctured: remove qubit at (0,0)
        shift: (dx, dy) offset
    """

    def _process_params(self):
        self.rx = self.params.get("rx")
        self.rz = self.params.get("rz")
        self.m = self.params.get("m")
        self.punctured = self.params.get("punctured", True)
        self.shift = self.params.get("shift", (0, 0))

        if self.rx is None or self.rz is None or self.m is None:
            raise ValueError("rx, rz, m must be provided.")
        key = (self.rx, self.rz, self.m)
        if key not in PQRM_SUPPORTED_PARAMS:
            raise ValueError(
                f"Unsupported PQRM parameters {key}. "
                f"Supported: {sorted(PQRM_SUPPORTED_PARAMS)}"
            )

        self.N = 2 ** self.m
        self.x_range = 2 ** (self.m // 2)
        self.y_range = 2 ** ((self.m + 1) // 2)

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        """X-type syndrome coords (PQRM has none - X stabs use post-select)."""
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        m = self.m
        punctured = self.punctured
        N = self.N

        # --- Phase 1: Hypercube data coords ---
        data_coords_raw = [(0, 0)]
        for r in range(m):
            if r % 2 == 0:
                coords_shifted = [(x, y + 2 ** (r // 2)) for (x, y) in data_coords_raw]
            else:
                coords_shifted = [(x + 2 ** (r // 2), y) for (x, y) in data_coords_raw]
            data_coords_raw += coords_shifted

        data_coords_scaled = [(2 * x, 2 * y) for (x, y) in data_coords_raw]

        # Index convention: coord -> hypercube index (0..N-1), used only during build
        coord_to_hypercube = {c: i for i, c in enumerate(data_coords_scaled)}

        if punctured:
            if (0, 0) in data_coords_scaled:
                data_coords_scaled.remove((0, 0))
                coord_to_hypercube.pop((0, 0), None)

        # --- Phase 2: Register qubits (preserve hypercube uid for data) ---
        # add_qubit populates self.qubit_coords (index->coord) and self.index_map (coord->index)
        for coord in data_coords_scaled:
            uid = coord_to_hypercube.get(coord)
            if uid is not None:
                self.add_qubit(coord[0], coord[1], role='data', uid=uid)

        # --- Phase 3: Z stabilizer ancillas ---
        bulk_syn = [
            (2 * x + 1, 2 * y + 1)
            for x in range(self.x_range - 1)
            for y in range(self.y_range - 1)
        ]
        if (1, 1) in bulk_syn:
            bulk_syn.remove((1, 1))

        bound_config = _SYNDROME_BOUND_CONFIG.get((self.x_range, self.y_range), [])
        bound_syn_coords = [s[0] for s in bound_config]

        all_z_syn = bulk_syn + bound_syn_coords
        all_z_syn = sorted(all_z_syn, key=lambda c: (c[1], c[0]))

        next_uid = N
        for coord in all_z_syn:
            self.add_qubit(coord[0], coord[1], role='syndrome_z', uid=next_uid)
            next_uid += 1

        # --- Phase 4: Z stabilizers ---
        # Bulk: syn at (2x+1, 2y+1), data at 4 corners
        for x in range(self.x_range - 1):
            for y in range(self.y_range - 1):
                syn = (2 * x + 1, 2 * y + 1)
                if syn == (1, 1):
                    continue
                targets = {
                    (2 * x, 2 * y): 'Z',
                    (2 * x + 2, 2 * y): 'Z',
                    (2 * x, 2 * y + 2): 'Z',
                    (2 * x + 2, 2 * y + 2): 'Z',
                }
                self.create_stim_stabilizer(targets, syn, 'Z')

        # Boundary: data deltas from pqrm_se_config (6-tick schedule source of truth)
        from .pqrm_se_config import get_boundary_data_deltas
        for (syn_coord, region_key) in bound_config:
            deltas = get_boundary_data_deltas(region_key)
            targets = {}
            for dx, dy in deltas:
                dc = self.snap_coord((syn_coord[0] + dx, syn_coord[1] + dy))
                if dc in self.index_map:
                    targets[dc] = 'Z'
            if targets:
                self.create_stim_stabilizer(targets, syn_coord, 'Z')

        # --- Phase 5: X stabilizers (from RM shortened, no syndrome ancilla) ---
        G_short = _RM_generator_matrix(self.rx, m, "shortened")
        # G_short: rows = X stabs, cols = data indices 1..N-1 (punctured)
        for row in G_short:
            targets = {}
            for col_idx in range(len(row)):
                if row[col_idx] != 0:
                    data_idx = col_idx + 1
                    if data_idx in self.qubit_coords:
                        coord = self.snap_coord(self.qubit_coords[data_idx])
                        if coord in self.index_map:
                            targets[coord] = 'X'
            if targets:
                stab_rec = {
                    "pauli": {self.index_map[c]: 'X' for c in targets},
                    "type": "X",
                    "data_indices": [self.index_map[c] for c in targets],
                    "syn_coord": None,
                    "syn_idx": None,
                }
                self.stabilizers.append(stab_rec)

        # --- Phase 6: Logical Z ---
        key = (self.rx, self.rz, m)
        if key not in LOG_PQRM_LEN_DICT:
            raise ValueError(
                f"Unsupported PQRM parameters {key}. "
                f"Supported: {sorted(LOG_PQRM_LEN_DICT.keys())}"
            )
        log_len = LOG_PQRM_LEN_DICT[key]
        lz_coords = [(0, 2 * (y + 1)) for y in range(log_len)]
        lz_targets = {c: 'Z' for c in lz_coords if c in self.index_map}
        if lz_targets:
            self.create_stim_logical(lz_targets, 'Z')

        lx_targets = {c: 'X' for c in self.data_coords if c in self.index_map}
        if lx_targets:
            self.create_stim_logical(lx_targets, 'X')

        self.num_logicals = 1

        # --- Phase 7: Shift ---
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def get_qubit_indices(self) -> List[int]:
        """All qubit indices (data + syndrome)."""
        return list(self.qubit_coords.keys())

    def get_info(self) -> Dict[str, Any]:
        """Match Surface Code format (unrotated/rotated): code-specific params + full structure."""
        info = super().get_info()
        info.update({
            # Code-specific
            'rx': self.rx,
            'rz': self.rz,
            'm': self.m,
            'punctured': self.punctured,
            'x_range': self.x_range,
            'y_range': self.y_range,
            # Structure (match Surface Code)
            'num_data_qubits': len(self.data_coords),
            'num_x_syndromes': len(self.syndrome_coords_x),
            'num_z_syndromes': len(self.syndrome_coords_z),
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
