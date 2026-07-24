from __future__ import annotations

from math import gcd
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lightstim.ir.qec_patch import QECPatch

from .presets import KASAI_CODE_PRESETS, get_preset


Affine = Tuple[int, int]


def apply_affine(map_params: Affine, x: int, P: int) -> int:
    """Apply ``a*x + b mod P``."""
    a, b = map_params
    return (a * x + b) % P


def invert_affine(map_params: Affine, P: int) -> Affine:
    """Return the inverse affine map over ``Z_P``."""
    a, b = map_params
    a_inv = pow(a, -1, P)
    return a_inv, (-a_inv * b) % P


def apply_affine_inverse(map_params: Affine, x: int, P: int) -> int:
    """Apply the inverse of ``a*x + b mod P``."""
    return apply_affine(invert_affine(map_params, P), x, P)


def affine_commutes(f: Affine, g: Affine, P: int) -> bool:
    """Return whether two affine permutations commute over ``Z_P``."""
    a, b = f
    c, d = g
    return (d * (a - 1) - b * (c - 1)) % P == 0


def gf2_rank_from_supports(row_supports: Sequence[Sequence[int]], n_cols: int) -> int:
    """Compute GF(2) rank from sparse row supports using bit-packed rows."""
    n_words = (n_cols + 63) // 64
    basis: Dict[int, np.ndarray] = {}
    rank = 0

    for support in row_supports:
        row = np.zeros(n_words, dtype=np.uint64)
        for col in support:
            row[col // 64] ^= np.uint64(1) << np.uint64(col % 64)

        while True:
            nonzero_words = np.flatnonzero(row)
            if len(nonzero_words) == 0:
                break

            word = int(nonzero_words[-1])
            value = int(row[word])
            pivot = word * 64 + value.bit_length() - 1
            pivot_row = basis.get(pivot)
            if pivot_row is None:
                basis[pivot] = row.copy()
                rank += 1
                break
            row ^= pivot_row

    return rank


class KasaiCode(QECPatch):
    """
    Kasai affine-permutation CSS LDPC code.

    The standard paper instances use ``J=3``, ``L=12``, and six affine
    permutations ``f_i`` and ``g_i`` over ``Z_P``. The CSS matrices keep
    ``J`` active rows from the ``L/2`` block-row parent matrices:

    ``H_X[s, j] = F_{j-s}``, ``H_X[s, L/2+j] = G_{j-s}``

    ``H_Z[s, j] = G_{s-j}^T``, ``H_Z[s, L/2+j] = F_{s-j}^T``

    all block indices modulo ``L/2``.

    Scope: this class provides the code construction (stabilizers, CSS
    matrices, GF(2) ranks) and ``num_logicals``, but **no explicit logical
    operator representatives** — ``get_info()`` reports
    ``logical_ops_available = False``. Logical observables are handled at the
    experiment level (e.g. ``MemoryExperiment``), which is sufficient for
    memory experiments and syndrome-extraction studies. Protocols that need
    named logical operators (transversal gates, lattice surgery) are not
    supported for Kasai codes yet.

    Small instances: the published presets are large (n >= 1152) and their
    circuit builds take minutes. Affine maps with multiplier ``a=1`` are
    translations, which commute unconditionally, so e.g.
    ``KasaiCode(P=6, L=4, J=2, f=[(1, 0), (1, 1)], g=[(1, 0), (1, 2)])``
    yields a valid [[24, 2]] instance that builds instantly — useful for
    smoke-testing the pipeline (see ``tests/test_kasai_code.py``).
    """

    def _process_params(self):
        preset_name = self.params.get("preset")
        if preset_name is not None:
            preset = get_preset(preset_name)
            if preset is None:
                known = ", ".join(sorted(KASAI_CODE_PRESETS))
                raise ValueError(f"Unknown Kasai preset {preset_name!r}. Known presets: {known}")
            merged = {**preset, **self.params}
            self.params = merged

        if self.params.get("P") is None:
            raise ValueError("'P' block size must be provided.")
        self.P = int(self.params.get("P"))
        self.J = int(self.params.get("J", 3))
        self.L = int(self.params.get("L", 12))
        self.f = self._normalize_affines(self.params.get("f"), "f")
        self.g = self._normalize_affines(self.params.get("g"), "g")
        self.active_rows = self._normalize_active_rows(
            self.params.get("active_rows", list(range(self.J)))
        )
        self.shift = self.params.get("shift", (0, 0))
        self.expected_k = self.params.get("expected_k", None)
        self.expected_n = self.params.get("expected_n", None)
        self.compute_k = bool(self.params.get("compute_k", True))

        if self.P <= 0:
            raise ValueError("'P' must be positive.")
        if self.L <= 0 or self.L % 2 != 0:
            raise ValueError("'L' must be a positive even integer.")
        if self.J <= 0:
            raise ValueError("'J' must be positive.")

        self.m = self.L // 2
        if self.J > self.m:
            raise ValueError("'J' cannot exceed L/2 for the standard Kasai template.")
        if len(self.f) != self.m or len(self.g) != self.m:
            raise ValueError(f"'f' and 'g' must each contain L/2={self.m} affine maps.")

        for family_name, family in (("f", self.f), ("g", self.g)):
            for idx, (a, _) in enumerate(family):
                if gcd(a, self.P) != 1:
                    raise ValueError(
                        f"{family_name}[{idx}] has non-invertible multiplier {a} modulo P={self.P}."
                    )

        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two numbers.")

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "KasaiCode":
        """Construct a published Kasai-code instance by preset name."""
        return cls(preset=name, **overrides)

    @staticmethod
    def preset_names() -> List[str]:
        return sorted(KASAI_CODE_PRESETS)

    @staticmethod
    def _normalize_affines(raw_affines: Optional[Iterable[Sequence[int]]], name: str) -> List[Affine]:
        if raw_affines is None:
            raise ValueError(f"'{name}' affine maps must be provided.")
        affines: List[Affine] = []
        for item in raw_affines:
            if len(item) != 2:
                raise ValueError(f"Each '{name}' affine map must be a pair (a, b).")
            affines.append((int(item[0]), int(item[1])))
        return affines

    def _normalize_active_rows(self, raw_rows: Iterable[int]) -> List[int]:
        rows = [int(row) for row in raw_rows]
        if len(rows) != self.J:
            raise ValueError("'active_rows' must contain exactly J entries.")
        if len(set(rows)) != len(rows):
            raise ValueError("'active_rows' entries must be distinct.")
        if any(row < 0 or row >= self.L // 2 for row in rows):
            raise ValueError("'active_rows' entries must lie in [0, L/2).")
        return rows

    @property
    def n_data(self) -> int:
        return self.L * self.P

    @property
    def num_x_checks(self) -> int:
        return self.J * self.P

    @property
    def num_z_checks(self) -> int:
        return self.J * self.P

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        self._register_qubits()
        hx_supports, hz_supports = self.get_css_row_supports()
        self._register_stabilizers(hx_supports, hz_supports)

        self.rank_x: Optional[int] = None
        self.rank_z: Optional[int] = None
        if self.compute_k:
            self.rank_x = gf2_rank_from_supports(hx_supports, self.n_data)
            self.rank_z = gf2_rank_from_supports(hz_supports, self.n_data)
            self.num_logicals = self.n_data - self.rank_x - self.rank_z
        elif self.expected_k is not None:
            self.num_logicals = int(self.expected_k)

        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def _register_qubits(self):
        for block in range(self.L):
            for x in range(self.P):
                self.add_qubit(x, block, role="data", uid=self._data_uid(block, x))

        syn_base = self.n_data
        for row_slot in range(self.J):
            for x in range(self.P):
                self.add_qubit(
                    x,
                    self.L + row_slot,
                    role="syndrome_x",
                    uid=syn_base + row_slot * self.P + x,
                )

        z_base = self.n_data + self.num_x_checks
        for row_slot in range(self.J):
            for x in range(self.P):
                self.add_qubit(
                    x,
                    self.L + self.J + row_slot,
                    role="syndrome_z",
                    uid=z_base + row_slot * self.P + x,
                )

    def _register_stabilizers(
        self,
        hx_supports: Sequence[Sequence[int]],
        hz_supports: Sequence[Sequence[int]],
    ):
        for row_index, support in enumerate(hx_supports):
            row_slot, x = divmod(row_index, self.P)
            syn_coord = self.qubit_coords[self.n_data + row_slot * self.P + x]
            targets = {self.qubit_coords[col]: "X" for col in support}
            self.create_stim_stabilizer(targets, syn_coord, "X")

        z_base = self.n_data + self.num_x_checks
        for row_index, support in enumerate(hz_supports):
            row_slot, x = divmod(row_index, self.P)
            syn_coord = self.qubit_coords[z_base + row_slot * self.P + x]
            targets = {self.qubit_coords[col]: "Z" for col in support}
            self.create_stim_stabilizer(targets, syn_coord, "Z")

    def _data_uid(self, block: int, x: int) -> int:
        return block * self.P + x

    def get_css_row_supports(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Return sparse row supports for ``H_X`` and ``H_Z`` on data columns."""
        hx_supports: List[List[int]] = []
        hz_supports: List[List[int]] = []

        for active_row in self.active_rows:
            for x in range(self.P):
                hx_row: List[int] = []
                hz_row: List[int] = []

                for j in range(self.m):
                    hx_idx = (j - active_row) % self.m
                    hx_row.append(self._data_uid(j, apply_affine(self.f[hx_idx], x, self.P)))
                    hx_row.append(self._data_uid(self.m + j, apply_affine(self.g[hx_idx], x, self.P)))

                    hz_idx = (active_row - j) % self.m
                    hz_row.append(
                        self._data_uid(j, apply_affine_inverse(self.g[hz_idx], x, self.P))
                    )
                    hz_row.append(
                        self._data_uid(self.m + j, apply_affine_inverse(self.f[hz_idx], x, self.P))
                    )

                hx_supports.append(hx_row)
                hz_supports.append(hz_row)

        return hx_supports, hz_supports

    def get_css_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build dense binary ``H_X`` and ``H_Z`` matrices on data columns."""
        hx_supports, hz_supports = self.get_css_row_supports()
        hx = np.zeros((len(hx_supports), self.n_data), dtype=np.uint8)
        hz = np.zeros((len(hz_supports), self.n_data), dtype=np.uint8)
        for row_index, support in enumerate(hx_supports):
            hx[row_index, support] ^= 1
        for row_index, support in enumerate(hz_supports):
            hz[row_index, support] ^= 1
        return hx, hz

    def compute_rank_pair(self) -> Tuple[int, int]:
        """Return ``rank(H_X), rank(H_Z)`` over GF(2)."""
        hx_supports, hz_supports = self.get_css_row_supports()
        return (
            gf2_rank_from_supports(hx_supports, self.n_data),
            gf2_rank_from_supports(hz_supports, self.n_data),
        )

    def required_commuting_pairs(self) -> List[Tuple[int, int]]:
        """Pairs ``(i, j)`` whose commutation is sufficient for active CSS orthogonality."""
        deltas = {
            (right - left) % self.m
            for left in self.active_rows
            for right in self.active_rows
        }
        pairs = {
            (u, (delta - u) % self.m)
            for delta in deltas
            for u in range(self.m)
        }
        return sorted(pairs)

    def noncommuting_pairs(self) -> List[Tuple[int, int]]:
        """Return all ``(i, j)`` for which ``F_i`` and ``G_j`` do not commute."""
        pairs = []
        for i, f_i in enumerate(self.f):
            for j, g_j in enumerate(self.g):
                if not affine_commutes(f_i, g_j, self.P):
                    pairs.append((i, j))
        return pairs

    def validate_required_commutativity(self) -> bool:
        """Check the sufficient active-orthogonality commutation condition."""
        return all(
            affine_commutes(self.f[i], self.g[j], self.P)
            for i, j in self.required_commuting_pairs()
        )

    def get_info(self):
        info = super().get_info()
        info.update({
            "P": self.P,
            "J": self.J,
            "L": self.L,
            "active_rows": self.active_rows,
            "f": self.f,
            "g": self.g,
            "k": self.num_logicals,
            "n_data": self.n_data,
            "rank_x": self.rank_x,
            "rank_z": self.rank_z,
            "num_x_syndromes": len(self.syndrome_indices_x),
            "num_z_syndromes": len(self.syndrome_indices_z),
            "data_coords": self.data_coords,
            "syndrome_coords_z": self.syndrome_coords_z,
            "syndrome_coords_x": self.syndrome_coords_x,
            "syndrome_coords": self.syndrome_coords,
            "stabilizers": self.stabilizers,
            "logical_ops": self.logical_ops,
            "logical_ops_available": False,
            "index_map": self.index_map,
            "qubit_coords": self.qubit_coords,
            "num_logicals": self.num_logicals,
        })
        return info
