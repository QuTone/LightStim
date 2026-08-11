"""GALA codes — Group-Action Lifts with Active orthogonality (arXiv:2608.07431).

GALA generalises the Kasai construction (arXiv:2604.16209): the block-circulant
parent matrices are identical in shape, but each lift entry is an element of the
group ring ``F_2[G]`` over ``G = H_k x C_m`` rather than a single affine
permutation of ``Z_P``. The small non-abelian ``H_k`` decides the orthogonality
pattern; the abelian ``C_m`` supplies shift symmetries (hence AOD schedules) and
is unconstrained by orthogonality.

With ``m = |C_m|`` lift points per ``H_k`` point, the code has ``n = L * k * m``
data qubits:

    [H_X]_{i,j} = F_{j-i},      [H_X]_{i,j+L/2} = G_{j-i}
    [H_Z]_{i,j} = G^T_{i-j},    [H_Z]_{i,j+L/2} = F^T_{i-j}

keeping the first ``J`` block rows (the *active* rows). Orthogonality holds
because the anticommutator sum ``Psi_r = sum_u [F_u, G_{r-u}]`` vanishes for
``r < J``; the remaining ``L/2 - J`` *latent* rows need not commute, and those
degrees of freedom are what become above-weight logicals.

A lift entry may be a sum of monomials (the *polynomial* family of
Definition 18), in which case the stabilizer weight is the total number of
terms rather than ``L``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.kasai_code.code_patch import gf2_rank_from_supports

from .group import LiftAlphabet, Monomial, commutes, s3, s4
from .presets import GALA_CODE_PRESETS, get_preset


def _resolve_h(label: str, degree: int):
    """Resolve an H_k element label to a permutation of ``[degree]``."""
    if degree == 1:
        if label != "e":
            raise ValueError(
                f"Abelian lift (degree 1) cannot use H_k element {label!r}."
            )
        return (0,)
    if degree == 3:
        return s3(label)
    if degree == 4:
        return s4(label)
    raise ValueError(
        f"Unsupported H_k degree {degree}; expected 1 (abelian), 3 (S_3) or 4 (S_4)."
    )


class GalaCode(QECPatch):
    """GALA two-block CSS code lifted over ``H_k x C_m``.

    Parameters (or a ``preset`` name):
        L:       number of data blocks (even).
        J:       number of active block rows kept, ``J <= L/2``.
        degree:  degree ``k`` of the ``H_k`` action (1 for a purely abelian lift).
        cyclic:  orders of the cyclic factors of ``C_m``.
        F, G:    the lifts, ``L/2`` entries each; every entry is a list of
                 monomials ``(h_label, shifts)`` summed over ``F_2[G]``.

    Scope: this provides the code construction (stabilizers, CSS matrices, GF(2)
    ranks) and ``num_logicals``, but no explicit logical-operator
    representatives — ``logical_ops_available`` is ``False``, as for Kasai
    codes. Syndrome extraction uses the generic CSS coloration block.
    """

    def _process_params(self):
        preset_name = self.params.get("preset")
        if preset_name is not None:
            preset = get_preset(preset_name)
            if preset is None:
                known = ", ".join(sorted(GALA_CODE_PRESETS))
                raise ValueError(
                    f"Unknown GALA preset {preset_name!r}. Known presets: {known}"
                )
            self.params = {**preset, **self.params}

        for required in ("L", "J", "degree", "cyclic", "F", "G"):
            if self.params.get(required) is None:
                raise ValueError(f"GALA code requires parameter {required!r}.")

        self.L = int(self.params["L"])
        self.J = int(self.params["J"])
        self.degree = int(self.params["degree"])
        self.cyclic: Tuple[int, ...] = tuple(int(c) for c in self.params["cyclic"])
        self.shift = self.params.get("shift", (0, 0))
        self.expected_n = self.params.get("expected_n")
        self.expected_k = self.params.get("expected_k")
        self.expected_d = self.params.get("expected_d")
        self.compute_k = bool(self.params.get("compute_k", True))

        if self.L <= 0 or self.L % 2 != 0:
            raise ValueError("'L' must be a positive even integer.")
        self.m_blocks = self.L // 2
        if not 1 <= self.J <= self.m_blocks:
            raise ValueError(f"'J' must satisfy 1 <= J <= L/2 = {self.m_blocks}.")

        self.alphabet = LiftAlphabet(self.degree, self.cyclic)
        self.f = self._normalize_lift(self.params["F"], "F")
        self.g = self._normalize_lift(self.params["G"], "G")

        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two numbers.")

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "GalaCode":
        """Construct a published GALA instance by preset name."""
        return cls(preset=name, **overrides)

    @staticmethod
    def preset_names() -> List[str]:
        return sorted(GALA_CODE_PRESETS)

    def _normalize_lift(self, raw: Sequence[Any], name: str) -> List[List[Monomial]]:
        if len(raw) != self.m_blocks:
            raise ValueError(
                f"'{name}' must contain L/2={self.m_blocks} entries, got {len(raw)}."
            )
        lift: List[List[Monomial]] = []
        for idx, entry in enumerate(raw):
            monomials: List[Monomial] = []
            for term in entry:
                label, shifts = term
                perm = _resolve_h(label, self.degree)
                shifts = tuple(int(s) for s in shifts)
                if len(shifts) != len(self.cyclic):
                    raise ValueError(
                        f"'{name}[{idx}]' shift {shifts} does not match cyclic "
                        f"factors {self.cyclic}."
                    )
                monomials.append((perm, shifts))
            if not monomials:
                raise ValueError(f"'{name}[{idx}]' must contain at least one monomial.")
            lift.append(monomials)
        return lift

    # -- geometry ---------------------------------------------------------

    @property
    def lift_size(self) -> int:
        """Points per lift block, ``k * m``."""
        return self.alphabet.size

    @property
    def n_data(self) -> int:
        return self.L * self.lift_size

    @property
    def num_x_checks(self) -> int:
        return self.J * self.lift_size

    @property
    def num_z_checks(self) -> int:
        return self.J * self.lift_size

    @property
    def stabilizer_weight(self) -> int:
        """Total lift terms per check: ``sum_i (|F_i| + |G_i|)``."""
        return sum(len(e) for e in self.f) + sum(len(e) for e in self.g)

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
        size = self.lift_size
        for block in range(self.L):
            for p in range(size):
                self.add_qubit(p, block, role="data", uid=self._data_uid(block, p))

        syn_base = self.n_data
        for row_slot in range(self.J):
            for p in range(size):
                self.add_qubit(p, self.L + row_slot, role="syndrome_x",
                               uid=syn_base + row_slot * size + p)

        z_base = self.n_data + self.num_x_checks
        for row_slot in range(self.J):
            for p in range(size):
                self.add_qubit(p, self.L + self.J + row_slot, role="syndrome_z",
                               uid=z_base + row_slot * size + p)

    def _register_stabilizers(
        self,
        hx_supports: Sequence[Sequence[int]],
        hz_supports: Sequence[Sequence[int]],
    ):
        size = self.lift_size
        for row_index, support in enumerate(hx_supports):
            row_slot, p = divmod(row_index, size)
            syn_coord = self.qubit_coords[self.n_data + row_slot * size + p]
            self.create_stim_stabilizer(
                {self.qubit_coords[col]: "X" for col in support}, syn_coord, "X"
            )

        z_base = self.n_data + self.num_x_checks
        for row_index, support in enumerate(hz_supports):
            row_slot, p = divmod(row_index, size)
            syn_coord = self.qubit_coords[z_base + row_slot * size + p]
            self.create_stim_stabilizer(
                {self.qubit_coords[col]: "Z" for col in support}, syn_coord, "Z"
            )

    def _data_uid(self, block: int, point: int) -> int:
        return block * self.lift_size + point

    # -- parity checks ----------------------------------------------------

    def get_css_row_supports(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Sparse row supports of ``H_X`` and ``H_Z`` over the data columns.

        Supports may repeat a column when two monomials of one lift entry land
        on it; the parity is resolved by XOR when the dense matrix is built,
        matching the group-ring arithmetic over ``F_2``.
        """
        hx_supports: List[List[int]] = []
        hz_supports: List[List[int]] = []
        mb, apply = self.m_blocks, self.alphabet.apply
        inv = self.alphabet.invert_monomial

        for active_row in range(self.J):
            for p in range(self.lift_size):
                hx_row: List[int] = []
                hz_row: List[int] = []
                for j in range(mb):
                    x_idx = (j - active_row) % mb
                    for mono in self.f[x_idx]:
                        hx_row.append(self._data_uid(j, apply(mono, p)))
                    for mono in self.g[x_idx]:
                        hx_row.append(self._data_uid(mb + j, apply(mono, p)))

                    z_idx = (active_row - j) % mb
                    for mono in self.g[z_idx]:
                        hz_row.append(self._data_uid(j, apply(inv(mono), p)))
                    for mono in self.f[z_idx]:
                        hz_row.append(self._data_uid(mb + j, apply(inv(mono), p)))

                hx_supports.append(hx_row)
                hz_supports.append(hz_row)

        return hx_supports, hz_supports

    def get_css_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dense binary ``H_X`` and ``H_Z`` over the data columns."""
        hx_supports, hz_supports = self.get_css_row_supports()
        hx = np.zeros((len(hx_supports), self.n_data), dtype=np.uint8)
        hz = np.zeros((len(hz_supports), self.n_data), dtype=np.uint8)
        for row_index, support in enumerate(hx_supports):
            for col in support:
                hx[row_index, col] ^= 1
        for row_index, support in enumerate(hz_supports):
            for col in support:
                hz[row_index, col] ^= 1
        return hx, hz

    # -- orthogonality ----------------------------------------------------

    def active_pairs(self) -> List[Tuple[int, int]]:
        """Index pairs ``(i, j)`` required to commute for the active rows.

        These are the pairs in the diagonal band ``Gamma_J`` of Definition 11:
        ``i + j = r mod L/2`` with ``r < J``.
        """
        mb = self.m_blocks
        return [(i, j) for i in range(mb) for j in range(mb)
                if (i + j) % mb < self.J]

    def validate_required_commutativity(self) -> bool:
        """Whether every active pair of lift monomials commutes in ``H_k``.

        This is the strict per-pair criterion of Definition 11
        (``[F_i, G_j] = 0`` for every ``(i, j)`` in the band ``Gamma_J``); for a
        direct product the abelian parts always commute, so by Lemma 2 it is
        decided entirely inside ``H_k``.

        It is **sufficient but not necessary**. Orthogonality only needs the
        sums ``Psi_r = sum_u [F_u, G_{r-u}]`` to vanish for ``r < J``, and
        individual commutators may cancel against each other within a sum —
        several published rate-1/2 instances do exactly that (e.g.
        ``gala_576_292_8``, where ``[F_1, G_5]`` and ``[F_2, G_4]`` are each
        non-zero but cancel in ``Psi_0``). Use :meth:`check_css_orthogonality`
        for the authoritative test.
        """
        for i, j in self.active_pairs():
            for fp, _ in self.f[i]:
                for gp, _ in self.g[j]:
                    if not commutes(fp, gp):
                        return False
        return True

    def check_css_orthogonality(self) -> bool:
        """Whether ``H_X H_Z^T = 0`` over GF(2) — the CSS condition."""
        hx, hz = self.get_css_matrices()
        return not np.any((hx @ hz.T) % 2)

    def get_info(self):
        info = super().get_info()
        info.update({
            "L": self.L,
            "J": self.J,
            "degree": self.degree,
            "cyclic": self.cyclic,
            "lift_size": self.lift_size,
            "stabilizer_weight": self.stabilizer_weight,
            "k": self.num_logicals,
            "n_data": self.n_data,
            "rank_x": getattr(self, "rank_x", None),
            "rank_z": getattr(self, "rank_z", None),
            "num_x_syndromes": len(self.syndrome_indices_x),
            "num_z_syndromes": len(self.syndrome_indices_z),
            "data_coords": self.data_coords,
            "syndrome_coords_x": self.syndrome_coords_x,
            "syndrome_coords_z": self.syndrome_coords_z,
            "syndrome_coords": self.syndrome_coords,
            "stabilizers": self.stabilizers,
            "logical_ops": self.logical_ops,
            "logical_ops_available": False,
            "index_map": self.index_map,
            "qubit_coords": self.qubit_coords,
            "num_logicals": self.num_logicals,
        })
        return info
