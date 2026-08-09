"""Hypergraph-product CSS code patch."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from lightstim.ir.qec_patch import QECPatch

from .algebra import CanonicalKernelBasis, canonical_kernel_basis, transpose_parity_check
from .binary_parity_check import BinaryParityCheck, as_binary_parity_check


ProductIndex = tuple[int, int]


class HGPCode(QECPatch):
    """Hypergraph-product code constructed from two binary parity checks.

    The original seed matrices define the physical Tanner graph.  Separate
    canonical kernel analyses define deterministic, paired logical operators;
    row reduction is never applied to the physical stabilizer supports.

    Qubit and check ordering
    ------------------------
    Data columns are ordered as ``V1 x V2`` followed by ``C1 x C2``.

    Z checks are indexed by ``C1 x V2`` and have parity-check matrix
    ``[H1 kron I_n2 | I_m1 kron H2.T]``.

    X checks are indexed by ``V1 x C2`` and have parity-check matrix
    ``[I_n1 kron H2 | H1.T kron I_m2]``.

    The first seed is drawn on the vertical axis and the second on the
    horizontal axis.  Bits occupy even coordinates and checks odd coordinates.
    """

    def __init__(self, H1: Any, H2: Any | None = None, **kwargs: Any):
        super().__init__(H1=H1, H2=H2, **kwargs)

    def _process_params(self) -> None:
        self.h1 = as_binary_parity_check(self.params["H1"])
        raw_h2 = self.params.get("H2")
        self.h2 = self.h1 if raw_h2 is None else as_binary_parity_check(raw_h2)
        self.shift = self.params.get("shift", (0, 0))
        self.code_distance = self.params.get("d")

        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("shift must be a two-number tuple.")
        if self.h1.num_bits == 0 or self.h2.num_bits == 0:
            raise ValueError("Each HGP seed must contain at least one bit column.")

        # Semantic product indices remain stable even if display coordinates
        # are shifted or transformed later.
        self.vv_qubits: Dict[ProductIndex, int] = {}
        self.cc_qubits: Dict[ProductIndex, int] = {}
        self.x_check_qubits: Dict[ProductIndex, int] = {}
        self.z_check_qubits: Dict[ProductIndex, int] = {}
        self.logical_pairs: List[Dict[str, Any]] = []

    @property
    def num_data_qubits(self) -> int:
        return self.h1.num_bits * self.h2.num_bits + self.h1.num_checks * self.h2.num_checks

    @property
    def data_index_order(self) -> tuple[int, ...]:
        vv = [self.vv_qubits[key] for key in sorted(self.vv_qubits)]
        cc = [self.cc_qubits[key] for key in sorted(self.cc_qubits)]
        return tuple(vv + cc)

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[index] for index in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[index] for index in sorted(self.syndrome_indices_z)]

    def build(self) -> None:
        self.kernel_h1 = canonical_kernel_basis(self.h1)
        self.kernel_h2 = canonical_kernel_basis(self.h2)
        self.kernel_h1_transpose = canonical_kernel_basis(transpose_parity_check(self.h1))
        self.kernel_h2_transpose = canonical_kernel_basis(transpose_parity_check(self.h2))

        self._register_qubits()
        self._build_stabilizers()
        self._build_logical_operators()

        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def _register_qubits(self) -> None:
        # Data sector V1 x V2: first seed is vertical, second horizontal.
        for bit_1 in range(self.h1.num_bits):
            for bit_2 in range(self.h2.num_bits):
                coord = (2 * bit_2, 2 * bit_1)
                self.vv_qubits[(bit_1, bit_2)] = self.add_qubit(*coord, role="data")

        # Data sector C1 x C2.
        for check_1 in range(self.h1.num_checks):
            for check_2 in range(self.h2.num_checks):
                coord = (2 * check_2 + 1, 2 * check_1 + 1)
                self.cc_qubits[(check_1, check_2)] = self.add_qubit(*coord, role="data")

        # X checks V1 x C2.
        for bit_1 in range(self.h1.num_bits):
            for check_2 in range(self.h2.num_checks):
                coord = (2 * check_2 + 1, 2 * bit_1)
                self.x_check_qubits[(bit_1, check_2)] = self.add_qubit(
                    *coord, role="syndrome_x"
                )

        # Z checks C1 x V2.
        for check_1 in range(self.h1.num_checks):
            for bit_2 in range(self.h2.num_bits):
                coord = (2 * bit_2, 2 * check_1 + 1)
                self.z_check_qubits[(check_1, bit_2)] = self.add_qubit(
                    *coord, role="syndrome_z"
                )

    def _build_stabilizers(self) -> None:
        h1_columns = self.h1.column_supports
        h2_columns = self.h2.column_supports

        # Hx = [I_n1 kron H2 | H1.T kron I_m2].
        for bit_1 in range(self.h1.num_bits):
            for check_2, h2_row in enumerate(self.h2.row_supports):
                targets = {
                    self.qubit_coords[self.vv_qubits[(bit_1, bit_2)]]: "X"
                    for bit_2 in h2_row
                }
                targets.update(
                    {
                        self.qubit_coords[self.cc_qubits[(check_1, check_2)]]: "X"
                        for check_1 in h1_columns[bit_1]
                    }
                )
                syndrome = self.qubit_coords[self.x_check_qubits[(bit_1, check_2)]]
                self.create_stim_stabilizer(targets, syndrome, "X")

        # Hz = [H1 kron I_n2 | I_m1 kron H2.T].
        for check_1, h1_row in enumerate(self.h1.row_supports):
            for bit_2 in range(self.h2.num_bits):
                targets = {
                    self.qubit_coords[self.vv_qubits[(bit_1, bit_2)]]: "Z"
                    for bit_1 in h1_row
                }
                targets.update(
                    {
                        self.qubit_coords[self.cc_qubits[(check_1, check_2)]]: "Z"
                        for check_2 in h2_columns[bit_2]
                    }
                )
                syndrome = self.qubit_coords[self.z_check_qubits[(check_1, bit_2)]]
                self.create_stim_stabilizer(targets, syndrome, "Z")

    def _register_logical_pair(
        self,
        *,
        x_indices: set[int],
        z_indices: set[int],
        sector: str,
        seed_indices: tuple[int, int],
    ) -> None:
        logical_id = len(self.logical_pairs)
        overlap = x_indices & z_indices
        if len(overlap) != 1:
            raise RuntimeError(
                f"Canonical logical pair {logical_id} must have one pivot, got {sorted(overlap)}."
            )

        x_targets = {self.qubit_coords[index]: "X" for index in sorted(x_indices)}
        self.create_stim_logical(x_targets, "X")
        x_record = self.logical_ops[-1]
        x_record.update(
            logical_id=logical_id,
            sector=sector,
            seed_indices=seed_indices,
        )

        z_targets = {self.qubit_coords[index]: "Z" for index in sorted(z_indices)}
        self.create_stim_logical(z_targets, "Z")
        z_record = self.logical_ops[-1]
        z_record.update(
            logical_id=logical_id,
            sector=sector,
            seed_indices=seed_indices,
        )

        self.logical_pairs.append(
            {
                "logical_id": logical_id,
                "sector": sector,
                "seed_indices": seed_indices,
                "pivot_index": next(iter(overlap)),
                "x": x_record,
                "z": z_record,
            }
        )

    @staticmethod
    def _support(vector: np.ndarray) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(vector))

    def _build_logical_operators(self) -> None:
        k1 = self.kernel_h1
        k2 = self.kernel_h2
        k1t = self.kernel_h1_transpose
        k2t = self.kernel_h2_transpose

        # Logical pairs on V1 x V2:
        #   X = kernel(H1) x pivot(H2)
        #   Z = pivot(H1) x kernel(H2).
        for logical_1, vector_1 in enumerate(k1.basis):
            pivot_1 = k1.pivots[logical_1]
            for logical_2, vector_2 in enumerate(k2.basis):
                pivot_2 = k2.pivots[logical_2]
                x_indices = {
                    self.vv_qubits[(bit_1, pivot_2)]
                    for bit_1 in self._support(vector_1)
                }
                z_indices = {
                    self.vv_qubits[(pivot_1, bit_2)]
                    for bit_2 in self._support(vector_2)
                }
                self._register_logical_pair(
                    x_indices=x_indices,
                    z_indices=z_indices,
                    sector="bit_bit",
                    seed_indices=(logical_1, logical_2),
                )

        # Logical pairs on C1 x C2:
        #   X = pivot(H1.T) x kernel(H2.T)
        #   Z = kernel(H1.T) x pivot(H2.T).
        for logical_1, vector_1 in enumerate(k1t.basis):
            pivot_1 = k1t.pivots[logical_1]
            for logical_2, vector_2 in enumerate(k2t.basis):
                pivot_2 = k2t.pivots[logical_2]
                x_indices = {
                    self.cc_qubits[(pivot_1, check_2)]
                    for check_2 in self._support(vector_2)
                }
                z_indices = {
                    self.cc_qubits[(check_1, pivot_2)]
                    for check_1 in self._support(vector_1)
                }
                self._register_logical_pair(
                    x_indices=x_indices,
                    z_indices=z_indices,
                    sector="check_check",
                    seed_indices=(logical_1, logical_2),
                )

        self.num_logicals = len(self.logical_pairs)

    def get_data_parity_check_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dense ``(Hx, Hz)`` restricted to data-qubit columns."""
        data_order = self.data_index_order
        data_to_column = {index: column for column, index in enumerate(data_order)}
        x_stabilizers = [stab for stab in self.stabilizers if stab["type"] == "X"]
        z_stabilizers = [stab for stab in self.stabilizers if stab["type"] == "Z"]

        def matrix_for(stabilizers: List[Dict[str, Any]]) -> np.ndarray:
            matrix = np.zeros((len(stabilizers), len(data_order)), dtype=np.uint8)
            for row, stabilizer in enumerate(stabilizers):
                for data_index in stabilizer["data_indices"]:
                    matrix[row, data_to_column[data_index]] = 1
            return matrix

        return matrix_for(x_stabilizers), matrix_for(z_stabilizers)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "h1_shape": self.h1.shape,
                "h2_shape": self.h2.shape,
                "rank_h1": self.kernel_h1.rank,
                "rank_h2": self.kernel_h2.rank,
                "num_data_qubits": self.num_data_qubits,
                "num_x_syndromes": len(self.syndrome_indices_x),
                "num_z_syndromes": len(self.syndrome_indices_z),
                "num_logicals": self.num_logicals,
                "code_distance": self.code_distance,
                "logical_pairs": self.logical_pairs,
            }
        )
        return info
