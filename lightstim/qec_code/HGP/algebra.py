"""GF(2) algebra used by hypergraph-product code patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .binary_parity_check import BinaryParityCheck, as_binary_parity_check


@dataclass(frozen=True)
class CanonicalKernelBasis:
    """Kernel basis paired with deterministic pivot unit vectors.

    ``basis`` and ``unit_complement`` store vectors as rows.  For every pair
    of rows ``i, j``, their binary inner product is ``delta(i, j)``.
    """

    basis: np.ndarray
    pivots: tuple[int, ...]
    unit_complement: np.ndarray
    rank: int

    @property
    def nullity(self) -> int:
        return self.basis.shape[0]


def canonical_kernel_basis(matrix: Any) -> CanonicalKernelBasis:
    """Find a strongly lower-triangular kernel basis over GF(2).

    This is the column-elimination construction from Algorithm 1 of
    Quintavalle, Webster, and Vasmer, Quantum 7, 1153 (2023).  The input is
    copied: canonical analysis never changes the parity checks used to build
    the physical HGP patch.
    """
    parity_check = as_binary_parity_check(matrix)
    work = parity_check.to_dense(dtype=np.uint8)
    num_bits = parity_check.num_bits

    # Column operations are accumulated on the right:
    #     original_matrix @ transform == work.
    transform = np.eye(num_bits, dtype=np.uint8)
    kernel_candidates = set(range(num_bits))

    for column in range(num_bits):
        nonzero_rows = np.flatnonzero(work[:, column])
        if nonzero_rows.size == 0:
            continue

        pivot_row = int(nonzero_rows[0])
        kernel_candidates.remove(column)
        for later_column in range(column + 1, num_bits):
            if work[pivot_row, later_column]:
                work[:, later_column] ^= work[:, column]
                transform[:, later_column] ^= transform[:, column]

    pivots = tuple(sorted(kernel_candidates))
    if pivots:
        basis = transform[:, list(pivots)].T.copy()
        unit_complement = np.eye(num_bits, dtype=np.uint8)[list(pivots)]
    else:
        basis = np.zeros((0, num_bits), dtype=np.uint8)
        unit_complement = np.zeros((0, num_bits), dtype=np.uint8)

    if np.any((parity_check.to_dense() @ basis.T) % 2):
        raise RuntimeError("Internal error: canonical basis is not in the kernel.")
    if not np.array_equal((unit_complement @ basis.T) % 2, np.eye(len(pivots), dtype=np.uint8)):
        raise RuntimeError("Internal error: kernel basis and pivot vectors are not paired.")

    return CanonicalKernelBasis(
        basis=basis,
        pivots=pivots,
        unit_complement=unit_complement,
        rank=num_bits - len(pivots),
    )


def transpose_parity_check(matrix: BinaryParityCheck) -> BinaryParityCheck:
    """Transpose a normalized parity-check matrix without densifying it."""
    return BinaryParityCheck.from_row_supports(
        (matrix.num_bits, matrix.num_checks),
        matrix.column_supports,
        source_metadata={
            "construction": "transpose",
            "source": dict(matrix.source_metadata),
        },
    )
