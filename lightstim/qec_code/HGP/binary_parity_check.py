"""Normalized binary parity-check matrices for HGP seed codes.

The canonical representation stores the support of each row.  Constructors
adapt dense arrays, sparse matrices, and structured polynomial descriptions to
that representation without making SciPy a required LightStim dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _toggle_support(indices: Iterable[int]) -> tuple[int, ...]:
    """Return sorted GF(2) support, cancelling repeated indices."""
    support: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index in support:
            support.remove(index)
        else:
            support.add(index)
    return tuple(sorted(support))


@dataclass(frozen=True)
class BinaryParityCheck:
    """A binary parity-check matrix represented by its nonzero row supports."""

    num_checks: int
    num_bits: int
    row_supports: tuple[tuple[int, ...], ...]
    source_metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.num_checks < 0:
            raise ValueError("num_checks must be non-negative.")
        if self.num_bits < 0:
            raise ValueError("num_bits must be non-negative.")
        if len(self.row_supports) != self.num_checks:
            raise ValueError(
                f"Expected {self.num_checks} row supports, got {len(self.row_supports)}."
            )

        normalized_rows = []
        for row_index, row in enumerate(self.row_supports):
            normalized = tuple(int(index) for index in row)
            if normalized != tuple(sorted(set(normalized))):
                raise ValueError(
                    f"Row {row_index} must contain unique, sorted bit indices."
                )
            if any(index < 0 or index >= self.num_bits for index in normalized):
                raise ValueError(
                    f"Row {row_index} contains an index outside [0, {self.num_bits})."
                )
            normalized_rows.append(normalized)

        object.__setattr__(self, "row_supports", tuple(normalized_rows))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_checks, self.num_bits

    @property
    def column_supports(self) -> tuple[tuple[int, ...], ...]:
        columns: list[list[int]] = [[] for _ in range(self.num_bits)]
        for check_index, row in enumerate(self.row_supports):
            for bit_index in row:
                columns[bit_index].append(check_index)
        return tuple(tuple(column) for column in columns)

    @classmethod
    def from_dense(
        cls,
        matrix: Any,
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "BinaryParityCheck":
        """Normalize a dense two-dimensional binary array."""
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise ValueError(f"Parity-check matrix must be two-dimensional, got {array.ndim}D.")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("Parity-check matrix entries must be binary (0 or 1).")

        rows = tuple(tuple(np.flatnonzero(row).tolist()) for row in array)
        metadata = {"construction": "dense"}
        if source_metadata:
            metadata.update(source_metadata)
        return cls(array.shape[0], array.shape[1], rows, metadata)

    @classmethod
    def from_row_supports(
        cls,
        shape: tuple[int, int],
        rows: Sequence[Iterable[int]],
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "BinaryParityCheck":
        """Build from the bit indices touched by each parity check."""
        if len(shape) != 2:
            raise ValueError("shape must be a (num_checks, num_bits) pair.")
        num_checks, num_bits = (int(value) for value in shape)
        normalized = tuple(_toggle_support(row) for row in rows)
        metadata = {"construction": "row_supports"}
        if source_metadata:
            metadata.update(source_metadata)
        return cls(num_checks, num_bits, normalized, metadata)

    @classmethod
    def from_scipy_sparse(
        cls,
        matrix: Any,
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "BinaryParityCheck":
        """Normalize a SciPy-like sparse matrix without importing SciPy.

        The object must expose ``shape`` and ``tocoo``.  Integer entries are
        interpreted over GF(2), so duplicate or even-valued contributions
        cancel.
        """
        if not hasattr(matrix, "shape") or not hasattr(matrix, "tocoo"):
            raise TypeError("Expected a SciPy-like sparse matrix with shape and tocoo().")
        if len(matrix.shape) != 2:
            raise ValueError("Parity-check matrix must be two-dimensional.")

        coo = matrix.tocoo(copy=True)
        row_entries: list[list[int]] = [[] for _ in range(int(matrix.shape[0]))]
        for row, column, value in zip(coo.row, coo.col, coo.data):
            if not np.isfinite(value) or int(value) != value:
                raise ValueError("Sparse parity-check entries must be finite integers.")
            if int(value) % 2:
                row_entries[int(row)].append(int(column))

        metadata = {"construction": "scipy_sparse"}
        if source_metadata:
            metadata.update(source_metadata)
        return cls.from_row_supports(
            (int(matrix.shape[0]), int(matrix.shape[1])),
            row_entries,
            source_metadata=metadata,
        )

    @classmethod
    def from_cyclic_polynomial(
        cls,
        exponents: Iterable[int],
        size: int,
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "BinaryParityCheck":
        """Build a circulant matrix from a polynomial modulo ``x**size - 1``.

        Row ``i`` has ones at columns ``(i + exponent) % size``.  Repeated
        monomials cancel over GF(2).
        """
        size = int(size)
        if size <= 0:
            raise ValueError("Cyclic polynomial size must be positive.")
        normalized_exponents = _toggle_support(int(exp) % size for exp in exponents)
        rows = [
            _toggle_support((row + exponent) % size for exponent in normalized_exponents)
            for row in range(size)
        ]
        metadata = {
            "construction": "cyclic_polynomial",
            "exponents": normalized_exponents,
            "size": size,
        }
        if source_metadata:
            metadata.update(source_metadata)
        return cls.from_row_supports((size, size), rows, source_metadata=metadata)

    @classmethod
    def from_bivariate_polynomial(
        cls,
        monomials: Iterable[tuple[int, int]],
        shape: tuple[int, int],
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "BinaryParityCheck":
        """Build a 2D circulant matrix over ``Z_l x Z_m``.

        Coordinates are flattened as ``(i, j) -> i*m + j``.  A monomial
        ``(a, b)`` maps row ``(i, j)`` to column ``(i+a, j+b)`` modulo
        ``(l, m)``.
        """
        if len(shape) != 2:
            raise ValueError("shape must be the cyclic dimensions (l, m).")
        l, m = (int(value) for value in shape)
        if l <= 0 or m <= 0:
            raise ValueError("Bivariate cyclic dimensions must be positive.")

        normalized_monomials: set[tuple[int, int]] = set()
        for a, b in monomials:
            monomial = (int(a) % l, int(b) % m)
            if monomial in normalized_monomials:
                normalized_monomials.remove(monomial)
            else:
                normalized_monomials.add(monomial)
        ordered_monomials = tuple(sorted(normalized_monomials))

        rows = []
        for i in range(l):
            for j in range(m):
                rows.append(
                    _toggle_support(
                        ((i + a) % l) * m + (j + b) % m
                        for a, b in ordered_monomials
                    )
                )

        metadata = {
            "construction": "bivariate_polynomial",
            "monomials": ordered_monomials,
            "cyclic_shape": (l, m),
        }
        if source_metadata:
            metadata.update(source_metadata)
        size = l * m
        return cls.from_row_supports((size, size), rows, source_metadata=metadata)

    def to_dense(self, *, dtype: Any = np.uint8) -> np.ndarray:
        matrix = np.zeros(self.shape, dtype=dtype)
        for row_index, support in enumerate(self.row_supports):
            matrix[row_index, list(support)] = 1
        return matrix


def as_binary_parity_check(matrix: Any) -> BinaryParityCheck:
    """Normalize supported public input forms to ``BinaryParityCheck``."""
    if isinstance(matrix, BinaryParityCheck):
        return matrix
    if hasattr(matrix, "tocoo") and hasattr(matrix, "shape"):
        return BinaryParityCheck.from_scipy_sparse(matrix)
    return BinaryParityCheck.from_dense(matrix)
