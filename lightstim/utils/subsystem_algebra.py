"""Pauli row-space operations retaining coefficients in the input state basis."""

import numpy as np

from .linear_algebra import kernel_gf2, row_echelon


def independent_row_indices(matrix):
    """Select original rows, in order, spanning a possibly redundant matrix."""
    if not matrix.size:
        return []
    return list(row_echelon(matrix.T)[3])


def reduce_modulo(rows, basis):
    """Return row residues modulo a Pauli span, without changing the inputs."""
    result = np.asarray(rows, dtype=np.uint8).copy()
    if basis.shape[0] and basis.shape[1]:
        reduced, rank, _, pivots = row_echelon(basis, reduced=True)
        for row, pivot in zip(reduced[:rank], pivots):
            result[result[:, pivot].astype(bool)] ^= row.astype(np.uint8)
    return result


def intersect_row_spaces(left, right):
    """Return independent rows of span(left) ∩ span(right), and their lineage.

    The second result C satisfies ``intersection == (C @ left) % 2``.
    Unlike filtering input rows by membership, this also finds intersections
    that only appear as products of several tracked Paulis. Coefficients can
    be applied to measurement-record parities using the same XOR operations.
    """
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Pauli spaces must be matrices of the same width.")
    if not left.shape[0] or not right.shape[0] or not left.shape[1]:
        return (np.zeros((0, left.shape[1]), dtype=np.uint8),
                np.zeros((0, left.shape[0]), dtype=np.uint8))
    residues = reduce_modulo(left, right)
    coefficients = kernel_gf2(residues.T)
    intersection = (coefficients @ left) % 2
    indices = independent_row_indices(intersection)
    return intersection[indices], coefficients[indices]


def combine_records(coefficients, records):
    """Apply a GF(2) basis change to absolute measurement-record parities."""
    result = []
    for coefficients_row in coefficients:
        parity = set()
        for idx in np.flatnonzero(coefficients_row):
            for record in records[idx]:
                if record in parity:
                    parity.remove(record)
                else:
                    parity.add(record)
        result.append(sorted(parity))
    return result
