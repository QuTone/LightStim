"""Optional native Pauli algebra under the existing Tracker role policy.

The persistent inverse-tableau architecture follows Hou Yuewen's scalability
contribution in LightStim PR #110. Packed source-coordinate algebra extends it
with exact S_bare/S_dressed/L classification and readout closure lineages,
including arbitrary noncommuting role spaces. Tracker still decides projection
pivots, detector/logical roles, postselection, and absorbed-history policy.

Set ``LIGHTSTIM_PAULI_BACKEND`` to ``reference`` to disable acceleration or to
``native`` to require the compiled extension. The default, ``auto``, uses the
extension when available. Older extensions retain the commuting-only fallback.
"""

from collections import OrderedDict
import os
from threading import RLock

import numpy as np


_NATIVE_IMPORT_ERROR = None
try:
    from .cpp._pauli_backend_cpp import (
        CommutingBasis as _CommutingBasis,
        propagate_paulis as _propagate_paulis_native,
    )
except (ImportError, OSError) as error:
    _CommutingBasis = None
    _propagate_paulis_native = None
    _NATIVE_IMPORT_ERROR = error

# The role algebra is optional independently of older compiled extensions.
try:
    from .cpp import _pauli_backend_cpp as _packed_native
    PackedRows = _packed_native.PackedRows
    PackedBasis = _packed_native.PackedBasis
except (ImportError, OSError, AttributeError):
    _packed_native = None
    PackedRows = PackedBasis = None

_packed_basis_cache = OrderedDict()
_packed_basis_cache_bytes = 0


_NATIVE_AVAILABLE = _CommutingBasis is not None
_CACHE_MAX_ENTRIES = 8
_CACHE_MAX_BYTES = 64 * 1024 * 1024
_basis_cache = OrderedDict()
_basis_cache_bytes = 0
_basis_cache_lock = RLock()


def _use_native():
    # Read at call time so one process can compare the two implementations.
    mode = os.environ.get("LIGHTSTIM_PAULI_BACKEND", "auto").strip().lower()
    if mode not in {"auto", "reference", "native"}:
        raise ValueError(
            "LIGHTSTIM_PAULI_BACKEND must be 'auto', 'reference', or 'native'; "
            f"got {mode!r}."
        )
    if mode == "reference":
        return False
    if mode == "native" and not _NATIVE_AVAILABLE:
        raise ImportError(
            "LIGHTSTIM_PAULI_BACKEND=native requires the optional "
            "lightstim.utils.cpp._pauli_backend_cpp extension. "
            "Build the native extensions or select 'auto'/'reference'."
        ) from _NATIVE_IMPORT_ERROR
    return _NATIVE_AVAILABLE


def _clear_basis_cache():
    """Release cached algebra state, including unsupported-basis entries."""
    global _basis_cache_bytes, _packed_basis_cache_bytes
    with _basis_cache_lock:
        _basis_cache.clear()
        _basis_cache_bytes = 0
        _packed_basis_cache.clear()
        _packed_basis_cache_bytes = 0


def _get_basis(matrix):
    # A native constructor currently holds the GIL, but the Python cache
    # lookup/eviction/accounting sequence can still interleave across threads.
    with _basis_cache_lock:
        return _get_basis_locked(matrix)


def _get_basis_locked(matrix):
    global _basis_cache_bytes

    # Tracker arrays are mutable.  An identity-based key would silently reuse
    # stale decompositions after projections, rebasing, or Clifford evolution.
    contents = matrix.tobytes()
    key = (matrix.shape, contents)
    if key in _basis_cache:
        instance, size = _basis_cache.pop(key)
        _basis_cache[key] = (instance, size)
        return instance

    # Measurement batches commonly grow by one row while the earlier source
    # rows stay unchanged. Extend the longest cached prefix, copying its native
    # tableau so future exact hits still see the unchanged original basis.
    parent = None
    parent_rows = -1
    for (shape, old_contents), (candidate, _) in reversed(_basis_cache.items()):
        if (candidate is not None
                and shape[1] == matrix.shape[1]
                and parent_rows < shape[0] < matrix.shape[0]
                and contents.startswith(old_contents)):
            parent = candidate
            parent_rows = shape[0]
    instance = (parent.extended(matrix[parent_rows:]) if parent is not None
                else _CommutingBasis(matrix))
    if instance.commuting:
        size = len(contents) + int(instance.estimated_bytes) + 128
    else:
        # Remember unsupported bases without retaining their native tableau.
        instance = None
        size = len(contents) + 128

    if _CACHE_MAX_ENTRIES <= 0 or size > _CACHE_MAX_BYTES:
        return instance
    while _basis_cache and (
        len(_basis_cache) >= _CACHE_MAX_ENTRIES
        or _basis_cache_bytes + size > _CACHE_MAX_BYTES
    ):
        _, (_, old_size) = _basis_cache.popitem(last=False)
        _basis_cache_bytes -= old_size
    _basis_cache[key] = (instance, size)
    _basis_cache_bytes += size
    return instance


def solve_commuting_decomposition(basis, targets):
    """Return an accelerated decomposition, or ``None`` for general solving.

    Coefficients refer to original input rows, including redundant rows.  New
    target directions are returned in target order; the public general solver
    retains responsibility for its historical ordering of independent blocks.
    This primitive does not perform greedy coefficient-weight reduction.
    """
    if not _use_native():
        return None
    basis = np.asarray(basis)
    targets = np.asarray(targets)
    if basis.ndim != 2 or targets.ndim != 2:
        raise ValueError("Basis and targets must be two-dimensional matrices.")
    if basis.shape[1] != targets.shape[1]:
        raise ValueError("Basis and targets must have the same number of columns.")
    if basis.shape[1] % 2:
        # The public GF(2) solver also accepts ordinary, non-Pauli vectors.
        return None
    basis = np.ascontiguousarray(basis, dtype=np.uint8) & np.uint8(1)
    targets = np.ascontiguousarray(targets, dtype=np.uint8) & np.uint8(1)
    instance = _get_basis(basis)
    if instance is None:
        return None
    return instance.decompose(targets)


def propagate_paulis(rows, forward):
    """Apply a forward binary Pauli transform, with a NumPy fallback.

    ``rows`` has shape ``(M, 2N)`` and ``forward`` has shape ``(2N, 2N)``.
    No measurement records or role metadata are inspected or modified.
    """
    use_native = _use_native()
    rows = np.asarray(rows)
    forward = np.asarray(forward)
    if rows.ndim != 2 or forward.ndim != 2:
        raise ValueError("Pauli rows and their transform must be two-dimensional.")
    width = rows.shape[1]
    if width % 2 or forward.shape != (width, width):
        raise ValueError(
            "Pauli rows require an even width and a matching square transform; "
            f"got rows {rows.shape} and transform {forward.shape}."
        )
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    forward = np.ascontiguousarray(forward, dtype=np.uint8)
    if use_native:
        return _propagate_paulis_native(rows, forward)
    return ((rows @ forward) % 2).astype(np.uint8)


def use_packed_algebra():
    """Whether the optional full-coordinate role algebra is selected."""
    return _use_native() and PackedRows is not None


def _get_packed_basis(matrix):
    """Cache immutable packed source bases by content, never mutable identity."""
    global _packed_basis_cache_bytes
    matrix = np.ascontiguousarray(matrix, dtype=np.uint8) & np.uint8(1)
    key = (matrix.shape, matrix.tobytes())
    with _basis_cache_lock:
        if key in _packed_basis_cache:
            result, size = _packed_basis_cache.pop(key)
            _packed_basis_cache[key] = result, size
            return result
        parent = None
        parent_rows = -1
        for (shape, contents), (candidate, _) in reversed(_packed_basis_cache.items()):
            if (shape[1] == matrix.shape[1] and parent_rows < shape[0] < matrix.shape[0]
                    and key[1].startswith(contents)):
                parent, parent_rows = candidate, shape[0]
        result = parent.extended(matrix[parent_rows:]) if parent is not None else PackedBasis(matrix)
        size = len(key[1]) + result.estimated_bytes + 128
        if _CACHE_MAX_ENTRIES > 0 and size <= _CACHE_MAX_BYTES:
            while _packed_basis_cache and (len(_packed_basis_cache) >= _CACHE_MAX_ENTRIES
                    or _packed_basis_cache_bytes + size > _CACHE_MAX_BYTES):
                _, (_, previous_size) = _packed_basis_cache.popitem(last=False)
                _packed_basis_cache_bytes -= previous_size
            _packed_basis_cache[key] = result, size
            _packed_basis_cache_bytes += size
        return result


def solve_packed_decomposition(basis, targets):
    """General source-coordinate decomposition, including noncommuting roles."""
    if not use_packed_algebra():
        return None
    basis, targets = np.asarray(basis), np.asarray(targets)
    if basis.ndim != 2 or targets.ndim != 2 or basis.shape[1] != targets.shape[1]:
        raise ValueError("Basis and targets must be matrices of the same width")
    return _get_packed_basis(basis).decompose(targets)


def multiply_binary(coefficients, rows):
    """GF(2) product using word XOR instead of dense integer multiplication."""
    if use_packed_algebra():
        return _packed_native.packed_multiply(PackedRows(coefficients), PackedRows(rows)).numpy()
    return ((np.asarray(coefficients, dtype=np.uint8) @ np.asarray(rows, dtype=np.uint8)) % 2).astype(np.uint8)
