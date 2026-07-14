"""Convert a stim DetectorErrorModel into parity-check / observable matrices.

Shared by every decoder backend that needs raw matrices instead of a stim DEM
(the GPU ``cudaqx`` backend and the :mod:`external` decoder facade). Keeping a
single implementation here avoids the subtle bit-ordering / contiguity bugs
that creep in when this conversion is re-derived per decoder.
"""

from __future__ import annotations

from array import array
from typing import Union

import numpy as np
import scipy.sparse as sp
import stim

Matrix = Union[np.ndarray, sp.csr_matrix]


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
    *,
    sparse: bool = False,
    merge_duplicates: bool = True,
) -> tuple[Matrix, Matrix, np.ndarray]:
    """Convert a stim DEM to ``(H, obs_matrix, priors)``.

    Args:
        dem: the detector error model to convert.
        sparse: if ``True``, return ``H`` and ``obs_matrix`` as
            ``scipy.sparse.csr_matrix`` built without ever materialising a dense
            array. Essential for large multi-round LDPC circuits, where the dense
            ``(n_detectors, n_error_mechanisms)`` matrix can reach many GB and OOM
            a worker. If ``False`` (default) they are dense ``uint8`` arrays —
            required by the CUDA (`cudaqx`) backend.
        merge_duplicates: merge error mechanisms with identical (detector,
            observable) footprints into one column, combining priors with the
            XOR rule ``p1(1-p2) + p2(1-p1)`` (default ``True``). stim does not always fuse
            such mechanisms itself — e.g. with ``z_only`` detectors, X- and
            Y-type data errors keep separate columns despite identical
            symptoms — and the resulting degenerate near-duplicate variables
            measurably degrade BP (a chen_p96 z_only run went from ~11% to
            ~2% heralded error per shot after merging). A no-op when the DEM
            is already fully merged. Column order preserves first occurrence.

    Returns:
        H          -- parity-check matrix, shape ``(n_detectors, n_error_mechanisms)``.
                      Dense C-contiguous ``uint8`` (default) or CSR (``sparse=True``).
        obs_matrix -- observable-flip matrix, shape ``(n_observables, n_error_mechanisms)``.
        priors     -- float64 prior error probabilities, shape ``(n_error_mechanisms,)``.

    Each error instruction in the (flattened) DEM becomes one column: ``H[d, e]``
    is 1 iff error ``e`` flips detector ``d``, and ``obs_matrix[o, e]`` is 1 iff
    error ``e`` flips observable ``o``.
    """
    n_dets = dem.num_detectors
    n_obs = dem.num_observables

    # Single streaming pass over the DEM into flat COO coordinate buffers — one
    # (row, col) pair per detector/observable flip. C-typed ``array('i')``
    # buffers, not Python lists: a list entry costs ~40 B (pointer + boxed int)
    # versus 4 B here, and a multi-round LDPC DEM can have tens of millions of
    # observable flips, so lists OOM before the CSR matrix is even built.
    priors_arr = array("d")
    h_rows = array("i")
    h_cols = array("i")
    o_rows = array("i")
    o_cols = array("i")

    e = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        priors_arr.append(instruction.args_copy()[0])
        for t in instruction.targets_copy():
            if t.is_relative_detector_id():
                h_rows.append(t.val)
                h_cols.append(e)
            elif t.is_logical_observable_id():
                o_rows.append(t.val)
                o_cols.append(e)
        e += 1

    n_err = e
    priors = np.asarray(priors_arr, dtype=np.float64)

    if sparse:
        H = _parity_csr(h_rows, h_cols, n_dets, n_err)
        obs_matrix = _parity_csr(o_rows, o_cols, n_obs, n_err)
        if merge_duplicates:
            H, obs_matrix, priors = _merge_duplicate_columns(H, obs_matrix, priors)
        return H, obs_matrix, priors

    # Explicitly C-contiguous (row-major) to match decoder expectations,
    # equivalent to scipy sparse_matrix.todense(order='C').
    H = np.zeros((n_dets, n_err), dtype=np.uint8, order="C")
    obs_matrix = np.zeros((n_obs, n_err), dtype=np.uint8, order="C")
    # add.at then & 1 == XOR/parity: stim treats a target listed an even number
    # of times as cancelling, so a repeated (row, col) flips the entry zero times.
    if h_rows:
        np.add.at(H, (np.asarray(h_rows), np.asarray(h_cols)), 1)
        H &= 1
    if o_rows:
        np.add.at(obs_matrix, (np.asarray(o_rows), np.asarray(o_cols)), 1)
        obs_matrix &= 1

    if merge_duplicates:
        H, obs_matrix, priors = _merge_duplicate_columns_dense(
            H, obs_matrix, priors)
    return np.ascontiguousarray(H), obs_matrix, priors


def _merge_duplicate_columns(
    H: sp.csr_matrix, obs_matrix: sp.csr_matrix, priors: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Fuse columns with identical (detector, observable) footprints.

    Priors combine as the probability of an odd number of the fused mechanisms
    firing (two mechanisms both firing cancel over GF(2)): for two,
    ``p1(1-p2) + p2(1-p1)`` — the same rule stim applies when it merges
    identical error instructions.
    """
    Hc = H.tocsc()
    Oc = obs_matrix.tocsc()
    n_err = H.shape[1]
    rep: dict[bytes, int] = {}          # footprint -> merged column index
    col_of = np.empty(n_err, dtype=np.int64)
    merged_priors: list[float] = []
    keep: list[int] = []                # original column giving the footprint
    for j in range(n_err):
        key = (Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]].tobytes()
               + b"|" + Oc.indices[Oc.indptr[j]:Oc.indptr[j + 1]].tobytes())
        jj = rep.get(key)
        if jj is None:
            jj = len(keep)
            rep[key] = jj
            keep.append(j)
            merged_priors.append(priors[j])
        else:
            a, b = merged_priors[jj], priors[j]
            merged_priors[jj] = a * (1 - b) + b * (1 - a)
        col_of[j] = jj
    if len(keep) == n_err:              # nothing to merge
        return H, obs_matrix, priors
    keep_idx = np.asarray(keep)
    return (Hc[:, keep_idx].tocsr(), Oc[:, keep_idx].tocsr(),
            np.asarray(merged_priors, dtype=np.float64))


def _merge_duplicate_columns_dense(
    H: np.ndarray, obs_matrix: np.ndarray, priors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense counterpart of :func:`_merge_duplicate_columns`.

    Vectorised via ``np.unique`` over stacked (H; obs) columns; first-seen
    column order is preserved so the result matches the sparse path.
    """
    n_err = H.shape[1]
    if n_err == 0:
        return H, obs_matrix, priors
    stacked = np.vstack([H, obs_matrix]).T          # (n_err, n_dets + n_obs)
    _, first_idx, inverse = np.unique(
        stacked, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)                   # numpy 2.0 axis quirk
    if len(first_idx) == n_err:
        return H, obs_matrix, priors
    order = np.argsort(first_idx)                   # first-occurrence order
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    group = rank[inverse]                           # old column -> new column
    keep = first_idx[order]
    # XOR-combine group priors: 1-2p is multiplicative under the odd-firing
    # rule, i.e. (1-2p_new) = prod_i (1-2p_i).
    bias = np.ones(len(keep), dtype=np.float64)
    np.multiply.at(bias, group, 1.0 - 2.0 * priors)
    return (np.ascontiguousarray(H[:, keep]),
            np.ascontiguousarray(obs_matrix[:, keep]),
            (1.0 - bias) / 2.0)


def _parity_csr(
    rows: "array[int]", cols: "array[int]", n_rows: int, n_err: int
) -> sp.csr_matrix:
    """Build a CSR ``(rows, columns=errors)`` GF(2) matrix from COO coordinates.

    Duplicate ``(row, col)`` pairs cancel mod 2 (matching the dense XOR path):
    COO sums duplicates on ``tocsr``, then ``data & 1`` reduces to parity and
    ``eliminate_zeros`` drops the cancelled entries. int32 coordinates are kept
    end-to-end (zero-copy views of the ``array('i')`` buffers) — with tens of
    millions of entries, int64 temporaries double the peak footprint.
    """
    row_idx = np.frombuffer(rows, dtype=np.int32) if len(rows) \
        else np.empty(0, dtype=np.int32)
    col_idx = np.frombuffer(cols, dtype=np.int32) if len(cols) \
        else np.empty(0, dtype=np.int32)
    data = np.ones(len(row_idx), dtype=np.uint8)
    coo = sp.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_rows, n_err),
        dtype=np.uint8,
    )
    csr = coo.tocsr()
    csr.data &= 1
    csr.eliminate_zeros()
    return csr
