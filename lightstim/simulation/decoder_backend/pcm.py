"""
PCM (parity check matrix) utilities for decoding.

Converts a stim DetectorErrorModel into sparse check matrices suitable for
BP-OSD and other matrix-based decoders.

Ported from decompose/gong_circuit.py (originally from gongaa/SlidingWindowDecoder).

Typical usage:

    dem = circuit.detector_error_model(decompose_errors=False)
    H, obs, priors = dem_to_check_matrices(dem)
    # H:      (num_detectors, num_mechanisms)  — parity check matrix
    # obs:    (num_observables, num_mechanisms) — logical flip matrix
    # priors: (num_mechanisms,)                 — error probabilities
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

import numpy as np
import stim
from scipy.sparse import csc_matrix


def dem_to_check_matrices(
    dem: stim.DetectorErrorModel,
    return_col_dict: bool = False,
):
    """
    Convert a stim DetectorErrorModel to sparse check matrices.

    Error mechanisms that trigger the same set of detectors and observables
    are merged into a single column (their probabilities are summed).

    Args:
        dem:             A stim DetectorErrorModel (from circuit.detector_error_model()).
        return_col_dict: If True, also return the column-key dict.

    Returns:
        check_matrix:       csc_matrix, shape (num_detectors, num_mechanisms).
                            Entry [d, e] = 1 iff error e flips detector d.
        observables_matrix: csc_matrix, shape (num_observables, num_mechanisms).
                            Entry [l, e] = 1 iff error e flips logical l.
        priors:             np.ndarray, shape (num_mechanisms,). Error probabilities.
        col_dict:           (only when return_col_dict=True) column-key → column index.
    """
    DL_ids: Dict[str, int] = {}
    L_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        key = " ".join(
            [f"D{s}" for s in sorted(detectors)] + [f"L{s}" for s in sorted(observables)]
        )
        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0
        hid = DL_ids[key]
        L_map[hid] = frozenset(observables)
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)

    num_cols = len(DL_ids)
    check_matrix = _build_csc(
        {v: [int(s[1:]) for s in k.split() if s.startswith("D")] for k, v in DL_ids.items()},
        shape=(dem.num_detectors, num_cols),
    )
    observables_matrix = _build_csc(L_map, shape=(dem.num_observables, num_cols))
    priors = np.zeros(num_cols)
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors


def _build_csc(elements: dict, shape: tuple) -> csc_matrix:
    nnz = sum(len(v) for v in elements.values())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, rows in elements.items():
        for row in rows:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)
