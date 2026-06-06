"""Convert a stim DetectorErrorModel into parity-check / observable matrices.

Shared by every decoder backend that needs raw matrices instead of a stim DEM
(the GPU ``cudaqx`` backend and the :mod:`external` decoder facade). Keeping a
single implementation here avoids the subtle bit-ordering / contiguity bugs
that creep in when this conversion is re-derived per decoder.
"""

from __future__ import annotations

import numpy as np
import stim


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a stim DEM to ``(H, obs_matrix, priors)``.

    Returns:
        H          -- uint8 parity-check matrix, shape ``(n_detectors, n_error_mechanisms)``,
                      C-contiguous (row-major) to match C++/CUDA decoder expectations.
        obs_matrix -- uint8 observable-flip matrix, shape ``(n_observables, n_error_mechanisms)``.
        priors     -- float64 prior error probabilities, shape ``(n_error_mechanisms,)``.

    Each error instruction in the (flattened) DEM becomes one column: ``H[d, e]``
    is 1 iff error ``e`` flips detector ``d``, and ``obs_matrix[o, e]`` is 1 iff
    error ``e`` flips observable ``o``.
    """
    n_dets = dem.num_detectors
    n_obs = dem.num_observables

    error_cols: list[dict] = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        dets: list[int] = []
        obs: list[int] = []
        for t in instruction.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        error_cols.append({"p": p, "dets": dets, "obs": obs})

    n_err = len(error_cols)
    # Explicitly C-contiguous (row-major) to match decoder expectations,
    # equivalent to scipy sparse_matrix.todense(order='C').
    H = np.zeros((n_dets, n_err), dtype=np.uint8, order="C")
    obs_matrix = np.zeros((n_obs, n_err), dtype=np.uint8, order="C")
    priors = np.zeros(n_err, dtype=np.float64)

    for e, col in enumerate(error_cols):
        priors[e] = col["p"]
        # XOR, not assignment: stim treats a target listed an even number of
        # times as cancelling (parity), so e.g. an error listing D0 twice flips
        # D0 zero times. Circuit-generated DEMs don't repeat targets, but a
        # hand-written / externally-built DEM can, and assignment would mis-set
        # those entries to 1.
        for d in col["dets"]:
            H[d, e] ^= 1
        for o in col["obs"]:
            obs_matrix[o, e] ^= 1

    return np.ascontiguousarray(H), obs_matrix, priors
