"""Shared shot-accounting for the custom decode loop.

Both the single-process loop (:mod:`pipeline`) and the multi-process worker
(:mod:`worker`) turn a batch of decoder predictions into a ``(kept, errors)``
pair. Keeping that arithmetic in one place stops the two paths from drifting
apart, and gives :class:`~.external.ExternalDecoder` failure flags a single
place to be honoured.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def count_batch(
    *,
    obs_filtered: np.ndarray,
    pred_packed: np.ndarray,
    post_select_corrected_observable_indices: Optional[List[int]],
    target_observable_indices: Optional[List[int]],
    flags: Optional[np.ndarray] = None,
    on_decode_failure: str = "error",
    observable_failure_mode: str = "any",
) -> tuple[int, int]:
    """Return ``(kept, errors)`` for one decoded batch.

    Args:
        obs_filtered: uint8 ground-truth observables, shape ``(n_shots, n_obs)``
            (already pre-decode post-selected).
        pred_packed: bit-packed observable predictions from the decoder,
            little-endian, one row per shot.
        post_select_corrected_observable_indices: if set, post-decode
            post-selection — keep only shots whose *corrected* observables at
            these indices are all 0.
        target_observable_indices: if set, only these observables count toward
            a logical error.
        flags: optional per-shot convergence flags from an external decoder,
            shape ``(n_shots,)``; ``False`` marks a failed decode. ``None`` means
            every shot converged.
        on_decode_failure: policy for flagged-failed shots — ``"error"`` counts
            them as logical errors, ``"discard"`` drops them from the
            denominator, ``"ignore"`` leaves the prediction untouched.
        observable_failure_mode: Count any target flip or the parity of all
            target flips, after recovery.

    Returns:
        ``(kept, errors)`` deltas to add to the running denominator and error
        count.
    """
    n_shots, n_obs = obs_filtered.shape
    pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]

    ps_corr = post_select_corrected_observable_indices
    target = target_observable_indices

    corrected = obs_filtered ^ pred_unpacked
    keep_mask = np.all(corrected[:, ps_corr] == 0, axis=1) if ps_corr else np.ones(n_shots, dtype=bool)
    selected = corrected[:, target] if target is not None else corrected
    if observable_failure_mode == "any":
        error_mask = np.any(selected, axis=1)
    elif observable_failure_mode == "parity":
        error_mask = np.logical_xor.reduce(selected, axis=1)
    else:
        raise ValueError(f"Unknown observable_failure_mode {observable_failure_mode!r}")

    if flags is not None and on_decode_failure != "ignore":
        failed = ~np.asarray(flags, dtype=bool)
        if on_decode_failure == "discard":
            keep_mask = keep_mask & ~failed
        else:  # "error"
            # This policy counts a failed decode as an error. It must stay in
            # the denominator even if post-decode post-selection would reject its
            # (untrusted) corrected observables — otherwise a failed-and-rejected
            # shot would silently vanish (kept=0, errors=0) instead of counting,
            # underestimating the LER.
            keep_mask = keep_mask | failed
            error_mask = error_mask | failed

    kept = int(keep_mask.sum())
    errors = int(np.sum(error_mask & keep_mask))
    return kept, errors
