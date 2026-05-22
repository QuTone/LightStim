"""Post-selection utilities for detector data."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import stim

# Must match tracker.POST_SELECT_TAG
DEFAULT_POST_SELECT_TAG = "post-select"


def get_post_select_detector_indices(
    circuit: stim.Circuit,
    tag: str = DEFAULT_POST_SELECT_TAG,
) -> List[int]:
    """
    Extract detector indices that have the post-select tag.

    Recursively walks the circuit (including REPEAT blocks) and returns the
    absolute detector indices of every DETECTOR instruction carrying the given
    tag, in order of appearance.

    Args:
        circuit: Stim circuit with DETECTOR instructions.
        tag: Tag string to look for (default "post-select").

    Returns:
        List of detector indices to use for post-selection.
    """
    indices = []
    det_count = [0]

    def _scan(circ: stim.Circuit) -> None:
        for instruction in circ:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                body = instruction.body_copy()
                for _ in range(instruction.repeat_count):
                    _scan(body)
            elif instruction.name == "DETECTOR":
                inst_tag = getattr(instruction, "tag", None)
                if inst_tag == tag or (isinstance(inst_tag, (list, tuple)) and tag in inst_tag):
                    indices.append(det_count[0])
                det_count[0] += 1

    _scan(circuit)
    return indices


def apply_post_selection(
    det_data: np.ndarray,
    obs_data: np.ndarray,
    post_select_indices: List[int],
    post_select_observable_indices: Optional[List[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter samples by post-selection: keep only rows where all
    post-select detectors AND observables are 0 (not flipped).

    Args:
        det_data: Shape (num_shots, num_detectors), binary detection events.
        obs_data: Shape (num_shots, num_observables), ground-truth observables.
        post_select_indices: Detector column indices used for post-selection.
        post_select_observable_indices: Observable column indices for post-selection.
            Shots where any of these observables is flipped are discarded.

    Returns:
        (filtered_det_data, filtered_obs_data) - only rows passing post-selection.
    """
    mask = np.ones(det_data.shape[0], dtype=bool)

    if post_select_indices:
        mask &= np.all(det_data[:, post_select_indices] == 0, axis=1)

    if post_select_observable_indices:
        mask &= np.all(obs_data[:, post_select_observable_indices] == 0, axis=1)

    return det_data[mask], obs_data[mask]
