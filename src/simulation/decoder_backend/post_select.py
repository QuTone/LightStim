"""Post-selection utilities for detector data."""

from typing import List

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

    Iterates over circuit instructions, finds DETECTOR instructions with
    the given tag, and returns their indices (in order of appearance).

    Args:
        circuit: Stim circuit with DETECTOR instructions.
        tag: Tag string to look for (default "post-select").

    Returns:
        List of detector indices to use for post-selection.
    """
    indices = []
    det_count = 0
    for instruction in circuit:
        if instruction.name == "DETECTOR":
            # Check if instruction has the tag (stim may use .tag or similar)
            inst_tag = getattr(instruction, "tag", None)
            if inst_tag == tag or (isinstance(inst_tag, (list, tuple)) and tag in inst_tag):
                indices.append(det_count)
            det_count += 1
    return indices


def apply_post_selection(
    det_data: np.ndarray,
    obs_data: np.ndarray,
    post_select_indices: List[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter samples by post-selection: keep only rows where all
    post-select detectors are 0 (not flipped).

    Args:
        det_data: Shape (num_shots, num_detectors), binary detection events.
        obs_data: Shape (num_shots, num_observables), ground-truth observables.
        post_select_indices: Detector column indices used for post-selection.

    Returns:
        (filtered_det_data, filtered_obs_data) - only rows passing post-selection.
    """
    if not post_select_indices:
        return det_data, obs_data

    # Keep samples where all post-select detectors are 0
    mask = np.all(det_data[:, post_select_indices] == 0, axis=1)
    return det_data[mask], obs_data[mask]
