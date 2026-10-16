"""Selection for the persistent record engine adapted from Hou Yuewen's PR 110.

The engine owns physical Clifford coordinates and record arithmetic. Tracker
role classification and output policy remain in the merged Python tracker.
This experimental integration is explicit opt-in until validated and timed.
"""

import os


def record_state_class():
    mode = os.environ.get("LIGHTSTIM_TRACKER_BACKEND", "reference").strip().lower()
    if mode not in {"reference", "persistent"}:
        raise ValueError("LIGHTSTIM_TRACKER_BACKEND must be 'reference' or 'persistent'; "
                         f"got {mode!r}")
    if mode == "reference":
        return None
    try:
        from .cpp._pauli_backend_cpp import PersistentRecordState
    except (ImportError, OSError) as error:
        raise ImportError(
            "LIGHTSTIM_TRACKER_BACKEND=persistent requires the native record "
            "extension; rebuild LightStim with the pinned Stim sources."
        ) from error
    return PersistentRecordState
