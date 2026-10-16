"""Selection for the persistent record engine adapted from Hou Yuewen's PR 110.

The engine owns physical Clifford coordinates, records and the packed execution
of the current Tracker's role/closure rules. The reference path remains an
independent policy oracle; native execution is explicitly selected.
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
        from .cpp._pauli_backend_cpp import PersistentRecordState, PackedRows
        required = ("commuting_relation", "coordinate_rows", "fold_rows", "trace_qubits",
                    "begin_measurement_batch", "measurement_anticommuting", "measurement_relation")
        if not all(hasattr(PersistentRecordState, name) for name in required):
            raise ImportError("The installed native Tracker extension is from an older source revision")
    except (ImportError, OSError) as error:
        raise ImportError(
            "LIGHTSTIM_TRACKER_BACKEND=persistent requires the matching native "
            "Tracker extension; rebuild LightStim with the pinned Stim sources."
        ) from error
    return PersistentRecordState
