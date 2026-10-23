"""
Observable analysis for distillation circuits.

Provides utilities to:
1. Build an obs-to-patch binary matrix from a circuit + QECSystem
2. Identify target (distilled output) vs post-select (outer-code stabilizer)
   observables via GF(2) Gaussian elimination
3. Apply the GF(2) transformation to sampled observable data

This generalizes the manual observable assignment used in LS_distillation
(where each observable maps to exactly 1 patch) to the TG_distillation case
(where observables span multiple patches and need GF(2) row operations).
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
import stim

from lightstim.ir.logical_history import (
    logical_history_observable_indices as _logical_history_observable_indices,
)


def build_obs_patch_matrix(
    circuit: stim.Circuit,
    system,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the observable-to-patch binary matrix.

    For each observable ID, combines all of its OBSERVABLE_INCLUDE contributions
    by XOR and maps the remaining records to qubits and patches. References are
    resolved at the annotation's location, including inside nested repeats.
    This is a support summary, not a proof of an observable's logical role.

    Args:
        circuit: Stim circuit with OBSERVABLE_INCLUDE instructions.
        system: QECSystem with index_to_owner_map.

    Returns:
        (matrix, patch_names):
            matrix: (num_obs × num_patches) GF(2) binary matrix.
            patch_names: ordered list of patch names (column labels).
    """
    # Resolve each annotation at its own circuit location. Several instructions
    # may contribute to one observable, including annotations inside REPEAT.
    meas_to_qubit: Dict[int, int] = {}
    meas_counter = 0
    records_by_observable = [set() for _ in range(circuit.num_observables)]
    for inst in circuit.flattened():
        if inst.name in ('M', 'MX', 'MY', 'MR', 'MRX', 'MRY'):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    meas_to_qubit[meas_counter] = t.value
                    meas_counter += 1
        elif inst.num_measurements:
            raise ValueError(
                f"Patch-support analysis does not support {inst.name} measurements."
            )
        elif inst.name == 'OBSERVABLE_INCLUDE':
            row = records_by_observable[int(inst.gate_args_copy()[0])]
            for t in inst.targets_copy():
                if not t.is_measurement_record_target:
                    raise ValueError("Patch-support analysis requires record-only observables.")
                absolute = meas_counter + t.value
                if not 0 <= absolute < meas_counter:
                    raise ValueError("Observable refers to an unavailable measurement.")
                row.symmetric_difference_update((absolute,))

    # Step 2: Collect patch names (columns) from the system
    patch_names = sorted(set(system.index_to_owner_map.values()))
    patch_to_col = {name: i for i, name in enumerate(patch_names)}

    matrix = np.zeros((circuit.num_observables, len(patch_names)), dtype=int)
    for observable, records in enumerate(records_by_observable):
        for record in records:
            patch = system.index_to_owner_map.get(meas_to_qubit[record])
            if patch in patch_to_col:
                matrix[observable, patch_to_col[patch]] = 1
    return matrix, patch_names


def logical_history_observable_indices(circuit: stim.Circuit) -> List[int]:
    """IDs of automatically exported history, distinct from protocol targets.

    A deterministic history relation is not automatically a distillation
    acceptance check. Distillation protocols use this list to preserve their
    specified output and outer-code postselection policy.
    """
    return list(_logical_history_observable_indices(circuit))


def identify_distillation_observables(
    obs_patch_matrix: np.ndarray,
    patch_names: List[str],
    target_patch_names: List[str],
    *,
    excluded_observable_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Identify target and post-select observables via GF(2) Gaussian elimination.

    Performs column elimination on the target patch columns so that exactly one
    observable row has support on the target patches (the distilled output),
    and all others have zero in those columns (outer-code stabilizers → post-select).

    Args:
        obs_patch_matrix: (num_obs × num_patches) GF(2) binary matrix.
        patch_names: ordered list of patch names (column labels).
        target_patch_names: patch name(s) that define the distillation output.
        excluded_observable_indices: IDs retained in the circuit but excluded
            from this protocol's output and acceptance policy, e.g. diagnostic
            logical-history relations. Their rows remain unchanged in T.

    Returns:
        (T, target_indices, post_select_indices):
            T: (num_obs × num_obs) GF(2) transformation matrix.
            target_indices: observable indices for the distilled output.
            post_select_indices: observable indices for post-selection.
    """
    n_obs = obs_patch_matrix.shape[0]
    excluded = set(excluded_observable_indices or ())
    if any(not 0 <= i < n_obs for i in excluded):
        raise ValueError("Excluded observable index is outside the observable matrix.")
    eligible = [i for i in range(n_obs) if i not in excluded]
    T = np.eye(n_obs, dtype=int)
    M = obs_patch_matrix.copy()

    for target_name in target_patch_names:
        if target_name not in patch_names:
            raise ValueError(f"Target patch '{target_name}' not found in patch_names: {patch_names}")
        col = patch_names.index(target_name)

        # Find pivot row (first row with 1 in target column)
        pivot = None
        for i in eligible:
            if M[i, col] == 1:
                pivot = i
                break
        if pivot is None:
            raise ValueError(
                f"No observable involves target patch '{target_name}'. "
                f"Column {col} of obs_patch_matrix is all zeros."
            )

        # Eliminate all other rows with 1 in this column
        for i in eligible:
            if i != pivot and M[i, col] == 1:
                M[i] = (M[i] + M[pivot]) % 2
                T[i] = (T[i] + T[pivot]) % 2

    # Classify: rows with any 1 in target columns → target, others → post-select
    target_cols = {patch_names.index(n) for n in target_patch_names}
    target_indices = [i for i in eligible if any(M[i, c] for c in target_cols)]
    ps_indices = [i for i in eligible if i not in target_indices]

    return T, target_indices, ps_indices


def transform_observables(
    obs_data: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """
    Apply GF(2) transformation to observable data.

    Args:
        obs_data: (shots × num_obs) binary array.
        T: (num_obs × num_obs) GF(2) transformation matrix.

    Returns:
        Transformed observable data (shots × num_obs).
    """
    return (obs_data @ T.T) % 2
