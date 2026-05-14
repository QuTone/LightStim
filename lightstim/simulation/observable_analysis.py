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
from typing import List, Tuple, Set, Optional, Dict
import numpy as np
import stim


def _count_measurements_in_body(body: stim.Circuit) -> int:
    """Count measurement instructions in a single-level circuit (no recursion)."""
    count = 0
    for inst in body:
        if isinstance(inst, stim.CircuitInstruction):
            if inst.name in ('M', 'MX', 'MY', 'MR', 'MRX'):
                count += len(inst.targets_copy())
    return count


def build_obs_patch_matrix(
    circuit: stim.Circuit,
    system,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the observable-to-patch binary matrix.

    For each OBSERVABLE_INCLUDE instruction, determines which patches are
    involved by mapping measurement record references → qubit indices → patches.

    Args:
        circuit: Stim circuit with OBSERVABLE_INCLUDE instructions.
        system: QECSystem with index_to_owner_map.

    Returns:
        (matrix, patch_names):
            matrix: (num_obs × num_patches) GF(2) binary matrix.
            patch_names: ordered list of patch names (column labels).
    """
    # Step 1: Build absolute measurement index → qubit index mapping
    meas_to_qubit: Dict[int, int] = {}
    meas_counter = 0

    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction):
            if inst.name in ('M', 'MX', 'MY', 'MR', 'MRX'):
                for t in inst.targets_copy():
                    if t.is_qubit_target:
                        meas_to_qubit[meas_counter] = t.value
                        meas_counter += 1
        elif isinstance(inst, stim.CircuitRepeatBlock):
            body = inst.body_copy()
            body_meas_per_rep = _count_measurements_in_body(body)
            for _ in range(inst.repeat_count):
                for sub in body:
                    if isinstance(sub, stim.CircuitInstruction):
                        if sub.name in ('M', 'MX', 'MY', 'MR', 'MRX'):
                            for t in sub.targets_copy():
                                if t.is_qubit_target:
                                    meas_to_qubit[meas_counter] = t.value
                                    meas_counter += 1

    total_meas = meas_counter

    # Step 2: Collect patch names (columns) from the system
    patch_names = sorted(set(system.index_to_owner_map.values()))
    patch_to_col = {name: i for i, name in enumerate(patch_names)}

    # Step 3: For each OBSERVABLE_INCLUDE, resolve rec[-k] → qubit → patch
    obs_list = []
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == 'OBSERVABLE_INCLUDE':
            row = [0] * len(patch_names)
            for t in inst.targets_copy():
                # rec[-k] has t.value = -k (negative)
                abs_idx = total_meas + t.value
                qubit = meas_to_qubit.get(abs_idx)
                if qubit is not None:
                    patch = system.index_to_owner_map.get(qubit)
                    if patch is not None and patch in patch_to_col:
                        row[patch_to_col[patch]] = 1
            obs_list.append(row)

    matrix = np.array(obs_list, dtype=int) if obs_list else np.zeros((0, len(patch_names)), dtype=int)
    return matrix, patch_names


def identify_distillation_observables(
    obs_patch_matrix: np.ndarray,
    patch_names: List[str],
    target_patch_names: List[str],
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

    Returns:
        (T, target_indices, post_select_indices):
            T: (num_obs × num_obs) GF(2) transformation matrix.
            target_indices: observable indices for the distilled output.
            post_select_indices: observable indices for post-selection.
    """
    n_obs = obs_patch_matrix.shape[0]
    T = np.eye(n_obs, dtype=int)
    M = obs_patch_matrix.copy()

    for target_name in target_patch_names:
        if target_name not in patch_names:
            raise ValueError(f"Target patch '{target_name}' not found in patch_names: {patch_names}")
        col = patch_names.index(target_name)

        # Find pivot row (first row with 1 in target column)
        pivot = None
        for i in range(n_obs):
            if M[i, col] == 1:
                pivot = i
                break
        if pivot is None:
            raise ValueError(
                f"No observable involves target patch '{target_name}'. "
                f"Column {col} of obs_patch_matrix is all zeros."
            )

        # Eliminate all other rows with 1 in this column
        for i in range(n_obs):
            if i != pivot and M[i, col] == 1:
                M[i] = (M[i] + M[pivot]) % 2
                T[i] = (T[i] + T[pivot]) % 2

    # Classify: rows with any 1 in target columns → target, others → post-select
    target_cols = {patch_names.index(n) for n in target_patch_names}
    target_indices = [i for i in range(n_obs) if any(M[i, c] for c in target_cols)]
    ps_indices = [i for i in range(n_obs) if i not in target_indices]

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
