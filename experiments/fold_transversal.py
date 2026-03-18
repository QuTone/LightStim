# experiments/fold_transversal.py

"""
Fold-transversal gate verification experiments for Unrotated and Rotated Surface Codes.

Provides circuit builders for:
- Single-patch gate verification (H, S) via init → SE → gate → SE → readout
- Bell pair verification (transversal CNOT) via H → CNOT → Bell readout
- S·S† roundtrip for fault-tolerant S gate benchmarking
"""

import stim
from typing import List, Literal, Optional

from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)
from src.ir.qec_system import QECSystem
from src.ir.tracker import SyndromeTracker
from src.ir.builder import CircuitBuilder
from src.ir.logical_executor import LogicalExecutor
from src.noise.config import NoiseConfig


def build_gate_verification_circuit(
    distance: int,
    gates: List[str],
    init_basis: Literal["Z", "X"] = "X",
    measure_basis: Literal["Z", "X", "Y"] = "Y",
    rounds: int = 2,
    unencode: bool = False,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a single-patch circuit for gate verification:
      init(basis) → SE → gate(s) → SE → readout

    When unencode=True, uses injection-style diagonal unencode + single-qubit
    measurement on the corner (for Y-basis logical readout).
    When unencode=False, uses transversal readout in measure_basis.

    Args:
        distance:      Code distance (must be odd, square patch).
        gates:         List of logical gate names (e.g. ['fold_transversal_s']).
        init_basis:    'Z' or 'X' for all data qubits.
        measure_basis: Final measurement basis ('Z', 'X', or 'Y').
        rounds:        SE rounds before and after gates.
        unencode:      If True, use diagonal unencode + single-qubit measurement.
        noise_params:  Optional NoiseConfig for noise injection.
        noise_model:   Noise model string (default 'circuit_level').

    Returns:
        stim.Circuit
    """
    patch_local = UnrotatedSurfaceCode(distance=distance)
    system = QECSystem()
    patch = system.add_patch(patch_local, name="patch")

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(UnrotatedSurfaceCode, UnrotatedSurfaceCodeLogicalOpSet())

    builder.write_coordinates()
    builder.initialize(
        {q: init_basis for q in system.data_indices},
        system.num_qubits,
    )

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    for gate in gates:
        executor.apply_logical_operation(gate, [patch])

    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    if unencode:
        # Injection-style diagonal unencode + single-qubit measurement
        data_global = sorted(patch.data_indices)
        coords_with_idx = [(system.qubit_coords[g], g) for g in data_global]
        corner_coord, corner_gidx = min(coords_with_idx, key=lambda t: (t[0][0], t[0][1]))
        ox, oy = corner_coord

        unencode_measurements = {}
        for gidx in data_global:
            if gidx == corner_gidx:
                continue
            cx, cy = system.qubit_coords[gidx]
            rel_x, rel_y = cx - ox, cy - oy
            if rel_y >= rel_x:
                unencode_measurements[gidx] = "X"
            else:
                unencode_measurements[gidx] = "Z"

        builder.apply_data_readout(
            final_measurements=unencode_measurements, noiseless=True
        )
        builder.apply_data_readout(
            final_measurements={corner_gidx: measure_basis}, noiseless=True
        )
    else:
        builder.apply_data_readout(
            {q: measure_basis for q in system.data_indices}
        )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit


def build_s_roundtrip_circuit(
    distance: int,
    rounds: int = 2,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a fault-tolerant S gate verification circuit via S·S† roundtrip:

      |+⟩ → SE → S_L → SE → S†_L → SE → transversal MX

    S·S† = I, so X_L should remain +1. Transversal MX readout provides full
    final-round detector coverage. LER_per_gate ≈ total_LER / 2.

    Args:
        distance:      Code distance (must be odd, square patch).
        rounds:        SE rounds between operations.
        noise_params:  Optional NoiseConfig for noise injection.
        noise_model:   Noise model string (default 'circuit_level').

    Returns:
        stim.Circuit
    """
    patch_local = UnrotatedSurfaceCode(distance=distance)
    system = QECSystem()
    patch = system.add_patch(patch_local, name="patch")

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(UnrotatedSurfaceCode, UnrotatedSurfaceCodeLogicalOpSet())

    builder.write_coordinates()
    builder.initialize(
        {q: "X" for q in system.data_indices},
        system.num_qubits,
    )

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)

    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    executor.apply_logical_operation("fold_transversal_s", [patch])
    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    executor.apply_logical_operation("fold_transversal_s_dag", [patch])
    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    builder.apply_data_readout(
        {q: "X" for q in system.data_indices}
    )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit


def build_bell_circuit(
    distance: int,
    measure_basis: Literal["Z", "X"] = "Z",
    rounds: int = 2,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a Bell pair circuit for two unrotated surface code patches.

    |00⟩ → SE → H_L(p1) → SE → CNOT_L(p1,p2) → SE → measure(basis)

    For |Φ+⟩ = (|00⟩+|11⟩)/√2:
    - Z readout: Z_L⊗Z_L = +1 (detects ZZ logical errors)
    - X readout: X_L⊗X_L = +1 (detects XX logical errors)

    Args:
        distance:      Code distance.
        measure_basis: 'Z' or 'X' for final data readout on both patches.
        rounds:        SE rounds between logical operations.
        noise_params:  Optional NoiseConfig for noise injection.
        noise_model:   Noise model string (default 'circuit_level').

    Returns:
        stim.Circuit
    """
    offset = (2 * distance + 2, 0)

    p1_local = UnrotatedSurfaceCode(distance=distance)
    p2_local = UnrotatedSurfaceCode(distance=distance)

    system = QECSystem()
    p1 = system.add_patch(p1_local, name="p1")
    p2 = system.add_patch(p2_local, name="p2", offset=offset)

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(UnrotatedSurfaceCode, UnrotatedSurfaceCodeLogicalOpSet())

    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)

    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    executor.apply_logical_operation("fold_transversal_hadamard", [p1])
    # builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    executor.apply_logical_operation("transversal_cnot", [p1, p2])
    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds)

    builder.apply_data_readout(
        {q: measure_basis for q in system.data_indices}
    )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit


# ==============================================================================
# Rotated Surface Code
# ==============================================================================

def build_rotated_gate_verification_circuit(
    distance: int,
    gates: List[str],
    init_basis: Literal["Z", "X"] = "X",
    measure_basis: Literal["Z", "X", "Y"] = "Y",
    rounds: int = 2,
    unencode: bool = False,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a single rotated-patch circuit for gate verification:
      init(basis) → SE → gate(s) → SE → readout

    For 'fold_transversal_s' / 'fold_transversal_s_dag' the gate IS a modified
    SE round (embedded), so the gate call replaces one SE round.
    When unencode=True, uses diagonal unencode + single-qubit Y measurement
    on the corner qubit (for Y-basis logical readout).

    Args:
        distance:      Code distance (must be odd, square patch).
        gates:         List of logical gate names.
        init_basis:    'Z' or 'X'.
        measure_basis: Final measurement basis ('Z', 'X', or 'Y').
        rounds:        SE rounds before and after each gate.
        unencode:      If True, use diagonal unencode + corner measurement.
        noise_params:  Optional NoiseConfig for noise injection.
        noise_model:   Noise model string (default 'circuit_level').

    Returns:
        stim.Circuit
    """
    patch_local = RotatedSurfaceCode(distance=distance)
    system = QECSystem()
    patch = system.add_patch(patch_local, name="patch")

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(RotatedSurfaceCode, RotatedSurfaceCodeLogicalOpSet())

    se = RotatedSurfaceCodeExtractionBlock(system)

    # When unencode=True, use the injection-style diagonal init so that the
    # unencode measurements are deterministic: corner gets init_basis, lower
    # diagonal (rel_y >= rel_x from corner) gets X, upper diagonal gets Z.
    # This matches the logical_unencode expectation in RotatedSurfaceCodeLogicalOpSet.
    builder.write_coordinates()
    if unencode:
        data_global = sorted(patch.data_indices)
        coords_with_idx = [(system.qubit_coords[g], g) for g in data_global]
        corner_coord, corner_gidx = min(coords_with_idx, key=lambda t: (t[0][0], t[0][1]))
        ox, oy = corner_coord
        init_dict = {}
        for gidx in data_global:
            cx, cy = system.qubit_coords[gidx]
            rel_x, rel_y = cx - ox, cy - oy
            if gidx == corner_gidx:
                init_dict[gidx] = init_basis
            elif rel_y >= rel_x:
                init_dict[gidx] = "X"
            else:
                init_dict[gidx] = "Z"
    else:
        corner_gidx = None
        init_dict = {q: init_basis for q in system.data_indices}

    builder.initialize(init_dict, system.num_qubits)
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    for gate in gates:
        if gate in ("fold_transversal_s", "fold_transversal_s_dag"):
            executor.apply_logical_operation(gate, [patch], se_block=se)
        else:
            executor.apply_logical_operation(gate, [patch])

    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    if unencode:
        unencode_measurements = {
            gidx: basis for gidx, basis in init_dict.items()
            if gidx != corner_gidx
        }
        builder.apply_data_readout(
            final_measurements=unencode_measurements, noiseless=True
        )
        builder.apply_data_readout(
            final_measurements={corner_gidx: measure_basis}, noiseless=True
        )
    else:
        builder.apply_data_readout(
            {q: measure_basis for q in system.data_indices}
        )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit


def build_rotated_s_roundtrip_circuit(
    distance: int,
    rounds: int = 2,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a fault-tolerant S gate verification circuit via S·S† roundtrip:

      |0⟩ → SE → S_L → SE → S†_L → SE → transversal MZ

    S·S† = I, so Z_L should remain +1. Z-initialization avoids the boundary
    X-syndrome tracking issue that occurs with X-initialized states (see
    boundary syndrome note in fold_transversal rotated SC tests). LER_per_gate
    ≈ total_LER / 2.

    Args:
        distance:      Code distance (must be odd, square patch).
        rounds:        SE rounds between operations.
        noise_params:  Optional NoiseConfig for noise injection.
        noise_model:   Noise model string (default 'circuit_level').

    Returns:
        stim.Circuit
    """
    patch_local = RotatedSurfaceCode(distance=distance)
    system = QECSystem()
    patch = system.add_patch(patch_local, name="patch")

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(RotatedSurfaceCode, RotatedSurfaceCodeLogicalOpSet())

    se = RotatedSurfaceCodeExtractionBlock(system)

    builder.write_coordinates()
    builder.initialize(
        {q: "Z" for q in system.data_indices},
        system.num_qubits,
    )
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    executor.apply_logical_operation("fold_transversal_s", [patch], se_block=se)
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    executor.apply_logical_operation("fold_transversal_s_dag", [patch], se_block=se)
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    builder.apply_data_readout(
        {q: "Z" for q in system.data_indices}
    )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit


def build_rotated_bell_circuit(
    distance: int,
    measure_basis: Literal["Z", "X"] = "Z",
    rounds: int = 2,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a Bell pair circuit for two rotated surface code patches:

      |00⟩ → SE → H_L(p1) → SE → CNOT_L(p1,p2) → SE → measure(basis)

    For |Φ+⟩ the joint parity observable (Z_L⊗Z_L or X_L⊗X_L) = +1 always.

    Args:
        distance:      Code distance.
        measure_basis: 'Z' or 'X' for final data readout on both patches.
        rounds:        SE rounds between logical operations.
        noise_params:  Optional NoiseConfig for noise injection.
        noise_model:   Noise model string (default 'circuit_level').

    Returns:
        stim.Circuit
    """
    offset = (2 * distance + 4, 0)

    p1_local = RotatedSurfaceCode(distance=distance)
    p2_local = RotatedSurfaceCode(distance=distance)

    system = QECSystem()
    p1 = system.add_patch(p1_local, name="p1")
    p2 = system.add_patch(p2_local, name="p2", offset=offset)

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(RotatedSurfaceCode, RotatedSurfaceCodeLogicalOpSet())

    se = RotatedSurfaceCodeExtractionBlock(system)

    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    executor.apply_logical_operation("fold_transversal_hadamard", [p1])
    executor.apply_logical_operation("transversal_cnot", [p1, p2])
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)

    builder.apply_data_readout(
        {q: measure_basis for q in system.data_indices}
    )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit
