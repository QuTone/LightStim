"""Logical S experiments for the rotated surface code.

The logical phase gate is the dynamical fold-transversal protocol inserted
halfway through one syndrome-extraction round (arXiv:2412.01391).

Two verification circuits are provided:

``build_rotated_s_two_way_circuit``
    Noisy ``S`` followed by noisy ``S†``.  The round-trip preserves ``|+>``.
    As in the historical unrotated benchmark, preparation, padding, and
    readout are also noisy, so this is a full-circuit consistency metric rather
    than an isolated per-gate estimate.

``build_rotated_s_y_injection_circuit``
    A gate-only experiment.  A logical ``+Y`` state is injected and padded
    noiselessly, the selected dynamical phase gate is noisy, and the
    deterministic ``+X`` (for ``S†``) or ``-X`` (for ``S``) result is read
    out noiselessly.  No physical Pauli correction is needed because the
    sampler measures flips relative to the noiseless reference.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)

PhaseGate = Literal["S", "S_DAG"]


def _setup(
    distance: int,
) -> Tuple[
    QECSystem,
    RotatedSurfaceCode,
    CircuitBuilder,
    LogicalExecutor,
]:
    patch_local = RotatedSurfaceCode(distance=distance)
    system = QECSystem()
    patch = system.add_patch(patch_local, name="patch")

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(
        tracker=tracker,
        system_config=system,
        if_detector=True,
    )
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(
        RotatedSurfaceCode,
        RotatedSurfaceCodeLogicalOpSet(
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        ),
    )
    builder.write_coordinates()
    return system, patch, builder, executor


def _fully_noiseless_extraction_round(
    system: QECSystem,
) -> stim.Circuit:
    """Return an SE round whose gates *and* idle boundary are noiseless.

    ``CircuitBuilder.apply_syndrome_extraction(noiseless=True)`` tags the
    quantum instructions as noiseless, but deliberately preserves structural
    tags.  Consequently an ``SE_start`` TICK would still receive ``p_idle``
    from the circuit-level noise injector.  Replacing that marker locally
    keeps gate-only experiments from leaking idle noise into their injection
    and padding rounds.
    """
    ordinary = RotatedSurfaceCodeExtractionBlock(system).circuit
    result = stim.Circuit()
    for instruction in ordinary:
        if not isinstance(instruction, stim.CircuitInstruction):
            raise ValueError("Expected an explicit syndrome-extraction round.")
        tag = (
            "noiseless"
            if instruction.name == "TICK" and instruction.tag == "SE_start"
            else instruction.tag
        )
        result.append(
            instruction.name,
            instruction.targets_copy(),
            instruction.gate_args_copy(),
            tag=tag,
        )
    return result


def build_rotated_s_two_way_circuit(
    distance: int,
    rounds: int = 2,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """Build ``|+> -> S -> S† -> MX`` with both phase gates noisy.

    Ordinary syndrome-extraction padding is placed before, between, and after
    the two dynamical S-SE rounds.  Initialization, padding, both gates, and
    readout all receive ``noise_params`` when supplied.
    """
    if rounds < 0:
        raise ValueError(f"rounds must be non-negative, got {rounds}.")

    system, patch, builder, executor = _setup(distance)
    builder.initialize(
        {qubit: "X" for qubit in system.data_indices},
        system.num_qubits,
    )

    ordinary = RotatedSurfaceCodeExtractionBlock(system).circuit
    builder.apply_syndrome_extraction(ordinary, rounds=rounds)
    executor.apply_logical_operation("fold_transversal_s", [patch])
    builder.apply_syndrome_extraction(ordinary, rounds=rounds)
    executor.apply_logical_operation("fold_transversal_s_dag", [patch])
    builder.apply_syndrome_extraction(ordinary, rounds=rounds)
    builder.apply_data_readout(
        {qubit: "X" for qubit in system.data_indices},
    )

    if noise_params is None:
        return builder.circuit
    return builder.build_noisy_circuit(noise_params, noise_model)


def build_rotated_s_y_injection_circuit(
    distance: int,
    gate: PhaseGate = "S_DAG",
    padding_rounds: int = 2,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
    injection_protocol: Literal["corner", "middle"] = "corner",
) -> stim.Circuit:
    """Build a gate-only ``+Y -> S/S† -> ∓X`` verification circuit.

    The same injected ``+Y`` state is used for both gates.  ``S_DAG`` produces
    deterministic ``+X`` and ``S`` produces deterministic ``-X``.  The
    injection, identity-SE padding, and final X readout are noiseless.  When
    ``noise_params`` is supplied, only the selected dynamical S-SE round
    receives circuit-level noise.
    """
    gate = gate.upper()
    if gate not in ("S", "S_DAG"):
        raise ValueError(f"gate must be 'S' or 'S_DAG', got {gate!r}.")
    if padding_rounds < 0:
        raise ValueError(
            f"padding_rounds must be non-negative, got {padding_rounds}."
        )

    system, patch, builder, executor = _setup(distance)
    executor.apply_logical_operation(
        "state_injection",
        [patch],
        inject_state="Y",
        protocol=injection_protocol,
        rounds=0,
        post_select_coords=set(),
        noiseless_init=True,
    )

    noiseless_round = _fully_noiseless_extraction_round(system)
    # One projection round completes the noiseless product-state injection.
    builder.apply_syndrome_extraction(
        noiseless_round,
        rounds=1,
        noiseless=True,
    )
    builder.apply_syndrome_extraction(
        noiseless_round,
        rounds=padding_rounds,
        noiseless=True,
    )
    executor.apply_logical_operation(
        "fold_transversal_s" if gate == "S" else "fold_transversal_s_dag",
        [patch],
    )
    builder.apply_syndrome_extraction(
        noiseless_round,
        rounds=padding_rounds,
        noiseless=True,
    )
    builder.apply_data_readout(
        {qubit: "X" for qubit in system.data_indices},
        noiseless=True,
    )

    if noise_params is None:
        return builder.circuit
    return builder.build_noisy_circuit(noise_params, noise_model)


__all__ = [
    "build_rotated_s_two_way_circuit",
    "build_rotated_s_y_injection_circuit",
]
