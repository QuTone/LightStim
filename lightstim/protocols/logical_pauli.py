# protocols/logical_pauli.py

"""
Logical Pauli memory experiment (Surface Code Handbook §7.1).

A memory experiment with logical Pauli operator layer(s) inserted mid-circuit:

  init(basis) → SE×⌈r/2⌉ → P̄ layer ×N → SE×⌊r/2⌋ → readout(basis)

Two modes build tick-identical clean circuits:

  physical — the Pauli string is applied as real gates; circuit-level noise
             injection adds single-qubit gate noise on the string qubits.
  frame    — the same layer is tagged 'noiseless' (skipped by noise
             injection): the circuit-model equivalent of tracking the
             operator classically in a Pauli frame, at zero physical cost.

The LER difference between the two modes therefore isolates exactly the
extra error locations introduced by applying the operator physically —
within this repo's noise convention, where idle noise is lumped once per
SE round: the layer tick adds gate noise on the weight-d string support
only, with no spectator idling and no extension of the experiment
duration. Under a time-step-resolved noise model the physical arm would
also accrue spectator idle noise during each layer, so the measured
per-layer cost is a lower bound on the real physical cost.
num_layers stacks N consecutive layers (one tick each) to amplify the
per-layer cost: LER vs N has positive slope for 'physical' and zero slope
for 'frame'.
"""

import stim
from typing import Literal, Optional, Type

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.noise.config import NoiseConfig


def build_pauli_memory_circuit(
    code_patch_class: Type,
    extraction_block_class: Type,
    code_params: dict,
    pauli: Literal["X", "Z"] = "X",
    mode: Literal["physical", "frame"] = "physical",
    num_layers: int = 1,
    rounds: int = 3,
    basis: Literal["Z", "X"] = "Z",
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
) -> stim.Circuit:
    """
    Build a memory circuit with mid-circuit logical Pauli layer(s).

    Args:
        code_patch_class:        QECPatch subclass (e.g. RotatedSurfaceCode).
        extraction_block_class:  Matching SE block class.
        code_params:             Constructor kwargs for the patch (e.g. {"distance": 3}).
        pauli:                   Which logical operator to apply ('X' or 'Z').
        mode:                    'physical' (noisy gates) or 'frame' (noiseless tag).
        num_layers:              Number of consecutive Pauli layers (one tick each).
        rounds:                  Total SE rounds, split ⌈r/2⌉ before / ⌊r/2⌋ after the layers.
        basis:                   Memory basis for init and readout.
        noise_params:            Optional NoiseConfig; None returns the clean circuit.
        noise_model:             Noise model strategy string.

    Returns:
        stim.Circuit
    """
    if mode not in ("physical", "frame"):
        raise ValueError(f"mode must be 'physical' or 'frame', got {mode!r}")
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")
    basis = basis.upper()

    system = QECSystem()
    patch = system.add_patch(code_patch_class(**code_params), name="patch")

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(code_patch_class, CSSLogicalOpSet())

    builder.write_coordinates()
    builder.initialize(
        {q: basis for q in system.data_indices},
        system.num_qubits,
    )

    se_block = extraction_block_class(system)
    rounds_before = rounds - rounds // 2
    rounds_after = rounds // 2

    builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds_before)

    for _ in range(num_layers):
        executor.apply_logical_operation(
            "logical_pauli", [patch],
            pauli=pauli, noiseless=(mode == "frame"),
        )

    if rounds_after > 0:
        builder.apply_syndrome_extraction(se_block.circuit, rounds=rounds_after)

    builder.apply_data_readout({q: basis for q in system.data_indices})

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit
