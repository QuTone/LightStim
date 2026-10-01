"""Verification circuits for logical gates on hypergraph-product patches.

Mirror of :func:`lightstim.protocols.fold_transversal.build_gate_verification_circuit`
for :class:`HGPCode` patches::

    init(basis) → SE rounds → logical gate(s) → SE rounds → transversal readout

A gate is addressed as ``(op_name, kwargs)`` where ``op_name`` is a method of
:class:`HGPCodeLogicalOpSet` and ``kwargs`` a dict of its keyword arguments,
e.g. ``("fold_transversal_h_swap", {})`` or
``("fold_transversal_h_swap", {"noisy_swap": False})``.  The tracker starts
from one logical operator per logical qubit (Z̄ for ``init_basis="Z"``, X̄
for ``"X"``), pushes each through the gates, and emits one observable for
every pushed operator that the transversal readout resolves.  With
``init_basis="Z"`` and ``measure_basis="X"`` an odd number of H-SWAPs makes
every logical qubit resolvable; a circuit in which nothing is resolvable is
rejected with ValueError.

What this harness does and does not verify
------------------------------------------
* It checks that the gates map the code's stabilizer group onto itself: every
  syndrome-extraction round after the gates must emit one detector per
  stabilizer, exactly like a steady-state memory round.  The tracker does not
  fail on its own here: a stabilizer that no longer matches the state after a
  gate is silently treated as a fresh projection and simply emits no detector
  in the first post-gate round.  The harness therefore counts the post-gate
  detectors and rejects the circuit with ValueError when the count is short
  (e.g. the H layer of the H-SWAP without its SWAP layer).
* Together with ``assert_noiseless`` on the returned circuit it checks that
  the resolved logical operators are deterministic.  Note that stim's
  detector sampler reports observable flips relative to a noiseless
  reference, so a residual logical Pauli is invisible to it; use the raw
  measurement parity of the observables to check the sign (see
  ``tests/test_hgp_fold_h_swap.py::_raw_observable_parity``).
* It does not check the exact Clifford action on each logical qubit; the
  stim-tableau tests do that.
* Observable ``k`` is the tracker's row order, not logical id ``k``; do not
  read the observable index as a logical id.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Type

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.HGP import (
    HGPCode,
    HGPCodeLogicalOpSet,
    HGPProductColorationExtractionBlock,
)


GateSpec = Tuple[str, dict]


def build_hgp_gate_verification_circuit(
    patch: HGPCode,
    gates: Sequence[GateSpec],
    init_basis: Literal["Z", "X"] = "Z",
    measure_basis: Literal["Z", "X"] = "X",
    rounds: int = 2,
    extraction_block_class: Type = HGPProductColorationExtractionBlock,
    se_block_kwargs: Optional[dict] = None,
    noise_params: Optional[NoiseConfig] = None,
    noise_model: str = "circuit_level",
    noiseless_gates: bool = False,
    gate_kwargs: Optional[dict] = None,
    patch_name: str = "hgp",
) -> stim.Circuit:
    """Build a single-patch HGP circuit that applies logical gates.

    Args:
        patch: Local HGPCode patch (it is placed into a fresh QECSystem).
        gates: Sequence of ``(op_name, kwargs)`` applied in order.
        init_basis: Transversal initialization basis of all data qubits.
        measure_basis: Transversal readout basis of all data qubits.
        rounds: Syndrome-extraction rounds before and after the gates.
        extraction_block_class: Extraction block class for this patch.
        se_block_kwargs: Extra keyword arguments for the extraction block.
        noise_params: Optional NoiseConfig; None returns the clean circuit.
        noise_model: Noise model string (default 'circuit_level').
        noiseless_gates: If True, tag the logical-gate layers as noiseless.
        gate_kwargs: Extra keyword arguments passed to every gate call, e.g.
            ``{"noisy_swap": False}``.  ``noiseless`` is not allowed here;
            use ``noiseless_gates``.
        patch_name: Name of the patch inside the QECSystem.

    Returns:
        stim.Circuit

    Raises:
        ValueError: on invalid bases/rounds/gate specs, when no logical is
            resolvable by the readout, or when the gates do not preserve the
            stabilizer group (a post-gate SE round emitted fewer detectors
            than there are stabilizers).
    """
    for name, basis in (("init_basis", init_basis), ("measure_basis", measure_basis)):
        if basis not in ("X", "Z"):
            raise ValueError(f"{name} must be 'X' or 'Z', got {basis!r}.")
    if rounds < 1:
        raise ValueError(
            f"rounds must be >= 1 so the tracker can establish the logical "
            f"operators before and after the gates; got {rounds}."
        )
    if gate_kwargs and "noiseless" in gate_kwargs:
        raise ValueError(
            "Pass noiseless_gates=... instead of gate_kwargs={'noiseless': ...}."
        )
    for op_name, target in gates:
        if not isinstance(target, dict):
            raise TypeError(
                f"Gate {op_name!r}: the gate spec must be (op_name, kwargs dict), "
                f"got {target!r}."
            )
        if "noiseless" in target:
            raise ValueError(
                f"Gate {op_name!r}: pass noiseless_gates=... instead of a "
                "'noiseless' entry in the gate spec."
            )
        clash = sorted(set(target) & set(gate_kwargs or {}))
        if clash:
            raise ValueError(
                f"Gate {op_name!r}: keys {clash} appear both in the gate spec "
                "and in gate_kwargs."
            )

    system = QECSystem()
    global_patch = system.add_patch(patch, name=patch_name)

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    executor = LogicalExecutor(builder)
    executor.register_op_set(
        HGPCode, HGPCodeLogicalOpSet(extraction_block_class=extraction_block_class)
    )

    builder.write_coordinates()
    se_block = extraction_block_class(system, **(se_block_kwargs or {}))
    measurement_blocks = getattr(se_block, "measurement_blocks", None)

    data_indices = sorted(system.data_indices)
    builder.initialize({q: init_basis for q in data_indices}, system.num_qubits)
    system.active_qubit_indices.update(data_indices)

    builder.apply_syndrome_extraction(
        se_block.circuit, rounds=rounds, measurement_blocks=measurement_blocks
    )
    for op_name, target in gates:
        executor.apply_logical_operation(
            op_name, [global_patch], noiseless=noiseless_gates,
            **dict(target), **(gate_kwargs or {}),
        )
    detectors_before_post_se = builder.circuit.num_detectors
    builder.apply_syndrome_extraction(
        se_block.circuit, rounds=rounds, measurement_blocks=measurement_blocks
    )
    # A steady-state round emits one detector per stabilizer (redundant
    # stabilizers included).  A gate that does not map the stabilizer group
    # onto itself makes the tracker drop the mismatched stabilizers from the
    # first post-gate round without raising, so count them here.
    num_stabilizers = len(global_patch.stabilizers)
    post_gate_detectors = builder.circuit.num_detectors - detectors_before_post_se
    expected_detectors = rounds * num_stabilizers
    if post_gate_detectors != expected_detectors:
        raise ValueError(
            "The gates do not preserve the code's stabilizer group: the "
            f"{rounds} syndrome-extraction round(s) after the gates emitted "
            f"{post_gate_detectors} detectors, expected {expected_detectors} "
            f"({rounds} x {num_stabilizers} stabilizers); gates={list(gates)!r}."
        )
    builder.apply_data_readout({q: measure_basis for q in data_indices})
    if builder.circuit.num_observables == 0:
        raise ValueError(
            "No logical qubit is resolvable by this circuit: with "
            f"init_basis={init_basis!r} and measure_basis={measure_basis!r} a "
            "logical is read out only if the gates map its initial logical "
            "operator onto the readout basis (e.g. an odd number of H-SWAPs "
            f"for X<->Z); gates={list(gates)!r}."
        )

    if noise_params is not None:
        return builder.build_noisy_circuit(noise_params, noise_model)
    return builder.circuit


__all__ = ["GateSpec", "build_hgp_gate_verification_circuit"]
