"""Build DEQ GADGET blocks for a single rotated-surface-code patch.

Every gadget body is derived from LightStim's own, already-tested circuit
construction code (``RotatedSurfaceCodeExtractionBlock`` for one syndrome
round, ``CircuitBuilder`` for data reset) rather than a second, hand-written
scheduler — the physical instruction sequence is LightStim's, only the text
rendering is new.
"""

from __future__ import annotations

from typing import Mapping

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock

from . import code as code_mod
from . import text
from .code import DeqExportError

# Stim bookkeeping instructions with no DEQ physical-instruction counterpart.
# DEQ derives its own check/hyperedge structure from the CODE + GADGET
# definitions; it does not consume Stim's DETECTOR/OBSERVABLE_INCLUDE stream.
_DROPPED_INSTRUCTIONS = frozenset({"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"})


def qubit_remap(patch: QECPatch) -> dict[int, int]:
    """Map every qubit LightStim uses for ``patch`` to a local DEQ index.

    Data qubits get the same 0..n-1 numbering as ``code.local_index_map`` (so
    CODE ports and gadget bodies agree); ancilla qubits are appended after,
    in sorted order. The same map must be reused for every gadget built
    against this patch so that ancilla indices line up across gadgets, the
    way they do in Microsoft's own generated libraries.
    """
    remap = dict(code_mod.local_index_map(patch))
    n = len(remap)
    ancillas = sorted(patch.syndrome_indices)
    remap.update({qubit: n + i for i, qubit in enumerate(ancillas)})
    return remap


def _instruction_lines(circuit: stim.Circuit, remap: Mapping[int, int]) -> list[str]:
    """Transliterate a flat, noiseless stim.Circuit into DEQ instruction lines."""
    lines: list[str] = []
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise DeqExportError(
                "a single-round gadget body may not contain a REPEAT block; "
                "repetition is expressed structurally in the COMPOSE/PROGRAM instead"
            )
        if instruction.name in _DROPPED_INSTRUCTIONS:
            continue
        if instruction.name == "TICK":
            lines.append("TICK")
            continue
        targets = [remap[target.value] for target in instruction.targets_copy()]
        lines.append(text.Instruction(instruction.name, tuple(targets)).render())
    return lines


def syndrome_extraction_gadget(
    se_circuit: stim.Circuit, remap: Mapping[int, int], code_name: str, port_qubits: list[int]
) -> text.GadgetBlock:
    """Bare, reusable one-round syndrome-extraction gadget (same code in and out)."""
    port = " ".join(str(q) for q in port_qubits)
    lines = [f"INPUT {code_name} {port}"]
    lines += _instruction_lines(se_circuit, remap)
    lines.append(f"OUTPUT {code_name} {port}")
    return text.GadgetBlock("SyndromeExtraction", lines)


def prepare_circuit(system: QECSystem, patch: QECPatch, se_block: RotatedSurfaceCodeExtractionBlock, basis: str) -> stim.Circuit:
    """Data reset (in ``basis``) fused with exactly one syndrome-extraction round.

    Reuses ``CircuitBuilder.initialize`` + ``apply_syndrome_extraction(rounds=1)``
    directly instead of reimplementing reset/SE fusion.
    """
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=False)
    init_dict = {qubit: basis for qubit in patch.data_indices}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=1)
    return builder.circuit


def prepare_gadget(
    name: str, circuit: stim.Circuit, remap: Mapping[int, int], code_name: str, port_qubits: list[int]
) -> text.GadgetBlock:
    """Terminal state-prep gadget: no INPUT (starts from nothing), OUTPUT only."""
    lines = _instruction_lines(circuit, remap)
    lines.append(f"OUTPUT {code_name} " + " ".join(str(q) for q in port_qubits))
    return text.GadgetBlock(name, lines)


def measure_lines(patch: QECPatch, remap: Mapping[int, int], basis: str) -> tuple[list[str], list[str]]:
    """Bare final data measurement in ``basis``, plus its READOUT rec[] tokens.

    Derived purely textually from the CODE's own logical representative: for
    a bare ``M<basis> <data in local order>`` gadget, the qubit at local
    index i lands at rec[-(n - i)] in that gadget's own measurement record.
    No LightStim tracker/OBSERVABLE_INCLUDE bookkeeping is needed here.
    """
    n = len(patch.data_indices)
    instruction_name = "MZ" if basis == "Z" else "MX"
    body = [text.Instruction(instruction_name, tuple(range(n))).render()]

    logical_op = next(op for op in patch.logical_ops if op["type"] == basis)
    support_local = sorted(remap[qubit] for qubit in logical_op["pauli"])
    readout = [f"rec[-{n - i}]" for i in support_local]
    return body, readout


def measure_gadget(
    name: str, body_lines: list[str], readout: list[str], code_name: str, port_qubits: list[int]
) -> text.GadgetBlock:
    """Terminal measurement gadget: INPUT only, ends in a READOUT statement."""
    lines = [f"INPUT {code_name} " + " ".join(str(q) for q in port_qubits)]
    lines += body_lines
    lines.append("READOUT " + " ".join(readout))
    return text.GadgetBlock(name, lines)
