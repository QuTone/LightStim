"""Validation helpers for exported DEQ text.

Two independent tiers, each usable without the other:

* Tier 0 (needs only the ``deq`` package): parse the emitted text with
  Microsoft's real parser and run its algebraic CODE validator. No
  ``deq_runtime`` (the compiled transpiler/simulator backend) required.
* Tier 1 (needs nothing beyond ``lightstim``/``stim``): reverse-translate the
  emitted GADGET bodies back into a ``stim.Circuit`` and compare against
  LightStim's own native circuit construction, the same
  ``assert circuit == native`` pattern ``tests/test_run_memory.py`` already
  uses for its CLI-vs-native cross-check.

A third, real-compiler tier (``python -m deq transpile`` + ``deq sample``,
needs ``deq_runtime``) is exercised directly by
``benchmarks/deq/run_deq_verify.py`` rather than wrapped here, since it
shells out to the ``deq`` CLI instead of calling a stable Python API.
"""

from __future__ import annotations

import importlib.util
import re

from typing import Mapping

import stim

from .code import DeqExportError

_INSTRUCTION_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$")


def deq_available() -> bool:
    """True if the real ``deq`` package (parser + spec validators) is importable."""
    return importlib.util.find_spec("deq") is not None


def parse_and_validate_codes(deq_text: str):
    """Tier 0: parse ``deq_text`` and run ``validate_code`` on every CODE block.

    Requires ``deq`` to be importable; raises whatever the real parser/validator
    raises (typically a ``SyntaxError`` or ``ValueError``) on invalid input.
    Returns the parsed ``deq.circuit.model.DeqFile``.
    """
    from deq.circuit.parser import parse
    from deq.transpiler.code_validation import validate_code

    deq_file = parse(deq_text)
    for definition in deq_file.definitions:
        if type(definition).__name__ == "CodeDefinition":
            validate_code(definition)
    return deq_file


def gadget_body_to_stim_circuit(lines: list[str]) -> stim.Circuit:
    """Tier 1 helper: replay a gadget's already-rendered body lines as a stim.Circuit.

    Only understands the physical-instruction subset ``gadgets.py`` emits
    (bare TICK, and ``NAME target target ...``); ``INPUT``/``OUTPUT``/``READOUT``
    lines are not physical instructions and must be filtered out by the caller
    before calling this.
    """
    circuit = stim.Circuit()
    for line in lines:
        if line == "TICK":
            circuit.append("TICK")
            continue
        match = _INSTRUCTION_LINE.match(line)
        if not match:
            raise DeqExportError(f"unrecognized instruction line: {line!r}")
        name, rest = match.groups()
        targets = [int(token) for token in rest.split()]
        circuit.append(name, targets)
    return circuit


def is_physical_instruction_line(line: str) -> bool:
    """True for a gadget body line that is a physical instruction (or TICK),
    false for a structural line (``INPUT``/``OUTPUT``/``READOUT``)."""
    head = line.split(None, 1)[0] if line else ""
    return head not in ("INPUT", "OUTPUT", "READOUT")


_ANNOTATION_ONLY_INSTRUCTIONS = frozenset({"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"})


def strip_annotations(circuit: stim.Circuit) -> stim.Circuit:
    """Drop the Stim-only bookkeeping instructions the exporter also drops.

    Lets a native LightStim circuit (built with ``if_detector=True``, which
    also unconditionally writes ``QUBIT_COORDS``) be compared fairly against
    a physical-only circuit rebuilt from exported gadget bodies.
    """
    stripped = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            stripped.append(
                stim.CircuitRepeatBlock(instruction.repeat_count, strip_annotations(instruction.body_copy()))
            )
            continue
        if instruction.name in _ANNOTATION_ONLY_INSTRUCTIONS:
            continue
        stripped.append(instruction)
    return stripped


def strip_ticks(circuit: stim.Circuit) -> stim.Circuit:
    """Drop every TICK, recursing into REPEAT blocks.

    DEQ's own compiler concatenates successive gadget calls with no implicit
    TICK at the boundary (confirmed against a real `python -m deq transpile`
    run), matching the reference surface-code-deq library's own generated
    gadgets (e.g. its ``MeasureZ`` has no leading TICK). LightStim's native
    ``CircuitBuilder.apply_data_readout``, by contrast, inserts an explicit
    TICK before the terminal data measurement. Both are physically valid
    noiseless circuits; TICK placement doesn't change unitary/measurement
    semantics, only how a later noise model would group error locations. Use
    this to compare the two conventions on physical content alone.
    """
    stripped = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            stripped.append(
                stim.CircuitRepeatBlock(instruction.repeat_count, strip_ticks(instruction.body_copy()))
            )
            continue
        if instruction.name == "TICK":
            continue
        stripped.append(instruction)
    return stripped


def relabel_qubits(circuit: stim.Circuit, remap: Mapping[int, int]) -> stim.Circuit:
    """Rewrite every plain-qubit target through ``remap`` (e.g. LightStim's own
    global qubit numbering -> the exporter's local, data-first-then-ancilla
    numbering), so a native LightStim circuit can be compared fairly against
    a circuit rebuilt from exported gadget bodies. Only handles plain qubit
    targets — call ``strip_annotations`` first to remove any rec[]-target
    instructions (DETECTOR/OBSERVABLE_INCLUDE)."""
    relabeled = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            relabeled.append(
                stim.CircuitRepeatBlock(instruction.repeat_count, relabel_qubits(instruction.body_copy(), remap))
            )
            continue
        targets = [remap[target.value] for target in instruction.targets_copy()]
        relabeled.append(instruction.name, targets, instruction.gate_args_copy())
    return relabeled
