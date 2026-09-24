"""
Integration tests for the DEQ export feature:

  1. Round-trip: replay the exported PrepareZ/SyndromeExtraction/MeasureZ
     gadget bodies back into a stim.Circuit and assert it matches LightStim's
     own native MemoryExperiment(...).build() circuit instruction-for-
     instruction — the same `assert circuit == native` pattern
     test_run_memory.py uses for its CLI-vs-native cross-check.
  2. CLI: benchmarks/deq/run_deq_export.py end-to-end (tiny case).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "benchmarks" / "deq" / "run_deq_export.py"
PYTHON = Path(sys.executable)

from lightstim.deq.export import export_rotated_surface_code_memory
from lightstim.deq.gadgets import qubit_remap
from lightstim.deq.validate import (
    gadget_body_to_stim_circuit,
    is_physical_instruction_line,
    relabel_qubits,
    strip_annotations,
    strip_ticks,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock


def _gadget_body(deq_text: str, gadget_name: str) -> list[str]:
    """Pull the raw body lines out of `GADGET <gadget_name> { ... }` in deq_text."""
    match = re.search(rf"GADGET {re.escape(gadget_name)} \{{\n(.*?)\n\}}", deq_text, re.DOTALL)
    assert match, f"GADGET {gadget_name} not found in exported text"
    return [line.strip() for line in match.group(1).splitlines()]


def _rebuild_experiment_circuit(deq_text: str, rounds: int, basis: str):
    prepare_name = "PrepareZ" if basis == "Z" else "PrepareX"
    measure_name = "MeasureZ" if basis == "Z" else "MeasureX"
    lines: list[str] = []
    lines += [l for l in _gadget_body(deq_text, prepare_name) if is_physical_instruction_line(l)]
    se_lines = [l for l in _gadget_body(deq_text, "SyndromeExtraction") if is_physical_instruction_line(l)]
    for _ in range(rounds - 1):
        lines += se_lines
    lines += [l for l in _gadget_body(deq_text, measure_name) if is_physical_instruction_line(l)]
    return gadget_body_to_stim_circuit(lines)


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_deq_roundtrip_matches_native_memory_experiment(basis, rounds):
    distance = 3
    deq_text = export_rotated_surface_code_memory(distance=distance, rounds=rounds, basis=basis)
    rebuilt = _rebuild_experiment_circuit(deq_text, rounds=rounds, basis=basis)

    # RotatedSurfaceCode.default_extraction_block_class is
    # RotatedSurfaceCodeExtractionBlock (set in surface_code/rotated/__init__.py),
    # so a bare MemoryExperiment(qec_patch=...) already builds with the same
    # code-specific, hook-avoiding SE block the exporter uses (gadgets.py) —
    # no need to pass extraction_block_class explicitly for a fair comparison.
    native = MemoryExperiment(
        qec_patch=RotatedSurfaceCode(distance=distance), basis=basis, rounds=rounds, if_detector=True,
    ).build()
    assert MemoryExperiment(qec_patch=RotatedSurfaceCode(distance=distance)).block_class is RotatedSurfaceCodeExtractionBlock

    # The exporter renumbers qubits (data-first, local 0..n-1, then ancilla)
    # for the gadget bodies; LightStim's own global numbering interleaves
    # data and ancilla qubits by construction (RotatedSurfaceCode.build()
    # registers ancilla rows between data rows). Relabel native's qubits
    # through the same remap the exporter used before comparing.
    patch = QECSystem().add_patch(RotatedSurfaceCode(distance=distance), name="patch")
    remap = qubit_remap(patch)
    native_physical_only = relabel_qubits(strip_annotations(native), remap)

    # DEQ's own compiler concatenates successive gadget calls with no implicit
    # TICK at the call boundary (confirmed against a real `deq transpile` run,
    # and matching the reference library's own MeasureZ, which has no leading
    # TICK either); LightStim's native apply_data_readout inserts one before
    # the terminal data measurement. Compare on physical content only — see
    # strip_ticks's docstring. .flattened() unrolls native's internal
    # stim.CircuitRepeatBlock (rounds > 2 wraps steady-state rounds in one)
    # so a REPEAT-wrapped round compares equal to the exporter's fully
    # inlined per-call gadget bodies.
    assert strip_ticks(rebuilt).flattened() == strip_ticks(native_physical_only).flattened()


def test_deq_measurez_readout_matches_native_observable_support():
    """The exporter derives MeasureZ's READOUT purely textually from the CODE's
    logical representative (no tracker consultation). Cross-check that choice
    independently against what LightStim's own tracker actually emits as
    OBSERVABLE_INCLUDE for the same experiment."""
    distance = 3
    rounds = 2
    native = MemoryExperiment(
        qec_patch=RotatedSurfaceCode(distance=distance), basis="Z", rounds=rounds, if_detector=True,
    ).build()
    observable_targets = [
        instruction.targets_copy()
        for instruction in native
        if instruction.name == "OBSERVABLE_INCLUDE"
    ]
    assert len(observable_targets) == 1
    native_rec_offsets = sorted(-target.value for target in observable_targets[0])  # target.value < 0

    deq_text = export_rotated_surface_code_memory(distance=distance, rounds=rounds, basis="Z")
    match = re.search(r"GADGET MeasureZ \{\n(.*?)\n\}", deq_text, re.DOTALL)
    readout_line = next(line for line in match.group(1).splitlines() if "READOUT" in line)
    exported_offsets = sorted(int(tok[5:-1]) for tok in readout_line.split()[1:])

    assert exported_offsets == native_rec_offsets


# ── CLI integration ─────────────────────────────────────────────────────────

def test_cli_writes_valid_deq_file(tmp_path):
    out_path = tmp_path / "d3.deq"
    result = subprocess.run(
        [str(PYTHON), str(RUNNER), "--distance", "3", "--rounds", "2", "--basis", "Z", "--out", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    content = out_path.read_text()
    assert "CODE RotatedSurfaceCodeD3" in content
    assert "PROGRAM RotatedSurfaceCodeD3MemoryExperimentZ2" in content


def test_cli_validate_flag_runs_real_deq_parser(tmp_path):
    pytest.importorskip("deq")
    out_path = tmp_path / "d3.deq"
    result = subprocess.run(
        [str(PYTHON), str(RUNNER), "--distance", "3", "--rounds", "2", "--basis", "Z",
         "--out", str(out_path), "--validate"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "validated OK" in result.stdout
