"""
Unit tests for lightstim.deq: exporting a rotated-surface-code memory
experiment to Microsoft's DEQ device DSL.

Layers:
  1. Shape: CODE/GADGET/COMPOSE/PROGRAM text contains the expected structure.
  2. Correctness: readout-offset formula, CSS-only guard, qubit remap.
  3. Real-tooling (skipped if `deq` isn't installed): the emitted text parses
     with Microsoft's own parser and passes its algebraic CODE validator.
"""

import importlib.util

import pytest

from lightstim.deq.code import DeqExportError, code_block, is_css, local_index_map
from lightstim.deq.export import export_rotated_surface_code_memory
from lightstim.deq.gadgets import measure_lines, qubit_remap
from lightstim.deq.validate import deq_available, parse_and_validate_codes
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.bacon_shor.code_patch import BaconShorCode
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode


# ── helpers ─────────────────────────────────────────────────────────────────

def _patch(distance: int = 3):
    system = QECSystem()
    return system, system.add_patch(RotatedSurfaceCode(distance=distance), name="patch")


# ── 1. shape ────────────────────────────────────────────────────────────────

def test_export_contains_expected_blocks():
    deq_text = export_rotated_surface_code_memory(distance=3, rounds=3, basis="Z")
    assert "CODE RotatedSurfaceCodeD3 [[9,1,3]] {" in deq_text
    assert "GADGET PrepareZ {" in deq_text
    assert "GADGET PrepareX {" in deq_text
    assert "GADGET SyndromeExtraction {" in deq_text
    assert "GADGET MeasureZ {" in deq_text
    assert "GADGET MeasureX {" in deq_text
    assert "COMPOSE RotatedSurfaceCodeD3Memory {" in deq_text
    assert "PROGRAM RotatedSurfaceCodeD3MemoryExperimentZ3 {" in deq_text


def test_export_drops_stim_only_bookkeeping_instructions():
    deq_text = export_rotated_surface_code_memory(distance=3, rounds=2, basis="Z")
    for forbidden in ("QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
        assert forbidden not in deq_text


def test_program_repeat_count_is_rounds_minus_one():
    # PrepareZ already contributes round 1, so REPEAT covers the rest.
    deq_text = export_rotated_surface_code_memory(distance=3, rounds=5, basis="Z")
    program = deq_text.split("PROGRAM RotatedSurfaceCodeD3MemoryExperimentZ5 {")[1]
    assert "REPEAT 4 {" in program
    assert "SyndromeExtraction 0" in program

    single_round = export_rotated_surface_code_memory(distance=3, rounds=1, basis="Z")
    program = single_round.split("PROGRAM RotatedSurfaceCodeD3MemoryExperimentZ1 {")[1]
    assert "REPEAT" not in program


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance", [3, 5])
def test_export_runs_for_several_configurations(basis, distance):
    deq_text = export_rotated_surface_code_memory(distance=distance, rounds=distance, basis=basis)
    assert f"CODE RotatedSurfaceCodeD{distance}" in deq_text


# ── 2. correctness ─────────────────────────────────────────────────────────

def test_code_block_logical_and_stabilizers_match_patch():
    _, patch = _patch(distance=3)
    block = code_block(patch, "RotatedSurfaceCodeD3", distance=3)
    assert block.n == 9
    assert block.k == 1
    assert block.d == 3
    assert len(block.stabilizers) == len(patch.stabilizers)
    # Logical X support is {0, 3, 6} and logical Z support is {0, 1, 2} in the
    # canonical local numbering (sorted global data indices -> 0..n-1).
    assert str(block.logical_x) == "X0*X3*X6"
    assert str(block.logical_z) == "Z0*Z1*Z2"


def test_measure_readout_offset_formula():
    _, patch = _patch(distance=3)
    remap = qubit_remap(patch)
    body, readout = measure_lines(patch, remap, "Z")
    assert body == ["MZ 0 1 2 3 4 5 6 7 8"]
    # Local support {0,1,2} in a 9-qubit bare measurement -> rec[-9],[-8],[-7].
    assert readout == ["rec[-9]", "rec[-8]", "rec[-7]"]


def test_qubit_remap_is_data_first_then_ancilla():
    _, patch = _patch(distance=3)
    remap = qubit_remap(patch)
    local_map = local_index_map(patch)
    # Data qubits keep the CODE's own local numbering.
    for global_q, local_q in local_map.items():
        assert remap[global_q] == local_q
    # Ancillas are appended after, contiguous, in sorted global order.
    n = len(local_map)
    ancilla_locals = sorted(remap[q] for q in patch.syndrome_indices)
    assert ancilla_locals == list(range(n, n + len(patch.syndrome_indices)))


def test_non_css_patch_is_rejected():
    system = QECSystem()
    patch = system.add_patch(BaconShorCode(distance=3), name="patch")
    if is_css(patch):
        pytest.skip("BaconShorCode is CSS in this configuration; guard not exercised")
    with pytest.raises(DeqExportError):
        code_block(patch, "BaconShorD3")


def test_invalid_basis_and_rounds_are_rejected():
    with pytest.raises(DeqExportError):
        export_rotated_surface_code_memory(distance=3, rounds=3, basis="Y")
    with pytest.raises(DeqExportError):
        export_rotated_surface_code_memory(distance=3, rounds=0, basis="Z")


# ── 3. real deq tooling (Tier 0) ────────────────────────────────────────────

@pytest.mark.skipif(not deq_available(), reason="deq package not installed")
def test_deq_parses_and_validates():
    deq_text = export_rotated_surface_code_memory(distance=3, rounds=3, basis="Z")
    deq_file = parse_and_validate_codes(deq_text)
    kinds = sorted(type(d).__name__ for d in deq_file.definitions)
    assert kinds == [
        "CodeDefinition",
        "ComposeDefinition",
        "GadgetDefinition",
        "GadgetDefinition",
        "GadgetDefinition",
        "GadgetDefinition",
        "GadgetDefinition",
        "ProgramDefinition",
    ]


@pytest.mark.skipif(not importlib.util.find_spec("deq_runtime"), reason="deq_runtime not installed")
def test_deq_transpile_and_sample_noiseless_program(tmp_path):
    """Real end-to-end check: Microsoft's own compiler + runtime accept the
    exported PROGRAM and a noiseless run gives a deterministic all-zero
    syndrome and readout, exactly as a correctly-built memory experiment
    should. Shells out to the `deq` CLI (the same commands
    `benchmarks/deq/run_deq_verify.py` runs) rather than reaching into
    internal compiler functions whose signatures aren't part of deq's
    public API."""
    import subprocess
    import sys

    deq_text = export_rotated_surface_code_memory(distance=3, rounds=3, basis="Z")
    deq_path = tmp_path / "d3_memory.deq"
    deq_path.write_text(deq_text)
    jit_path = tmp_path / "d3_memory.deq.jit"
    program_name = "RotatedSurfaceCodeD3MemoryExperimentZ3"

    subprocess.run(
        [sys.executable, "-m", "deq", "transpile", str(deq_path), "--program", program_name, "--out", str(jit_path)],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "deq", "sample", str(deq_path), "--program", program_name,
         "--shots", "50", "--noiseless", "--interpret"],
        check=True, capture_output=True, text=True,
    )
    syndromes = [line for line in result.stdout.splitlines() if line.startswith("Syndrome:")]
    readouts = [line for line in result.stdout.splitlines() if line.startswith("Readout:")]
    assert len(syndromes) == 50
    assert all(line == syndromes[0] for line in syndromes), "expected an all-zero syndrome every noiseless shot"
    assert all("0b0" == line.split()[-1] for line in readouts), "expected a zero logical readout every noiseless shot"
