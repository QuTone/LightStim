"""Top-level orchestration: LightStim rotated-surface-code memory experiment
exported as a DEQ device library.
"""

from __future__ import annotations

from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock

from . import code as code_mod
from . import compose as compose_mod
from . import gadgets
from . import text
from .code import DeqExportError

__all__ = ["export_rotated_surface_code_memory", "DeqExportError"]


def export_rotated_surface_code_memory(distance: int, rounds: int, *, basis: str = "Z") -> str:
    """Export a distance-``distance`` rotated-surface-code memory experiment to DEQ text.

    Produces one CODE block; PrepareZ, PrepareX, SyndromeExtraction, MeasureZ,
    and MeasureX GADGETs; a reusable REPEAT-only memory-register COMPOSE
    (matching the reference surface-code-deq library's own shape); and a full
    runnable PROGRAM for ``rounds`` total syndrome-extraction rounds in
    ``basis``, matching ``MemoryExperiment(rounds=rounds).build()``'s round
    count exactly.

    Always uses ``RotatedSurfaceCodeExtractionBlock`` (the code-specific,
    hook-error-avoiding schedule) — this is also ``RotatedSurfaceCode``'s own
    ``default_extraction_block_class``, so it matches what an unconfigured
    ``MemoryExperiment(qec_patch=RotatedSurfaceCode(...))`` builds too.
    """
    if basis not in ("X", "Z"):
        raise DeqExportError(f"basis must be 'X' or 'Z'; got {basis!r}")
    if rounds < 1:
        raise DeqExportError("rounds must be >= 1")

    system = QECSystem()
    patch = system.add_patch(RotatedSurfaceCode(distance=distance), name="patch")
    code_name = f"RotatedSurfaceCodeD{distance}"
    code_block = code_mod.code_block(patch, code_name, distance=distance)

    remap = gadgets.qubit_remap(patch)
    port_qubits = list(range(len(patch.data_indices)))

    se_block = RotatedSurfaceCodeExtractionBlock(system)
    se_gadget = gadgets.syndrome_extraction_gadget(se_block.circuit, remap, code_name, port_qubits)

    prepare_z_circuit = gadgets.prepare_circuit(system, patch, se_block, basis="Z")
    prepare_x_circuit = gadgets.prepare_circuit(system, patch, se_block, basis="X")
    prepare_z_gadget = gadgets.prepare_gadget("PrepareZ", prepare_z_circuit, remap, code_name, port_qubits)
    prepare_x_gadget = gadgets.prepare_gadget("PrepareX", prepare_x_circuit, remap, code_name, port_qubits)

    measure_z_body, measure_z_readout = gadgets.measure_lines(patch, remap, "Z")
    measure_x_body, measure_x_readout = gadgets.measure_lines(patch, remap, "X")
    measure_z_gadget = gadgets.measure_gadget("MeasureZ", measure_z_body, measure_z_readout, code_name, port_qubits)
    measure_x_gadget = gadgets.measure_gadget("MeasureX", measure_x_body, measure_x_readout, code_name, port_qubits)

    register_compose = compose_mod.memory_register_compose(code_name, rounds=rounds, name=f"{code_name}Memory")
    program_name = f"{code_name}MemoryExperiment{basis}{rounds}"
    experiment_program = compose_mod.memory_experiment_program(program_name, rounds=rounds, basis=basis)

    return text.document(
        code_block,
        prepare_z_gadget,
        prepare_x_gadget,
        se_gadget,
        measure_z_gadget,
        measure_x_gadget,
        register_compose,
        experiment_program,
    )
