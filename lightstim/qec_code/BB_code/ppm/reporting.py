"""Drawing helpers for gross-code Pauli product measurement circuits."""

from io import BytesIO
from pathlib import Path

import stim


def compact_circuit_layout(circuit: stim.Circuit, *, bb_qubits: int) -> stim.Circuit:
    """Center the LPU below the BB memory lattice in a drawing copy.

    BB PPM circuits number the BB patch first and the LPU second. Translate
    the LPU as a group, including detector labels at its check sites.
    """
    sites = circuit.get_final_qubit_coordinates()
    memory = [xy for q, xy in sites.items() if q < bb_qubits]
    lpu = [xy for q, xy in sites.items() if q >= bb_qubits]
    if not lpu:
        return circuit.copy()

    memory_x = [xy[0] for xy in memory]
    lpu_x = [xy[0] for xy in lpu]
    dx = (min(memory_x) + max(memory_x) - min(lpu_x) - max(lpu_x)) / 2
    dy = max(xy[1] for xy in memory) + 2 - min(xy[1] for xy in lpu)
    moved = {(x, y): (x + dx, y + dy) for x, y in lpu}

    drawing = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name in ('QUBIT_COORDS', 'DETECTOR'):
            coords = instruction.gate_args_copy()
            location = tuple(coords[:2])
            if location in moved:
                drawing.append(instruction.name, instruction.targets_copy(),
                               [*moved[location], *coords[2:]], tag=instruction.tag)
                continue
        drawing.append(instruction)
    return drawing


def write_circuit_preview(svg_path: Path, png_path: Path) -> None:
    """Rasterize the full SVG at notebook width and keep the embedded PNG small."""
    from PIL import Image
    from resvg_py import svg_to_bytes

    png = svg_to_bytes(svg_path=str(svg_path), width=1280, background="white")
    with Image.open(BytesIO(png)) as image:
        image.convert("RGB").quantize(colors=32).save(png_path, format="PNG", optimize=True)
