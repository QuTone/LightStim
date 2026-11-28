"""Check the compact drawing and the full circuit shown in the BB PPM example."""

import base64
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import stim

from lightstim.qec_code.BB_code.ppm.reporting import compact_circuit_layout, write_circuit_preview


def test_circuit_layout_moves_lpu_and_its_detector_together():
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(3, 1) 1
        QUBIT_COORDS(100, 100) 2
        QUBIT_COORDS(101, 100) 3
        REPEAT 2 {
            CX 0 2
            M 2
            SHIFT_COORDS(0, 0, 1)
            DETECTOR(100, 100, 0) rec[-1]
            TICK
        }
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    original = circuit.copy()
    drawing = compact_circuit_layout(circuit, bb_qubits=2)
    assert circuit == original
    assert drawing.get_final_qubit_coordinates() == {
        0: [0, 0], 1: [3, 1], 2: [1, 3], 3: [2, 3],
    }
    assert drawing.get_detector_coordinates() == {0: [1, 3, 1], 1: [1, 3, 2]}
    for before, after in zip(original.flattened(), drawing, strict=True):
        if before.name in ('DETECTOR', 'QUBIT_COORDS'):
            assert (before.name, before.targets_copy()) == (after.name, after.targets_copy())
        else:
            assert before == after
    assert compact_circuit_layout(circuit, bb_qubits=4) == original


@pytest.fixture(scope='module')
def circuit_diagram_cell():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / 'notebooks/LogicalOps/logical_PPM_BB_code.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    [cell] = [c for c in notebook['cells']
              if c['cell_type'] == 'code' and 'detslice-with-ops-svg' in ''.join(c['source'])]
    return cell


def png_size(data):
    assert data.startswith(b'\x89PNG\r\n\x1a\n')
    assert data[12:16] == b'IHDR'
    return struct.unpack('>II', data[16:24])


def test_notebook_embeds_full_circuit_preview(circuit_diagram_cell):
    [preview] = [out for out in circuit_diagram_cell['outputs'] if 'image/png' in out.get('data', {})]
    data = base64.b64decode(''.join(preview['data']['image/png']))
    assert png_size(data) == (1280, 11558)
    assert preview['metadata']['image/png']['width'] == 1000


def test_diagram_cell_renders_every_tick_without_reference_files(circuit_diagram_cell, tmp_path):
    pytest.importorskip('resvg_py')
    ipython = pytest.importorskip('IPython.display')
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        TICK
        H 0
        TICK
        CX 0 1
        TICK
        M 0 1
    """)
    displayed = []
    exec(''.join(circuit_diagram_cell['source']), {
        'circuit': circuit, 'gross': SimpleNamespace(num_qubits=circuit.num_qubits),
        'ROOT': tmp_path, 'OUTPUT': tmp_path, 'compact_circuit_layout': compact_circuit_layout,
        'write_circuit_preview': write_circuit_preview, 'display': displayed.append, 'Image': ipython.Image,
    })

    assert stim.Circuit.from_file(tmp_path / 'full_circuit.stim') == circuit
    svg = (tmp_path / 'full_circuit.svg').read_text(encoding='utf-8')
    panels = circuit.num_ticks + 1
    assert svg.count('id="qubit_dot:') == panels * circuit.num_qubits
    assert svg.count('id="tick_border:') == panels
    _, _, w, h = map(float, ET.fromstring(svg).attrib['viewBox'].split())
    [image] = displayed
    assert image.data == (tmp_path / 'full_circuit.png').read_bytes()
    width, height = png_size(image.data)
    assert width == 1280
    assert height == pytest.approx(width * h / w, abs=1)
