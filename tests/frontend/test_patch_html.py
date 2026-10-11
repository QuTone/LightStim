"""The standalone patch viewer must preserve geometry and non-CSS algebra."""

import json
import re
import subprocess
import sys

import pytest
import stim

from lightstim.frontend.patch_html import export_patch_html, patch_html, patch_view_data
from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.toric import ToricCode
from lightstim.qec_code.surface_code.xzzx import XZZXSurfaceCode


class SparsePatch(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        self.add_qubit(0, 0, "data", uid=2)
        self.add_qubit(2, 0, "data", uid=8)
        self.add_qubit(1, 1, "syndrome_x", uid=20)
        self.stabilizers = [{"name": "mixed signed", "type": "X", "sign": -1,
                             "pauli": {2: "X", 8: "Y"}, "syn_idx": 20}]
        logical = stim.PauliString(9)
        logical[2], logical[8], logical.sign = "Y", "Z", -1
        self.logical_ops = [{"name": "logical representative", "pauli": logical}]
        self.gauges = [{"pauli": {2: "X"}}, {"pauli": {2: "Z"}}]
        self.num_logicals = 2


def _embedded(document):
    return json.loads(re.search(r'<script id="patch-data" type="application/json">(.*?)</script>',
                               document, re.S)[1])


def test_sparse_mixed_signed_operators_and_noncommuting_gauges_are_preserved():
    patch = SparsePatch()
    before = str(patch.logical_ops[0]["pauli"])
    data = patch_view_data(patch, overlays=[
        {"name": "Source", "qubits": [2, 8]},
        {"name": "Output", "qubits": [8], "color": "#188578"},
    ])
    assert [(q["id"], q["x"], q["y"]) for q in data["qubits"]] == [(2, 0, 0), (8, 2, 0), (20, 1, 1)]
    assert data["num_logicals"] == 2
    check, logical, gx, gz = data["operators"]
    assert check["support"] == [[2, "X"], [8, "Y"]]
    assert check["basis"] == "mixed"  # Not the misleading record.type X label.
    assert check["sign"] == logical["sign"] == -1
    assert check["syndrome_qubit"] == 20
    assert logical["support"] == [[2, "Y"], [8, "Z"]]
    assert check["pauli_string"] == "−X2 Y8"
    assert gx["category"] == gz["category"] == "gauge"
    assert data["overlays"][0]["qubits"] == [2, 8]
    assert data["overlays"][1]["qubits"] == [8]
    assert str(patch.logical_ops[0]["pauli"]) == before


@pytest.mark.parametrize("factory", [RotatedSurfaceCode, XZZXSurfaceCode, ToricCode])
def test_native_patches_keep_every_pauli_factor_and_coordinate(factory):
    patch = factory(distance=3)
    data = patch_view_data(patch)
    for q in data["qubits"]:
        assert (q["x"], q["y"]) == patch.qubit_coords[q["id"]]
    assert data["num_logicals"] == patch.num_logicals
    records = patch.stabilizers + patch.logical_ops + patch.gauges
    assert len(data["operators"]) == len(records)
    for actual, expected in zip(data["operators"], records):
        assert dict(actual["support"]) == expected["pauli"]
    if factory is XZZXSurfaceCode:
        assert any(o["basis"] == "mixed" for o in data["operators"])
    if factory is ToricCode:
        assert len([o for o in data["operators"] if o["category"] == "logical"]) == 4


def test_standalone_embedded_payload_round_trips_and_cannot_end_script(tmp_path):
    patch = SparsePatch()
    title = '</script><img src=x onerror="alert(1)">'
    patch.stabilizers[0]["name"] = title
    path = export_patch_html(patch, tmp_path / "nested" / "code.html", title=title)
    document = path.read_text()
    data = _embedded(document)
    assert data["title"] == data["operators"][0]["name"] == title
    assert title not in document
    assert "fetch(" not in document
    assert '<script src=' not in document
    assert '<link ' not in document


@pytest.mark.parametrize("change,match", [
    (lambda p: p.stabilizers[0].update(pauli={999: "Z"}), "without coordinates"),
    (lambda p: p.stabilizers[0].update(pauli={2: "A"}), "Invalid Pauli"),
    (lambda p: p.stabilizers[0].update(sign=1j), "real sign"),
    (lambda p: p.logical_ops[0]["pauli"].__setattr__("sign", 1j), "real sign"),
    (lambda p: p.qubit_coords.update({2: (float("nan"), 0)}), "finite coordinates"),
])
def test_invalid_physics_or_geometry_is_rejected(change, match):
    patch = SparsePatch()
    change(patch)
    with pytest.raises(ValueError, match=match):
        patch_html(patch)


def test_empty_patch_is_renderable():
    patch = SparsePatch()
    patch.stabilizers.clear()
    patch.logical_ops.clear()
    patch.gauges.clear()
    data = _embedded(patch_html(patch))
    assert data["operators"] == []


def test_cli_builds_patch_with_supplied_configuration(tmp_path):
    output = tmp_path / "code.html"
    result = subprocess.run([
        sys.executable, "-m", "lightstim.frontend.patch_html",
        "--factory", "lightstim.qec_code.surface_code.rotated:RotatedSurfaceCode",
        "--kwargs", '{"distance": 3}', "--output", str(output),
    ], check=True, text=True, capture_output=True)
    assert str(output) in result.stdout
    data = _embedded(output.read_text())
    assert len([q for q in data["qubits"] if q["role"] == "data"]) == 9
