"""The patch API produces portable, independent notebook and file views."""

from copy import deepcopy
from html.parser import HTMLParser
import json
import re
import subprocess
import sys

import pytest

from lightstim.frontend.patch_html import export_patch_html, patch_html, patch_view_data
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedTwoPatchCoupler
from lightstim.qec_code.surface_code.toric import ToricCode
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode
from lightstim.qec_code.surface_code.xzzx import XZZXSurfaceCode


def _payload(document):
    match = re.search(
        r'<script id="patch-data" type="application/json">(.*?)</script>',
        document,
        re.S,
    )
    assert match is not None
    return json.loads(match[1])


class _Frames(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.frames = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        if tag == "iframe":
            self.frames.append(dict(attrs))


class _GridPatch(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        for index in range(self.params["qubits"]):
            role = "data" if index < 250 else "syndrome_x"
            # Sparse IDs ensure the limit counts registered qubits, not max ID.
            self.add_qubit(index % 25, index // 25, role, uid=1000 + 3 * index)


def test_500_registered_qubits_including_ancillas_are_supported(tmp_path):
    patch = _GridPatch(qubits=500)
    output = export_patch_html(patch, tmp_path / "patch.html")
    snapshots = [
        patch_view_data(patch),
        _payload(patch_html(patch)),
        _payload(patch.visualize().html),
        _payload(output.read_text(encoding="utf-8")),
    ]

    for data in snapshots:
        assert {q["id"] for q in data["qubits"]} == set(patch.qubit_coords)
        assert sum(q["role"] == "data" for q in data["qubits"]) == 250
        assert sum(q["role"] == "syndrome X" for q in data["qubits"]) == 250


@pytest.mark.parametrize("render", [patch_view_data, patch_html, QECPatch.visualize])
def test_501_registered_qubits_including_ancillas_are_rejected(render):
    patch = _GridPatch(qubits=501)
    assert len(patch.data_indices) == 250

    with pytest.raises(ValueError, match="500"):
        render(patch)


def test_oversized_export_rejects_before_creating_output_directory(tmp_path):
    output = tmp_path / "nested" / "patch.html"

    with pytest.raises(ValueError, match="500"):
        export_patch_html(_GridPatch(qubits=501), output)

    assert not output.parent.exists()


@pytest.mark.parametrize("registered", [False, True], ids=["raw", "registered"])
def test_native_couplers_are_rejected_as_not_self_contained(registered, tmp_path):
    first = RotatedSurfaceCode(distance=3)
    second = RotatedSurfaceCode(distance=3)
    second.shift_coords(8, 0)
    coupler = RotatedTwoPatchCoupler().create_coupler_patch(
        [first, second], name="seam", interaction_type="XX"
    )
    if registered:
        system = QECSystem()
        system.add_patch(first, name="first")
        system.add_patch(second, name="second")
        coupler = system.add_patch(coupler, name="seam", is_active=False)
        assert any(set(record["pauli"]) - set(coupler.qubit_coords)
                   for record in coupler.stabilizers)

    for render in (patch_view_data, patch_html, QECPatch.visualize):
        with pytest.raises(ValueError, match="(?i)coupler") as error:
            render(coupler)
        assert "self-contained" in str(error.value)

    output = tmp_path / "coupler" / "patch.html"
    with pytest.raises(ValueError, match="(?i)coupler") as error:
        export_patch_html(coupler, output)
    assert "self-contained" in str(error.value)
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "factory", [RotatedSurfaceCode, UnrotatedSurfaceCode, XZZXSurfaceCode, ToricCode]
)
def test_visualize_native_unnamed_patches_preserves_all_operators(factory):
    patch = factory(distance=3)
    records = patch.stabilizers + patch.logical_ops + patch.gauges
    for record in records:
        record.pop("name", None)
        record.pop("label", None)
    before_coords = deepcopy(patch.qubit_coords)
    before_records = deepcopy(records)

    data = _payload(patch.visualize().html)

    assert patch.qubit_coords == before_coords
    assert records == before_records
    assert data["overlays"] == []
    assert data["num_logicals"] == patch.num_logicals
    assert {q["id"]: (q["x"], q["y"]) for q in data["qubits"]} == before_coords
    assert len(data["operators"]) == len(records)
    for actual, expected in zip(data["operators"], records):
        assert dict(actual["support"]) == expected["pauli"]
        assert actual["name"].startswith({"stabilizer": "S_", "logical": "L_", "gauge": "G_"}[actual["category"]])
    if factory is XZZXSurfaceCode:
        assert any(op["basis"] == "mixed" for op in data["operators"])
    if factory is ToricCode:
        assert sum(op["category"] == "logical" for op in data["operators"]) == 4


def test_saved_document_and_notebook_document_are_the_same_snapshot(tmp_path):
    patch = RotatedSurfaceCode(distance=3)
    view = patch.visualize(title="A portable patch")
    document = view.html
    first_qubit = next(iter(patch.qubit_coords))
    patch.qubit_coords[first_qubit] = (100, 100)
    patch.stabilizers.clear()

    output = tmp_path / "nested" / "patch.html"
    assert view.save_html(output) == output
    assert output.read_text(encoding="utf-8") == document == str(view)
    frames = _Frames(view._repr_html_()).frames
    assert len(frames) == 1
    assert frames[0]["srcdoc"] == document
    assert "allow-scripts" in frames[0]["sandbox"].split()
    assert _payload(document)["operators"]
    assert _payload(document)["qubits"][0]["x"] != 100


def test_multiple_notebook_views_keep_distinct_documents_and_escape_titles():
    title = '\"<script>not executable</script>& a title'
    first = RotatedSurfaceCode(distance=3).visualize(title=title)
    second = ToricCode(distance=3).visualize(title="Two logical qubits")
    frames = _Frames(first._repr_html_() + second._repr_html_()).frames

    assert len(frames) == 2
    assert frames[0]["srcdoc"] == first.html
    assert frames[1]["srcdoc"] == second.html
    assert title not in first.html
    assert _payload(frames[0]["srcdoc"])["title"] == title
    assert _payload(frames[1]["srcdoc"])["num_logicals"] == 2


def test_explicit_region_annotation_is_optional_and_not_inferred():
    patch = RotatedSurfaceCode(distance=3)
    selected = sorted(patch.data_indices)[:2]
    patch.source = selected  # Application metadata has no automatic visual meaning.
    assert _payload(patch.visualize().html)["overlays"] == []
    data = _payload(patch.visualize(overlays=[{"name": "A user annotation", "qubits": selected}]).html)
    assert data["overlays"][0]["name"] == "A user annotation"
    assert data["overlays"][0]["qubits"] == selected


def test_patch_import_and_visualization_do_not_require_ipython():
    subprocess.run(
        [sys.executable, "-c", """
import sys
from lightstim.ir.qec_patch import QECPatch
assert not any(name.startswith('lightstim.frontend') for name in sys.modules)
assert not any(name.startswith('IPython') for name in sys.modules)
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
view = RotatedSurfaceCode(distance=3).visualize()
assert '<iframe' in view._repr_html_()
assert not any(name.startswith('IPython') for name in sys.modules)
"""],
        check=True,
        capture_output=True,
        text=True,
    )
