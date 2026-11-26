import json
import re
from copy import deepcopy

import numpy as np
import pytest

from lightstim.frontend.patch_html import system_view_data
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedTwoPatchCoupler
from lightstim.qec_code.surface_code.xzzx import XZZXSurfaceCode


def test_system_snapshot_matches_registered_geometry_and_operator_factors(tmp_path):
    system = QECSystem()
    system.add_patch(BaconShorCode(distance=3), name="subsystem", is_active=False)
    second = XZZXSurfaceCode(distance=3)
    placed = system.add_patch(second, name="second", offset=(np.float32(20.25), -4.5))
    overlay = [{"name": "edge", "qubits": [max(placed.data_indices)]}]
    before = deepcopy(system.stabilizers)
    snapshot = system_view_data(system, title="Placed pair", overlays=overlay)

    assert snapshot["schema"] == "lightstim.qec-system-view.v1"
    assert snapshot["num_logicals"] == 2
    assert snapshot["patches"][1] == {
        "id": "second", "kind": "patch", "patch_class": "XZZXSurfaceCode",
        "qubits": sorted(placed.qubit_coords), "offset": [20.25, -4.5],
        "parameters": {"distance": 3},
    }
    assert {q["id"]: (q["x"], q["y"]) for q in snapshot["qubits"]} == system.qubit_coords
    assert len({op["id"] for op in snapshot["operators"]}) == len(snapshot["operators"])
    checks = [op for op in snapshot["operators"]
              if op["patch"] == "second" and op["category"] == "stabilizer"]
    assert any(op["basis"] == "mixed" for op in checks)
    for op, record in zip(checks, placed.stabilizers, strict=True):
        assert dict(op["support"]) == record["pauli"]
        assert op["syndrome_qubit"] == record["syn_idx"]
        assert op["active"] is True
    gauges = [op for op in snapshot["operators"] if op["category"] == "gauge"]
    assert len(gauges) == len(system.gauges) > 0
    assert all(not op["active"] and op["patch"] == "subsystem" for op in gauges)
    assert system.stabilizers == before
    assert snapshot["overlays"][0]["qubits"] == overlay[0]["qubits"]

    path = system.visualize(title="Placed pair", overlays=overlay).save_html(tmp_path / "pair.html")
    match = re.search(r'<script id="patch-data" type="application/json">(.*?)</script>',
                      path.read_text(encoding="utf-8"), re.S)
    assert json.loads(match[1]) == snapshot


def test_registered_coupler_resolves_external_sites_and_current_masks():
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="left")
    system.add_patch(RotatedSurfaceCode(distance=3), name="right", offset=(8, 0))
    system.register_coupler(RotatedTwoPatchCoupler(), ["left", "right"], "seam",
                            interaction_type="XX")

    initial = system_view_data(system)
    seam = initial["patches"][2]
    assert seam["kind"] == "coupler"
    checks = [op for op in initial["operators"] if op["patch"] == "seam"]
    assert checks and all(not op["active"] for op in checks)
    raw, _ = system.patches["seam"]
    for op, record in zip(checks, raw.stabilizers, strict=True):
        assert dict(op["support"]) == {system.index_map[xy]: p for xy, p in record["pauli"].items()}
        assert op["syndrome_qubit"] == system.index_map[record["syn_coord"]]
    assert any(set(dict(op["support"])) - set(seam["qubits"]) for op in checks)

    system.activate_coupler("seam")
    active = system_view_data(system)
    assert any(op["active"] for op in active["operators"] if op["patch"] == "seam")
    assert any(not op["active"] for op in active["operators"]
               if op["category"] == "stabilizer" and op["patch"] != "seam")
    system.deactivate_coupler("seam")
    assert system_view_data(system) == initial

    system.remove_coupler("seam")
    cleaned = system_view_data(system)
    assert [p["id"] for p in cleaned["patches"]] == ["left", "right"]
    assert all(op["patch"] != "seam" for op in cleaned["operators"])
    assert not set(seam["qubits"]).intersection(q["id"] for q in cleaned["qubits"])


def test_shared_qubits_and_deduplicated_checks_keep_both_declaring_patches():
    system = QECSystem()
    patch = RotatedSurfaceCode(distance=3)
    system.add_patch(patch, name="first")
    system.add_patch(patch, name="reused")
    data = system_view_data(system)
    assert data["patches"][0]["qubits"] == data["patches"][1]["qubits"]
    assert len(data["qubits"]) == patch.num_qubits
    assert all(q["owner"] == "reused" for q in data["qubits"])
    checks = [op for op in data["operators"] if op["category"] == "stabilizer"]
    assert len(checks) == len(patch.stabilizers)
    assert all(op["patches"] == ["first", "reused"] for op in checks)


def test_reused_site_displays_the_current_owners_role_and_preserves_other_declarations():
    class OneSite(QECPatch):
        def _process_params(self):
            pass

        def build(self):
            self.add_qubit(1, 2, self.params["role"])

    system = QECSystem()
    system.add_patch(OneSite(role="data"), name="previous", is_active=False)
    system.add_patch(OneSite(role="syndrome_x"), name="current")
    site, = system_view_data(system)["qubits"]
    assert site["role"] == "syndrome X"
    assert site["owner"] == "current"
    assert site["roles_by_patch"] == {"previous": "data", "current": "syndrome X"}

    del system.qubit_coords[site["id"]]
    with pytest.raises(ValueError, match=r"Registered qubit 0.*no global system coordinates"):
        system_view_data(system)


def test_disabled_site_projects_checks_without_changing_the_registered_definition():
    system = QECSystem()
    placed = system.add_patch(RotatedSurfaceCode(distance=3), name="memory")
    q = min(placed.data_indices)
    declared = deepcopy(system.stabilizers)
    system.disable_data_qubits([q])
    data = system_view_data(system)
    assert [site["id"] for site in data["qubits"] if site.get("disabled")] == [q]
    assert all(q not in dict(op["support"])
               for op in data["operators"] if op["category"] == "stabilizer")
    assert system.stabilizers == declared


def test_empty_system_and_invalid_global_overlay():
    system = QECSystem()
    empty = system_view_data(system)
    assert empty["qubits"] == empty["patches"] == empty["operators"] == []
    with pytest.raises(ValueError, match="unknown qubits"):
        system_view_data(system, overlays=[{"name": "unknown", "qubits": [0]}])


def test_size_limit_counts_unique_sites_across_all_patches():
    system = QECSystem()
    patch = RotatedSurfaceCode(distance=13)
    system.add_patch(patch, name="first")
    system.add_patch(patch, name="overlap")
    assert len(system_view_data(system)["qubits"]) == patch.num_qubits
    system.add_patch(patch, name="distinct", offset=(50, 0))
    with pytest.raises(ValueError, match="System visualization supports at most 500"):
        system_view_data(system)
