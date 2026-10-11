"""Registered patch views use system IDs consistently in geometry and physics."""

from copy import deepcopy
import json
import re
from types import SimpleNamespace

import pytest
import stim

from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.qec_code.HGP import (
    fold_diagonal_qubits_by_sector,
    fold_mirror_pairs,
    hgp_13_1_3,
)
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)


class _IndexedPatch(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        ids = self.params.get("ids", (0, 1, 2))
        for uid, (x, y, role) in zip(
            ids, ((0, 0, "data"), (2, 0, "data"), (1, 1, "syndrome_x"))
        ):
            self.add_qubit(x, y, role, uid=uid)
        self.create_stim_stabilizer({(0, 0): "X", (2, 0): "X"}, (1, 1), "X")
        self.create_stim_logical({(0, 0): "X"}, "X")
        self.create_stim_logical({(0, 0): "Z", (2, 0): "Z"}, "Z")
        self.num_logicals = 1


def _visualization_data(patch):
    document = patch.visualize().html
    return json.loads(re.search(
        r'<script id="patch-data" type="application/json">(.*?)</script>',
        document, re.S,
    )[1])


def _assert_geometry(system, patch, expected):
    assert patch.qubit_coords == expected
    assert patch.index_map == {coord: q for q, coord in expected.items()}
    assert patch.grid_map == {patch.get_grid_key(coord): q for q, coord in expected.items()}
    assert patch.data_coords == [expected[q] for q in sorted(patch.data_indices)]
    assert patch.syndrome_coords == [expected[q] for q in sorted(patch.syndrome_indices)]
    assert all(system.qubit_coords[q] == coord for q, coord in expected.items())
    assert patch.num_qubits == len(expected)


@pytest.mark.parametrize("preceding_patch", [False, True], ids=["first", "second"])
@pytest.mark.parametrize("ids", [(0, 1, 2), (40, 7, 99), (1, 0, 2)],
                         ids=["sequential", "sparse", "permuted"])
def test_registered_geometry_and_visualization_follow_global_ids(preceding_patch, ids):
    source = _IndexedPatch(ids=ids)
    original = deepcopy(source.__dict__)
    system = QECSystem()
    if preceding_patch:
        system.add_patch(RotatedSurfaceCode(distance=3), name="first")
    placed = system.add_patch(source, name="placed", offset=(10.25, -20.5))
    mapping = system.local_to_global_map["placed"]
    shifted = {q: (x + 10.25, y - 20.5) for q, (x, y) in source.qubit_coords.items()}
    expected = {mapping[q]: coord for q, coord in shifted.items()}

    _assert_geometry(system, placed, expected)
    assert source.__dict__ == original
    stored, offset = system.patches["placed"]
    assert offset == (10.25, -20.5)
    assert stored.qubit_coords == shifted
    assert stored.index_map == {coord: q for q, coord in shifted.items()}
    assert stored.grid_map == {stored.get_grid_key(coord): q for q, coord in shifted.items()}
    assert stored.data_indices == source.data_indices
    assert stored.stabilizers[0]["pauli"] == source.stabilizers[0]["pauli"]

    payload = _visualization_data(placed)
    assert {q["id"]: (q["x"], q["y"]) for q in payload["qubits"]} == expected
    assert {q["id"] for q in payload["qubits"] if q["role"] == "data"} == placed.data_indices
    for rendered, raw in zip(payload["operators"], source.stabilizers + source.logical_ops):
        assert dict(rendered["support"]) == {mapping[q]: p for q, p in raw["pauli"].items()}
    assert payload["operators"][0]["syndrome_qubit"] == mapping[ids[2]]

    # The returned geometry is an independent view, just like its operators.
    placed.qubit_coords.clear()
    placed.index_map.clear()
    placed.grid_map.clear()
    assert stored.qubit_coords == shifted
    assert all(system.qubit_coords[q] == coord for q, coord in expected.items())
    assert source.__dict__ == original


def test_dormant_reuse_preserves_nonidentity_geometry_mapping():
    source = _IndexedPatch()
    previous = deepcopy(source)
    previous.qubit_coords = dict(reversed(list(previous.qubit_coords.items())))
    system = QECSystem()
    system.add_patch(previous, name="previous", offset=(10, 20))
    count = system.num_qubits
    placed = system.add_patch(source, name="reused", offset=(10, 20))

    assert system.num_qubits == count
    assert system.local_to_global_map["reused"] == {0: 2, 1: 1, 2: 0}
    _assert_geometry(system, placed, {2: (10, 20), 1: (12, 20), 0: (11, 21)})
    assert placed.stabilizers[0]["syn_idx"] == 0
    assert placed.stabilizers[0]["pauli"] == {2: "X", 1: "X"}


def test_registered_subsystem_gauge_geometry_is_self_consistent():
    source = BaconShorCode(distance=3)
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="first")
    placed = system.add_patch(source, name="subsystem", offset=(20, 30))
    mapping = system.local_to_global_map["subsystem"]
    expected = {mapping[q]: (x + 20, y + 30) for q, (x, y) in source.qubit_coords.items()}
    _assert_geometry(system, placed, expected)

    for gauge in placed.gauges:
        assert gauge["syn_coord"] == placed.qubit_coords[gauge["syn_idx"]]
        assert set(gauge["pauli"]) <= placed.data_indices
    payload = _visualization_data(placed)
    assert sum(record["category"] == "gauge" for record in payload["operators"]) == len(source.gauges)


def test_rotated_hadamard_permutation_uses_second_patch_geometry():
    system = QECSystem()
    first = system.add_patch(RotatedSurfaceCode(distance=3), name="first")
    placed = system.add_patch(RotatedSurfaceCode(distance=3), name="second", offset=(20, 30))
    ops = RotatedSurfaceCodeLogicalOpSet(RotatedSurfaceCodeExtractionBlock)
    layers = ops._rot90_swap_layers(system, placed)
    circuit = stim.Circuit()
    circuit.append("I", range(system.num_qubits))
    for layer in layers:
        assert all(q in placed.data_indices for pair in layer for q in pair)
        circuit.append("SWAP", [q for pair in layer for q in pair])
    tableau = stim.Tableau.from_circuit(circuit)
    for q in placed.data_indices:
        x, y = system.qubit_coords[q]
        rotated = (56 - y, x + 10)
        image = stim.PauliString(system.num_qubits)
        image[system.index_map[rotated]] = "X"
        assert tableau.x_output(q) == image
    for q in first.data_indices:
        image = stim.PauliString(system.num_qubits)
        image[q] = "X"
        assert tableau.x_output(q) == image


@pytest.mark.parametrize("reuse", [False, True], ids=["second", "dormant-reuse"])
def test_hgp_fold_mapping_preserves_local_sector_metadata(reuse):
    source = hgp_13_1_3()
    system = QECSystem()
    previous = deepcopy(source)
    if reuse:
        # Same global ID set, with a different qubit at each index.
        previous.qubit_coords = dict(reversed(list(previous.qubit_coords.items())))
    system.add_patch(previous, name="previous")
    placed = system.add_patch(source, name="placed", offset=(0, 0) if reuse else (20, 30))
    builder = SimpleNamespace(system=system)
    mapping = system.local_to_global_map["placed"]

    assert any(local != global_ for local, global_ in mapping.items())
    assert placed.vv_qubits == source.vv_qubits
    assert placed.cc_qubits == source.cc_qubits
    local_pairs = fold_mirror_pairs(source)
    assert fold_mirror_pairs(placed) == local_pairs
    assert fold_mirror_pairs(placed, builder) == [(mapping[a], mapping[b]) for a, b in local_pairs]
    local_diagonals = fold_diagonal_qubits_by_sector(source)
    assert fold_diagonal_qubits_by_sector(placed, builder) == tuple(
        [mapping[q] for q in sector] for sector in local_diagonals
    )
