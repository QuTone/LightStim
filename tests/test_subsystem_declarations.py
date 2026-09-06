"""Subsystem declarations preserve algebra, placement, and ancilla roles."""

import copy

import numpy as np
import pytest

from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem


class SmallSubsystem(QECPatch):
    """One protected qubit and one gauge qubit, with centre Z0 Z1."""

    def _process_params(self):
        pass

    def build(self):
        for x in range(3):
            self.add_qubit(x, 0, "data")
        self.add_qubit(0, 1, "syndrome_x")
        self.add_qubit(1, 1, "syndrome_z")
        self.add_qubit(2, 1, "syndrome_z")
        self.create_stim_stabilizer({(0, 0): "Z", (1, 0): "Z"}, type="Z")
        self.create_stim_gauge({(0, 0): "X", (1, 0): "X"}, (0, 1), "X")
        self.create_stim_gauge({(0, 0): "Z"}, (1, 1), "Z")
        self.create_stim_gauge({(1, 0): "Z"}, (2, 1), "Z")
        self.create_stim_logical({(2, 0): "X"}, "X")
        self.create_stim_logical({(2, 0): "Z"}, "Z")
        self.num_logicals = 1


class MixedSubsystem(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        self.add_qubit(0, 0, "data")
        self.add_qubit(1, 0, "data")
        first = {(0, 0): "X", (1, 0): "Z"}
        if self.params.get("commuting"):
            second = {(0, 0): "Z", (1, 0): "X"}
            self.create_stim_stabilizer(first, type="Mixed")
            self.create_stim_stabilizer(second, type="Mixed")
            self.num_logicals = 0
        else:
            second = {(0, 0): "Y", (1, 0): "Z"}
            self.num_logicals = 1
        self.create_stim_gauge(first, type="Mixed")
        self.create_stim_gauge(second, type="Mixed")


def test_registration_translates_gauges_and_exposes_ancillas():
    patch = SmallSubsystem()
    original_gauges = copy.deepcopy(patch.gauges)
    system = QECSystem()
    system.add_patch(SmallSubsystem(), name="first")
    placed = system.add_patch(patch, name="second", offset=(10, 20))

    assert patch.gauges == original_gauges
    assert len(system.stabilizers) == 2
    assert len(system.gauges) == 6
    assert placed.gauges == system.gauges[3:]
    assert placed.gauges[0]["pauli"] == {6: "X", 7: "X"}
    assert placed.gauges[0]["syn_idx"] == 9
    assert placed.gauges[0]["syn_coord"] == (10, 21)
    assert all(gauge["patch_name"] == "second" for gauge in placed.gauges)
    assert system.active_syndrome_indices == [3, 4, 5, 9, 10, 11]
    assert system.active_syndrome_indices_x == [3, 9]
    assert system.active_syndrome_indices_z == [4, 5, 10, 11]
    assert len(system.active_gauges_x) == 2
    assert len(system.active_gauges_z) == 4
    placed.gauges[0]["pauli"].clear()
    assert system.gauges[3]["pauli"] == {6: "X", 7: "X"}


def test_inactive_patch_does_not_activate_gauge_readout_ancillas():
    system = QECSystem()
    system.add_patch(SmallSubsystem(), name="inactive", is_active=False)
    assert len(system.gauges) == 3
    assert system.active_gauges == []
    assert system.active_syndrome_indices == []
    assert system.active_syndrome_indices_x == []
    assert system.active_syndrome_indices_z == []


def test_gauge_geometry_follows_shift_transpose_and_rotation():
    patch = SmallSubsystem()
    paulis = [dict(gauge["pauli"]) for gauge in patch.gauges]
    patch.shift_coords(7, 11)
    patch.transpose_coords()
    patch.rotate_coords(np.pi / 2, center=(0, 0))
    for gauge, pauli in zip(patch.gauges, paulis):
        assert gauge["pauli"] == pauli
        assert gauge["syn_coord"] == patch.qubit_coords[gauge["syn_idx"]]
    assert patch.gauges[0]["syn_coord"] == (-7, 12)
    assert patch.stabilizers[0]["syn_idx"] is None
    patch.validate_subsystem_declaration()


@pytest.mark.parametrize("commuting", [False, True])
def test_same_support_mixed_paulis_do_not_collapse(commuting):
    patch = MixedSubsystem(commuting=commuting)
    system = QECSystem()
    system.add_patch(patch)
    assert len(system.gauges) == 2
    assert system.gauges[0]["pauli"] != system.gauges[1]["pauli"]
    assert len(system.stabilizers) == (2 if commuting else 0)


@pytest.mark.parametrize("invalid", [
    "num_logicals", "missing_centre", "outside_span", "noncentral",
    "noncommuting", "syndrome_support", "unknown_support", "signed_factor",
    "support_metadata", "duplicate_support", "sign_metadata", "phase_metadata",
    "missing_data_geometry", "syndrome_metadata",
])
def test_invalid_declaration_fails_before_system_mutation(invalid):
    patch = SmallSubsystem()
    if invalid == "num_logicals":
        patch.num_logicals = 2
    elif invalid == "missing_centre":
        patch.stabilizers.clear()
    elif invalid == "outside_span":
        patch.create_stim_stabilizer({(2, 0): "Z"}, type="Z")
    elif invalid == "noncentral":
        patch.create_stim_stabilizer({(0, 0): "X", (1, 0): "X"}, type="X")
    elif invalid == "noncommuting":
        patch.create_stim_stabilizer({(0, 0): "X"}, type="X")
    elif invalid in ("syndrome_support", "unknown_support"):
        qubit = 3 if invalid == "syndrome_support" else 100
        patch.gauges[0]["pauli"] = {qubit: "X"}
        patch.gauges[0]["data_indices"] = [qubit]
    elif invalid == "signed_factor":
        patch.gauges[0]["pauli"][0] = "-X"
    elif invalid == "duplicate_support":
        patch.gauges[0]["data_indices"] = [0, 1, 1]
    elif invalid in ("sign_metadata", "phase_metadata"):
        patch.gauges[0][invalid.removesuffix("_metadata")] = -1
    elif invalid == "missing_data_geometry":
        del patch.qubit_coords[2]
    elif invalid == "syndrome_metadata":
        patch.gauges[0]["syn_coord"] = (100, 100)
    else:
        patch.gauges[0]["data_indices"] = [0]

    system = QECSystem()
    with pytest.raises(ValueError, match="Subsystem"):
        system.add_patch(patch, name="bad")
    assert system.patches == {}
    assert system.num_qubits == 0
    assert system.stabilizers == []
    assert system.gauges == []
    assert system.num_logicals == 0


def test_redundant_gauge_generators_are_allowed():
    patch = SmallSubsystem()
    patch.gauges.append(copy.deepcopy(patch.gauges[0]))
    patch.validate_subsystem_declaration()


@pytest.mark.parametrize("targets,syndrome", [
    ({(99, 0): "X"}, None),
    ({(0, 1): "X"}, None),
    ({(0, 0): "-X"}, None),
    ({(0, 0): "X"}, (99, 0)),
    ({(0, 0): "X"}, (0, 0)),
])
def test_gauge_helper_rejects_invalid_physical_support(targets, syndrome):
    patch = SmallSubsystem()
    with pytest.raises(ValueError, match="Gauge"):
        patch.create_stim_gauge(targets, syndrome)
    assert len(patch.gauges) == 3


def test_no_gauge_declarations_preserve_existing_registration_behavior():
    patch = SmallSubsystem()
    patch.gauges.clear()
    patch.num_logicals = 3  # The existing static path accepts caller-supplied k.
    system = QECSystem()
    system.add_patch(patch)
    assert system.num_logicals == 3
    assert system.gauges == []
    assert system.active_syndrome_indices == []
