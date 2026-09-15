"""H-family algebra, geometry, compatibility, and concurrent extraction."""

from itertools import product

import numpy as np
import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode, HCodeExtractionBlock, HSixCode

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("n", [6, 8, 10, 16, 32, 64])
def test_family_algebra_and_distance(n):
    patch = HCode(n=n)
    assert patch.num_qubits == n + 4 and patch.num_logicals == n - 4
    hx, hz = patch.get_parity_check_matrix()
    assert hx.shape == hz.shape == (2, n + 4)
    assert not hx[:, n:].any() and not hz[:, n:].any()
    hx, hz = hx[:, :n], hz[:, :n]
    assert np.array_equal(hx, hz) and not ((hx @ hz.T) % 2).any()
    assert hx.tolist() == [[1] * 4 + [0] * (n - 4), [0, 0] + [1] * (n - 2)]

    def pauli(record):
        p = stim.PauliString(n + 4)
        for q, basis in record["pauli"].items():
            p[q] = basis
        return p

    checks = [pauli(s) for s in patch.stabilizers]
    logicals = [pauli(s) for s in patch.logical_ops]
    # Four independent generators over GF(2), not real-valued matrix rank.
    group = set()
    for choices in product((0, 1), repeat=4):
        p = stim.PauliString(n + 4)
        for include, check in zip(choices, checks):
            if include:
                p *= check
        group.add(str(p))
    assert len(group) == 16
    assert all(a.commutes(b) for a in checks for b in checks)
    assert len(logicals) == 2 * (n - 4)
    for i, a in enumerate(logicals):
        assert all(a.commutes(b) for b in checks)
        for j, b in enumerate(logicals):
            assert a.commutes(b) == (i == j or i // 2 != j // 2)
    for q, p in product(range(n), "XYZ"):
        error = stim.PauliString(n + 4)
        error[q] = p
        assert any(not error.commutes(s) for s in checks)
    witness = stim.PauliString("XX" + "I" * (n + 2))
    assert all(witness.commutes(s) for s in checks)
    assert any(not witness.commutes(op) for op in logicals)

    rng = np.random.default_rng(98)
    for bits in rng.integers(0, 2, size=(32, n)):
        for helper, records in (
            (patch.get_stabs, [s for s in patch.stabilizers if s["type"] == "Z"]),
            (patch.get_logicals, [s for s in patch.logical_ops if s["type"] == "Z"]),
        ):
            assert helper(bits) == [sum(bits[q] for q in s["data_indices"]) % 2 for s in records]


@pytest.mark.parametrize("n", [None, True, 4, 5, 7, 8.0, "8"])
def test_invalid_family_size(n):
    with pytest.raises(ValueError, match="even integer"):
        HCode(n=n)


def test_h6_compatibility_paths_and_exact_patch():
    from lightstim.qec_code.H_six import HSixCode as LegacyCode
    from lightstim.qec_code.H_six.code_patch import HSixCode as LegacyPatch
    from lightstim.qec_code.H_six.SE_block import HSixExtractionBlock

    assert LegacyCode is LegacyPatch is HSixCode
    assert HSixExtractionBlock is HCodeExtractionBlock
    family = HCode.from_config({"n": 6, "shift": (20, 4)})
    old = HSixCode(shift=(20, 4))
    for attr in ("qubit_coords", "data_indices", "syndrome_indices", "stabilizers", "logical_ops"):
        assert getattr(family, attr) == getattr(old, attr)
    assert isinstance(old, HCode)
    with pytest.raises(ValueError, match="requires n=6"):
        HSixCode(n=8)


@pytest.mark.parametrize("n", [6, 8, 16])
def test_coordinates_center_checks_and_accumulate_shift(n):
    patch = HCode(n, shift=(20, 4))
    patch.shift_coords(3, -2)
    assert patch.shift == (23, 2)
    assert patch.data_coords == [(2 * q + 23, 2) for q in range(n)]
    assert len(set(patch.qubit_coords.values())) == n + 4
    assert patch.syndrome_coords_x == [(26, 3), (n + 24, 3)]
    assert patch.syndrome_coords_z == [(26, 1), (n + 24, 1)]
    for check in patch.stabilizers:
        xs = [patch.qubit_coords[q][0] for q in check["data_indices"]]
        assert check["syn_coord"][0] == sum(xs) / len(xs)
        assert check["syn_coord"] == patch.qubit_coords[check["syn_idx"]]


def _assert_ideal_extraction(system, se):
    # Compare the entire Clifford unitary on arbitrary data and ancilla states,
    # including logical action. Deterministic memory detectors alone cannot do this.
    actual = stim.Circuit()
    for inst in se.circuit:
        if inst.name in ("H", "CX"):
            actual.append(inst)
    expected = stim.Circuit()
    xs = sorted(system.active_syndrome_indices_x)
    if xs:
        expected.append("H", xs)
    for basis, checks in (("X", system.active_stabilizers_x), ("Z", system.active_stabilizers_z)):
        for s in checks:
            for q in s["data_indices"]:
                expected.append("CX", [s["syn_idx"], q] if basis == "X" else [q, s["syn_idx"]])
    if xs:
        expected.append("H", xs)
    assert stim.Tableau.from_circuit(actual) == stim.Tableau.from_circuit(expected)
    expected_edges = sorted((s["type"], s["syn_idx"], q) for s in system.active_stabilizers for q in s["data_indices"])
    actual_edges = sorted((b, a, q) for b, layers in (("X", se.x_layers), ("Z", se.z_layers)) for layer in layers for a, q in layer)
    assert actual_edges == expected_edges
    for layer in se.cnot_layers:
        targets = [q for pair in layer for q in pair]
        assert len(targets) == len(set(targets))
    assert se.circuit[-1].name == "M"
    assert {t.value for t in se.circuit[-1].targets_copy()} == set(system.active_syndrome_indices)


@pytest.mark.parametrize("n", [6, 8, 10, 16, 32, 64])
def test_concurrent_extraction_measures_exact_checks(n):
    system = QECSystem()
    system.add_patch(HCode(n=n), name="h")
    se = HCodeExtractionBlock(system)
    assert se.cnot_depth == n + 2
    assert sum(len(layer) for layer in se.cnot_layers) == 2 * n + 4
    assert any(xs and zs for xs, zs in zip(se.x_layers, se.z_layers))
    _assert_ideal_extraction(system, se)


def test_extraction_multiple_sizes_global_indices_and_inactive_checks():
    system = QECSystem()
    system.add_patch(HCode(6), name="six")
    system.add_patch(HCode(10), offset=(0, 6), name="ten")
    system.add_patch(HCode(16), offset=(0, 12), name="inactive", is_active=False)
    se = HCodeExtractionBlock(system)
    assert se.cnot_depth == 12
    _assert_ideal_extraction(system, se)
    # Mask individual checks too, not just whole patches.
    system.active_stabilizer_indices.remove(min(system.active_stabilizer_indices))
    _assert_ideal_extraction(system, HCodeExtractionBlock(system))


def test_extraction_rejects_modified_or_foreign_checks():
    for before_registration in (False, True):
        system = QECSystem()
        patch = HCode(8)
        if before_registration:
            patch.stabilizers[0]["pauli"].pop(0)
        system.add_patch(patch, name="h")
        if not before_registration:
            system.stabilizers[0]["pauli"].pop(0)
        with pytest.raises(ValueError, match="unmodified"):
            HCodeExtractionBlock(system)

    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="surface")
    with pytest.raises(ValueError, match="H-code checks only"):
        HCodeExtractionBlock(system)


@pytest.mark.parametrize("n", [6, 8, 12])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_family_default_memory(n, basis, rounds):
    exp = MemoryExperiment(qec_patch=HCode(n), basis=basis, rounds=rounds)
    assert exp.block_class is HCodeExtractionBlock
    c = exp.build()
    assert c.num_qubits == n + 4 and c.num_observables == n - 4
    assert c.num_detectors == 4 * rounds
    c.detector_error_model()
    dets, obs = c.compile_detector_sampler(seed=98).sample(128, separate_observables=True)
    assert not dets.any() and not obs.any()
