"""H6 patch algebra, placement, and native memory integration."""

from itertools import product

import numpy as np
import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_six import HSixCode, HSixExtractionBlock

pytestmark = pytest.mark.smoke


def test_patch_algebra_and_code_distance():
    patch = HSixCode()
    assert patch.num_qubits == 10
    assert len(patch.data_indices) == 6
    assert len(patch.syndrome_indices_x) == len(patch.syndrome_indices_z) == 2
    assert patch.num_logicals == 2

    def as_pauli(record):
        pauli = stim.PauliString(patch.num_qubits)
        for q, factor in record["pauli"].items():
            pauli[q] = factor
        return pauli

    checks = [as_pauli(s) for s in patch.stabilizers]
    logicals = [as_pauli(op) for op in patch.logical_ops]
    assert len(checks) == len(logicals) == 4
    hx, hz = patch.get_parity_check_matrix()
    assert hx[:, :6].tolist() == [[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]]
    assert np.array_equal(hx, hz)
    assert not ((hx @ hz.T) % 2).any()
    # Enumerate over GF(2), rather than use real-valued matrix rank.
    group = set()
    for choices in product((0, 1), repeat=4):
        p = stim.PauliString(patch.num_qubits)
        for include, check in zip(choices, checks):
            if include:
                p *= check
        group.add(str(p))
    assert len(group) == 16  # Four independent generators: k = 6 - 4.

    expected = ("XIXIXI", "ZIZIZI", "IXIXIX", "IZIZIZ")
    for op, paulis in zip(logicals, expected):
        assert op == stim.PauliString(paulis + "IIII")
        assert all(op.commutes(check) for check in checks)
    for i, a in enumerate(logicals):
        for j, b in enumerate(logicals):
            assert a.commutes(b) == (i == j or i // 2 != j // 2)

    # Every weight-one Pauli is detected; X0 X1 is a weight-two logical.
    for q, pauli in product(range(6), (1, 2, 3)):
        error = stim.PauliString(patch.num_qubits)
        error[q] = pauli
        assert any(not error.commutes(check) for check in checks)
    witness = stim.PauliString("XXIIIIIIII")
    assert all(witness.commutes(check) for check in checks)
    assert any(not witness.commutes(op) for op in logicals)


def test_extraction_uses_disjoint_cnot_layers_and_global_indices():
    system = QECSystem()
    system.add_patch(HSixCode(), name="first")
    system.add_patch(HSixCode(shift=(20, 4)), name="second")
    se = HSixExtractionBlock(system)
    assert se.cnot_depth == 8
    assert any(xs and zs for xs, zs in zip(se.x_layers, se.z_layers))
    for layers, stabilizers in (
        (se.x_layers, system.active_stabilizers_x),
        (se.z_layers, system.active_stabilizers_z),
    ):
        assert sorted(edge for layer in layers for edge in layer) == sorted(
            (s["syn_idx"], q) for s in stabilizers for q in s["data_indices"]
        )
        for layer in layers:
            targets = [q for edge in layer for q in edge]
            assert len(targets) == len(set(targets))
    assert se.circuit[-1].name == "M"


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_memory_experiment_wrapper(basis, rounds):
    experiment = MemoryExperiment(
        qec_patch=HSixCode(shift=(20, 4)), rounds=rounds, basis=basis
    )
    assert experiment.block_class is HSixExtractionBlock
    circuit = experiment.build()
    assert circuit.num_observables == 2
    assert circuit.num_detectors == 4 * rounds
    circuit.detector_error_model()
    dets, obs = circuit.compile_detector_sampler(seed=98).sample(
        256, separate_observables=True
    )
    assert not dets.any() and not obs.any()


def test_shift_metadata_matches_geometry_and_config():
    patch = HSixCode.from_config({"shift": [20, 4]})
    assert patch.shift == (20, 4)
    patch.shift_coords(3, -2)
    assert patch.shift == (23, 2)
    for q, (x, y) in HSixCode().qubit_coords.items():
        assert patch.qubit_coords[q] == (x + 23, y + 2)
    for check in patch.stabilizers:
        assert check["syn_coord"] == patch.qubit_coords[check["syn_idx"]]


@pytest.mark.parametrize("shift", [None, (1,), (1, 2, 3), ("a", 0), (float("nan"), 0)])
def test_invalid_shift_is_rejected(shift):
    with pytest.raises(ValueError, match="finite real"):
        HSixCode(shift=shift)


def test_fixed_code_rejects_protocol_parameters():
    with pytest.raises(ValueError, match="Unknown"):
        HSixCode(h_check_ancillas=2)


def test_classical_parities_match_registered_supports():
    patch = HSixCode()
    z_checks = [s for s in patch.stabilizers if s["type"] == "Z"]
    z_logicals = [s for s in patch.logical_ops if s["type"] == "Z"]
    for bits in product((0, 1), repeat=6):
        for helper, operators in (
            (HSixCode.get_stabs, z_checks),
            (HSixCode.get_logicals, z_logicals),
        ):
            assert helper(bits) == [
                sum(bits[q] for q in op["data_indices"]) % 2 for op in operators
            ]
