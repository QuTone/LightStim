"""Fold-transversal H-SWAP on symmetric hypergraph-product patches (Xu et al. Table I)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import stim

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import assert_noiseless, build_quiet

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.hgp_logical_gates import build_hgp_gate_verification_circuit
from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock
from lightstim.qec_code.HGP import (
    BinaryParityCheck,
    HGPCode,
    HGPCodeLogicalOpSet,
    fold_diagonal_qubits,
    fold_h_layer_circuit,
    fold_h_swap_circuit,
    fold_logical_permutation,
    fold_mirror_pairs,
    fold_swap_layer_circuit,
    hgp_13_1_3,
    hgp_18_2_3,
    hgp_225_9_4,
    is_symmetric_hgp,
)
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode


HS = "fold_transversal_h_swap"
NOISE = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)

HAMMING = BinaryParityCheck.from_dense(
    np.array(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]],
        dtype=np.uint8,
    )
)
REP3 = BinaryParityCheck.from_dense(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))


def hgp_hamming_squared() -> HGPCode:
    """[[58,16,3]]: Hamming [7,4,3] with itself, a 4 x 4 logical grid."""
    return HGPCode(HAMMING, d=3)


INSTANCES = {
    "hgp_13_1_3": hgp_13_1_3,          # rep3 x rep3, full rank, k = 1 (diagonal only)
    "hgp_18_2_3": hgp_18_2_3,          # cyc3 x cyc3, redundant check, one C1xC2 logical
    "hgp_225_9_4": hgp_225_9_4,        # 3 x 3 logical grid
    "hgp_hamming2": hgp_hamming_squared,  # 4 x 4 logical grid, explicit H2 = H1
}


def _global_patch(factory, name="hgp"):
    system = QECSystem()
    patch = system.add_patch(factory(), name=name)
    return system, patch


class _SystemOnly:
    """Minimal stand-in for a builder when only ``builder.system`` is needed."""

    def __init__(self, system):
        self.system = system


def _pauli(num_qubits, indices, letter):
    chars = ["_"] * num_qubits
    for q in indices:
        chars[q] = letter
    return stim.PauliString("".join(chars))


def _padded_tableau(circuit, num_qubits):
    padded = stim.Circuit()
    padded.append("I", list(range(num_qubits)))
    padded += circuit
    return stim.Tableau.from_circuit(padded)


def _logical_records(patch):
    records = {}
    for record in patch.logical_ops:
        records.setdefault(int(record["logical_id"]), {})[record["type"]] = tuple(
            sorted(int(q) for q in record["data_indices"])
        )
    return records


def _raw_observable_parity(circuit, shots=64, seed=1):
    """Raw measurement parity of every OBSERVABLE_INCLUDE (no reference subtraction)."""
    raw = circuit.compile_sampler(seed=seed).sample(shots)
    columns = []
    for inst in circuit.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            recs = [circuit.num_measurements + t.value for t in inst.targets_copy()]
            columns.append(np.bitwise_xor.reduce(raw[:, recs], axis=1))
    return np.array(columns).T


# ---------------------------------------------------------------------------
# Geometry of the fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_fold_pairs_and_diagonal_partition_the_data_qubits(name):
    system, patch = _global_patch(INSTANCES[name])
    builder = _SystemOnly(system)
    pairs = fold_mirror_pairs(patch, builder)
    diagonal = fold_diagonal_qubits(patch, builder)

    swapped = [q for pair in pairs for q in pair]
    assert len(set(swapped)) == len(swapped), "a qubit appears in two SWAP pairs"
    assert not set(swapped) & set(diagonal)
    assert sorted(set(swapped) | set(diagonal)) == sorted(patch.data_indices)

    # Diagonal = n1 (V1xV2) + m1 (C1xC2) qubits, pairs = the rest halved.
    n1, m1 = patch.h1.num_bits, patch.h1.num_checks
    assert len(diagonal) == n1 + m1
    assert len(pairs) == (n1 * n1 - n1) // 2 + (m1 * m1 - m1) // 2

    # Every pair is a coordinate transposition inside one sector.
    local_pairs = fold_mirror_pairs(patch)
    for sector in (patch.vv_qubits, patch.cc_qubits):
        inverse = {index: key for key, index in sector.items()}
        for a, b in local_pairs:
            if a in inverse:
                assert inverse[b] == inverse[a][::-1]
                assert inverse[a][0] < inverse[a][1]


def test_local_and_global_pairs_agree_through_the_system_index_map():
    system, patch = _global_patch(hgp_225_9_4)
    local_pairs = fold_mirror_pairs(patch)
    global_pairs = fold_mirror_pairs(patch, _SystemOnly(system))
    remap = {
        local: system.index_map[patch.qubit_coords[local]]
        for sector in (patch.vv_qubits, patch.cc_qubits)
        for local in sector.values()
    }
    assert [(remap[a], remap[b]) for a, b in local_pairs] == global_pairs


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_logical_permutation_is_the_transposition_of_seed_indices(name):
    _, patch = _global_patch(INSTANCES[name])
    permutation = fold_logical_permutation(patch)
    assert sorted(permutation) == sorted(permutation.values()) == list(range(patch.num_logicals))
    by_id = {int(r["logical_id"]): r for r in patch.logical_pairs}
    for logical_id, mirror in permutation.items():
        assert permutation[mirror] == logical_id                      # involution
        assert by_id[mirror]["sector"] == by_id[logical_id]["sector"]
        assert tuple(by_id[mirror]["seed_indices"]) == tuple(by_id[logical_id]["seed_indices"])[::-1]
    fixed = [i for i, j in permutation.items() if i == j]
    k1 = len(patch.kernel_h1.basis)                # V1xV2 grid is k1 x k1
    k1t = len(patch.kernel_h1_transpose.basis)     # C1xC2 grid is k1t x k1t
    assert len(fixed) == k1 + k1t                  # the logical diagonal


def test_hgp_225_permutation_transposes_the_3x3_grid():
    _, patch = _global_patch(hgp_225_9_4)
    assert fold_logical_permutation(patch) == {
        0: 0, 1: 3, 2: 6, 3: 1, 4: 4, 5: 7, 6: 2, 7: 5, 8: 8
    }


# ---------------------------------------------------------------------------
# Exact Clifford action (stim tableau)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_h_swap_permutes_x_checks_onto_z_checks_exactly(name):
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    data = sorted(patch.data_indices)
    tableau = _padded_tableau(fold_h_swap_circuit(data, fold_mirror_pairs(patch, _SystemOnly(system))), n)

    x_checks = [_pauli(n, s["data_indices"], "X") for s in patch.stabilizers if s["type"] == "X"]
    z_checks = [_pauli(n, s["data_indices"], "Z") for s in patch.stabilizers if s["type"] == "Z"]
    assert {str(tableau(s)) for s in x_checks} == {str(s) for s in z_checks}
    assert {str(tableau(s)) for s in z_checks} == {str(s) for s in x_checks}
    assert all(tableau(s).sign == 1 for s in x_checks + z_checks)


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_h_swap_maps_every_logical_pair_to_hadamard_of_its_mirror_exactly(name):
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    data = sorted(patch.data_indices)
    tableau = _padded_tableau(fold_h_swap_circuit(data, fold_mirror_pairs(patch, _SystemOnly(system))), n)
    records = _logical_records(patch)
    permutation = fold_logical_permutation(patch)
    assert len(records) == system.num_logicals

    for logical_id, support in records.items():
        mirror = records[permutation[logical_id]]
        # X̄(l1,l2) -> +Z̄(l2,l1) and Z̄(l1,l2) -> +X̄(l2,l1), as exact Pauli strings.
        assert tableau(_pauli(n, support["X"], "X")) == _pauli(n, mirror["Z"], "Z")
        assert tableau(_pauli(n, support["Z"], "Z")) == _pauli(n, mirror["X"], "X")


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_h_swap_is_an_involution(name):
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    circuit = fold_h_swap_circuit(sorted(patch.data_indices), fold_mirror_pairs(patch, _SystemOnly(system)))
    assert _padded_tableau(circuit + circuit, n) == stim.Tableau(n)


def test_diagonal_logical_of_13_1_3_gets_a_plain_hadamard():
    system, patch = _global_patch(hgp_13_1_3)
    n = system.num_qubits
    (record,) = _logical_records(patch).values()
    tableau = _padded_tableau(
        fold_h_swap_circuit(sorted(patch.data_indices), fold_mirror_pairs(patch, _SystemOnly(system))), n
    )
    assert tableau(_pauli(n, record["X"], "X")) == _pauli(n, record["Z"], "Z")
    assert tableau(_pauli(n, record["Z"], "Z")) == _pauli(n, record["X"], "X")


def test_13_1_3_fold_equals_the_unrotated_surface_code_fold_hadamard():
    """rep3 x rep3 is the d=3 unrotated surface patch; same H and SWAP counts as its fold H."""
    system, patch = _global_patch(hgp_13_1_3)
    hgp_circuit = fold_h_swap_circuit(sorted(patch.data_indices), fold_mirror_pairs(patch, _SystemOnly(system)))

    from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCodeLogicalOpSet
    ref_system = QECSystem()
    ref_patch = ref_system.add_patch(UnrotatedSurfaceCode(distance=3), name="usc")
    tracker = SyndromeTracker(num_qubits=ref_system.num_qubits, expected_num_logicals=ref_system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=ref_system, if_detector=True)
    ref_system.register_tracker(tracker)
    ref_system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in sorted(ref_system.data_indices)}, ref_system.num_qubits)
    UnrotatedSurfaceCodeLogicalOpSet().fold_transversal_hadamard(builder, ref_patch)

    def counts(circuit):
        out = {}
        for inst in circuit.flattened():
            if inst.name in ("H", "SWAP"):
                width = 2 if inst.name == "SWAP" else 1
                out[inst.name] = out.get(inst.name, 0) + len(inst.targets_copy()) // width
        return out

    assert counts(hgp_circuit) == {"H": 13, "SWAP": 4}     # 3 V1xV2 pairs + 1 C1xC2 pair
    assert counts(builder.circuit) == counts(hgp_circuit)


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------

def test_circuit_helpers_layer_structure():
    pairs = [(0, 3), (1, 2)]
    circuit = fold_h_swap_circuit([0, 1, 2, 3, 4], pairs)
    names = [inst.name for inst in circuit]
    assert names == ["H", "TICK", "SWAP"]
    assert circuit[0].targets_copy() == [stim.GateTarget(q) for q in range(5)]
    assert [t.value for t in circuit[2].targets_copy()] == [0, 3, 1, 2]
    assert len(fold_h_layer_circuit([])) == 0
    assert len(fold_swap_layer_circuit([])) == 0
    assert [inst.name for inst in fold_h_swap_circuit([1], [])] == ["H"]


# ---------------------------------------------------------------------------
# Op set through the builder / executor
# ---------------------------------------------------------------------------

def _builder_for(factory, name="hgp"):
    system = QECSystem()
    patch = system.add_patch(factory(), name=name)
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in data}, system.num_qubits)
    system.active_qubit_indices.update(data)
    return system, patch, builder


def test_op_set_emits_h_then_swap_and_returns_the_same_circuit():
    system, patch, builder = _builder_for(hgp_18_2_3)
    before = len(builder.circuit.flattened())
    emitted = HGPCodeLogicalOpSet().fold_transversal_h_swap(builder, patch)
    expected = fold_h_swap_circuit(sorted(patch.data_indices), fold_mirror_pairs(patch, builder))
    assert emitted == expected
    appended = [inst for inst in builder.circuit.flattened()[before:] if inst.name in ("H", "SWAP")]
    assert [inst.name for inst in appended] == ["H", "SWAP"]
    assert sorted(t.value for t in appended[0].targets_copy()) == sorted(patch.data_indices)
    assert len(appended[1].targets_copy()) == 2 * len(fold_mirror_pairs(patch, builder))


def test_noiseless_and_noisy_swap_flags_drop_the_right_gate_noise():
    base = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(HS, {})], noise_params=NOISE))
    quiet_swap = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(HS, {"noisy_swap": False})], noise_params=NOISE))
    all_quiet = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(HS, {})], noise_params=NOISE, noiseless_gates=True))

    def count(circuit, name, width):
        return sum(len(inst.targets_copy()) // width for inst in circuit.flattened() if inst.name == name)

    pairs = len(fold_mirror_pairs(_global_patch(hgp_13_1_3)[1]))
    # noisy_swap=False removes exactly the two-qubit noise of the SWAP pairs.
    assert count(base, "DEPOLARIZE2", 2) - count(quiet_swap, "DEPOLARIZE2", 2) == pairs
    assert count(all_quiet, "DEPOLARIZE2", 2) == count(quiet_swap, "DEPOLARIZE2", 2)
    # noiseless=True additionally removes the single-qubit noise of the 13 H gates.
    assert count(base, "DEPOLARIZE1", 1) - count(all_quiet, "DEPOLARIZE1", 1) >= 13
    assert count(quiet_swap, "DEPOLARIZE1", 1) >= count(all_quiet, "DEPOLARIZE1", 1)


def test_executor_dispatch_matches_direct_call():
    system, patch, builder = _builder_for(hgp_18_2_3)
    executor = LogicalExecutor(builder)
    executor.register_op_set(HGPCode, HGPCodeLogicalOpSet())
    executor.apply_logical_operation(HS, [patch])
    direct = fold_h_swap_circuit(sorted(patch.data_indices), fold_mirror_pairs(patch, builder))
    gates = [inst for inst in builder.circuit.flattened() if inst.name in ("H", "SWAP")]
    assert [inst.name for inst in gates] == [inst.name for inst in direct if inst.name != "TICK"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_non_symmetric_hgp_is_rejected():
    system = QECSystem()
    patch = system.add_patch(HGPCode(REP3, HAMMING), name="hgp")
    assert not is_symmetric_hgp(patch)
    with pytest.raises(ValueError, match="symmetric"):
        fold_mirror_pairs(patch)
    with pytest.raises(ValueError, match="symmetric"):
        fold_logical_permutation(patch)
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in sorted(system.data_indices)}, system.num_qubits)
    with pytest.raises(ValueError, match="symmetric"):
        HGPCodeLogicalOpSet().fold_transversal_h_swap(builder, patch)


def test_same_shape_different_entries_is_rejected():
    twisted = BinaryParityCheck.from_dense(np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8))
    patch = HGPCode(REP3, twisted)
    assert not is_symmetric_hgp(patch)
    with pytest.raises(ValueError, match="different entries"):
        fold_mirror_pairs(patch)


def test_explicit_equal_h2_counts_as_symmetric():
    assert is_symmetric_hgp(HGPCode(HAMMING, HAMMING))
    assert is_symmetric_hgp(HGPCode(HAMMING))


def test_local_patch_and_wrong_type_are_rejected():
    system, patch, builder = _builder_for(hgp_13_1_3)
    with pytest.raises(ValueError, match="global patch"):
        HGPCodeLogicalOpSet().fold_transversal_h_swap(builder, hgp_13_1_3())
    with pytest.raises(TypeError, match="HGPCode"):
        HGPCodeLogicalOpSet().fold_transversal_h_swap(builder, UnrotatedSurfaceCode(distance=3))
    # The exported helpers apply the same guard when asked for global indices.
    with pytest.raises(ValueError, match="global patch"):
        fold_mirror_pairs(hgp_13_1_3(), builder)
    with pytest.raises(ValueError, match="global patch"):
        fold_diagonal_qubits(hgp_13_1_3(), builder)
    with pytest.raises(TypeError, match="HGPCode"):
        fold_logical_permutation(UnrotatedSurfaceCode(distance=3))
    # Local indices are still available without a builder.
    assert fold_mirror_pairs(hgp_13_1_3()) == fold_mirror_pairs(patch)


def test_helpers_do_not_return_another_patchs_qubits_for_a_local_patch():
    """A local hgp patch whose local coordinates fall inside a registered surface patch."""
    system = QECSystem()
    system.add_patch(UnrotatedSurfaceCode(distance=9), name="usc")
    system.add_patch(hgp_18_2_3(), offset=(40, 0), name="hgp")
    with pytest.raises(ValueError, match="global patch"):
        fold_mirror_pairs(hgp_18_2_3(), _SystemOnly(system))


# ---------------------------------------------------------------------------
# End to end through the verification harness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(INSTANCES))
@pytest.mark.parametrize("bases", [("Z", "X"), ("X", "Z")])
def test_harness_noiseless_h_swap_reads_out_every_logical(name, bases):
    init_basis, measure_basis = bases
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        INSTANCES[name](), [(HS, {})], init_basis=init_basis, measure_basis=measure_basis
    ))
    _, patch = _global_patch(INSTANCES[name])
    assert circuit.num_observables == patch.num_logicals
    assert_noiseless(circuit)
    assert not _raw_observable_parity(circuit).any()   # +H̄: no logical Pauli left behind


@pytest.mark.parametrize("name", ["hgp_13_1_3", "hgp_18_2_3", "hgp_hamming2"])
def test_harness_two_h_swaps_cancel(name):
    for init_basis in ("Z", "X"):
        circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
            INSTANCES[name](), [(HS, {}), (HS, {})], init_basis=init_basis, measure_basis=init_basis
        ))
        _, patch = _global_patch(INSTANCES[name])
        assert circuit.num_observables == patch.num_logicals
        assert_noiseless(circuit)
        assert not _raw_observable_parity(circuit).any()


def test_harness_rejects_the_h_layer_without_its_swap_layer(monkeypatch):
    """H^{⊗n} alone maps the code onto a different code; the tracker would
    silently drop the mismatched stabilizers, so the harness must refuse."""

    def h_only(self, builder, patch, noiseless=False):
        builder.apply_unitary_block(
            fold_h_layer_circuit(sorted(patch.data_indices)), noiseless=noiseless
        )

    monkeypatch.setattr(HGPCodeLogicalOpSet, "h_only", h_only, raising=False)
    for name in ("hgp_13_1_3", "hgp_18_2_3", "hgp_225_9_4"):
        with pytest.raises(ValueError, match="stabilizer group"):
            build_quiet(lambda: build_hgp_gate_verification_circuit(
                INSTANCES[name](), [("h_only", {})]
            ))
    # The genuine gate passes the same check with every rounds setting.
    for rounds in (1, 2, 3):
        circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
            hgp_18_2_3(), [(HS, {})], rounds=rounds
        ))
        assert_noiseless(circuit)


def test_harness_rejects_unresolvable_readout():
    with pytest.raises(ValueError, match="resolvable"):
        build_quiet(lambda: build_hgp_gate_verification_circuit(
            hgp_13_1_3(), [(HS, {})], init_basis="Z", measure_basis="Z"
        ))


def test_harness_noisy_circuit_has_a_detector_error_model():
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_18_2_3(), [(HS, {})], noise_params=NOISE
    ))
    dem = circuit.detector_error_model(decompose_errors=False)
    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == 2


def _circuit_level_distance(circuit):
    """Exact minimum number of faults giving an undetected logical error.

    The search caps must exceed the largest number of detectors any single
    DEM error touches (HGP circuits have hyperedges), otherwise errors are
    excluded from the search and the distance can be over-reported.
    """
    dem = circuit.detector_error_model(decompose_errors=False)
    max_degree = max(
        (sum(1 for target in inst.targets_copy() if target.is_relative_detector_id())
         for inst in dem.flattened() if inst.type == "error"),
        default=0,
    )
    cap = max_degree + 1
    errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=cap,
        dont_explore_edges_with_degree_above=cap,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    return len(errors), max_degree


@pytest.mark.parametrize("noisy_swap", [True, False])
def test_circuit_level_distance_of_18_2_3_is_three(noisy_swap):
    """Full circuit-level distance of the gate equals the code distance (memory: 3)."""
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_18_2_3(), [(HS, {"noisy_swap": noisy_swap})], rounds=3, noise_params=NOISE
    ))
    distance, max_degree = _circuit_level_distance(circuit)
    assert max_degree <= 8
    assert distance == 3


def test_circuit_level_distance_of_13_1_3_is_three():
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(HS, {})], rounds=3, noise_params=NOISE
    ))
    assert _circuit_level_distance(circuit)[0] == 3


# ---------------------------------------------------------------------------
# Multi-patch: the gate acts on one patch only
# ---------------------------------------------------------------------------

def test_two_patch_system_gate_on_one_patch_leaves_the_other_alone():
    system = QECSystem()
    patch_a = system.add_patch(hgp_18_2_3(), name="a")
    patch_b = system.add_patch(hgp_18_2_3(), offset=(20, 0), name="b")
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in data}, system.num_qubits)
    system.active_qubit_indices.update(data)
    # The product-coloration block serves one HGP patch; two patches use the
    # generic CSS coloration block.
    se_block = GenericCSSColorationExtractionBlock(system)
    blocks = getattr(se_block, "measurement_blocks", None)
    builder.apply_syndrome_extraction(se_block.circuit, rounds=2, measurement_blocks=blocks)

    emitted = HGPCodeLogicalOpSet().fold_transversal_h_swap(builder, patch_b)
    touched = {t.value for inst in emitted.flattened() for t in inst.targets_copy()}
    assert touched == set(patch_b.data_indices)
    assert not touched & set(patch_a.data_indices)

    builder.apply_syndrome_extraction(se_block.circuit, rounds=2, measurement_blocks=blocks)
    # Patch b was Hadamard-ed: read it in X; patch a stayed in Z.
    readout = {q: "Z" for q in patch_a.data_indices}
    readout.update({q: "X" for q in patch_b.data_indices})
    builder.apply_data_readout(readout)
    assert builder.circuit.num_observables == 4
    assert_noiseless(builder.circuit)
