"""Fold-transversal CZ-S on symmetric hypergraph-product patches (Xu et al. Table I)."""

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
    fold_cz_s_circuit,
    fold_diagonal_qubits,
    fold_diagonal_qubits_by_sector,
    fold_logical_cz_s_action,
    fold_logical_permutation,
    fold_mirror_pairs,
    hgp_13_1_3,
    hgp_18_2_3,
    hgp_225_9_4,
)
from lightstim.utils.linear_algebra import row_echelon


CZS, CZS_DAG, HS = "fold_transversal_cz_s", "fold_transversal_cz_s_dag", "fold_transversal_h_swap"
NOISE = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)
HAMMING = BinaryParityCheck.from_dense(
    np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
)
REP3 = BinaryParityCheck.from_dense(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))

INSTANCES = {
    "hgp_13_1_3": hgp_13_1_3,
    "hgp_18_2_3": hgp_18_2_3,                     # redundant check: one C1xC2 diagonal pair -> S̄†
    "hgp_225_9_4": hgp_225_9_4,
    "hgp_hamming2": lambda: HGPCode(HAMMING, d=3),
}


class _SystemOnly:
    def __init__(self, system):
        self.system = system


def _global_patch(factory):
    system = QECSystem()
    return system, system.add_patch(factory(), name="hgp")


def _pauli(n, indices, letter):
    chars = ["_"] * n
    for q in indices:
        chars[q] = letter
    return stim.PauliString("".join(chars))


def _padded_tableau(circuit, n):
    padded = stim.Circuit()
    padded.append("I", list(range(n)))
    return stim.Tableau.from_circuit(padded + circuit)


def _records(patch):
    out = {}
    for r in patch.logical_ops:
        out.setdefault(int(r["logical_id"]), {})[r["type"]] = tuple(sorted(int(q) for q in r["data_indices"]))
    return out


def _layer(patch, system, dagger=False):
    vv, cc = fold_diagonal_qubits_by_sector(patch, _SystemOnly(system))
    return fold_cz_s_circuit(vv, cc, fold_mirror_pairs(patch, _SystemOnly(system)), dagger=dagger)


def _raw_observable_parity(circuit, shots=64, seed=1):
    """Raw measurement parity of every OBSERVABLE_INCLUDE (no reference subtraction)."""
    raw = circuit.compile_sampler(seed=seed).sample(shots)
    columns = []
    for inst in circuit.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            recs = [circuit.num_measurements + t.value for t in inst.targets_copy()]
            columns.append(np.bitwise_xor.reduce(raw[:, recs], axis=1))
    return np.array(columns).T


def _symplectic(pauli, n):
    v = np.zeros(2 * n, dtype=np.uint8)
    for q, ch in enumerate(str(pauli)[1:]):
        if ch in "XY":
            v[q] = 1
        if ch in "ZY":
            v[n + q] = 1
    return v


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_diagonals_by_sector_partition_the_fold_diagonal(name):
    system, patch = _global_patch(INSTANCES[name])
    vv, cc = fold_diagonal_qubits_by_sector(patch, _SystemOnly(system))
    assert sorted(vv + cc) == sorted(fold_diagonal_qubits(patch, _SystemOnly(system)))
    assert len(vv) == patch.h1.num_bits and len(cc) == patch.h1.num_checks
    local_vv, local_cc = fold_diagonal_qubits_by_sector(patch)
    assert local_vv == [patch.vv_qubits[(i, i)] for i in range(patch.h1.num_bits)]
    assert local_cc == [patch.cc_qubits[(i, i)] for i in range(patch.h1.num_checks)]


def test_circuit_layer_structure():
    circuit = fold_cz_s_circuit([0, 1], [5], [(2, 3)])
    assert [inst.name for inst in circuit] == ["S", "S_DAG", "TICK", "CZ"]
    assert [t.value for t in circuit[0].targets_copy()] == [0, 1]
    assert [t.value for t in circuit[1].targets_copy()] == [5]
    assert [t.value for t in circuit[3].targets_copy()] == [2, 3]
    dagger = fold_cz_s_circuit([0, 1], [5], [(2, 3)], dagger=True)
    assert [inst.name for inst in dagger] == ["S_DAG", "S", "TICK", "CZ"]
    assert [inst.name for inst in fold_cz_s_circuit([0], [], [])] == ["S"]
    assert len(fold_cz_s_circuit([], [], [])) == 0


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_logical_action_table(name):
    _, patch = _global_patch(INSTANCES[name])
    action = fold_logical_cz_s_action(patch)
    dagger = fold_logical_cz_s_action(patch, dagger=True)
    permutation = fold_logical_permutation(patch)
    sector = {int(r["logical_id"]): r["sector"] for r in patch.logical_pairs}
    assert sorted(action) == list(range(patch.num_logicals))
    for logical_id, (kind, partner) in action.items():
        if permutation[logical_id] == logical_id:
            assert partner is None
            assert kind == ("S" if sector[logical_id] == "bit_bit" else "S_DAG")
            assert dagger[logical_id] == ("S_DAG" if kind == "S" else "S", None)
        else:
            assert kind == "CZ" and partner == permutation[logical_id]
            assert action[partner] == ("CZ", logical_id)
            assert dagger[logical_id] == action[logical_id]


def test_hgp_225_action_is_three_s_and_three_cz_pairs():
    _, patch = _global_patch(hgp_225_9_4)
    action = fold_logical_cz_s_action(patch)
    assert {i for i, (k, _) in action.items() if k == "S"} == {0, 4, 8}
    assert {frozenset((i, p)) for i, (k, p) in action.items() if k == "CZ"} == {
        frozenset((1, 3)), frozenset((2, 6)), frozenset((5, 7))
    }


# ---------------------------------------------------------------------------
# Exact Clifford action
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(INSTANCES))
@pytest.mark.parametrize("dagger", [False, True])
def test_cz_s_acts_as_s_on_the_diagonal_and_cz_on_mirror_pairs_exactly(name, dagger):
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    tableau = _padded_tableau(_layer(patch, system, dagger), n)
    records = _records(patch)
    for logical_id, (kind, partner) in fold_logical_cz_s_action(patch).items():
        x = _pauli(n, records[logical_id]["X"], "X")
        z = _pauli(n, records[logical_id]["Z"], "Z")
        if kind == "CZ":
            expected = x * _pauli(n, records[partner]["Z"], "Z")      # CZ̄: X̄_a -> X̄_a Z̄_b (no phase)
        else:
            kind_here = fold_logical_cz_s_action(patch, dagger=dagger)[logical_id][0]
            expected = (x * z) * (1j if kind_here == "S" else -1j)     # S̄: X̄ -> +Ȳ = i X̄ Z̄; S̄†: -Ȳ
        assert tableau(x) == expected
        assert tableau(z) == z


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_cz_s_maps_the_stabilizer_group_into_itself(name):
    """Each X check maps to itself times the transposed Z check; Z checks are fixed."""
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    tableau = _padded_tableau(_layer(patch, system), n)
    rows = np.array([_symplectic(_pauli(n, s["data_indices"], s["type"]), n) for s in patch.stabilizers])
    rank = row_echelon(rows)[1]
    for s in patch.stabilizers:
        pauli = _pauli(n, s["data_indices"], s["type"])
        image = tableau(pauli)
        if s["type"] == "Z":
            assert image == pauli
        else:
            assert row_echelon(np.vstack([rows, _symplectic(image, n)]))[1] == rank
            # It is the product with exactly one Z check (the transposed one).
            z_part = _pauli(n, [q for q, ch in enumerate(str(image)[1:]) if ch in "ZY"], "Z")
            assert any(z_part == _pauli(n, t["data_indices"], "Z") for t in patch.stabilizers if t["type"] == "Z")
    # Signs are checked on a code state below (the tracker and the detector
    # sampler are both blind to them).


def _code_state(patch, n, basis="Z"):
    """A stabilizer state of the code with every stabilizer at +1 (logicals fixed in ``basis``)."""
    sim = stim.TableauSimulator()
    generators = [_pauli(n, s["data_indices"], s["type"]) for s in patch.stabilizers]
    generators += [_pauli(n, r[basis], basis) for r in _records(patch).values()]
    sim.set_state_from_stabilizers(generators, allow_redundant=True, allow_underconstrained=True)
    return sim


@pytest.mark.parametrize("name", sorted(INSTANCES))
@pytest.mark.parametrize("dagger", [False, True])
def test_cz_s_keeps_every_stabilizer_at_plus_one(name, dagger):
    """Sign-aware check on a code state: the S† on the C1xC2 diagonal cancels the
    phase of the S on the V1xV2 diagonal in every X-check image.  A layer with S
    on both diagonals flips exactly weight(H) X checks to −1 (negative control)."""
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    stabilizers = [_pauli(n, s["data_indices"], s["type"]) for s in patch.stabilizers]
    for basis in ("Z", "X"):
        sim = _code_state(patch, n, basis)
        sim.do(_layer(patch, system, dagger))
        assert all(sim.peek_observable_expectation(s) == 1 for s in stabilizers)
        # Z̄ (Z init) is fixed by the layer; X̄ (X init) is not, so only check Z̄.
        if basis == "Z":
            assert all(sim.peek_observable_expectation(_pauli(n, r["Z"], "Z")) == 1 for r in _records(patch).values())

    # Negative control: S on both diagonals.
    vv, cc = fold_diagonal_qubits_by_sector(patch, _SystemOnly(system))
    wrong = stim.Circuit()
    wrong.append("S", sorted(vv + cc))
    wrong.append("TICK")
    wrong.append("CZ", [q for pair in fold_mirror_pairs(patch, _SystemOnly(system)) for q in pair])
    sim = _code_state(patch, n, "Z")
    sim.do(wrong)
    flipped = sum(1 for s in stabilizers if sim.peek_observable_expectation(s) == -1)
    assert flipped == int(patch.h1.to_dense().sum())        # one −1 per (check, bit) with H[check, bit] = 1


def _min_weight_logicals_contain_no_mirror_pair(patch, system, distance):
    """Code capacity: no minimum-weight X or Z logical operator contains a fold mirror pair,
    so a weight-2 fault on one CZ cannot complete a logical error with d−2 further faults."""
    import itertools

    n = system.num_qubits
    data = sorted(patch.data_indices)
    pairs = {frozenset(p) for p in fold_mirror_pairs(patch, _SystemOnly(system))}
    for letter, other in (("X", "Z"), ("Z", "X")):
        checks = [set(s["data_indices"]) for s in patch.stabilizers if s["type"] == other]
        own = np.array([_symplectic(_pauli(n, s["data_indices"], letter), n) for s in patch.stabilizers if s["type"] == letter])
        own_rank = row_echelon(own)[1]
        found = 0
        for support in itertools.combinations(data, distance):
            support_set = set(support)
            if any(len(support_set & c) % 2 for c in checks):
                continue                                   # anticommutes with a stabilizer
            v = _symplectic(_pauli(n, support, letter), n)
            if row_echelon(np.vstack([own, v]))[1] == own_rank:
                continue                                   # a stabilizer, not a logical
            found += 1
            assert not any(pair <= support_set for pair in pairs), (letter, support)
        assert found > 0                                    # the enumeration did find the distance


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_cz_s_and_its_dagger_cancel(name):
    system, patch = _global_patch(INSTANCES[name])
    n = system.num_qubits
    assert _padded_tableau(_layer(patch, system) + _layer(patch, system, dagger=True), n) == stim.Tableau(n)


def test_13_1_3_diagonal_logical_gets_exactly_s():
    system, patch = _global_patch(hgp_13_1_3)
    n = system.num_qubits
    (record,) = _records(patch).values()
    tableau = _padded_tableau(_layer(patch, system), n)
    x, z = _pauli(n, record["X"], "X"), _pauli(n, record["Z"], "Z")
    assert tableau(x) == (x * z) * 1j and tableau(z) == z
    twice = _padded_tableau(_layer(patch, system) + _layer(patch, system), n)
    assert twice(x) == -x                                             # S̄² = Z̄ acting on X̄ gives −X̄


# ---------------------------------------------------------------------------
# Op set, executor, validation
# ---------------------------------------------------------------------------

def _builder_for(factory, basis="Z"):
    system = QECSystem()
    patch = system.add_patch(factory(), name="hgp")
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: basis for q in data}, system.num_qubits)
    system.active_qubit_indices.update(data)
    return system, patch, builder


def test_op_set_emits_the_two_layers_and_returns_the_circuit():
    system, patch, builder = _builder_for(hgp_225_9_4)
    before = len(builder.circuit.flattened())
    emitted = HGPCodeLogicalOpSet().fold_transversal_cz_s(builder, patch)
    assert emitted == _layer(patch, system)
    gates = [inst for inst in builder.circuit.flattened()[before:] if inst.name in ("S", "S_DAG", "CZ")]
    assert [inst.name for inst in gates] == ["S", "S_DAG", "CZ"]
    vv, cc = fold_diagonal_qubits_by_sector(patch, builder)
    assert sorted(t.value for t in gates[0].targets_copy()) == sorted(vv)
    assert sorted(t.value for t in gates[1].targets_copy()) == sorted(cc)
    assert len(gates[2].targets_copy()) == 2 * len(fold_mirror_pairs(patch, builder))
    dagger = HGPCodeLogicalOpSet().fold_transversal_cz_s_dag(builder, patch)
    assert dagger == _layer(patch, system, dagger=True)


def test_executor_dispatch_and_validation():
    system, patch, builder = _builder_for(hgp_18_2_3)
    executor = LogicalExecutor(builder)
    executor.register_op_set(HGPCode, HGPCodeLogicalOpSet())
    executor.apply_logical_operation(CZS, [patch])
    executor.apply_logical_operation(CZS_DAG, [patch])
    names = [inst.name for inst in builder.circuit.flattened() if inst.name in ("S", "S_DAG", "CZ")]
    assert names == ["S", "S_DAG", "CZ", "S_DAG", "S", "CZ"]

    with pytest.raises(ValueError, match="global patch"):
        HGPCodeLogicalOpSet().fold_transversal_cz_s(builder, hgp_18_2_3())
    asymmetric = QECSystem().add_patch(HGPCode(REP3, HAMMING), name="a")
    with pytest.raises(ValueError, match="symmetric"):
        fold_diagonal_qubits_by_sector(asymmetric)
    with pytest.raises(ValueError, match="symmetric"):
        fold_logical_cz_s_action(asymmetric)


def test_noise_tags():
    base = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(CZS, {}), (CZS_DAG, {})], init_basis="X", measure_basis="X", noise_params=NOISE))
    quiet = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(CZS, {}), (CZS_DAG, {})], init_basis="X", measure_basis="X", noise_params=NOISE,
        noiseless_gates=True))

    def count(circuit, name, width):
        return sum(len(inst.targets_copy()) // width for inst in circuit.flattened() if inst.name == name)

    _, patch = _global_patch(hgp_13_1_3)
    pairs = len(fold_mirror_pairs(patch))
    assert count(base, "DEPOLARIZE2", 2) - count(quiet, "DEPOLARIZE2", 2) == 2 * pairs
    assert count(base, "DEPOLARIZE1", 1) - count(quiet, "DEPOLARIZE1", 1) >= 2 * len(fold_diagonal_qubits(patch))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_harness_cz_s_preserves_the_code_and_z_logicals(name):
    """Z init, CZ-S, Z readout: Z̄ is fixed, all k logicals resolvable, stabilizer signs right."""
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        INSTANCES[name](), [(CZS, {})], init_basis="Z", measure_basis="Z"))
    _, patch = _global_patch(INSTANCES[name])
    assert circuit.num_observables == patch.num_logicals
    assert_noiseless(circuit)


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_harness_cz_s_round_trip_restores_x_logicals(name):
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        INSTANCES[name](), [(CZS, {}), (CZS_DAG, {})], init_basis="X", measure_basis="X"))
    _, patch = _global_patch(INSTANCES[name])
    assert circuit.num_observables == patch.num_logicals
    assert_noiseless(circuit)
    # Signs: CZ-S·CZ-S† is the identity, so no logical Pauli is left behind
    # (CZ-S·CZ-S = S̄² = Z̄ on the diagonal would flip X̄, invisible to the
    # detector sampler but visible in the raw parity).
    assert not _raw_observable_parity(circuit).any()


def test_harness_cz_s_twice_leaves_a_logical_z_on_the_diagonal():
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_13_1_3(), [(CZS, {}), (CZS, {})], init_basis="X", measure_basis="X"))
    assert circuit.num_observables == 1
    assert_noiseless(circuit)                              # deterministic ...
    assert _raw_observable_parity(circuit).all()           # ... but X̄ -> −X̄


def test_harness_single_cz_s_on_x_init_leaves_nothing_x_resolvable_for_k1():
    """[[13,1,3]]: X̄ -> Ȳ, which a transversal X readout does not resolve."""
    with pytest.raises(ValueError, match="resolvable"):
        build_quiet(lambda: build_hgp_gate_verification_circuit(
            hgp_13_1_3(), [(CZS, {})], init_basis="X", measure_basis="X"))


def test_cz_s_composes_with_h_swap():
    """S̄·H̄ on the diagonal: Z init, H-SWAP then CZ-S then CZ-S† then H-SWAP, Z readout."""
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        hgp_225_9_4(), [(HS, {}), (CZS, {}), (CZS_DAG, {}), (HS, {})], init_basis="Z", measure_basis="Z"))
    assert circuit.num_observables == 9
    assert_noiseless(circuit)
    assert not _raw_observable_parity(circuit).any()


# ---------------------------------------------------------------------------
# Multi-patch / offset: the gate acts on one patch only, with global indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", [CZS, CZS_DAG])
def test_two_patch_system_gate_on_one_patch_leaves_the_other_alone(op_name):
    """Patch b sits at a non-zero offset, so local and global indices differ."""
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
    se_block = GenericCSSColorationExtractionBlock(system)
    blocks = getattr(se_block, "measurement_blocks", None)
    builder.apply_syndrome_extraction(se_block.circuit, rounds=2, measurement_blocks=blocks)

    emitted = getattr(HGPCodeLogicalOpSet(), op_name)(builder, patch_b)
    touched = {t.value for inst in emitted.flattened() for t in inst.targets_copy()}
    assert touched == set(patch_b.data_indices)
    assert not touched & set(patch_a.data_indices)
    # The S/S† targets are patch b's diagonals and the CZ targets its mirror pairs, in global indices.
    vv, cc = fold_diagonal_qubits_by_sector(patch_b, builder)
    names = {inst.name: sorted(t.value for t in inst.targets_copy()) for inst in emitted.flattened() if inst.name != "TICK"}
    assert names["S_DAG" if op_name == CZS_DAG else "S"] == sorted(vv)
    assert names["S" if op_name == CZS_DAG else "S_DAG"] == sorted(cc)
    assert names["CZ"] == sorted(q for pair in fold_mirror_pairs(patch_b, builder) for q in pair)

    before = builder.circuit.num_detectors
    builder.apply_syndrome_extraction(se_block.circuit, rounds=2, measurement_blocks=blocks)
    assert builder.circuit.num_detectors - before == 2 * (len(patch_a.stabilizers) + len(patch_b.stabilizers))
    builder.apply_data_readout({q: "Z" for q in data})     # Z̄ is fixed by CZ-S on both patches
    assert builder.circuit.num_observables == 4
    assert_noiseless(builder.circuit)
    assert not _raw_observable_parity(builder.circuit).any()


def test_exact_action_at_a_non_zero_offset():
    system = QECSystem()
    patch = system.add_patch(hgp_225_9_4(), offset=(7, 11), name="hgp")
    n = system.num_qubits
    tableau = _padded_tableau(_layer(patch, system), n)
    records = _records(patch)
    for logical_id, (kind, partner) in fold_logical_cz_s_action(patch).items():
        x, z = _pauli(n, records[logical_id]["X"], "X"), _pauli(n, records[logical_id]["Z"], "Z")
        expected = x * _pauli(n, records[partner]["Z"], "Z") if kind == "CZ" else (x * z) * 1j
        assert tableau(x) == expected and tableau(z) == z


def _circuit_level_distance(circuit):
    dem = circuit.detector_error_model(decompose_errors=False)
    max_degree = max(
        (sum(1 for t in inst.targets_copy() if t.is_relative_detector_id())
         for inst in dem.flattened() if inst.type == "error"),
        default=0,
    )
    errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=max_degree + 1,
        dont_explore_edges_with_degree_above=max_degree + 1,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    return len(errors)


@pytest.mark.parametrize("name", ["hgp_13_1_3", "hgp_18_2_3", "hgp_hamming2"])
def test_no_minimum_weight_logical_contains_a_mirror_pair(name):
    system, patch = _global_patch(INSTANCES[name])
    _min_weight_logicals_contain_no_mirror_pair(patch, system, 3)


@pytest.mark.parametrize("name,rounds", [("hgp_13_1_3", 3), ("hgp_18_2_3", 3), ("hgp_hamming2", 2)])
@pytest.mark.parametrize("gates,bases", [
    ([(CZS, {})], ("Z", "Z")),
    ([(CZS, {}), (CZS_DAG, {})], ("X", "X")),
], ids=["single_Z", "round_trip_X"])
def test_circuit_level_distance_is_three(name, rounds, gates, bases):
    """Z/Z sees the X part of the phase- and CZ-layer noise and the X-check hyperedges;
    the X/X round trip additionally sees Z⊗Z faults of the CZ layer on mirror pairs."""
    circuit = build_quiet(lambda: build_hgp_gate_verification_circuit(
        INSTANCES[name](), gates, init_basis=bases[0], measure_basis=bases[1], rounds=rounds, noise_params=NOISE))
    assert _circuit_level_distance(circuit) == 3
