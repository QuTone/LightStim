"""Signed physical-flow and channel checks for the paper gross LPU."""
from itertools import product

import numpy as np
import pytest
import stim

from lightstim.qec_code.BB_code.ppm.native import build_native_measurement
from lightstim.qec_code.BB_code.ppm.shift import build_tour_de_gross_code


LABELS = [a + b for a, b in product("IXYZ", repeat=2) if a + b != "II"]


def _ps(support, n):
    p = stim.PauliString(n)
    for q, b in support.items():
        p[q] = b
    return p


def _rank(paulis):
    pivots = {}
    for p in paulis:
        x, z = p.to_numpy()
        value = sum(int(b) << i for i, b in enumerate(np.concatenate((x, z))))
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


@pytest.fixture(scope="module")
def gross():
    return build_tour_de_gross_code()


@pytest.fixture(scope="module")
def natives(gross):
    offsets = {"L": (0, 1), "R": (1, 0), "X": (0, 0), "Z": (1, 1)}
    maps = {s: {(s, i, j): gross.index_map[(2 * i + dx, 2 * j + dy)]
                for i in range(12) for j in range(6)} for s, (dx, dy) in offsets.items()}
    return {label: build_native_measurement(*label,
                data_index=maps["L"] | maps["R"],
                bb_check_indices=maps["X"] | maps["Z"], ancilla_start=288)
            for label in LABELS}


@pytest.mark.parametrize("label", LABELS)
def test_checks_have_correct_rank_target_and_physical_backprop(label, natives, gross):
    native = natives[label]
    native.validate()
    assert len(native.all_lpu_qubits) == 90
    assert _rank([ch.pauli for ch in native.checks]) == 132 + len(native.edge_qubits) + 1
    expected = stim.PauliString(native.num_qubits)
    xs = [_ps(op["pauli"], native.num_qubits) for op in gross.logical_ops if op["type"] == "X"]
    zs = [_ps(op["pauli"], native.num_qubits) for op in gross.logical_ops if op["type"] == "Z"]
    for j, p in zip((0, 6), label):
        if p == "X":
            expected *= xs[j]
        elif p == "Z":
            expected *= zs[j]
        elif p == "Y":
            expected *= 1j * xs[j] * zs[j]
    assert native.target_pauli == expected
    for ch in native.checks:
        assert len(ch.ancillas) <= 2
        if len(ch.ancillas) == 2:
            assert set(ch.parts[0]).isdisjoint(ch.parts[1])
            assert max(map(len, ch.parts)) <= 4
    assert all(inst.name not in {"MPP", "DETECTOR", "OBSERVABLE_INCLUDE"}
               for inst in native.round_circuit)


def test_subgraphs_and_fixed_allocation(natives):
    for label in ("XI", "IZ", "XZ", "IX", "ZI", "ZX"):
        n = natives[label]
        assert len(n.graph_vertices) == 12
        assert len(n.graph_edges) == 18
        assert not n.bell_pairs
    for label in set(LABELS) - {"XI", "IZ", "XZ", "IX", "ZI", "ZX"}:
        n = natives[label]
        assert len(n.graph_vertices) == 23
        assert len(n.graph_edges) == 47
        assert len(n.bell_pairs) == 1
    assert len({tuple(n.auxiliary_roles.items()) for n in natives.values()}) == 1


@pytest.mark.parametrize("label", LABELS)
def test_gauging_preserves_all_commuting_logical_information(label, natives, gross):
    """Choi-style check: a random PPM preserves all 23 commuting Bell checks.

    This tests actual ancilla CNOT/CY/CZ/readout and gauge-out corrections, not
    just algebraic commutation or the desired measurement bit in one input basis.
    """
    native = natives[label]
    n = native.num_qubits + 12
    initial_stabilizers = [_ps(op["pauli"], n) for op in gross.stabilizers]
    logical_bell = []
    for basis in "XZ":
        for j, op in enumerate(op for op in gross.logical_ops if op["type"] == basis):
            bell = _ps(op["pauli"], n)
            bell[native.num_qubits + j] = basis
            logical_bell.append(bell)
    initial = stim.Tableau.from_stabilizers(initial_stabilizers + logical_bell,
                                          allow_redundant=True, allow_underconstrained=True)
    target = native.target_pauli + stim.PauliString(12)
    commuting = [p for p in logical_bell if p.commutes(target)]
    anti = [p for p in logical_bell if not p.commutes(target)]
    commuting += [anti[0] * p for p in anti[1:]]
    assert len(commuting) == 23
    for seed in (81, 117):
        sim = stim.TableauSimulator(seed=seed)
        sim.set_inverse_tableau(initial.inverse())
        sim.do(native.prepare_circuit)
        sim.do(native.round_circuit)
        first = sim.current_measurement_record()[-len(native.syndrome_qubits):]
        sim.do(native.round_circuit)
        second = sim.current_measurement_record()[-len(native.syndrome_qubits):]
        # Bell halves can fluctuate independently; their parity cannot.
        for ch in native.checks:
            positions = [native.syndrome_qubits.index(q) for q in ch.ancillas]
            assert sum(first[i] for i in positions) % 2 == sum(second[i] for i in positions) % 2
        outcome = sum(second[i] for i in native.outcome_parity) % 2
        sim.do(native.gaugeout_circuit)
        edge_results = sim.current_measurement_record()[-len(native.edge_qubits):]
        for correction in native.frame_corrections:
            if sum(edge_results[i] for i in correction.edge_result_indices) % 2:
                for q, basis in correction.pauli.items():
                    {"X": sim.x, "Y": sim.y, "Z": sim.z}[basis](q)
        assert sim.peek_observable_expectation(target) == (-1 if outcome else 1)
        assert all(sim.peek_observable_expectation(p) == 1 for p in initial_stabilizers)
        assert all(sim.peek_observable_expectation(p) == 1 for p in commuting)


def test_bad_native_and_overlapping_allocations_are_rejected():
    for a, b in (("I", "I"), ("XX", "I"), ("x", "Z")):
        with pytest.raises(ValueError):
            build_native_measurement(a, b)
    with pytest.raises(ValueError, match="distinct"):
        build_native_measurement("X", "I", ancilla_start=140)
