"""Paper-basis algebra and physical two-CNOT shift checks."""
import numpy as np
import pytest
import stim

from lightstim.qec_code.BB_code import BBCode
from lightstim.qec_code.BB_code.ppm.shift import (
    basic_shift_recipes,
    build_tour_de_gross_code,
    decompose_logical_shift,
    logical_shift_matrices,
    logical_shift_tableau,
    logical_shift_tableaus,
    physical_shift_permutation,
    two_cnot_transfer_circuit,
)


@pytest.fixture(scope='module')
def gross():
    return build_tour_de_gross_code()


def _support_mask(support):
    return sum(1 << int(q) for q in support)


def _binary_span(supports):
    pivots = {}
    for support in supports:
        value = _support_mask(support)
        while value:
            bit = value.bit_length() - 1
            if bit in pivots:
                value ^= pivots[bit]
            else:
                pivots[bit] = value
                break
    return pivots


def _in_span(support, pivots):
    value = _support_mask(support)
    while value:
        bit = value.bit_length() - 1
        if bit not in pivots:
            return False
        value ^= pivots[bit]
    return True


def test_paper_logicals_are_canonical_and_rank_66(gross):
    xs = [set(r['pauli']) for r in gross.logical_ops if r['type'] == 'X']
    zs = [set(r['pauli']) for r in gross.logical_ops if r['type'] == 'Z']
    assert len(gross.data_indices) == 144
    assert gross.num_logicals == 12
    assert {len(s) for s in xs + zs} == {12}
    assert np.array_equal([[len(x & z) % 2 for z in zs] for x in xs], np.eye(12))
    for kind, logicals in [('X', zs), ('Z', xs)]:
        stabs = [set(r['pauli']) for r in gross.stabilizers if r['type'] == kind]
        assert len(stabs) == 72
        assert len(_binary_span(stabs)) == 66
        assert all(len(s & logical) % 2 == 0 for s in stabs for logical in logicals)


def test_paper_shift_matrices_match_equation_37(gross):
    ax = np.array([[0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1],
                   [0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1]], dtype=np.uint8)
    ay = np.array([[1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 1],
                   [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1]], dtype=np.uint8)
    for delta, expected in [((1, 0), ax), ((0, 1), ay)]:
        mx, mz = logical_shift_matrices(gross, *delta)
        assert np.array_equal(mx[:6, :6], expected)
        assert np.array_equal(mx[6:, 6:], expected.T)
        assert not mx[:6, 6:].any() and not mx[6:, :6].any()
        assert np.array_equal((mx.T @ mz) % 2, np.eye(12))
    mx, mz = logical_shift_matrices(gross, 1, -1)
    assert list(np.flatnonzero(mx[:, 6]) + 1) == [11, 12]
    assert list(np.flatnonzero(mz[:, 0]) + 1) == [1, 2, 4, 5]


def test_all_shift_images_agree_modulo_stabilizers(gross):
    # Symplectic coefficients alone are not enough: the remaining physical
    # support must actually lie in the stabilizer span.
    for kind, matrix_index in [('X', 0), ('Z', 1)]:
        logicals = [set(r['pauli']) for r in gross.logical_ops if r['type'] == kind]
        pivots = _binary_span([r['pauli'] for r in gross.stabilizers if r['type'] == kind])
        for dx in range(6):
            for dy in range(6):
                permutation = physical_shift_permutation(gross, dx, dy)
                matrix = logical_shift_matrices(gross, dx, dy)[matrix_index]
                for j, support in enumerate(logicals):
                    residual = {permutation[q] for q in support}
                    for i in np.flatnonzero(matrix[:, j]):
                        residual ^= logicals[i]
                    assert _in_span(residual, pivots)


def test_36_logical_shifts_and_two_instruction_synthesis(gross):
    table = logical_shift_tableaus(gross)
    assert len({str(value) for value in table.values()}) == 36
    assert logical_shift_tableau(gross, 6, 0) == stim.Tableau(12)
    assert logical_shift_tableau(gross, 0, 6) == stim.Tableau(12)
    for key, target in table.items():
        recipes = decompose_logical_shift(gross, *key)
        assert len(recipes) <= 2
        actual = stim.Tableau(12)
        for recipe in recipes:
            actual = actual.then(logical_shift_tableau(gross, *recipe.delta))
        assert actual == target
        assert target.inverse() == table[(-key[0] % 6, -key[1] % 6)]


def test_transfers_use_existing_tanner_edges_and_translate(gross):
    edges = {frozenset((stab['syn_idx'], data))
             for stab in gross.stabilizers for data in stab['pauli']}
    for recipe in basic_shift_recipes(gross):
        assert len(recipe.first_pairs) == len(recipe.second_pairs) == 144
        assert all(frozenset(pair) in edges for pair in recipe.first_pairs + recipe.second_pairs)
        second = dict(recipe.second_pairs)
        actual = {source: second[middle] for source, middle in recipe.first_pairs}
        assert actual == physical_shift_permutation(gross, *recipe.delta)
        for pairs in (recipe.first_pairs, recipe.second_pairs):
            assert len({q for pair in pairs for q in pair}) == 288


@pytest.mark.parametrize('source_state', ['X', 'Y', 'Z', 'entangled'])
def test_two_cnot_transfer_preserves_quantum_state(source_state):
    circuit = stim.Circuit('R 0 1 2')
    if source_state == 'X':
        circuit.append('H', [0])
    elif source_state == 'Y':
        circuit.append('H', [0])
        circuit.append('S', [0])
    elif source_state == 'entangled':
        circuit.append('H', [0])
        circuit.append('CX', [0, 2])
    circuit += two_cnot_transfer_circuit([(0, 1)])
    sim = stim.TableauSimulator()
    sim.do(circuit)
    assert sim.current_measurement_record() == [False]
    assert sim.peek_observable_expectation(stim.PauliString('Z__')) == 1
    if source_state == 'entangled':
        assert sim.peek_observable_expectation(stim.PauliString('_XX')) == 1
        assert sim.peek_observable_expectation(stim.PauliString('_ZZ')) == 1
    else:
        assert sim.peek_observable_expectation(stim.PauliString('_' + source_state + '_')) == 1


def test_physical_shift_preserves_entanglement_with_all_reference_qubits(gross):
    # This tests the 576 physical CX gates plus measurements, not a free
    # relabeling of coordinates or a three-CX SWAP approximation.
    recipe = basic_shift_recipes(gross)[0]
    circuit = stim.Circuit()
    refs = {q: gross.num_qubits + i for i, q in enumerate(sorted(gross.data_indices))}
    for q, ref in refs.items():
        circuit.append('H', [q])
        circuit.append('CX', [q, ref])
    circuit += recipe.transfer_circuit()
    sim = stim.TableauSimulator()
    sim.do(circuit)
    assert sim.current_measurement_record() == [False] * 288
    permutation = physical_shift_permutation(gross, *recipe.delta)
    for q, ref in refs.items():
        for kind in ('X', 'Z'):
            pauli = stim.PauliString(gross.num_qubits + 144)
            pauli[permutation[q]] = kind
            pauli[ref] = kind
            assert sim.peek_observable_expectation(pauli) == 1
    assert circuit.num_ticks == 6
    assert all(inst.name != 'SWAP' for inst in circuit)


def test_named_preset_does_not_change_existing_memory_basis():
    memory = BBCode(l=12, m=6, A=[[3, 0], [0, 1], [0, 2]], B=[[0, 3], [1, 0], [2, 0]])
    assert memory.num_logicals == 12
    assert memory.logical_preset is None
    with pytest.raises(ValueError, match='polynomials'):
        BBCode(l=12, m=6, A=[[3, 0], [0, 1], [0, 2]], B=[[0, 3], [1, 0], [2, 0]], logical_preset='tour_de_gross')
