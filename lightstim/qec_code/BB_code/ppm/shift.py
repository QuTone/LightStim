"""Physical shifts and induced logical Clifford maps for Tour de gross.

The code basis is Eqs. (27), (32), (34), (35) of arXiv:2506.03094.
Logical matrices are derived from physical support and symplectic pairing,
not used as a substitute for a noisy physical shift circuit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np
import stim

from ..code_patch import BBCode
from ..logical_presets import TOUR_DE_GROSS_CONFIG


def build_tour_de_gross_code(**kwargs) -> BBCode:
    """Construct the paper's gross-code geometry and logical labels 1..12."""
    return BBCode(**(deepcopy(TOUR_DE_GROSS_CONFIG) | kwargs))


def _cell_qubit(code: BBCode, kind: str, x: int, y: int) -> int:
    ox, oy = code.shift
    offsets = {'L': (0, 1), 'R': (1, 0), 'X': (0, 0), 'Z': (1, 1)}
    u, v = offsets[kind]
    return code.index_map[(ox + 2 * (x % code.l) + u,
                           oy + 2 * (y % code.m) + v)]


def physical_shift_permutation(code: BBCode, dx: int, dy: int) -> dict[int, int]:
    """Map each local data-qubit index to its translated destination."""
    return {
        _cell_qubit(code, kind, x, y): _cell_qubit(code, kind, x + dx, y + dy)
        for kind in ('L', 'R') for x in range(code.l) for y in range(code.m)
    }


def logical_shift_matrices(code: BBCode, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Mx, Mz with column j giving W X_j W† and W Z_j W†.

    A coefficient is the binary overlap with the complementary logical.
    Thus Mz = inverse(Mx).T for the CSS-preserving physical permutation.
    This only applies to canonical paired CSS logical representatives.
    """
    xs = [set(op['pauli']) for op in code.logical_ops if op['type'] == 'X']
    zs = [set(op['pauli']) for op in code.logical_ops if op['type'] == 'Z']
    k = code.num_logicals
    if len(xs) != k or len(zs) != k:
        raise ValueError('Shift matrices require k X and k Z logical representatives.')
    pairing = np.array([[len(x & z) % 2 for z in zs] for x in xs], dtype=np.uint8)
    if not np.array_equal(pairing, np.eye(k, dtype=np.uint8)):
        raise ValueError('Logical representatives must have canonical X/Z pairing.')
    permutation = physical_shift_permutation(code, dx, dy)
    shifted_x = [{permutation[q] for q in support} for support in xs]
    shifted_z = [{permutation[q] for q in support} for support in zs]
    mx = np.array([[len(z & sx) % 2 for sx in shifted_x] for z in zs], dtype=np.uint8)
    mz = np.array([[len(x & sz) % 2 for sz in shifted_z] for x in xs], dtype=np.uint8)
    if not np.array_equal((mx.T @ mz) % 2, np.eye(k, dtype=np.uint8)):
        raise ValueError('Physical shift does not induce a symplectic logical map.')
    return mx, mz


def logical_shift_tableau(code: BBCode, dx: int, dy: int) -> stim.Tableau:
    """Return a k-qubit tableau whose action is W P W† (not W† P W)."""
    mx, mz = logical_shift_matrices(code, dx, dy)
    return stim.Tableau.from_conjugated_generators(
        xs=[stim.PauliString(''.join('X' if bit else '_' for bit in col)) for col in mx.T],
        zs=[stim.PauliString(''.join('Z' if bit else '_' for bit in col)) for col in mz.T],
    )


def logical_shift_tableaus(code: BBCode | None = None) -> dict[tuple[int, int], stim.Tableau]:
    """The 36 distinct gross-code logical shifts, indexed by dx,dy modulo 6."""
    code = build_tour_de_gross_code() if code is None else code
    if code.logical_preset != 'tour_de_gross':
        raise ValueError('The 36-element shift table requires the tour_de_gross basis.')
    return {(dx, dy): logical_shift_tableau(code, dx, dy)
            for dx in range(6) for dy in range(6)}


@dataclass(frozen=True)
class BasicShiftRecipe:
    """One Tanner-edge shift: two parallel data transfers, then one QEC round.

    Indices are patch-local; use remap() for a QECSystem's local-to-global map.
    delta is the *physical* displacement modulo (l,m), not modulo logical 6.
    first/second_pairs are (source,destination). Each pair transfers through
    CX(source,destination), CX(destination,source), then M(source).
    Both rounds need destinations in the tracked |0> state. The second
    round returns data to the data-qubit lattice. The caller must follow it
    with a normal QEC cycle; this recipe alone is the six-step transfer part.
    """
    delta: tuple[int, int]
    family: str
    i: int
    j: int
    first_pairs: tuple[tuple[int, int], ...]
    second_pairs: tuple[tuple[int, int], ...]

    def remap(self, local_to_global: Mapping[int, int]) -> BasicShiftRecipe:
        remap_pairs = lambda ps: tuple((local_to_global[s], local_to_global[t]) for s, t in ps)
        return BasicShiftRecipe(self.delta, self.family, self.i, self.j,
                                remap_pairs(self.first_pairs), remap_pairs(self.second_pairs))

    def transfer_circuit(self, *, reset_targets: bool = True) -> stim.Circuit:
        """Reference circuit including ideal classical updates of vacated X frames.

        The record-controlled X operations are tagged noiseless because they
        represent tracked Pauli-frame updates, not extra physical gates.
        A protocol must still propagate these frames in its own tracker.
        Optional initial reset is preparation, outside the six transfer steps.
        No DETECTOR or OBSERVABLE_INCLUDE is inserted here.
        """
        circuit = stim.Circuit()
        if reset_targets:
            circuit.append('R', [t for _, t in self.first_pairs])
        for pairs in (self.first_pairs, self.second_pairs):
            circuit += two_cnot_transfer_circuit(pairs)
        return circuit


def two_cnot_transfer_circuit(pairs, *, clear_vacated_frame: bool = True) -> stim.Circuit:
    """Transfer arbitrary (possibly entangled) source states to zero targets.

    CX(s,t); CX(t,s) maps |psi>_s|0>_t to |0>_s|psi>_t.
    M_Z(s) can detect faults without measuring the transferred state.
    A noisy outcome 1 leaves a known X frame on the vacated qubit; the
    feedback below clears that frame before it is reused as a destination.
    This is distinct from one-CNOT X-measurement teleportation.
    """
    pairs = tuple(pairs)
    occupied = [q for pair in pairs for q in pair]
    if len(set(occupied)) != len(occupied):
        raise ValueError('A transfer layer must consist of disjoint source-target pairs.')
    circuit = stim.Circuit()
    if not pairs:
        return circuit
    circuit.append('CX', [q for s, t in pairs for q in (s, t)])
    circuit.append('TICK')
    circuit.append('CX', [q for s, t in pairs for q in (t, s)])
    circuit.append('TICK')
    circuit.append('M', [s for s, _ in pairs])
    if clear_vacated_frame:
        circuit.append('CX', [q for i, (s, _) in enumerate(pairs)
                              for q in (stim.target_rec(i - len(pairs)), s)], tag='noiseless')
    circuit.append('TICK')
    return circuit


def basic_shift_recipes(code: BBCode | None = None) -> tuple[BasicShiftRecipe, ...]:
    """The 12 Tanner-edge circuits of Appendix A.2, including inverses."""
    code = build_tour_de_gross_code() if code is None else code
    result = []
    for family, terms in (('A', code.A), ('B', code.B)):
        for i, j in product(range(3), repeat=2):
            if i == j:
                continue
            ai, bi = terms[i]
            aj, bj = terms[j]
            first, second = [], []
            for x in range(code.l):
                for y in range(code.m):
                    if family == 'A':
                        # L -> X along A_j then A_i; R -> Z along A_i then A_j.
                        paths = [('L', 'X', x - aj, y - bj),
                                 ('R', 'Z', x + ai, y + bi)]
                    else:
                        # L -> Z along B_i then B_j; R -> X along B_j then B_i.
                        paths = [('L', 'Z', x + ai, y + bi),
                                 ('R', 'X', x - aj, y - bj)]
                    for data_kind, check_kind, cx, cy in paths:
                        source = _cell_qubit(code, data_kind, x, y)
                        middle = _cell_qubit(code, check_kind, cx, cy)
                        target = _cell_qubit(code, data_kind, x + ai - aj, y + bi - bj)
                        first.append((source, middle))
                        second.append((middle, target))
            result.append(BasicShiftRecipe(((ai - aj) % code.l, (bi - bj) % code.m),
                                           family, i, j, tuple(first), tuple(second)))
    return tuple(result)


def decompose_logical_shift(code: BBCode, dx: int, dy: int) -> tuple[BasicShiftRecipe, ...]:
    """Compile a logical shift into zero, one or two physical basic shifts.

    Physically equivalent shifts can differ by x^6, which acts identically
    on the code space; each returned recipe records its actual displacement.
    """
    if code.logical_preset != 'tour_de_gross':
        raise ValueError('The two-instruction synthesis is specific to the gross paper preset.')
    key = (dx % 6, dy % 6)
    if key == (0, 0):
        return ()
    recipes = basic_shift_recipes(code)
    for recipe in recipes:
        if tuple(v % 6 for v in recipe.delta) == key:
            return (recipe,)
    for first, second in product(recipes, repeat=2):
        if tuple((a + b) % 6 for a, b in zip(first.delta, second.delta)) == key:
            return first, second
    raise ValueError(f'No physical decomposition for logical shift {key}.')
