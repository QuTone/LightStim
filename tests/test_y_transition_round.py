"""Acceptance tests for the XZXZ <-> XZZX Y-transition round.

Port target
-----------
Craig Gidney, ``code/src/midout/circuits/steps/_measure_y_transition_round.py``
(:func:`make_y_transition_round_nesw_xzxz_to_xzzx`, zenodo 7487893) — the single
native round that converts an ordinary rotated-surface-code patch
(``make_xtop_qubit_patch``) into the degenerate XXZZ "Y boundary" patch
(``make_ztop_yboundary_patch``) while measuring ``Y_L``.

``direction='measure'`` is that round verbatim.  ``direction='init'`` is its
time reverse — exactly what ``gen.Chunk.inverted()`` produces and what the
published ``basis='Y'`` memory circuit uses to *prepare* ``Y_L``.

Verification
------------
(a) d=3 and d=5: the emitted chunk must equal the corresponding round sliced
    out of Gidney's own **native (uncompiled)** circuit, per TICK layer, after
    mapping qubit indices to coordinates.  Skipped if his package will not
    import.
(b) d=3 and d=5: a source-independent spec expressed with ``stim`` flows — the
    old patch's checks are consumed, the new patch's checks are prepared, and
    ``Y_L`` is measured (``measure``) / prepared (``init``).
"""
import collections
import sys

import numpy as np
import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.y_boundary_patch import (
    make_degenerate_y_boundary_patch,
)
from lightstim.qec_code.surface_code.rotated.y_transition_round import (
    make_y_transition_chunk,
)

GIDNEY_SRC = "/nvme2n1/yuehan_zhang/zenodo_7487893/code/src"

ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}


# =============================================================================
# system construction
# =============================================================================

def build_two_patch_system(d):
    """A fresh canvas holding the qubit patch and the degenerate patch on the
    same footprint.  Nothing is initialised, so the overlapping coordinates are
    ``dormant`` and ``add_patch`` reuses their global indices."""
    system = QECSystem()
    qubit_view = system.add_patch(RotatedSurfaceCode(distance=d), name='Q')
    deg_view = system.add_patch(make_degenerate_y_boundary_patch(d), name='D')
    return system, qubit_view, deg_view


def g2l(g: complex) -> tuple:
    """Gidney complex coord -> LightStim coordinate."""
    return (2.0 * g.real + 1.0, 2.0 * g.imag + 1.0)


def l2g(c) -> complex:
    return complex((c[0] - 1.0) / 2.0, (c[1] - 1.0) / 2.0)


# =============================================================================
# canonical per-TICK-layer form
# =============================================================================

def canonical_layers(circuit: stim.Circuit, idx_to_key):
    """Split ``circuit`` on TICK; per layer return a Counter of atomic ops.

    A one-qubit op contributes ``(name, key)``; a two-qubit op contributes
    ``(name, key_a, key_b)`` with the control/target orientation preserved.
    Order inside a layer is deliberately ignored (Gidney's builder sorts, and
    stim fuses adjacent same-gate instructions).
    """
    layers = [collections.Counter()]
    for inst in circuit.flattened():
        if inst.name == 'TICK':
            layers.append(collections.Counter())
            continue
        if inst.name in ANNOTATIONS:
            continue
        targets = [t.qubit_value for t in inst.targets_copy()]
        assert all(t is not None for t in targets), inst
        gate = stim.gate_data(inst.name)
        if gate.is_two_qubit_gate:
            assert len(targets) % 2 == 0
            for k in range(0, len(targets), 2):
                layers[-1][(inst.name, idx_to_key(targets[k]),
                            idx_to_key(targets[k + 1]))] += 1
        else:
            for t in targets:
                layers[-1][(inst.name, idx_to_key(t))] += 1
    return [dict(layer) for layer in layers]


# =============================================================================
# (a) golden fixture: Gidney's own native circuit
# =============================================================================

def _import_gidney():
    if not hasattr(np, 'bool8'):        # his package predates the numpy 2 removal
        np.bool8 = np.bool_
    if GIDNEY_SRC not in sys.path:
        sys.path.insert(0, GIDNEY_SRC)
    try:
        from midout._make_circuit import make_circuit
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"Gidney's midout package is not importable here: {e!r}")
    return make_circuit


def _split_rounds(circuit: stim.Circuit):
    """Slice a compiled circuit into per-round gate segments.

    Rounds are delimited by the annotation block (DETECTOR / OBSERVABLE_INCLUDE
    / SHIFT_COORDS) that closes them; the TICK the compiler emits *before* a
    round's gates is dropped, so a segment starts on the round's first reset.
    """
    rounds = []
    cur = []
    saw_annotation = False
    for inst in circuit.flattened():
        if inst.name in ANNOTATIONS:
            if cur:
                rounds.append(cur)
                cur = []
            saw_annotation = True
            continue
        if inst.name == 'TICK' and not cur:
            continue                       # leading TICK belongs to the separator
        cur.append(inst)
        saw_annotation = False
    if cur:
        rounds.append(cur)
    out = []
    for r in rounds:
        while r and r[-1].name == 'TICK':
            r.pop()
        if r:
            c = stim.Circuit()
            for inst in r:
                c.append(inst)
            out.append(c)
    return out


def gidney_native_transition_rounds(d):
    """Return ``{'init': circuit, 'measure': circuit}`` sliced out of Gidney's
    native (``convert_to_cz=False``) ``basis='Y'`` memory circuit, together with
    his index -> Gidney-coordinate map."""
    make_circuit = _import_gidney()
    full = make_circuit(basis='Y', distance=d, memory_rounds=3,
                        boundary_rounds=1, noise=None, convert_to_cz=False)
    coords = full.get_final_qubit_coordinates()
    idx2g = {i: complex(c[0], c[1]) for i, c in coords.items()}

    found = {}
    for r in _split_rounds(full):
        names = {inst.name for inst in r}
        if 'RY' in names:
            assert 'init' not in found, "more than one Y-init round"
            found['init'] = r
        elif 'MY' in names:
            assert 'measure' not in found, "more than one Y-measure round"
            found['measure'] = r
    assert set(found) == {'init', 'measure'}, sorted(found)
    return found, idx2g


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("direction", ['init', 'measure'])
def test_matches_gidney_native_round(d, direction):
    golden, idx2g = gidney_native_transition_rounds(d)
    system, qubit_view, deg_view = build_two_patch_system(d)
    mine = make_y_transition_chunk(system, 'Q', 'D', direction=direction)

    want = canonical_layers(golden[direction], lambda i: idx2g[i])
    got = canonical_layers(mine, lambda i: l2g(system.qubit_coords[i]))

    assert len(got) == len(want), (
        f"{len(got)} tick layers, expected {len(want)}"
    )
    for k, (a, b) in enumerate(zip(got, want)):
        assert a == b, f"layer {k} differs\n got {sorted(a)}\nwant {sorted(b)}"


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("direction", ['init', 'measure'])
def test_matches_gidney_instruction_stream_verbatim(d, direction):
    """Stronger than the per-layer multiset: after relabelling his indices to
    ours the instruction stream is identical, gate for gate, target for target.
    """
    golden, idx2g = gidney_native_transition_rounds(d)
    system, _, _ = build_two_patch_system(d)
    mine = make_y_transition_chunk(system, 'Q', 'D', direction=direction)

    relabelled = stim.Circuit()
    for inst in golden[direction].flattened():
        if inst.name == 'TICK':
            relabelled.append('TICK')
            continue
        relabelled.append(inst.name,
                          [system.index_map[g2l(idx2g[t.qubit_value])]
                           for t in inst.targets_copy()],
                          inst.gate_args_copy())
    assert str(mine) == str(relabelled)


@pytest.mark.smoke
def test_emits_native_gates_not_cz_compiled():
    system, _, _ = build_two_patch_system(3)
    for direction in ('init', 'measure'):
        names = {inst.name for inst in
                 make_y_transition_chunk(system, 'Q', 'D', direction=direction)}
        assert 'CZ' not in names
        assert 'CX' in names and 'XCY' in names and 'H' in names
        assert 'SQRT_X' in names
    fwd = {inst.name for inst in make_y_transition_chunk(system, 'Q', 'D',
                                                         direction='measure')}
    inv = {inst.name for inst in make_y_transition_chunk(system, 'Q', 'D',
                                                         direction='init')}
    assert {'MX', 'MY', 'M'} <= fwd and {'RX', 'R'} <= fwd
    assert {'RX', 'RY', 'R'} <= inv and {'MX', 'M'} <= inv


@pytest.mark.smoke
def test_rejects_unknown_direction():
    system, _, _ = build_two_patch_system(3)
    with pytest.raises(ValueError):
        make_y_transition_chunk(system, 'Q', 'D', direction='sideways')


@pytest.mark.smoke
def test_targets_are_global_indices():
    """A shifted placement must move every target, proving the chunk speaks
    global indices rather than Gidney's or a patch's local ones."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name='Q', offset=(20, 10))
    system.add_patch(make_degenerate_y_boundary_patch(3), name='D', offset=(20, 10))
    chunk = make_y_transition_chunk(system, 'Q', 'D', direction='measure')
    used = {t.qubit_value for inst in chunk for t in inst.targets_copy()}
    assert used == set(system.qubit_coords)
    for i in used:
        x, y = system.qubit_coords[i]
        assert 19.0 <= x <= 27.0 and 9.0 <= y <= 17.0


# =============================================================================
# (b) source-independent flow spec
# =============================================================================

class FlowSolver:
    """Answers "is there a set of measurement records M making
    ``p_in -> p_out xor rec[M]`` a flow of this circuit?" by reducing the
    circuit's flow generators to row echelon form over GF(2)."""

    def __init__(self, circuit: stim.Circuit, n: int):
        self.circuit = circuit
        self.n = n
        self.rows = []
        self.meas = []
        self.pivots = {}
        for g in circuit.flow_generators():
            v = np.concatenate([self._bits(g.input_copy()),
                                self._bits(g.output_copy())])
            m = set(g.measurements_copy())
            while True:
                nz = np.nonzero(v)[0]
                if not len(nz):
                    break
                p = int(nz[0])
                if p in self.pivots:
                    j = self.pivots[p]
                    v = v ^ self.rows[j]
                    m = m ^ self.meas[j]
                else:
                    self.pivots[p] = len(self.rows)
                    self.rows.append(v)
                    self.meas.append(m)
                    break

    def _bits(self, p: stim.PauliString):
        xs, zs = p.to_numpy(bit_packed=False)
        v = np.zeros(2 * self.n, dtype=np.uint8)
        v[:len(xs)] = xs
        v[self.n:self.n + len(zs)] = zs
        return v

    def solve(self, p_in: stim.PauliString, p_out: stim.PauliString):
        t = np.concatenate([self._bits(p_in), self._bits(p_out)])
        acc = set()
        while True:
            nz = np.nonzero(t)[0]
            if not len(nz):
                return sorted(acc)
            p = int(nz[0])
            if p not in self.pivots:
                return None
            j = self.pivots[p]
            t = t ^ self.rows[j]
            acc ^= self.meas[j]

    def assert_flow(self, p_in, p_out, what):
        m = self.solve(p_in, p_out)
        assert m is not None, f"no flow {what}"
        flow = stim.Flow(input=p_in, output=p_out, measurements=m)
        assert self.circuit.has_flow(flow, unsigned=True), \
            f"stim rejects the reconstructed flow {what}: {flow}"
        return m


def check_ps(stab, n):
    p = stim.PauliString(n)
    for idx, b in stab['pauli'].items():
        p[idx] = b
    return p


def y_logical(system, d):
    """Gidney's ``qubit_obs``: Y on the corner, Z along the g-real axis, X along
    the g-imaginary axis."""
    n = system.num_qubits
    p = stim.PauliString(n)
    x0, y0 = min(system.qubit_coords[i] for i in system.data_indices)
    p[system.index_map[(x0, y0)]] = 'Y'
    for k in range(1, d):
        p[system.index_map[(x0 + 2 * k, y0)]] = 'Z'
        p[system.index_map[(x0, y0 + 2 * k)]] = 'X'
    return p


@pytest.mark.parametrize("d", [3, 5])
def test_measure_direction_flow_spec(d):
    system, qubit_view, deg_view = build_two_patch_system(d)
    n = system.num_qubits
    chunk = make_y_transition_chunk(system, 'Q', 'D', direction='measure')
    solver = FlowSolver(chunk, n)
    ident = stim.PauliString(n)

    # every qubit-patch (old) check is measured out
    for s in qubit_view.stabilizers:
        solver.assert_flow(check_ps(s, n), ident,
                           f"consuming qubit check {s['type']}@{s['syn_coord']}")
    # every degenerate (new) check is prepared
    for s in deg_view.stabilizers:
        solver.assert_flow(ident, check_ps(s, n),
                           f"preparing degenerate check {s['type']}@{s['syn_coord']}")
    # Y_L is measured: Y_L -> rec[...]
    m = solver.assert_flow(y_logical(system, d), ident, "measuring Y_L")
    assert m, "Y_L flow must consume at least one measurement record"
    # ...and it is NOT merely passed through
    assert solver.solve(y_logical(system, d), y_logical(system, d)) is None


@pytest.mark.parametrize("d", [3, 5])
def test_init_direction_flow_spec(d):
    system, qubit_view, deg_view = build_two_patch_system(d)
    n = system.num_qubits
    chunk = make_y_transition_chunk(system, 'Q', 'D', direction='init')
    solver = FlowSolver(chunk, n)
    ident = stim.PauliString(n)

    # every degenerate (old) check is measured out
    for s in deg_view.stabilizers:
        solver.assert_flow(check_ps(s, n), ident,
                           f"consuming degenerate check {s['type']}@{s['syn_coord']}")
    # every qubit-patch (new) check is prepared
    for s in qubit_view.stabilizers:
        solver.assert_flow(ident, check_ps(s, n),
                           f"preparing qubit check {s['type']}@{s['syn_coord']}")
    # Y_L is prepared: 1 -> Y_L xor rec[...]
    m = solver.assert_flow(ident, y_logical(system, d), "preparing Y_L")
    assert m, "Y_L flow must consume at least one measurement record"


@pytest.mark.parametrize("d", [3, 5])
def test_directions_are_time_reverses(d):
    system, _, _ = build_two_patch_system(d)
    fwd = make_y_transition_chunk(system, 'Q', 'D', direction='measure')
    inv = make_y_transition_chunk(system, 'Q', 'D', direction='init')
    # measurement count of one equals reset count of the other
    def n_reset(c):
        return sum(len(i.targets_copy()) for i in c.flattened()
                   if stim.gate_data(i.name).is_reset)
    def n_meas(c):
        return sum(len(i.targets_copy()) for i in c.flattened()
                   if stim.gate_data(i.name).produces_measurements)
    assert n_reset(fwd) == n_meas(inv)
    assert n_meas(fwd) == n_reset(inv)
    assert fwd.num_ticks == inv.num_ticks
