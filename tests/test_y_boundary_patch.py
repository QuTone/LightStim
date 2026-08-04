"""Acceptance tests for the degenerate XXZZ ("Y boundary") rotated-surface-code patch.

Port target
-----------
Craig Gidney, ``code/src/midout/circuits/steps/_patches.py``
(:func:`make_ztop_yboundary_patch`, zenodo 7487893) — the ``top=Z left=Z
bottom=X right=X`` rectangular surface-code patch whose ``(0,0)`` corner data
qubit falls out of the construction, leaving a **degenerate** patch with
``d*d - 1`` data qubits, ``d*d - 1`` independent checks and **zero** logical
qubits.

Coordinate bridge (verified, see ``gidney_probe/01_compare_qubit_patch.py``)
---------------------------------------------------------------------------
Gidney data qubit ``g = gx + gy*1j``  ->  LightStim ``(2*gx + 1, 2*gy + 1)``.
His measure qubits sit at half-integer ``g`` and therefore land on even
LightStim coordinates.  Under this map ``RotatedSurfaceCode(distance=d)`` is
tile-for-tile equal to his ``make_xtop_qubit_patch(distance=d)``.
"""
import collections
import sys

import numpy as np
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.rotated.y_boundary_patch import (
    make_degenerate_y_boundary_patch,
)

GIDNEY_SRC = "/nvme2n1/yuehan_zhang/zenodo_7487893/code/src"


# --- golden d=3 tile set, dumped straight out of Gidney's patch object -------
# (basis, syndrome coord, sorted data coords) in LightStim coordinates.
D3_GOLDEN_TILES = {
    ('X', (2.0, 4.0), ((1.0, 3.0), (1.0, 5.0), (3.0, 3.0), (3.0, 5.0))),
    ('X', (4.0, 2.0), ((3.0, 1.0), (3.0, 3.0), (5.0, 1.0), (5.0, 3.0))),
    ('X', (4.0, 6.0), ((3.0, 5.0), (5.0, 5.0))),
    ('X', (6.0, 4.0), ((5.0, 3.0), (5.0, 5.0))),
    ('Z', (0.0, 4.0), ((1.0, 3.0), (1.0, 5.0))),
    ('Z', (2.0, 2.0), ((1.0, 3.0), (3.0, 1.0), (3.0, 3.0))),          # weight 3
    ('Z', (4.0, 0.0), ((3.0, 1.0), (5.0, 1.0))),
    ('Z', (4.0, 4.0), ((3.0, 3.0), (3.0, 5.0), (5.0, 3.0), (5.0, 5.0))),
}
D3_GOLDEN_DATA = sorted({q for _, _, ds in D3_GOLDEN_TILES for q in ds})

CORNER = (1.0, 1.0)
DIAGONAL_W3_SYN = (2.0, 2.0)   # Gidney 0.5+0.5j, for every distance


def tiles_of(patch):
    """(type, syn_coord, sorted data coords) for every stabilizer record."""
    out = set()
    for s in patch.stabilizers:
        syn = tuple(float(v) for v in s['syn_coord'])
        data = tuple(sorted(
            tuple(float(v) for v in patch.qubit_coords[i]) for i in s['data_indices']
        ))
        out.add((s['type'], syn, data))
    return out


# =============================================================================
# d = 3: exact equality with Gidney's dump
# =============================================================================

@pytest.mark.smoke
def test_d3_tiles_exactly_match_gidney():
    p = make_degenerate_y_boundary_patch(3)
    assert tiles_of(p) == D3_GOLDEN_TILES


@pytest.mark.smoke
def test_d3_data_qubits_exclude_the_corner():
    p = make_degenerate_y_boundary_patch(3)
    data = sorted(tuple(float(v) for v in p.qubit_coords[i]) for i in p.data_indices)
    assert data == D3_GOLDEN_DATA
    assert len(data) == 8
    assert CORNER not in data
    assert CORNER not in p.index_map


@pytest.mark.smoke
def test_d3_syndrome_roles_split_x_and_z():
    p = make_degenerate_y_boundary_patch(3)
    xs = sorted(tuple(float(v) for v in p.qubit_coords[i]) for i in p.syndrome_indices_x)
    zs = sorted(tuple(float(v) for v in p.qubit_coords[i]) for i in p.syndrome_indices_z)
    assert xs == sorted(syn for b, syn, _ in D3_GOLDEN_TILES if b == 'X')
    assert zs == sorted(syn for b, syn, _ in D3_GOLDEN_TILES if b == 'Z')
    assert p.syndrome_indices == p.syndrome_indices_x | p.syndrome_indices_z


@pytest.mark.smoke
def test_pauli_record_is_uniform_css_and_matches_data_indices():
    p = make_degenerate_y_boundary_patch(3)
    for s in p.stabilizers:
        assert set(s['pauli'].keys()) == set(s['data_indices'])
        assert set(s['pauli'].values()) == {s['type']}
        assert s['syn_idx'] == p.index_map[s['syn_coord']]


# =============================================================================
# general d: structural spec
# =============================================================================

@pytest.mark.parametrize("d", [3, 5, 7])
def test_zero_logicals(d):
    p = make_degenerate_y_boundary_patch(d)
    assert p.num_logicals == 0
    assert p.logical_ops == []


@pytest.mark.parametrize("d", [3, 5, 7])
def test_corner_data_qubit_is_absent(d):
    p = make_degenerate_y_boundary_patch(d)
    assert CORNER not in p.index_map
    assert len(p.data_indices) == d * d - 1


@pytest.mark.parametrize("d", [3, 5, 7])
def test_check_count_and_weights(d):
    p = make_degenerate_y_boundary_patch(d)
    assert len(p.stabilizers) == d * d - 1

    weights = collections.Counter(len(s['data_indices']) for s in p.stabilizers)
    assert set(weights) <= {2, 3, 4}, weights
    assert weights[3] == 1, "exactly one weight-3 check (the diagonal corner)"
    assert weights[2] == 2 * (d - 1)
    assert weights[4] == (d * d - 1) - weights[2] - weights[3]

    w3 = [s for s in p.stabilizers if len(s['data_indices']) == 3]
    assert tuple(float(v) for v in w3[0]['syn_coord']) == DIAGONAL_W3_SYN


@pytest.mark.parametrize("d", [3, 5, 7])
def test_x_and_z_counts_are_symmetric(d):
    p = make_degenerate_y_boundary_patch(d)
    kinds = collections.Counter(s['type'] for s in p.stabilizers)
    assert kinds['X'] == kinds['Z'] == (d * d - 1) // 2
    assert len(p.syndrome_indices_x) == kinds['X']
    assert len(p.syndrome_indices_z) == kinds['Z']


@pytest.mark.parametrize("d", [3, 5, 7])
def test_no_stabilizer_silently_lost_a_data_qubit(d):
    """``create_stim_stabilizer`` drops coordinates it cannot resolve; the
    patch must never rely on that."""
    p = make_degenerate_y_boundary_patch(d)
    for s in p.stabilizers:
        assert len(s['data_indices']) >= 2
        assert all(i in p.data_indices for i in s['data_indices'])
    # every data qubit is touched by at least two checks
    touched = collections.Counter(i for s in p.stabilizers for i in s['data_indices'])
    assert set(touched) == set(p.data_indices)
    assert min(touched.values()) >= 2


# =============================================================================
# cross-check against Gidney's own patch object, at every distance
# =============================================================================

def gidney_tiles(d):
    """``make_ztop_yboundary_patch(distance=d)`` in LightStim coordinates."""
    if not hasattr(np, 'bool8'):        # his package predates the numpy 2 removal
        np.bool8 = np.bool_
    if GIDNEY_SRC not in sys.path:
        sys.path.insert(0, GIDNEY_SRC)
    try:
        from midout.circuits.steps._patches import make_ztop_yboundary_patch
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"Gidney's midout package is not importable here: {e!r}")

    def g2l(g):
        return (2.0 * g.real + 1.0, 2.0 * g.imag + 1.0)

    out = set()
    for t in make_ztop_yboundary_patch(distance=d).tiles:
        data = tuple(sorted(g2l(q) for q in t.ordered_data_qubits if q is not None))
        out.add((t.basis, g2l(t.measurement_qubit), data))
    return out


@pytest.mark.smoke
def test_golden_constants_agree_with_gidney():
    assert gidney_tiles(3) == D3_GOLDEN_TILES


@pytest.mark.parametrize("d", [3, 5, 7])
def test_tiles_exactly_match_gidney_at_every_distance(d):
    assert tiles_of(make_degenerate_y_boundary_patch(d)) == gidney_tiles(d)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_interaction_order_matches_gidney(d):
    """LightStim's record format has no idle-layer slot, so ``None`` entries are
    dropped — but the surviving interaction order must be Gidney's."""
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    if GIDNEY_SRC not in sys.path:
        sys.path.insert(0, GIDNEY_SRC)
    try:
        from midout.circuits.steps._patches import make_ztop_yboundary_patch
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"Gidney's midout package is not importable here: {e!r}")

    want = {
        (2.0 * t.measurement_qubit.real + 1.0, 2.0 * t.measurement_qubit.imag + 1.0):
            tuple((2.0 * q.real + 1.0, 2.0 * q.imag + 1.0)
                  for q in t.ordered_data_qubits if q is not None)
        for t in make_ztop_yboundary_patch(distance=d).tiles
    }
    p = make_degenerate_y_boundary_patch(d)
    for s in p.stabilizers:
        syn = tuple(float(v) for v in s['syn_coord'])
        got = tuple(tuple(float(v) for v in p.qubit_coords[i])
                    for i in s['data_indices'])
        assert got == want[syn], f"order differs at {syn}"


@pytest.mark.parametrize("d", [3, 5, 7])
def test_registers_into_a_qec_system_with_zero_logicals(d):
    p = make_degenerate_y_boundary_patch(d)
    system = QECSystem()
    view = system.add_patch(p, name='D')
    assert system.num_logicals == 0
    assert len(system.stabilizers) == d * d - 1
    assert len(system.active_stabilizer_indices) == d * d - 1
    assert len(system.data_indices) == d * d - 1
    assert len(system.syndrome_indices) == d * d - 1
    assert CORNER not in system.index_map
    for s in view.stabilizers:
        assert s['syn_idx'] == system.index_map[s['syn_coord']]
        assert set(s['pauli'].keys()) == set(s['data_indices'])
