"""ColorCodeLogicalOpSet -- transversal Clifford gates for the triangular code."""

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.color_code import (
    ColorCode,
    ColorCodeExtractionBlock,
    ColorCodeLogicalOpSet,
)
from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock


def _builder(patch):
    system = QECSystem()
    p = system.add_patch(patch, name="c")
    tr = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    b = CircuitBuilder(tr, system, if_detector=False)
    b.write_coordinates()
    b.initialize({q: "Z" for q in sorted(p.data_indices)}, n=system.num_qubits)
    return b, p


def test_d3_is_steane_and_doubly_even():
    c = ColorCode(distance=3)
    assert len(c.data_indices) == 7 and c.num_logicals == 1
    assert all(len(s["data_indices"]) % 4 == 0 for s in c.stabilizers)   # all weight 4


def test_transversal_h_is_physical_h_on_all_data():
    b, p = _builder(ColorCode(distance=3))
    ColorCodeLogicalOpSet().transversal_hadamard(b, p)
    last = b.circuit[-1]
    assert last.name == "H"
    assert {t.value for t in last.targets_copy()} == set(sorted(p.data_indices))


def test_transversal_s_is_uniform_s_dag_at_d3():
    b, p = _builder(ColorCode(distance=3))
    ops = ColorCodeLogicalOpSet()
    ops.transversal_s(b, p)                       # logical S  (n=7 -> physical S_DAG)
    assert b.circuit[-1].name == "S_DAG"
    ops.transversal_s_dag(b, p)                   # logical S_DAG
    assert b.circuit[-1].name == "S"


def test_transversal_s_raises_for_non_doubly_even():
    with pytest.raises(NotImplementedError):
        ColorCodeLogicalOpSet().transversal_s(*_builder(ColorCode(distance=5)))


def test_inherited_transversal_x_z_use_logical_support():
    c = ColorCode(distance=3)
    b, p = _builder(c)
    ops = ColorCodeLogicalOpSet()
    xl = sorted(next(o["data_indices"] for o in p.logical_ops if o["type"] == "X"))
    ops.transversal_x(b, p)
    assert b.circuit[-1].name == "X"
    assert sorted(t.value for t in b.circuit[-1].targets_copy()) == xl


def test_transversal_h_is_logical_h_by_tableau():
    """H_L : |0>_L -> |+>_L (Z_L eigenstate -> X_L eigenstate) on a d=3 patch."""
    c = ColorCode(distance=3)
    b, p = _builder(c)
    n = max(p.data_indices) + 1
    xs = set(next(o["data_indices"] for o in p.logical_ops if o["type"] == "X"))
    zs = set(next(o["data_indices"] for o in p.logical_ops if o["type"] == "Z"))
    XL = stim.PauliString("".join("X" if i in xs else "_" for i in range(n)))
    ZL = stim.PauliString("".join("Z" if i in zs else "_" for i in range(n)))

    sim = stim.TableauSimulator(); sim.do(b.circuit)
    assert sim.peek_observable_expectation(ZL) == 1        # |0>_L

    ColorCodeLogicalOpSet().transversal_hadamard(b, p)
    sim = stim.TableauSimulator(); sim.do(b.circuit)
    assert sim.peek_observable_expectation(XL) == 1        # -> |+>_L
