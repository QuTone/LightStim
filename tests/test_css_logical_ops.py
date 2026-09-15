"""Shared logical Pauli gates: slot semantics, global indices, and builder use."""

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.H_code import HCode
from lightstim.qec_code.color_code import ColorCode
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode

pytestmark = pytest.mark.smoke


def make_builder(code):
    system = QECSystem()
    other = system.add_patch(HCode(6), name="other")
    patch = system.add_patch(code, name="target", offset=(100, 20))
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    return builder, patch, other


def as_pauli(record, n):
    p = stim.PauliString(n)
    for q, basis in record["pauli"].items():
        p[q] = basis
    return p


@pytest.mark.parametrize("factory", [lambda: HCode(8), lambda: ColorCode(distance=5),
                                     lambda: RotatedSurfaceCode(distance=3),
                                     lambda: RotatedSurfaceCode(distance=1)])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("noiseless", [False, True])
def test_logical_pauli_action_on_every_slot(factory, basis, noiseless):
    builder, patch, other = make_builder(factory())
    executor = LogicalExecutor(builder)
    executor.register_op_set(type(patch), CSSLogicalOpSet())
    for slot in range(patch.num_logicals):
        start = len(builder.circuit)
        executor.apply_logical_operation(f"transversal_{basis.lower()}", [patch],
                                         slot=slot, noiseless=noiseless)
        chunk = builder.circuit[start:]
        gate = next(inst for inst in chunk if inst.name != "TICK")
        assert gate.tag == ("noiseless" if noiseless else "")
        assert {t.value for t in gate.targets_copy()} <= patch.data_indices
        for check in (*patch.stabilizers, *other.stabilizers):
            p = as_pauli(check, builder.system.num_qubits)
            assert p.after(chunk) == p
        for logical_basis in ("X", "Z"):
            ops = [op for op in patch.logical_ops if op["type"] == logical_basis]
            for j, op in enumerate(ops):
                p = as_pauli(op, builder.system.num_qubits)
                sign = -1 if j == slot and logical_basis != basis else 1
                assert p.after(chunk) == sign * p
        for op in other.logical_ops:
            p = as_pauli(op, builder.system.num_qubits)
            assert p.after(chunk) == p


@pytest.mark.parametrize("slot", [-1, 2, 0.5, True])
def test_invalid_logical_slot(slot):
    builder, patch, _ = make_builder(HCode(6))
    with pytest.raises(ValueError, match="slot"):
        CSSLogicalOpSet().transversal_x(builder, patch, slot=slot)


def test_rejects_local_patch_and_mixed_pauli_support():
    builder, patch, _ = make_builder(HCode(6))
    local = builder.system.patches["target"][0]
    with pytest.raises(ValueError, match="global patch"):
        CSSLogicalOpSet().transversal_x(builder, local)
    op = next(op for op in patch.logical_ops if op["type"] == "X")
    op["pauli"][next(iter(op["pauli"]))] = "Y"
    with pytest.raises(ValueError, match="pure X"):
        CSSLogicalOpSet().transversal_x(builder, patch)
