"""Signed logical action of color-code Clifford gates, extending PR #98 tests."""

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.H_code import HCode
from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock, ColorCodeLogicalOpSet

pytestmark = pytest.mark.smoke


def setup_code(distance=3, layout="superdense", basis="Z"):
    system = QECSystem()
    other = system.add_patch(HCode(6), name="other", is_active=False)
    patch = system.add_patch(ColorCode(distance=distance, layout=layout), name="color", offset=(100, 20))
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.write_coordinates()
    builder.initialize({q: basis for q in system.data_indices}, n=system.num_qubits)
    return builder, patch, other


def pauli(record, n):
    p = stim.PauliString(n)
    for q, factor in record["pauli"].items():
        p[q] = factor
    return p


@pytest.mark.parametrize("distance", [3, 5, 7, 9])
@pytest.mark.parametrize("layout", ["superdense", "raw", "triangular", "rectangle"])
@pytest.mark.parametrize("gate", ["x", "z", "hadamard", "s", "s_dag"])
def test_signed_clifford_action_at_all_code_boundaries(distance, layout, gate):
    builder, patch, other = setup_code(distance, layout)
    start = len(builder.circuit)
    getattr(ColorCodeLogicalOpSet(), f"transversal_{gate}")(builder, patch, noiseless=True)
    c = builder.circuit[start:]
    n = builder.system.num_qubits
    xl, zl = [pauli(next(o for o in patch.logical_ops if o["type"] == b), n) for b in ("X", "Z")]
    yl = 1j * xl * zl
    expected = {"x": (xl, -zl), "z": (-xl, zl),
                "hadamard": (zl, xl), "s": (yl, zl), "s_dag": (-yl, zl)}[gate]
    assert xl.after(c) == expected[0] and zl.after(c) == expected[1]
    xchecks = {frozenset(s["pauli"]): pauli(s, n) for s in patch.stabilizers if s["type"] == "X"}
    zchecks = {frozenset(s["pauli"]): pauli(s, n) for s in patch.stabilizers if s["type"] == "Z"}
    for support, x in xchecks.items():
        z = zchecks[support]
        expected_x = z if gate == "hadamard" else x * z if gate in ("s", "s_dag") else x
        assert x.after(c) == expected_x
        assert z.after(c) == (x if gate == "hadamard" else z)
    for op in other.logical_ops:
        p = pauli(op, n)
        assert p.after(c) == p
    # One parallel layer of single-qubit gates; no syndrome qubit touched.
    gates = [inst for inst in c if inst.name != "TICK"]
    targets = [t.value for inst in gates for t in inst.targets_copy()]
    assert len(targets) == len(set(targets)) and set(targets) == patch.data_indices
    assert all(inst.tag == "noiseless" for inst in gates)


def test_d3_uniform_phase_convention_from_original_pr():
    builder, patch, _ = setup_code()
    ops = ColorCodeLogicalOpSet()
    ops.transversal_s(builder, patch)
    assert builder.circuit[-1].name == "S_DAG"
    ops.transversal_s_dag(builder, patch)
    assert builder.circuit[-1].name == "S"


@pytest.mark.parametrize("gate,initial,final", [("x", "Z", "Z"), ("z", "X", "X"),
                                               ("hadamard", "Z", "X"),
                                               ("s", "X", "Y"), ("s_dag", "X", "Y")])
def test_clifford_between_extraction_rounds_uses_tracker(gate, initial, final):
    system = QECSystem()
    patch = system.add_patch(ColorCode(distance=5), name="color")
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.write_coordinates()
    builder.initialize({q: initial for q in patch.data_indices}, n=system.num_qubits)
    se = ColorCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=1)
    executor = LogicalExecutor(builder)
    executor.register_op_set(ColorCode, ColorCodeLogicalOpSet())
    executor.apply_logical_operation(f"transversal_{gate}", [patch])
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=2)
    builder.apply_data_readout({q: final for q in patch.data_indices})
    c = builder.circuit
    c.detector_error_model()
    dets, obs = c.compile_detector_sampler(seed=98).sample(128, separate_observables=True)
    assert c.num_observables == 1 and not dets.any() and not obs.any()


def test_color_gates_reject_local_and_mixed_supports():
    builder, patch, _ = setup_code()
    ops = ColorCodeLogicalOpSet()
    with pytest.raises(ValueError, match="global patch"):
        ops.transversal_x(builder, builder.system.patches["color"][0])
    logical = next(op for op in patch.logical_ops if op["type"] == "X")
    logical["pauli"][next(iter(logical["pauli"]))] = "Y"
    with pytest.raises(ValueError, match="pure X"):
        ops.transversal_x(builder, patch)
