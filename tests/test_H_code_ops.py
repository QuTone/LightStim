"""Signed H-family logical action and tracker integration at code boundaries."""

from itertools import product

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.H_code import (
    FlagAncillaPatch, HCode, HCodeExtractionBlock, HCodeLogicalOpSet,
    HSixCode, HSixLogicalOpSet,
)
from lightstim.qec_code.color_code import ColorCode

pytestmark = pytest.mark.smoke


def _builder(code):
    system = QECSystem()
    other = system.add_patch(HCode(6), name="other", is_active=False)
    patch = system.add_patch(code, name="h", offset=(100, 20))
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    return builder, patch, other


def _pauli(record, n):
    pauli = stim.PauliString(n)
    for q, basis in record["pauli"].items():
        pauli[q] = basis
    return pauli


@pytest.mark.parametrize("n", [6, 8, 12, 64])
@pytest.mark.parametrize("noiseless", [False, True])
def test_h_exchanges_every_signed_logical_pair_and_check(n, noiseless):
    builder, patch, other = _builder(HCode(n))
    executor = LogicalExecutor(builder)
    executor.register_op_set(HCode, HCodeLogicalOpSet())
    start = len(builder.circuit)
    executor.apply_logical_operation("transversal_hadamard", [patch], noiseless=noiseless)
    circuit = builder.circuit[start:]
    for records in (patch.stabilizers, patch.logical_ops):
        xs = [_pauli(r, builder.system.num_qubits) for r in records if r["type"] == "X"]
        zs = [_pauli(r, builder.system.num_qubits) for r in records if r["type"] == "Z"]
        for x, z in zip(xs, zs):
            assert x.after(circuit) == z
            assert z.after(circuit) == x
    for x_record, z_record in zip(patch.logical_ops[::2], patch.logical_ops[1::2]):
        y = 1j * _pauli(x_record, builder.system.num_qubits) * _pauli(z_record, builder.system.num_qubits)
        assert y.after(circuit) == -y
    for record in (*other.stabilizers, *other.logical_ops):
        pauli = _pauli(record, builder.system.num_qubits)
        assert pauli.after(circuit) == pauli
    gates = [g for g in circuit if g.name != "TICK"]
    assert len(gates) == 1 and gates[0].name == "H"
    assert {t.value for t in gates[0].targets_copy()} == patch.data_indices
    assert gates[0].tag == ("noiseless" if noiseless else "")


@pytest.mark.parametrize("code", [HCode(8), HSixCode()])
@pytest.mark.parametrize("initial,final", [("Z", "X"), ("X", "Z")])
def test_h_between_extraction_rounds_preserves_tracker_relations(code, initial, final):
    system = QECSystem()
    patch = system.add_patch(code, name="h")
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.initialize({q: initial for q in patch.data_indices}, n=system.num_qubits)
    se = HCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(se.circuit, rounds=1)
    executor = LogicalExecutor(builder)
    executor.register_op_set(type(patch), HCodeLogicalOpSet())
    executor.apply_logical_operation("transversal_hadamard", [patch])
    builder.apply_syndrome_extraction(se.circuit, rounds=2)
    builder.apply_data_readout({q: final for q in patch.data_indices})
    circuit = builder.circuit
    circuit.detector_error_model()
    dets, obs = circuit.compile_detector_sampler(seed=98).sample(128, separate_observables=True)
    assert circuit.num_observables == patch.n - 4
    assert not dets.any() and not obs.any()


def test_h6_operation_exports():
    from lightstim.qec_code.H_code.operation import HSixLogicalOpSet as SixOps
    assert HSixLogicalOpSet is SixOps
    assert issubclass(HSixLogicalOpSet, HCodeLogicalOpSet)


def test_h_rejects_local_foreign_or_mismatched_patch():
    ops = HCodeLogicalOpSet()
    builder, patch, _ = _builder(HCode(8))
    with pytest.raises(ValueError, match="global patch"):
        ops.transversal_hadamard(builder, builder.system.patches["h"][0])
    foreign_builder, foreign, _ = _builder(ColorCode(distance=3))
    with pytest.raises(ValueError, match="requires an HCode"):
        ops.transversal_hadamard(foreign_builder, foreign)
    patch.logical_ops[1]["pauli"] = dict(patch.logical_ops[3]["pauli"])
    patch.logical_ops[1]["data_indices"] = list(patch.logical_ops[3]["data_indices"])
    with pytest.raises(ValueError, match="matching X/Z"):
        ops.transversal_hadamard(builder, patch)


@pytest.mark.parametrize("code", [HSixCode(), HCode(6)])
@pytest.mark.parametrize("bases", list(product("XYZ", repeat=2)))
def test_h6_encoder_prepares_independent_signed_logical_states(code, bases):
    system = QECSystem()
    other = system.add_patch(HCode(6), name="other", is_active=False)
    patch = system.add_patch(code, name="h", offset=(100, 20))
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.initialize({q: "Y" for q in other.data_indices}, n=system.num_qubits)
    start = len(builder.circuit)
    executor = LogicalExecutor(builder)
    executor.register_op_set(type(patch), HSixLogicalOpSet())
    executor.apply_logical_operation("encode", [patch], logical_bases=bases)

    sim = stim.TableauSimulator()
    sim.do(builder.circuit)
    for record in patch.stabilizers:
        assert sim.peek_observable_expectation(_pauli(record, system.num_qubits)) == 1
    for slot, basis in enumerate(bases):
        x = _pauli(patch.logical_ops[2 * slot], system.num_qubits)
        z = _pauli(patch.logical_ops[2 * slot + 1], system.num_qubits)
        logical = {"X": x, "Y": 1j * x * z, "Z": z}[basis]
        assert sim.peek_observable_expectation(logical) == 1
    for q in other.data_indices:
        y = stim.PauliString(system.num_qubits)
        y[q] = "Y"
        assert sim.peek_observable_expectation(y) == 1
    for inst in builder.circuit[start:]:
        assert {t.value for t in inst.targets_copy()} <= patch.data_indices


@pytest.mark.parametrize("bases", list(product("XYZ", repeat=2)))
def test_h6_encoded_memory_uses_tracker_for_all_basis_pairs(bases):
    system = QECSystem()
    patch = system.add_patch(HSixCode(), name="h")
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    HSixLogicalOpSet().encode(builder, patch, logical_bases=bases)
    builder.stabilizer_canonicalization()
    builder.apply_syndrome_extraction(HCodeExtractionBlock(system).circuit, rounds=2)
    # The two registered logical supports are the even and odd data labels.
    builder.apply_data_readout({
        q: bases[i % 2] for i, q in enumerate(sorted(patch.data_indices))
    })
    circuit = builder.circuit
    circuit.detector_error_model()
    dets, obs = circuit.compile_detector_sampler(seed=98).sample(128, separate_observables=True)
    assert circuit.num_observables == 2
    assert circuit.num_detectors == (10 if bases[0] == bases[1] else 8)
    assert not dets.any() and not obs.any()


def _flagged_encode_builder():
    system = QECSystem()
    patch = system.add_patch(HSixCode(), name="h")
    flag_patch = system.add_patch(FlagAncillaPatch(), name="flags", offset=(20, 20))
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    return system, patch, flag_patch, builder


def test_h6_flagged_encoder_rejects_invalid_requests():
    system, patch, flag_patch, builder = _flagged_encode_builder()
    flags = sorted(flag_patch.syndrome_indices)
    ops = HSixLogicalOpSet()
    with pytest.raises(ValueError, match="only prepares"):
        ops.encode(builder, patch, ("X", "X"), flagged=True, flag_qubits=flags)
    with pytest.raises(ValueError, match="exactly 2 global qubit indices"):
        ops.encode(builder, patch, ("Z", "Z"), flagged=True, flag_qubits=flags[:1])
    with pytest.raises(ValueError, match="exactly 2 global qubit indices"):
        ops.encode(builder, patch, ("Z", "Z"), flagged=True, flag_qubits=None)


def test_h6_flagged_encoder_produces_tagged_flag_detectors_at_distinct_coords():
    system, patch, flag_patch, builder = _flagged_encode_builder()
    flags = sorted(flag_patch.syndrome_indices)
    HSixLogicalOpSet().encode(builder, patch, ("Z", "Z"), flagged=True, flag_qubits=flags)
    builder.stabilizer_canonicalization()
    builder.apply_data_readout({q: "Z" for q in patch.data_indices})
    circuit = builder.circuit
    flag_detectors = [
        inst for inst in circuit.flattened()
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "DETECTOR"
        and inst.tag == "post-select"
    ]
    assert len(flag_detectors) == 2
    coords = [tuple(inst.gate_args_copy()) for inst in flag_detectors]
    assert coords[0] != coords[1]  # each flag gets its own coordinate
    dets, obs = circuit.compile_detector_sampler().sample(2000, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_h6_flagged_encoder_reaches_fault_distance_two_with_one_se_round():
    """The whole point of Fig. 5: a single SE round after the flagged encoder
    is already enough to reach fault distance 2, unlike the unflagged Fig.
    1(d) encoder (see lightstim/protocols/h6_distillation.py, which needs an
    extra Bell-pair check on top of Fig. 1(d) to close this same gap)."""
    from lightstim.noise.config import NoiseConfig
    from lightstim.noise.injector import NoiseInjector

    system, patch, flag_patch, builder = _flagged_encode_builder()
    flags = sorted(flag_patch.syndrome_indices)
    HSixLogicalOpSet().encode(builder, patch, ("Z", "Z"), flagged=True, flag_qubits=flags)
    builder.stabilizer_canonicalization()
    ps_coords = set()
    for stab in patch.stabilizers:
        syn_idx = stab.get("syn_idx")
        if syn_idx is None:
            continue
        xy = tuple(system.qubit_coords[syn_idx])
        ps_coords.add(xy + (0.0,))
        ps_coords.add(xy + (1.0,))
    builder.tracker.post_select_detector_coords |= ps_coords
    builder.apply_syndrome_extraction(HCodeExtractionBlock(system).circuit, rounds=1)
    builder.apply_data_readout({q: "Z" for q in patch.data_indices})
    circuit = builder.circuit
    noisy = NoiseInjector.from_circuit_level(
        NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3),
        list(range(circuit.num_qubits)),
    ).inject_noise(circuit)
    assert len(noisy.shortest_graphlike_error()) == 2


@pytest.mark.parametrize("basis", "XYZ")
def test_h6_preparation_wrappers_and_noiseless_layers(basis):
    system = QECSystem()
    patch = system.add_patch(HSixCode(), name="h")
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    getattr(HSixLogicalOpSet(), f"prepare_logical_{basis.lower() * 2}")(
        builder, patch, noiseless=True,
    )
    gates = [g for g in builder.circuit if g.name not in ("QUBIT_COORDS", "TICK")]
    assert all(g.tag == "noiseless" for g in gates)
    # Four disjoint CNOT layers, separated so circuit-level noise follows
    # each physical gate before any later gate on the same qubit.
    layers = [g for g in gates if g.name == "CX"]
    assert len(layers) == 4
    assert all(len(set(g.targets_copy())) == 4 for g in layers)
    sim = stim.TableauSimulator()
    sim.do(builder.circuit)
    for slot in range(2):
        x = _pauli(patch.logical_ops[2 * slot], system.num_qubits)
        z = _pauli(patch.logical_ops[2 * slot + 1], system.num_qubits)
        assert sim.peek_observable_expectation({"X": x, "Y": 1j*x*z, "Z": z}[basis]) == 1


def test_h6_encoder_rejects_invalid_requests_before_mutation():
    system = QECSystem()
    patch = system.add_patch(HSixCode(), name="h")
    larger = system.add_patch(HCode(8), name="larger", offset=(100, 20))
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    ops = HSixLogicalOpSet()
    original = builder.circuit.copy()
    with pytest.raises(ValueError, match="requires HSixCode"):
        ops.encode(builder, larger)
    with pytest.raises(ValueError, match="global patch"):
        ops.encode(builder, HSixCode())
    for bases in [("Z",), ("X", "Y", "Z"), ("Z", "H")]:
        with pytest.raises(ValueError, match="two bases"):
            ops.encode(builder, patch, bases)
    assert builder.circuit == original
    ops.encode(builder, patch)
    encoded = builder.circuit.copy()
    with pytest.raises(ValueError, match="uninitialized"):
        ops.encode(builder, patch)
    assert builder.circuit == encoded
