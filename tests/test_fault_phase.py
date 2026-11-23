"""Independent analytic checks for phase propagation and the measurement instrument."""
import math

import pytest
import stim

from lightstim.analysis.faults._catalogue import catalogue
from lightstim.analysis.faults._nonclifford import BornEvaluator
from lightstim.analysis.faults._phase import (
    QuantumEnumerationLimitError, QuantumLimits, evaluate_fixed_circuit, parse_quantum_circuit,
)


def evaluator(text, reference=()):
    circuit = parse_quantum_circuit(text)
    return BornEvaluator(circuit, catalogue(circuit), limits=QuantumLimits(), detector_reference=reference)


def test_joint_faults_straddling_t_preserve_phase():
    circuit = evaluator("H 0\nX_ERROR(.1) 0\nT 0\nX_ERROR(.1) 0\nEXP_VAL X0 Y0")
    expected = math.sqrt(.5)
    for chosen, probe_y in [((), expected), (((0, 0), (1, 0)), -expected)]:
        (probability, syndrome, observable, probes), = circuit.responses(chosen)
        assert (probability, syndrome, observable) == (1., 0, 0)
        assert probes == pytest.approx((expected, probe_y))


def test_two_phase_layers_interfere_and_probes_do_not_consume_records():
    circuit = evaluator("H 0\nT 0\nH 0\nEXP_VAL Z0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    branches = tuple(circuit.responses(()))
    assert {observable: probability for probability, _, observable, _ in branches} == pytest.approx({0: .75, 1: .25})
    for _, _, _, probes in branches:
        assert probes == pytest.approx((math.sqrt(.5),))


@pytest.mark.parametrize(
    "setup,measurement,probe",
    [("R 0", "M", "Z0"), ("RX 0", "MX", "X0"), ("RY 0", "MY", "Y0"),
     ("X 0", "MR", "Z0"), ("RX 0\nZ 0", "MRX", "X0"), ("RY 0\nX 0", "MRY", "Y0")],
)
def test_forced_readout_flips_feedback_and_preserves_postmeasurement_state(setup, measurement, probe):
    # The baseline MR input has negative eigenvalue, while M input is positive.
    baseline_bit = measurement.startswith("MR")
    circuit = evaluator(
        f"{setup}\n{measurement}(.2) !0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1]\n"
        f"OBSERVABLE_INCLUDE(0) rec[-2]\nEXP_VAL {probe}", reference=(True,),
    )
    for fault, reported in [((), not baseline_bit), (((0, 0),), baseline_bit)]:
        (probability, syndrome, observable, probes), = circuit.responses(fault)
        assert probability == pytest.approx(1.)
        assert syndrome == (reported ^ True)
        assert observable == reported
        assert probes == pytest.approx((1.,))


@pytest.mark.parametrize(
    "measurement,targets,probes",
    [("MXX", "0 1 2 3", "X0*X1 X2*X3"), ("MYY", "0 1 2 3", "Y0*Y1 Y2*Y3"),
     ("MZZ", "0 1 2 3", "Z0*Z1 Z2*Z3"), ("MPP", "X0*X1 Z2*Z3", "X0*X1 Z2*Z3")],
)
def test_readout_fault_on_second_product_preserves_its_quantum_state(measurement, targets, probes):
    circuit = evaluator(
        f"{measurement}(.2) {targets}\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
        f"OBSERVABLE_INCLUDE(1) rec[-1]\nEXP_VAL {probes}"
    )
    for p, _, bits, expectations in circuit.responses(((1, 0),)):
        assert p > 0
        assert expectations == pytest.approx(((-1) ** (bits & 1), -((-1) ** (bits >> 1))))


def test_reset_and_continued_projected_branch_keep_joint_probability():
    branches = evaluate_fixed_circuit("H 0\nCX 0 1\nR 0\nM 0 1")
    by_record = {branch.records: branch.probability for branch in branches}
    assert by_record == pytest.approx({(False, False): .5, (False, True): .5})
    accepted = tuple(branch for branch in branches if branch.records[-1])
    continued, = evaluate_fixed_circuit("T_DAG 1\nEXP_VAL Z1", initial_branches=accepted)
    assert continued.probability == pytest.approx(.5)
    assert continued.expectations == pytest.approx((-1.,))


def test_resource_guards_bound_circuit_expansion_and_born_branches():
    repeated = stim.Circuit("REPEAT 1000000000 {\nTICK\n}")
    with pytest.raises(QuantumEnumerationLimitError, match="max_instructions"):
        parse_quantum_circuit(repeated, max_instructions=16)
    with pytest.raises(QuantumEnumerationLimitError, match="max_instructions"):
        parse_quantum_circuit("REPEAT 1000000000 {\nREPEAT 2 {\n}\n}")
    with pytest.raises(QuantumEnumerationLimitError, match="max_targets"):
        parse_quantum_circuit("REPEAT 5 {\nH 0 1 2 3\n}", max_targets=19)
    with pytest.raises(QuantumEnumerationLimitError, match="max_branches"):
        evaluate_fixed_circuit("H 0 1\nM 0 1", limits=QuantumLimits(max_branches=2))
    with pytest.raises(QuantumEnumerationLimitError, match="max_amplitudes"):
        evaluate_fixed_circuit("H 0 1", limits=QuantumLimits(max_amplitudes=2))
    with pytest.raises(QuantumEnumerationLimitError, match="max_work"):
        evaluate_fixed_circuit("H 0", limits=QuantumLimits(max_work=1))
    with pytest.raises(QuantumEnumerationLimitError, match="max_total_amplitudes"):
        evaluate_fixed_circuit("H 0\nM 0\nH 1", limits=QuantumLimits(max_total_amplitudes=3))


@pytest.mark.parametrize("gate", ["T", "T_DAG", "EXP_VAL"])
def test_empty_extended_gate_is_rejected(gate):
    with pytest.raises(ValueError, match="target"):
        parse_quantum_circuit(gate)


def test_parser_rejects_native_reserved_tags_and_preserves_probability_precision():
    with pytest.raises(ValueError, match="reserved"):
        parse_quantum_circuit(stim.Circuit("REPEAT 2 {\nS[ENUM_T] 0\n}"))
    circuit = stim.Circuit()
    p = .12345678901234567
    circuit.append("X_ERROR", [0], p)
    assert parse_quantum_circuit(circuit)[0].gate_args_copy() == [p]


def test_born_rejects_silent_loss_of_output_or_noise():
    with pytest.raises(ValueError, match="record-backed"):
        evaluator("T 0\nOBSERVABLE_INCLUDE(0) Z0")
    with pytest.raises(ValueError, match="stochastic"):
        evaluate_fixed_circuit("X_ERROR(.1) 0")
