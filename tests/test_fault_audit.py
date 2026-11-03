"""Self-tests for the code-agnostic low-weight fault audit in
lightstim.utils.fault_audit.

These use toy circuits only -- stim's built-in repetition-code generator, and
a couple of hand-built minimal circuits -- rather than any QEC code family in
this repo, since the audit tool itself is meant to work on any annotated
circuit. See tests/test_h6_distillation.py for its application to the H6
distillation protocol.
"""

import stim

from lightstim.utils.fault_audit import (
    atomise,
    low_weight_fault_audit,
    single_fault_audit,
)


def _repetition_code_circuit(distance, rounds=2):
    return stim.Circuit.generated(
        "repetition_code:memory", distance=distance, rounds=rounds
    )


def _undefended_logical_qubit_circuit():
    """A single logical qubit measured with no detectors at all: a bare
    weight-1 fault before the measurement flips the observable with zero
    detectors firing -- the sharpest possible undetected-logical case."""
    circuit = stim.Circuit()
    circuit.append("R", [0])
    circuit.append("TICK")
    circuit.append("M", [0])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


def test_atomise_preserves_determinism_and_splits_layers():
    circuit = _repetition_code_circuit(distance=3)
    atomic = atomise(circuit)
    dets, obs = atomic.compile_detector_sampler().sample(256, separate_observables=True)
    assert not dets.any() and not obs.any()
    # every two-qubit gate instruction now carries exactly one operand pair
    for inst in atomic.flattened():
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "CX":
            assert len(inst.targets_copy()) == 2


def test_audit_reports_both_detected_and_harmless_classes():
    result = single_fault_audit(_repetition_code_circuit(distance=3))
    assert result.audited_max_weight == 1
    assert any(r.detected for r in result.records)
    assert len(result.undetected_harmless_faults) > 0


def test_low_weight_fault_audit_certifies_distance_two_for_larger_code():
    result = low_weight_fault_audit(_repetition_code_circuit(distance=5))
    assert result.num_locations > 0
    assert result.has_circuit_fault_distance_two, result.summary()


def test_undefended_circuit_is_not_distance_two():
    result = low_weight_fault_audit(_undefended_logical_qubit_circuit())
    assert not result.has_circuit_fault_distance_two
    assert result.min_undetectable_logical_weight == 1
    assert len(result.undetectable_logical_faults) > 0


def test_weight_one_only_audit_cannot_certify_distance_two():
    result = single_fault_audit(_repetition_code_circuit(distance=5))
    assert result.audited_max_weight == 1
    assert not result.has_circuit_fault_distance_two
