"""Stage-2 fault-distance / O(p^2) checks for the [[6, 2, 2]] H-code patch.

The exact low-weight fault audit is the primary correctness criterion; the
Monte-Carlo slope tests are marked-slow sanity checks on top of it. Both agree
because the audit atomises merged gate layers before probing.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_six import (
    HSixCode,
    HSixExtractionBlock,
    HSixLogicalXCheckBlock,
    encoded_memory_circuit,
    get_dist_circ,
)
from lightstim.utils.fault_audit import (
    atomise,
    low_weight_fault_audit,
    single_fault_audit,
)


def _baseline_memory_circuit(basis: str, rounds: int = 2):
    return MemoryExperiment(
        qec_patch=HSixCode(), rounds=rounds, basis=basis, noise_params=None
    ).build()


def _distillation_encoder_circuit():
    """Magic-H6 level-1 non-FT path: |0>^6 -> get_dist_circ -> |++>_L, one SE
    round, the Bell-pair H-check (which registers the X0_L / X1_L observables),
    then destructive X readout. This is roadmap stage 3, kept here only as the
    contrast case for the fault audit."""
    system = QECSystem()
    system.add_patch(HSixCode(h_check_ancillas=2), name="c622")
    tracker = SyndromeTracker(
        system.num_qubits, expected_num_logicals=system.num_logicals
    )
    builder = CircuitBuilder(tracker, system, if_detector=True)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in data}, n=system.num_qubits)
    builder.apply_unitary_block(get_dist_circ(data))
    builder.apply_syndrome_extraction(
        circuit_chunk=HSixExtractionBlock(system).circuit, rounds=1
    )
    builder.apply_syndrome_extraction(
        circuit_chunk=HSixLogicalXCheckBlock(system).circuit, rounds=1
    )
    builder.apply_data_readout({q: "X" for q in data})
    return builder.circuit


def _conditional_ler_slope(circuit_fn, ps, shots=400_000):
    logs_p, logs_ler = [], []
    for p in ps:
        circuit = circuit_fn()
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
        noisy = NoiseInjector.from_circuit_level(
            cfg, list(range(circuit.num_qubits))
        ).inject_noise(circuit)
        dets, obs = noisy.compile_detector_sampler().sample(
            shots, separate_observables=True
        )
        accepted = ~dets.any(axis=1)
        assert accepted.sum() > 2_000
        ler = obs[accepted].any(axis=1).mean()
        assert ler > 0
        logs_p.append(np.log(p))
        logs_ler.append(np.log(ler))
    return float(np.polyfit(logs_p, logs_ler, 1)[0])


# --- the audit tool -------------------------------------------------------

@pytest.mark.smoke
def test_atomise_preserves_determinism_and_splits_layers():
    circ, _ = encoded_memory_circuit(basis="Z", rounds=1, encoder="zero_zero")
    atomic = atomise(circ)
    dets, obs = atomic.compile_detector_sampler().sample(
        256, separate_observables=True
    )
    assert not dets.any() and not obs.any()
    # every gate instruction now carries at most one 1q target or one 2q pair
    for inst in atomic.flattened():
        if inst.name in ("CX",):
            assert len(inst.targets_copy()) == 2
        if inst.name in ("H",):
            assert len(inst.targets_copy()) == 1


@pytest.mark.smoke
def test_audit_reports_both_detected_and_harmless_classes():
    result = single_fault_audit(_baseline_memory_circuit("Z"))
    assert any(r.detected for r in result.records)
    assert len(result.undetected_harmless_faults) > 0
    assert result.audited_max_weight == 1


# --- O(p^2) circuits: bare, encoded, flag-verified ----------------------

@pytest.mark.smoke
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_baseline_memory_is_circuit_fault_distance_two(basis):
    result = low_weight_fault_audit(_baseline_memory_circuit(basis))
    assert result.num_locations > 0
    assert result.has_circuit_fault_distance_two, result.summary()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder": None},
        {"encoder": "zero_zero"},
        {"encoder": "zero_zero", "flag_verified": True},
    ],
)
def test_encoded_memory_is_circuit_fault_distance_two(kwargs):
    circ, info = encoded_memory_circuit(basis="Z", rounds=2, **kwargs)
    assert info["num_observables"] == 2
    dets, obs = circ.compile_detector_sampler().sample(4096, separate_observables=True)
    assert not dets.any() and not obs.any()
    circ.detector_error_model(decompose_errors=True)
    assert low_weight_fault_audit(circ).has_circuit_fault_distance_two


@pytest.mark.smoke
def test_flag_verified_prep_adds_two_postselected_flag_detectors():
    plain, _ = encoded_memory_circuit(basis="Z", rounds=2, encoder="zero_zero")
    flagged, info = encoded_memory_circuit(
        basis="Z", rounds=2, encoder="zero_zero", flag_verified=True
    )
    assert info["flag_verified"] is True
    assert info["flag_detector_indices"] == [0, 1]
    assert flagged.num_detectors == plain.num_detectors + 2
    # the flag detectors are deterministic-0 noiseless, like every other detector
    dets, _ = flagged.compile_detector_sampler().sample(512, separate_observables=True)
    assert not dets[:, info["flag_detector_indices"]].any()


# --- contrast: the |++>_L distillation path is genuinely distance 1 -----

@pytest.mark.smoke
def test_distillation_encoder_path_is_only_distance_one():
    """Roadmap stage 3 motivation: get_dist_circ + H-check has weight-1 faults
    on the encoder spine / H-check ancilla that flip a logical undetected."""
    result = single_fault_audit(_distillation_encoder_circuit())
    assert len(result.undetectable_logical_faults) > 0
    assert all(
        r.gate in {"R", "H", "CX"} for r in result.undetectable_logical_faults
    )


# --- O(p^2) scaling under post-selection (slow Monte-Carlo) -------------

@pytest.mark.slow
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_baseline_memory_conditional_ler_is_quadratic(basis):
    ps = [0.002, 0.004, 0.008, 0.016]
    slope = _conditional_ler_slope(
        lambda: _baseline_memory_circuit(basis, rounds=2), ps
    )
    assert 1.6 <= slope <= 2.6, f"baseline slope {slope:.2f} not ~2"


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs",
    [{"encoder": "zero_zero"}, {"encoder": "zero_zero", "flag_verified": True}],
)
def test_encoded_memory_conditional_ler_is_quadratic(kwargs):
    ps = [0.002, 0.004, 0.008, 0.016]
    slope = _conditional_ler_slope(
        lambda: encoded_memory_circuit(basis="Z", rounds=2, **kwargs)[0], ps
    )
    assert 1.6 <= slope <= 2.6, f"encoded slope {slope:.2f} not ~2"


@pytest.mark.slow
def test_distillation_encoder_path_is_linear():
    ps = [0.001, 0.002, 0.004, 0.008]
    slope = _conditional_ler_slope(_distillation_encoder_circuit, ps)
    assert slope < 1.6, f"expected ~O(p), got slope {slope:.2f}"
