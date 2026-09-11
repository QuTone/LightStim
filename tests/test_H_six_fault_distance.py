"""Stage-2 fault-distance / O(p^2) checks for the [[6, 2, 2]] H-code patch.

The primary correctness criterion is stim's built-in
:meth:`stim.Circuit.shortest_graphlike_error`: circuit-level noise is injected
with :class:`~lightstim.noise.injector.NoiseInjector`, and the length of the
shortest graphlike error is the circuit fault distance. Length ``>= 2`` means no
single fault flips a logical undetected, hence an ``O(p^2)`` conditional
logical-error rate under detector post-selection. The marked-slow Monte-Carlo
slope tests are sanity checks on top of it.
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


def _baseline_memory_circuit(basis: str, rounds: int = 2):
    return MemoryExperiment(
        qec_patch=HSixCode(), rounds=rounds, basis=basis, noise_params=None
    ).build()


def _distillation_encoder_circuit():
    """Magic-H6 level-1 non-FT path: |0>^6 -> get_dist_circ -> |++>_L, one SE
    round, the Bell-pair H-check (which registers the X0_L / X1_L observables),
    then destructive X readout. This is roadmap stage 3, kept here only as the
    contrast case for the fault-distance check."""
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


def _circuit_level_noise(circuit, p: float):
    cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    return NoiseInjector.from_circuit_level(
        cfg, list(range(circuit.num_qubits))
    ).inject_noise(circuit)


def _circuit_fault_distance(circuit, p: float = 1e-3) -> int:
    """Length of the shortest graphlike error of the circuit-level-noised circuit."""
    return len(_circuit_level_noise(circuit, p).shortest_graphlike_error())


def _conditional_ler_slope(circuit_fn, ps, shots=400_000):
    logs_p, logs_ler = [], []
    for p in ps:
        noisy = _circuit_level_noise(circuit_fn(), p)
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


# --- noiseless determinism ---------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_baseline_memory_is_deterministic_noiseless(basis):
    circ = _baseline_memory_circuit(basis)
    dets, obs = circ.compile_detector_sampler().sample(256, separate_observables=True)
    assert not dets.any() and not obs.any()
    circ.detector_error_model(decompose_errors=True)


# --- O(p^2) circuits: bare, encoded, flag-verified -------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_baseline_memory_is_circuit_fault_distance_two(basis):
    assert _circuit_fault_distance(_baseline_memory_circuit(basis)) >= 2


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
    assert _circuit_fault_distance(circ) >= 2


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


# --- contrast: the |++>_L distillation path is genuinely distance 1 -------

@pytest.mark.smoke
def test_distillation_encoder_path_is_only_distance_one():
    """Roadmap stage 3 motivation: get_dist_circ + H-check has a single fault
    on the encoder spine / H-check ancilla that flips a logical undetected."""
    noisy = _circuit_level_noise(_distillation_encoder_circuit(), 1e-3)
    err = noisy.shortest_graphlike_error()
    assert len(err) == 1
    # the lone undetected error flips a logical observable
    assert any("L" in str(term) for term in err[0].dem_error_terms)


# --- O(p^2) scaling under post-selection (slow Monte-Carlo) --------------

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
