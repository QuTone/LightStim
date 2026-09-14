"""H6 memory audits with all-detector postselection.

The undecomposed DEM includes all single-fault signatures, including hyperedges.
Absence of an undetected logical term gives a lower bound of two. An actual
two-fault graphlike witness gives the upper bound. This applies to this memory
schedule with independent Pauli gate/SPAM noise and no idle noise.
"""

import numpy as np
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_six import HSixCode


def _noisy_memory(basis, rounds=2, p=1e-3):
    return MemoryExperiment(
        qec_patch=HSixCode(), rounds=rounds, basis=basis,
        noise_params=NoiseConfig(
            p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=0
        ),
        noise_model="circuit_level",
    ).build()


@pytest.mark.smoke
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_all_single_faults_detected_and_two_fault_witness(basis, rounds):
    noisy = _noisy_memory(basis, rounds)
    errors = [
        inst for inst in noisy.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=False
        ).flattened() if inst.type == "error"
    ]
    assert errors
    assert any(any(t.is_logical_observable_id() for t in e.targets_copy()) for e in errors)
    for error in errors:
        targets = error.targets_copy()
        assert not any(t.is_separator() for t in targets)
        if any(t.is_logical_observable_id() for t in targets):
            assert any(t.is_relative_detector_id() for t in targets), str(error)

    witness = noisy.shortest_graphlike_error(canonicalize_circuit_errors=True)
    assert len(witness) == 2
    assert all(error.circuit_error_locations for error in witness)
    # Batched Stim instructions contain independent channels on each target
    # (or target pair). Exclude mutually exclusive outcomes of one channel.
    locations = [error.circuit_error_locations[0] for error in witness]
    if locations[0].stack_frames == locations[1].stack_frames:
        a, b = (loc.instruction_targets for loc in locations)
        assert a.target_range_end <= b.target_range_start or b.target_range_end <= a.target_range_start


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_baseline_memory_conditional_ler_is_quadratic(basis):
    rates = (0.002, 0.004, 0.008, 0.016)
    lers = []
    for index, p in enumerate(rates):
        noisy = _noisy_memory(basis, p=p)
        dets, obs = noisy.compile_detector_sampler(seed=98 + index).sample(
            400_000, separate_observables=True
        )
        accepted = ~dets.any(axis=1)
        failures = int(obs[accepted].any(axis=1).sum())
        assert accepted.sum() > 2_000 and failures > 0
        ler = failures / accepted.sum()
        lers.append(ler)
        print(f"{basis}: p={p:g}, accepted={accepted.sum()}, failures={failures}, conditional_ler={ler:.8g}")
    slope = float(np.polyfit(np.log(rates), np.log(lers), 1)[0])
    print(f"{basis}: fitted slope={slope:.4f}")
    assert 1.6 <= slope <= 2.6, f"baseline slope {slope:.2f} not near 2"
