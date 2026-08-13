"""Failure-accounting regressions for TG's transformed observable loop."""

import numpy as np
import pytest
import stim

from lightstim.protocols.tg_distillation import run_simulation


def _two_observable_circuit() -> stim.Circuit:
    return stim.Circuit(
        """
        R 0 1
        X_ERROR(0.5) 0
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )


@pytest.mark.parametrize(
    "policy, expected_kept, expected_errors",
    [
        ("error", 12, 12),
        ("discard", 0, 0),
        ("ignore", 12, 0),
    ],
)
def test_transformed_tg_loop_honors_decode_failure_policy(
    policy, expected_kept, expected_errors
):
    # Swapping observables forces the custom transform loop. Its target is the
    # always-zero original observable 1, so trusting the zero timeout fallback
    # under "ignore" produces no errors; the other policies remain observable.
    transform = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    stats = run_simulation(
        circuit=_two_observable_circuit(),
        magic_qubits=set(),
        p=0.0,
        p_injected=0.0,
        mode="injection",
        T=transform,
        ps_indices=[],
        target_indices=[0],
        decoder_name="mle-ilp",
        max_shots=12,
        max_errors=100,
        num_workers=1,
        backend="cpu",
        batch_size=12,
        decoder_params={"time_limit": 1e-12},
        on_decode_failure=policy,
    )

    assert stats.shots == 12
    assert stats.post_selected_shots == expected_kept
    assert stats.errors == expected_errors
