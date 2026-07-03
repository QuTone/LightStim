import pytest
import stim

from lightstim.noise.injector import NoiseInjector


def test_xz_biased_model_uses_z_dominant_gate_and_idle_channels():
    cfg = NoiseInjector.compute_XZ_biased_params(
        p_1q=0.01,
        p_2q=0.02,
        p_meas=0.0,
        p_reset=0.0,
        eta=0.1,
        p_idle=0.03,
    )

    assert cfg.custom_params["p_1q_z"] > cfg.custom_params["p_1q_x"]
    assert cfg.custom_params["p_2q_z"] > cfg.custom_params["p_2q_x"]
    assert cfg.custom_params["p_idle_z"] > cfg.custom_params["p_idle_x"]

    circuit = stim.Circuit(
        """
        R 0
        TICK[SE_start]
        H 0
        """
    )
    noisy = NoiseInjector.from_XZ_biased(cfg, [0]).inject_noise(circuit)

    assert not any(inst.name == "DEPOLARIZE1" for inst in noisy)
    assert any(
        inst.name == "X_ERROR"
        and inst.gate_args_copy()[0] == pytest.approx(cfg.custom_params["p_idle_x"])
        for inst in noisy
    )
    assert any(
        inst.name == "Z_ERROR"
        and inst.gate_args_copy()[0] == pytest.approx(cfg.custom_params["p_idle_z"])
        for inst in noisy
    )
