"""Background and imperfect-state preparation channels shared by the 7-to-1 protocols."""

from dataclasses import replace

from lightstim.noise.config import NOISE_MODELS, NoiseConfig
from lightstim.noise.injector import NoiseInjector


def inject_distillation_noise(circuit, reset_rule, magic_qubits, p, p_injected, mode, *,
                              noise_model, background_noise, background_data_indices):
    """Apply a reset rule on the preparation qubits, a background model, or both."""
    if mode == "injection":
        injector = NoiseInjector(NoiseConfig(p_reset=p_injected))
        injector.add_rule(reset_rule(magic_qubits, param_name="p_reset"))
        return injector.inject_noise(circuit)
    if mode not in ("full", "both"):
        raise ValueError(f"Unknown distillation noise mode {mode!r}")
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"Unknown background noise model {noise_model!r}")

    # The circuit-level distillation baselines leave idle noise disabled.
    config = background_noise or NoiseConfig(
        p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=0 if noise_model == "circuit_level" else p,
    )
    if mode == "both":
        config = replace(config, custom_params={**config.custom_params, "p_injected": p_injected})
    targets = background_data_indices
    if targets is None or noise_model == "circuit_level_with_idling":
        targets = range(circuit.num_qubits)
    injector = getattr(NoiseInjector, f"from_{noise_model}")(config, list(targets))
    if mode == "both":
        injector.add_rule(reset_rule(magic_qubits, param_name="p_injected"))
    return injector.inject_noise(circuit)
