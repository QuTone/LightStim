from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict


NOISE_MODELS = (
    "code_capacity", "phenomenological", "circuit_level",
    "circuit_level_with_idling", "XZ_biased",
)
NOISE_RATES = ("p_1q", "p_2q", "p_meas", "p_reset", "p_idle")


@dataclass
class NoiseConfig:
    """
    Data class holding noise parameters.
    Standardized naming convention for broad applicability.
    """
    # --- Standard Parameters ---
    p_1q: float = 0.0        # Depolarizing noise after 1-qubit gates (H, S, etc.)
    p_2q: float = 0.0        # Depolarizing noise after 2-qubit gates (CX, CZ, etc.)
    p_meas: float = 0.0      # Probability of measurement outcome flip (Readout error)
    p_reset: float = 0.0     # Probability of state prep flip (Reset error)
    p_idle: float = 0.0      # Idle depolarization; placement depends on the selected noise model
    
    # --- Custom / Biased Parameters ---
    # Store specialized rates here (e.g., 'p_1q_x', 'p_1q_z', 'p_2q_zz')
    custom_params: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, noise: Mapping[str, Any]) -> "NoiseConfig":
        """Build all five rates from a resolved experiment noise section."""
        from lightstim.config.loader import _validate

        _validate(dict(noise), "noise")
        base = noise["p"]
        if isinstance(base, dict):
            raise ValueError("Resolve the parameter reference in noise.p before constructing NoiseConfig")
        overrides = noise.get("overrides", {})
        rates = {
            key: base if overrides.get(key) is None else overrides[key]
            for key in NOISE_RATES
        }
        return cls(**rates, custom_params=dict(noise.get("custom_params", {})))

    def get(self, param_name: str, default: float = 0.0) -> float:
        """Safe accessor for parameters."""
        if hasattr(self, param_name): # standard parameters
            val = getattr(self, param_name)
            return val if val is not None else default
        return self.custom_params.get(param_name, default) # custom parameters
