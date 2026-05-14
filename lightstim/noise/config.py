from typing import Optional, Dict, Any
from dataclasses import dataclass, field

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
    p_idle: float = 0.0      # Depolarizing noise applied to idle qubits (e.g., on data qubits before SE)
    
    # --- Custom / Biased Parameters ---
    # Store specialized rates here (e.g., 'p_1q_x', 'p_1q_z', 'p_2q_zz')
    custom_params: Dict[str, float] = field(default_factory=dict)

    def get(self, param_name: str, default: float = 0.0) -> float:
        """Safe accessor for parameters."""
        if hasattr(self, param_name): # standard parameters
            val = getattr(self, param_name)
            return val if val is not None else default
        return self.custom_params.get(param_name, default) # custom parameters