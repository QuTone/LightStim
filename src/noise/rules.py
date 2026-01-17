import stim
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set
from .config import NoiseConfig

class NoiseRule(ABC):
    """
    Abstract base class for noise rules.
    """
    @abstractmethod
    def apply(self, instruction: stim.CircuitInstruction, config: NoiseConfig, active_qubits: Set[int]) -> Tuple[List[stim.CircuitInstruction], List[stim.CircuitInstruction]]:
        """
        Input: A single instruction.
        Output: A tuple (pre_noise, post_noise).
        
        - pre_noise: Instructions to insert BEFORE the target.
        - post_noise: Instructions to insert AFTER the target.
        """
        pass

# --- Implementation ---

# Gate Error Rules
class DepolarizeAfterGate(NoiseRule):
    """
    Applies a specific depolarizing channel (1-qubit or 2-qubit) AFTER specific gates
    with a explicitly specified error rate.
    """
    def __init__(self, target_gates: List[str], param_name: str, noise_op: str):
        """
        Args:
            target_gates: List of gate names (e.g., ["CX", "CZ"] or ["H", "X"]).
            param_name: The key in NoiseConfig to look up (e.g., "p_2q").
            noise_op: The Stim noise instruction name (e.g., "DEPOLARIZE2").
        """
        self.target_gates = set(target_gates)
        self.param_name = param_name
        self.noise_op = noise_op

    def apply(self, instruction: stim.CircuitInstruction, config: NoiseConfig, active_qubits: Set[int]) -> Tuple[List[stim.CircuitInstruction], List[stim.CircuitInstruction]]:
        # Check if the current instruction is one of the targets
        if instruction.name in self.target_gates:
            # Retrieve the specific error rate for this group from the config
            p = config.get(self.param_name)
            
            if p > 0:
                # Construct the noise instruction
                # We apply the noise to the same targets as the gate
                noise = stim.CircuitInstruction(self.noise_op, instruction.targets_copy(), [p])
                return [], [noise]  # Return as post-noise
        
        return [], []

class GeneralPauliAfterGate(NoiseRule):
    """
    Applies a general Pauli channel (PAULI_CHANNEL_1 or PAULI_CHANNEL_2) AFTER a gate
    with an explicitly specified SET of probabilities (unlike DepolarizeAfterGate, which uses a single error rate).
    """

    # Stim expects 2-qubit Pauli probabilities in this exact order (IX, IY, IZ, XI...)
    ORDER_2Q = [
        "IX", "IY", "IZ", 
        "XI", "XX", "XY", "XZ", 
        "YI", "YX", "YY", "YZ", 
        "ZI", "ZX", "ZY", "ZZ"
    ]
    
    # Stim expects 1-qubit Pauli probabilities in this order (X, Y, Z)
    ORDER_1Q = ["X", "Y", "Z"]

    def __init__(self, target_gates: List[str], params_map: Dict[str, str]):
        """
        Args:
            target_gates: List of gate names to apply noise to (e.g., ["CX"]).
            params_map: A dictionary mapping Pauli strings to NoiseConfig parameter names.
                        
            Example for 1Q (X error only):
            { "X": "p_bitflip" }
                        
            Example for 2Q (Correlated ZZ error):
            { "ZZ": "p_zz_error", "IX": "p_residual" }
        """
        self.target_gates = set(target_gates)
        self.params_map = params_map

    def apply(self, instruction: stim.CircuitInstruction, config: NoiseConfig, active_qubits: Set[int]) -> Tuple[List[stim.CircuitInstruction], List[stim.CircuitInstruction]]:
        if instruction.name not in self.target_gates:
            return [], []

        # Determine if we need 1-qubit or 2-qubit channel logic based on the user's mapping keys
        # (Heuristic: check length of the first key in the map)
        first_key = next(iter(self.params_map))
        is_2q_channel = len(first_key) == 2

        if is_2q_channel:
            # --- Case 2Q: Build the 15-float list ---
            op_name = "PAULI_CHANNEL_2"
            args = []
            for pauli_str in self.ORDER_2Q:
                # 1. Get param name from map (e.g., "ZZ" -> "p_zz")
                param_name = self.params_map.get(pauli_str)
                
                # 2. Retrieve value from config, default to 0.0 if not set in map or config
                prob = 0.0
                if param_name:
                    prob = config.get(param_name, 0.0)
                args.append(prob)
        else:
            # --- Case 1Q: Build the 3-float list ---
            op_name = "PAULI_CHANNEL_1"
            args = []
            for pauli_str in self.ORDER_1Q:
                param_name = self.params_map.get(pauli_str)
                prob = 0.0
                if param_name:
                    prob = config.get(param_name, 0.0)
                args.append(prob)

        # Optimization: If all probabilities are 0, don't insert the instruction
        if all(p == 0.0 for p in args):
            return [], []

        # Construct and return the noise instruction
        noise = stim.CircuitInstruction(op_name, instruction.targets_copy(), args)
        return [], [noise]

# Measurement Error Rules

class FlipBeforeMeasurement(NoiseRule):
    """
    Applies the appropriate Pauli flip noise BEFORE a measurement to simulate readout error.
    Handles standard measurements (M) and composite Measure-Reset (MR).
    
    Logic:
    - Z-basis (M, MZ, MR, MRZ) -> X_ERROR
    - X-basis (MX, MRX)        -> Z_ERROR
    - Y-basis (MY, MRY)        -> Z_ERROR
    """
    def __init__(self, param_name: str = 'p_meas'):
        self.param_name = param_name
        # Define sets once for efficiency
        self.z_meas = {"M", "MZ", "MR", "MRZ"}
        self.x_meas = {"MX", "MRX"}
        self.y_meas = {"MY", "MRY"}

    def apply(self, instruction: stim.CircuitInstruction, config: NoiseConfig, active_qubits: Set[int]) -> Tuple[List[stim.CircuitInstruction], List[stim.CircuitInstruction]]:
        noise_op = None
        
        if instruction.name in self.z_meas:
            noise_op = "X_ERROR"
        elif instruction.name in self.x_meas:
            noise_op = "Z_ERROR"
        elif instruction.name in self.y_meas:
            noise_op = "Z_ERROR"
        
        if noise_op:
            p = config.get(self.param_name)
            if p > 0:
                # Pre-noise: Flip the state BEFORE measurement
                noise = stim.CircuitInstruction(noise_op, instruction.targets_copy(), [p])
                return [noise], []
                
        return [], []


class FlipAfterReset(NoiseRule):
    """
    Applies the appropriate Pauli flip error AFTER a reset to simulate state preparation error.
    Handles standard resets (R) and composite Measure-Reset (MR).
    
    Logic:
    - Z-basis (R, RZ, MR, MRZ) -> X_ERROR (flips |0> to |1> and vice versa)
    - X-basis (RX, MRX)        -> Z_ERROR (flips |+> to |-> and vice versa)
    - Y-basis (RY, MRY)        -> Z_ERROR (flips |i+> to |i-> and vice versa)

    Note: Including FlipBeforeMeasurement and FlipAfterReset rules is sufficient to cover the noise
    before and after a measurement reset (MRX, MRY, MRZ).
    """

    def __init__(self, param_name: str = 'p_reset'):
        self.param_name = param_name
        # Define sets once for efficiency
        self.z_reset = {"R", "RZ", "MR", "MRZ"}
        self.x_reset = {"RX", "MRX"}
        self.y_reset = {"RY", "MRY"}

    def apply(self, instruction: stim.CircuitInstruction, config: NoiseConfig, active_qubits: Set[int]) -> Tuple[List[stim.CircuitInstruction], List[stim.CircuitInstruction]]:
        noise_op = None
        
        if instruction.name in self.z_reset:
            noise_op = "X_ERROR"
        elif instruction.name in self.x_reset:
            noise_op = "Z_ERROR"
        elif instruction.name in self.y_reset:
            noise_op = "Z_ERROR"

        if noise_op:
            p = config.get(self.param_name)
            if p > 0:
                # Post-noise: Flip the state AFTER reset (dirty state prep)
                noise = stim.CircuitInstruction(noise_op, instruction.targets_copy(), [p])
                return [], [noise]

        return [], []

# Insert Idling Error at specific moments on specific qubits, controlled by tags

class TaggedIdling(NoiseRule):
    """
    Applies noise to specific qubits when a TICK instruction with a specific tag is encountered.
    
    Useful for:
    - Code Capacity Noise (apply error before syndrome extraction).
    - Idle Noise (apply error only at specific idle moments).

    This rule can be very versatile when combining tags and target_qubits.
    """
    def __init__(self, target_qubits: List[int], param_name: str, tag: str, noise_op: str = "DEPOLARIZE1"):
        """
        Args:
            target_qubits: List of qubit indices to apply the error to (e.g., data qubits).
            param_name: The parameter name in NoiseConfig (e.g., "p_idle").
            tag: The tag string to look for in the TICK instruction, e.g. "SE_start".
            noise_op: The noise operation to apply (default: DEPOLARIZE1).
        """
        self.target_qubits = set[int](target_qubits)
        self.param_name = param_name
        self.tag = tag
        self.noise_op = noise_op

    def apply(self, instruction: stim.CircuitInstruction, config: NoiseConfig, active_qubits: Set[int]) -> Tuple[List[stim.CircuitInstruction], List[stim.CircuitInstruction]]:
        # Check if it is a TICK and if it has the matching tag
        if instruction.name == "TICK" and instruction.tag == self.tag:
            p = config.get(self.param_name)

            if p > 0:
                # Create noise instruction on the specified target qubits that are currently active
                real_targets = sorted(list(self.target_qubits.intersection(active_qubits)))
                
                if real_targets:
                    noise = stim.CircuitInstruction(self.noise_op, real_targets, [p])
                    return [], [noise]
                
        return [], []