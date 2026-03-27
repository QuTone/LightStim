import stim
from typing import List, Dict, Set
from .config import NoiseConfig
# Import your defined rules
from .rules import (
    NoiseRule, 
    DepolarizeAfterGate, 
    GeneralPauliAfterGate,
    FlipBeforeMeasurement, 
    FlipAfterReset, 
    TaggedIdling 
)

class NoiseInjector:
    """
    Injects noise into a circuit based on a NoiseModel using a set of NoiseRules.
    """
    def __init__(self, model: NoiseConfig):
        self.model = model
        self.rules: List[NoiseRule] = []

    def add_rule(self, rule: NoiseRule):
        self.rules.append(rule)

    def inject_noise(self, circuit: stim.Circuit, active_qubits: Set[int] = None) -> stim.Circuit:
        # Initialize active_qubits if not provided
        if active_qubits is None:
            active_qubits = set[int]()
        
        # Insert noise
        noisy_circuit = stim.Circuit()
        for item in circuit:
            # Handle repeat blocks recursively
            if isinstance(item, stim.CircuitRepeatBlock):
                noisy_body = self.inject_noise(item.body_copy(), active_qubits)
                noisy_circuit.append(stim.CircuitRepeatBlock(item.repeat_count, noisy_body))
            elif isinstance(item, stim.CircuitInstruction):
                # --- Tracking active qubits ---
                if item.name in {"R", "RX", "RY", "RZ"} :
                    # Add newly initialized qubits to active_qubits (use index not stim.target_rec)
                    for t in item.targets_copy():
                        if t.is_qubit_target:
                            active_qubits.add(t.value)
                
                # Apply rules to the instruction
                pre_accum, post_accum = [], []
                for rule in self.rules:
                    pre, post = rule.apply(item, self.model, active_qubits)
                    pre_accum.extend(pre)
                    post_accum.extend(post)
                for inst in pre_accum: noisy_circuit.append(inst)
                noisy_circuit.append(item)
                for inst in post_accum: noisy_circuit.append(inst)

                # Concern about the order of rules applying to the same instruction, generating different noise sequences.
                # Since we only consider Pauli error, changing the order of the noise sequence only potentially brings a -1 global phase,
                # which is not physically observable.
        return noisy_circuit

    # =========================================================================
    # Factory: Compose Rules into for Noise Injector for Standard Error Models
    # =========================================================================

    @classmethod
    def from_code_capacity(cls, config: NoiseConfig, data_qubit_indices: List[int]) -> 'NoiseInjector':
        """
        Model 1: Code Capacity
        - Noise: Only Idling noise on data qubits before Syndrome Extraction (SE).
        - Rule: TaggedIdling
        """
        injector = cls(config)
        
        # 1. Data qubit noise: Apply p_idle (depolarizing) to data qubits when TICK["SE_start"] is encountered
        injector.add_rule(TaggedIdling(
            target_qubits=data_qubit_indices,
            param_name="p_idle",
            tag="SE_start"
        ))
        
        return injector

    @classmethod
    def from_phenomenological(cls, config: NoiseConfig, data_qubit_indices: List[int]) -> 'NoiseInjector':
        """
        Model 2: Phenomenological
        - Noise: Code Capacity (Idling) + Measurement Error.
        - Rules: TaggedIdleRule, FlipBeforeMeasurement
        """
        injector = cls(config)
        
        # 1. Data qubit noise
        injector.add_rule(TaggedIdling(
            target_qubits=data_qubit_indices,
            param_name="p_idle",
            tag="SE_start"
        ))
        
        # 2. Measurement error (Readout flip)
        injector.add_rule(FlipBeforeMeasurement(param_name="p_meas"))
        
        return injector

    @classmethod
    def from_circuit_level(cls, config: NoiseConfig, data_qubit_indices: List[int]) -> 'NoiseInjector':
        """
        Model 3: Standard Circuit-level (Depolarizing)
        - Noise: Gates (1Q/2Q), Measurement, Reset, and Idling (depolarizing before SE on data qubits).
        - Rules: All standard depolarizing/flip rules.
        """
        injector = cls(config)
        
        # 1. Idling (optional context-aware noise before SE)
        injector.add_rule(TaggedIdling(
            target_qubits=data_qubit_indices,
            param_name="p_idle",
            tag="SE_start"
        ))
        
        # 2. Measurement & Reset
        injector.add_rule(FlipBeforeMeasurement(param_name="p_meas"))
        injector.add_rule(FlipAfterReset(param_name="p_reset"))
        
        # 3. 1-Qubit Gates
        injector.add_rule(DepolarizeAfterGate(
            target_gates=["H", "X", "Y", "Z", "S", "S_DAG"],
            param_name="p_1q",
            noise_op="DEPOLARIZE1"
        ))
        
        # 4. 2-Qubit Gates
        injector.add_rule(DepolarizeAfterGate(
            target_gates=["CX", "CY", "CZ", "SWAP", "CNOT"],
            param_name="p_2q",
            noise_op="DEPOLARIZE2"
        ))
        
        return injector

    @classmethod
    def from_XZ_biased(cls, config: NoiseConfig, data_qubit_indices: List[int]) -> 'NoiseInjector':
        """
        Model 4: Bit/Phase-flip (Biased Circuit-level)
        - Noise: Similar to circuit-level, but uses GeneralPauliAfterGate for gates.
        - Assumption: User provides 'p_1q_x', 'p_1q_z', etc. in model.custom_params.
        """
        injector = cls(config)

        # 1. Idling (still typically depolarizing, or you can write a PauliIdleRule)
        injector.add_rule(TaggedIdling(
            target_qubits=data_qubit_indices,
            param_name="p_idle",
            tag="SE_start"
        ))

        # 2. Measurement & Reset (Using standard flip rules)
        injector.add_rule(FlipBeforeMeasurement(param_name="p_meas"))
        injector.add_rule(FlipAfterReset(param_name="p_reset"))

        # 3. Biased 1-Qubit Gate Noise
        # Separate X_ERROR + Z_ERROR for consistency with 2Q gate treatment.
        injector.add_rule(DepolarizeAfterGate(
            target_gates=["H", "R", "RX", "RY", "RZ", "X", "Y", "Z", "S", "S_DAG"],
            param_name="p_1q_x",
            noise_op="X_ERROR"
        ))
        injector.add_rule(DepolarizeAfterGate(
            target_gates=["H", "R", "RX", "RY", "RZ", "X", "Y", "Z", "S", "S_DAG"],
            param_name="p_1q_z",
            noise_op="Z_ERROR"
        ))

        # 4. Biased 2-Qubit Gate Noise
        # Independent X_ERROR + Z_ERROR on each qubit after 2Q gates.
        # Using separate X_ERROR/Z_ERROR instead of PAULI_CHANNEL_1/2 avoids
        # Stim's DEM composite-error symptom limit.
        injector.add_rule(DepolarizeAfterGate(
            target_gates=["CX", "CZ", "SWAP"],
            param_name="p_2q_x",
            noise_op="X_ERROR"
        ))
        injector.add_rule(DepolarizeAfterGate(
            target_gates=["CX", "CZ", "SWAP"],
            param_name="p_2q_z",
            noise_op="Z_ERROR"
        ))

        return injector

    @staticmethod
    def compute_XZ_biased_params(p_1q: float, p_2q: float, p_meas: float, p_reset: float, eta: float) -> NoiseConfig:
        """
        Compute NoiseConfig for XZ-biased noise model from physical error rates and bias ratio.

        Treats X and Z errors as independent channels on each qubit.
        - 1-qubit gates: p_x = eta * p_z, p_x + p_z = p_1q.
        - 2-qubit gates: each qubit independently gets X/Z errors with the same
          bias ratio. Per-qubit rates satisfy: 2*(p_x + p_z) ≈ p_2q (small-p approx).

        Args:
            p_1q:  Total 1-qubit gate error rate.
            p_2q:  Total 2-qubit gate error rate.
            p_meas: Measurement error rate (symmetric flip).
            p_reset: Reset error rate (symmetric flip).
            eta:   Bias ratio p_X / p_Z (>1 means X-biased, <1 means Z-biased).

        Returns:
            NoiseConfig with custom_params filled for from_XZ_biased().
        """
        # 1-qubit gates: p_x + p_z = p_1q, p_x = eta * p_z
        p_1q_z = p_1q / (1 + eta)
        p_1q_x = eta * p_1q_z

        # 2-qubit gates: independent per-qubit errors.
        # Two qubits each get (p_x, p_z), so total 2Q error rate ≈ 2*(p_x + p_z).
        # Solve: 2*(1 + eta)*p_z_per_qubit = p_2q
        p_2q_z = p_2q / (2 * (1 + eta))
        p_2q_x = eta * p_2q_z

        return NoiseConfig(
            p_meas=p_meas,
            p_reset=p_reset,
            p_idle=0.0,
            custom_params={
                "p_1q_x": p_1q_x,
                "p_1q_z": p_1q_z,
                "p_2q_x": p_2q_x,
                "p_2q_z": p_2q_z,
            },
        )
    
    # =========================================================================
    # Compose Rules into for Noise Injector for Custom Error Models
    # =========================================================================