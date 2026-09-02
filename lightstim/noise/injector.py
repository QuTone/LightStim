import stim
from typing import List, Optional, Set
from .config import NoiseConfig
# Import your defined rules
from .rules import (
    NoiseRule,
    MomentNoiseRule,
    DepolarizeAfterGate,
    GeneralPauliAfterGate,
    FlipBeforeMeasurement,
    FlipAfterReset,
    FlipAfterResetFiltered,
    FlipAfterYResetFiltered,
    TaggedIdling,
    MomentIdling,
)


class _UnsupportedMomentRepeat(Exception):
    """Signals that exact moment noise requires expanding a REPEAT block."""


class NoiseInjector:
    """
    Injects noise into a circuit based on a NoiseModel using a set of NoiseRules.
    """
    def __init__(self, model: NoiseConfig):
        self.model = model
        self.rules: List[NoiseRule] = []
        self.moment_rules: List[MomentNoiseRule] = []

    def add_rule(self, rule: NoiseRule):
        self.rules.append(rule)

    def add_moment_rule(self, rule: MomentNoiseRule):
        """Add a rule that needs all operations in a TICK-delimited moment."""
        self.moment_rules.append(rule)

    def _append_moment_noise(
        self,
        noisy_circuit: stim.Circuit,
        moment: List[stim.CircuitInstruction],
        boundary: Optional[stim.CircuitInstruction] = None,
    ) -> None:
        for rule in self.moment_rules:
            for instruction in rule.apply(moment, self.model, boundary):
                noisy_circuit.append(instruction)

    @staticmethod
    def _moment_has_physical_operation(
        moment: List[stim.CircuitInstruction],
    ) -> bool:
        return any(MomentIdling._operated_qubits(item) for item in moment)

    @classmethod
    def _repeat_boundary_shape(
        cls,
        body: stim.Circuit,
    ) -> tuple[Optional[stim.CircuitInstruction], bool]:
        """Return the leading TICK and whether the body has a physical suffix."""
        leading_tick = None
        suffix: List[stim.CircuitInstruction] = []
        for index, item in enumerate(body):
            if isinstance(item, stim.CircuitRepeatBlock):
                raise _UnsupportedMomentRepeat(
                    "Moment-aware idle noise does not support nested REPEAT blocks"
                )
            if index == 0 and item.name == "TICK":
                leading_tick = item
            if item.name == "TICK":
                suffix = []
            else:
                suffix.append(item)
        return leading_tick, cls._moment_has_physical_operation(suffix)

    def inject_noise(
        self,
        circuit: stim.Circuit,
        active_qubits: Set[int] = None,
    ) -> stim.Circuit:
        """Return a noisy circuit while preserving compact safe REPEAT blocks.

        Some valid Stim repeat bodies join one physical moment across iteration
        boundaries. If that topology cannot carry exact moment noise while
        remaining compressed, the circuit is flattened before injection.
        """
        if active_qubits is None:
            active_qubits = set()

        if self.moment_rules and any(
            isinstance(item, stim.CircuitRepeatBlock) for item in circuit
        ):
            trial_active_qubits = set(active_qubits)
            try:
                result = self._inject_noise(circuit, trial_active_qubits)
            except _UnsupportedMomentRepeat:
                return self._inject_noise(circuit.flattened(), active_qubits)
            active_qubits.clear()
            active_qubits.update(trial_active_qubits)
            return result

        return self._inject_noise(circuit, active_qubits)

    def _inject_noise(
        self,
        circuit: stim.Circuit,
        active_qubits: Set[int],
        *,
        _suppress_first_tick_idle: bool = False,
    ) -> stim.Circuit:
        # Insert noise
        noisy_circuit = stim.Circuit()
        moment: List[stim.CircuitInstruction] = []
        suppress_next_empty_tick_idle = _suppress_first_tick_idle
        for item in circuit:
            # Handle repeat blocks recursively
            if isinstance(item, stim.CircuitRepeatBlock):
                if not self.moment_rules:
                    noisy_body = self._inject_noise(item.body_copy(), active_qubits)
                    noisy_circuit.append(
                        stim.CircuitRepeatBlock(item.repeat_count, noisy_body)
                    )
                    continue
                if suppress_next_empty_tick_idle:
                    raise _UnsupportedMomentRepeat(
                        "A REPEAT suffix with operations must be followed by TICK"
                    )
                leading_tick, has_physical_suffix = self._repeat_boundary_shape(
                    item.body_copy()
                )
                has_physical_prefix = self._moment_has_physical_operation(moment)
                if leading_tick is None and has_physical_suffix:
                    raise _UnsupportedMomentRepeat(
                        "A REPEAT body must begin or end at a TICK boundary"
                    )
                if leading_tick is None and has_physical_prefix:
                    raise _UnsupportedMomentRepeat(
                        "Moment-aware idle noise requires a TICK between operations "
                        "and a following REPEAT block"
                    )
                if leading_tick is not None and not has_physical_suffix:
                    if has_physical_prefix:
                        raise _UnsupportedMomentRepeat(
                            "Cannot preserve a REPEAT whose leading TICK closes "
                            "a non-empty outer moment but whose body has no "
                            "physical suffix"
                        )
                elif leading_tick is not None:
                    # The body's first TICK closes the outer moment on its first
                    # iteration. Later iterations close the physical suffix,
                    # whose idle channel is emitted at the end of the body.
                    self._append_moment_noise(
                        noisy_circuit,
                        moment,
                        leading_tick,
                    )
                moment = []
                noisy_body = self._inject_noise(
                    item.body_copy(),
                    active_qubits,
                    _suppress_first_tick_idle=(
                        leading_tick is not None and has_physical_suffix
                    ),
                )
                noisy_circuit.append(stim.CircuitRepeatBlock(item.repeat_count, noisy_body))
                suppress_next_empty_tick_idle = has_physical_suffix
            elif isinstance(item, stim.CircuitInstruction):
                # Moment-aware rules need to see the complete set of operations
                # before the boundary itself is emitted.
                if item.name == "TICK":
                    if suppress_next_empty_tick_idle:
                        if self._moment_has_physical_operation(moment):
                            raise _UnsupportedMomentRepeat(
                                "A REPEAT suffix must be followed directly by TICK"
                            )
                        suppress_next_empty_tick_idle = False
                    else:
                        self._append_moment_noise(noisy_circuit, moment, item)
                    moment = []
                else:
                    if (
                        suppress_next_empty_tick_idle
                        and self._moment_has_physical_operation([item])
                    ):
                        raise _UnsupportedMomentRepeat(
                            "A REPEAT suffix must be followed directly by TICK"
                        )
                    moment.append(item)

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

                # A destructive measurement ends the qubit's active lifetime.
                # Measure-reset gates intentionally remain active.
                if item.name in {"M", "MX", "MY", "MZ"}:
                    for target in item.targets_copy():
                        if target.is_qubit_target:
                            active_qubits.discard(target.value)

                # Concern about the order of rules applying to the same instruction, generating different noise sequences.
                # Since we only consider Pauli error, changing the order of the noise sequence only potentially brings a -1 global phase,
                # which is not physically observable.
        self._append_moment_noise(noisy_circuit, moment)
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
        
        injector._add_standard_circuit_level_rules()

        return injector

    @classmethod
    def from_circuit_level_with_idling(
        cls,
        config: NoiseConfig,
        qubit_indices: List[int],
    ) -> 'NoiseInjector':
        """Circuit-level noise with idle depolarization after every moment.

        At each TICK, ``p_idle`` is applied once to every qubit in
        ``qubit_indices`` that has no unitary, reset, or measurement operation.
        This includes empty moments. A non-empty final moment without a trailing
        TICK is also covered. Moments containing only ``noiseless``-tagged
        operations are left noiseless.

        Unlike :meth:`from_circuit_level`, this model does not add the older
        ``SE_start``-tagged data-idle channel. The five rates remain
        independently configurable. Repeat layouts whose moments cross
        iteration boundaries are flattened to preserve exact semantics.
        """
        injector = cls(config)
        # Keep p_idle=0 a strict drop-in replacement for the historical
        # circuit-level gate/SPAM rules. In particular, repeat blocks remain
        # compressed and do not need moment-boundary analysis in this case.
        if config.get("p_idle") > 0:
            injector.add_moment_rule(MomentIdling(
                target_qubits=qubit_indices,
                param_name="p_idle",
            ))
        injector._add_standard_circuit_level_rules()
        return injector

    def _add_standard_circuit_level_rules(self) -> None:
        """Add the gate, measurement, and reset rules shared by CL models."""
        # 1. Measurement & Reset
        self.add_rule(FlipBeforeMeasurement(param_name="p_meas"))
        self.add_rule(FlipAfterReset(param_name="p_reset"))
        
        # 2. 1-Qubit Gates
        self.add_rule(DepolarizeAfterGate(
            target_gates=["H", "X", "Y", "Z", "S", "S_DAG"],
            param_name="p_1q",
            noise_op="DEPOLARIZE1"
        ))
        
        # 3. 2-Qubit Gates
        self.add_rule(DepolarizeAfterGate(
            target_gates=["CX", "CY", "CZ", "SWAP", "CNOT"],
            param_name="p_2q",
            noise_op="DEPOLARIZE2"
        ))

    @classmethod
    def from_XZ_biased(cls, config: NoiseConfig, data_qubit_indices: List[int]) -> 'NoiseInjector':
        """
        Model 4: Bit/Phase-flip (Biased Circuit-level)
        - Noise: Similar to circuit-level, but uses GeneralPauliAfterGate for gates.
        - Assumption: User provides 'p_1q_x', 'p_1q_z', etc. in model.custom_params.
        """
        injector = cls(config)

        # 1. Idling. Prefer biased X/Z idle channels when provided; otherwise
        # preserve the older depolarizing p_idle behavior.
        if config.get("p_idle_x") > 0 or config.get("p_idle_z") > 0:
            injector.add_rule(TaggedIdling(
                target_qubits=data_qubit_indices,
                param_name="p_idle_x",
                tag="SE_start",
                noise_op="X_ERROR",
            ))
            injector.add_rule(TaggedIdling(
                target_qubits=data_qubit_indices,
                param_name="p_idle_z",
                tag="SE_start",
                noise_op="Z_ERROR",
            ))
        else:
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
    def compute_XZ_biased_params(
        p_1q: float,
        p_2q: float,
        p_meas: float,
        p_reset: float,
        eta: float,
        p_idle: float = 0.0,
    ) -> NoiseConfig:
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
            p_idle: Optional idle error budget split into biased X/Z channels.

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

        p_idle_z = p_idle / (1 + eta)
        p_idle_x = eta * p_idle_z

        return NoiseConfig(
            p_meas=p_meas,
            p_reset=p_reset,
            p_idle=0.0,
            custom_params={
                "p_1q_x": p_1q_x,
                "p_1q_z": p_1q_z,
                "p_2q_x": p_2q_x,
                "p_2q_z": p_2q_z,
                "p_idle_x": p_idle_x,
                "p_idle_z": p_idle_z,
            },
        )
    
    # =========================================================================
    # Compose Rules into for Noise Injector for Custom Error Models
    # =========================================================================
