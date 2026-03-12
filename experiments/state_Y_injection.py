# experiments/state_injection.py

"""
State Injection Experiment for Rotated Surface Code.

Implements corner and middle injection protocols to prepare logical |0> or |+>
into a rotated surface code patch. The circuit construction follows the protocol
where data qubits are split by a diagonal pattern for initialization, with the
injection site (corner or center) receiving the target state.
"""

import stim
from typing import Type, Literal, Optional, Any

from src.ir.builder import CircuitBuilder
from src.ir.tracker import SyndromeTracker
from src.ir.qec_system import QECSystem
from src.noise.config import NoiseConfig
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)


class StateInjectionExperiment:
    """
    Orchestrates a State Injection Experiment for Rotated Surface Code.

    Prepares logical |0> or |+> via corner or middle injection protocol:
    1. Diagonal-split initialization of data qubits (|0> and |+> regions).
    2. Injection site receives the target state.
    3. Syndrome extraction rounds.
    4. Data qubit measurement (X or Z basis depending on inject_state).

    Circuit only - no noise, detectors, or observables are required for basic validation.
    """

    def __init__(
        self,
        distance: int = 3,
        rounds: int = 2,
        injection_protocol: Literal["corner", "middle"] = "corner",
        inject_state: Literal["Z", "X", "Y"] = "Z",
        extraction_block_class: Type = RotatedSurfaceCodeExtractionBlock,
        noise_params: Optional[NoiseConfig] = None,
        noise_model: Optional[str] = "circuit_level",
        if_detector: bool = True,
    ):
        """
        Args:
            distance: Code distance (odd integer).
            rounds: Number of syndrome extraction rounds.
            injection_protocol: 'corner' (inject at (1,1)) or 'middle' (inject at center).
            inject_state: Target logical state ('Z' -> |0>, 'X' -> |+>, 'Y' -> |+i>).
            extraction_block_class: SE block class (default RotatedSurfaceCodeExtractionBlock).
            noise_params: Optional noise configuration.
            noise_model: Noise model string.
            if_detector: Whether to generate detectors.
        """
        self.distance = distance
        self.rounds = rounds
        self.injection_protocol = injection_protocol.lower()
        self.inject_state = inject_state.upper()
        self.block_class = extraction_block_class
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.if_detector = if_detector

        if self.injection_protocol not in ("corner", "middle"):
            raise ValueError("injection_protocol must be 'corner' or 'middle'")
        if self.inject_state not in ("Z", "X", "Y"):
            raise ValueError("inject_state must be 'Z', 'X', or 'Y'")

        self.system: Optional[Any] = None
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None

    def build(self) -> stim.Circuit:
        """Constructs the full Stim circuit for the state injection experiment."""
        # 1. Create patch and register in QECSystem
        patch = RotatedSurfaceCode(distance=self.distance)
        self.system = QECSystem()
        self.system.add_patch(patch, name="surface_code")

        num_qubits = self.system.num_qubits
        num_logicals = self.system.num_logicals

        # 2. Setup tracker and builder
        self.tracker = SyndromeTracker(
            num_qubits=num_qubits,
            expected_num_logicals=num_logicals,
        )
        self.builder = CircuitBuilder(
            tracker=self.tracker,
            system_config=self.system,
            if_detector=self.if_detector,
        )

        op_set = RotatedSurfaceCodeLogicalOpSet()

        # 3. Write coordinates
        self.builder.write_coordinates()

        # 4. Initialize data qubits and tag syndrome detectors for post-selection
        op_set.state_injection(
            self.builder, patch,
            inject_state=self.inject_state,
            protocol=self.injection_protocol,
        )

        # 5. Syndrome extraction
        se_block = self.block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds,
        )

        # 6. Final readout: measure all data qubits in injection basis
        op_set.logical_unencode(self.builder, patch, inject_state=self.inject_state)

        # 7. Optional noise
        if self.noise_params is not None:
            return self.builder.build_noisy_circuit(
                noise_params=self.noise_params,
                noise_model=self.noise_model,
            )
        return self.builder.circuit
