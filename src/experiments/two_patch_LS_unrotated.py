# src/experiments/two_patch_LS_unrotated.py

import stim
import numpy as np
from typing import Type, Literal, Optional, Any, Dict, Tuple

from ..ir.builder import CircuitBuilder
from ..ir.tracker import SyndromeTracker
from ..ir.qec_system import QECSystem
from ..noise.config import NoiseConfig
from ..qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler
)


class TwoPatchLSExperiment:
    """
    Orchestrates a Two-Patch Lattice Surgery Experiment.
    
    This class acts as the 'Director' for Lattice Surgery experiments:
    1. Creates and configures two Unrotated Surface Code patches.
    2. Sets up their relative positions via offset.
    3. Registers a coupler protocol between them.
    4. Uses CircuitBuilder to layout the circuit (Init -> SE Loops -> Coupler Activation -> SE Loops -> Readout).
    5. Injects Noise using the configured strategy.
    """

    def __init__(self,
                 patch1_config: Dict[str, Any],
                 patch2_config: Dict[str, Any],
                 offset: Tuple[float, float],
                 interaction_type: Literal["XX", "ZZ"] = "XX",
                 coupler_protocol: Optional[Any] = UnrotatedTwoPatchCoupler(),
                 initial_state_patch1: Literal["X", "Z"] = "X",
                 initial_state_patch2: Literal["X", "Z"] = "X",
                 measure_state_patch1: Literal["X", "Z"] = "X",
                 measure_state_patch2: Literal["X", "Z"] = "X",
                 extraction_block_class: Type = UnrotatedSurfaceCodeExtractionBlock,
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True,
                 rotate_patch1: bool = True):
        """
        Args:
            patch1_config: Configuration dict for the first patch (passed to UnrotatedSurfaceCode).
            patch2_config: Configuration dict for the second patch (passed to UnrotatedSurfaceCode).
            offset: (dx, dy) offset for patch2 relative to patch1. Both dx and dy must be non-negative.
            interaction_type: "XX" or "ZZ" for the coupler interaction.
            coupler_protocol: Coupler protocol instance. Defaults to UnrotatedTwoPatchCoupler().
            initial_state_patch1: Initial state basis for patch1 data qubits ("X" or "Z").
            initial_state_patch2: Initial state basis for patch2 data qubits ("X" or "Z").
            extraction_block_class: Class for syndrome extraction block. Defaults to UnrotatedSurfaceCodeExtractionBlock.
            rounds: Number of QEC rounds before and after coupler activation.
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('circuit_level', 'code_capacity', etc.).
            if_detector: Whether to generate detectors.
            rotate_patch1: Whether to rotate patch1 by pi and reset rotation angle (default True).
        """
        # Validate offset
        dx, dy = offset
        if interaction_type == "XX":
            if dx < 0:
                raise ValueError(f"Offset dx component for XX interaction must be non-negative. Got (dx={dx}, dy={dy})")
        elif interaction_type == "ZZ":
            if dy < 0:
                raise ValueError(f"Offset dy component for ZZ interaction must be non-negative. Got (dx={dx}, dy={dy})")
        
        self.patch1_config = patch1_config
        self.patch2_config = patch2_config
        self.offset = offset
        self.interaction_type = interaction_type
        self.coupler_protocol = coupler_protocol
        self.initial_state_patch1 = initial_state_patch1.upper()
        self.initial_state_patch2 = initial_state_patch2.upper()
        self.measure_state_patch1 = measure_state_patch1.upper()
        self.measure_state_patch2 = measure_state_patch2.upper()
        self.extraction_block_class = extraction_block_class
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.if_detector = if_detector
        self.rotate_patch1 = rotate_patch1
        
        # Internal state
        self.system: Optional[QECSystem] = None
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None
        self.patch1_name = "surface_code_1"
        self.patch2_name = "surface_code_2"
        self.coupler_name = "coupler_1_2"

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the two-patch LS experiment.
        """
        # 1. Create patches
        # ----------------------------------------------------------------------
        print("Creating patches...")
        surface_code_1 = UnrotatedSurfaceCode(**self.patch1_config)
        surface_code_2 = UnrotatedSurfaceCode(**self.patch2_config)
        
        # Rotate patch1 if requested (to move logical operators closer)
        if self.rotate_patch1:
            print("Rotating patch1 by pi...")
            surface_code_1.rotate_coords(np.pi)
            surface_code_1.reset_rotation_angle()
        
        # 2. Create system and add patches
        # ----------------------------------------------------------------------
        print("Setting up QEC system...")
        self.system = QECSystem()
        self.system.add_patch(surface_code_1, name=self.patch1_name)
        self.system.add_patch(surface_code_2, name=self.patch2_name, offset=self.offset)
        
        # 3. Register coupler
        # ----------------------------------------------------------------------
        print(f"Registering coupler with interaction type {self.interaction_type}...")
        code_patches = [self.patch1_name, self.patch2_name]
        self.system.register_coupler(
            self.coupler_protocol,
            patch_names=code_patches,
            name=self.coupler_name,
            interaction_type=self.interaction_type
        )
        
        # 4. Setup tracker and builder
        # ----------------------------------------------------------------------
        num_qubits = self.system.num_qubits
        num_logicals = self.system.num_logicals
        
        self.tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=num_logicals)
        self.builder = CircuitBuilder(
            tracker=self.tracker,
            system_config=self.system,
            if_detector=self.if_detector
        )
        
        # 5. Write coordinates
        # ----------------------------------------------------------------------
        print("Writing coordinates...")
        self.builder.write_coordinates()
        
        # 6. Initialize code patch data qubits
        # ----------------------------------------------------------------------
        print("Initializing code patch data qubits...")
        
        # Only initialize qubits from code patches, not from coupler
        init_dict = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch1_name:
                init_dict[q] = self.initial_state_patch1
            elif owner == self.patch2_name:
                init_dict[q] = self.initial_state_patch2
            # Skip coupler qubits - they will be initialized later
        self.builder.initialize(init_dict=init_dict, n=num_qubits)
        
        # 7. Syndrome extraction rounds (before coupler activation)
        # ----------------------------------------------------------------------
        print(f"Building {self.rounds} rounds of syndrome extraction (before coupler activation)...")
        se_block = self.extraction_block_class(self.system)
        
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 8. Activate coupler
        # ----------------------------------------------------------------------
        print("Activating the coupler...")
        self.builder.activate_coupler(self.coupler_name)
        
        # 9. Initialize coupler data qubits
        # ----------------------------------------------------------------------
        print("Initializing coupler data qubits...")
        coupler_data_indices_local = self.system.coupler_patches[self.coupler_name].data_indices
        coupler_data_indices = [
            self.system.local_to_global_map[self.coupler_name][q]
            for q in coupler_data_indices_local
        ]
        # Coupler data qubits are initialized in Z basis for lattice surgery
        if self.interaction_type == "XX":
            coupler_init_dict = {q: "Z" for q in coupler_data_indices}
        elif self.interaction_type == "ZZ":
            coupler_init_dict = {q: "X" for q in coupler_data_indices}
        self.builder.initialize(init_dict=coupler_init_dict, n=num_qubits)
        
        # 10. Syndrome extraction rounds (after coupler activation)
        # ----------------------------------------------------------------------
        print(f"Building {self.rounds} rounds of syndrome extraction (after coupler activation)...")
        se_block = self.extraction_block_class(self.system)
        
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 11. Final readout
        # ----------------------------------------------------------------------
        print("Measuring data qubits...")
        # Measure code patch qubits in their initial basis, coupler qubits in Z basis
        measure_dict = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch1_name:
                measure_dict[q] = self.measure_state_patch1
            elif owner == self.patch2_name:
                measure_dict[q] = self.measure_state_patch2
        measure_dict.update(coupler_init_dict)
        
        self.builder.apply_data_readout(final_measurements=measure_dict)
        
        # 12. Noise injection
        # ----------------------------------------------------------------------
        if self.noise_params is not None:
            print("Injecting noise...")
            noisy_circuit = self.builder.build_noisy_circuit(
                noise_params=self.noise_params,
                noise_model=self.noise_model
            )
            return noisy_circuit
        else:
            return self.builder.circuit
