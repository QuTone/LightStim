# experiments/CNOT_LS.py

import stim
import numpy as np
from typing import Type, Literal, Optional, Any, Dict, Tuple

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler
)


class CNOTLSExperiment:
    """
    Implement a CNOT gate using Lattice Surgery with 3 Surface Code patches.
    
    This implements CNOT using Lattice Surgery protocol:
    1. Three patches: Control (C), Target (T), and Ancilla (A)
    Layout: A -- T (XX)
            |
            C (ZZ)
    2. Protocol: (1) Prepara A in |+>, ZZ on C-A, XX on T-A, Z on A; 
                 (2) Prepare A in |0>, XX on T-A, ZZ on C-A, X on A; 
    """

    def __init__(self,
                 patch_configs: Dict[str, Dict[str, Any]],
                 offset_ta: Tuple[float, float],  # Offset for Target relative to Ancilla
                 offset_ca: Tuple[float, float],   # Offset for Control relative to Ancilla
                 initial_state_dict: Dict[str, Literal["X", "Z"]] = {"a": "X", "c": "X", "t": "X"},
                 measure_state_dict: Dict[str, Literal["X", "Z"]] = {"a": "Z", "c": "X", "t": "X"},
                 extraction_block_class: Type = UnrotatedSurfaceCodeExtractionBlock,
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True,
                 rotate_patches: bool = True):
        """
        Args:
            patch_config: Configuration dict for all Ancilla, Control, and Target patches (passed to UnrotatedSurfaceCode).
            offset_ct: (dx, dy) offset for Target patch relative to Control patch.
            offset_ta: (dx, dy) offset for Ancilla patch relative to Target patch.
            initial_state_c: Initial state basis for Control patch data qubits ("X" or "Z").
            initial_state_t: Initial state basis for Target patch data qubits ("X" or "Z").
            initial_state_a: Initial state basis for Ancilla patch data qubits ("X" or "Z").
            extraction_block_class: Class for syndrome extraction block. Defaults to UnrotatedSurfaceCodeExtractionBlock.
            rounds: Number of QEC rounds before and after coupler activations.
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('circuit_level', 'code_capacity', etc.).
            if_detector: Whether to generate detectors.
            rotate_patches: Whether to rotate ancilla patches by pi and reset rotation angle (default True).
        """
        # Sanity check
        # Offsets
        dx_ta, dy_ta = offset_ta
        dx_ca, dy_ca = offset_ca
        
        # if dx_ta < 0:
        #     raise ValueError(f"Offset dx_ta for Target-Ancilla must be non-negative for XX interaction. Got (dx={dx_ta}, dy={dy_ta})")
        # if dy_ca < 0:
        #     raise ValueError(f"Offset dy_ca for Control-Ancilla must be non-negative for ZZ interaction. Got (dx={dx_ca}, dy={dy_ca})")
        
        self.patch_configs = patch_configs
        self.offset_ta = offset_ta
        self.offset_ca = offset_ca
        self.initial_state_dict = initial_state_dict
        self.measure_state_dict = measure_state_dict
        self.extraction_block_class = extraction_block_class
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.if_detector = if_detector
        self.rotate_patches = rotate_patches
        
        # Internal state
        self.system: Optional[QECSystem] = None
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None
        self.patch_c_name = "control"
        self.patch_t_name = "target"
        self.patch_a_name = "ancilla"
        self.coupler_ca_name = "coupler_ca"  # Control-Ancilla coupler (ZZ)
        self.coupler_ta_name = "coupler_ta"  # Target-Ancilla coupler (XX)

        if self.initial_state_dict["a"] == "X":
            self.interaction_types = ["ZZ", "XX"]
            if self.measure_state_dict["a"] != "Z":
                print(f"Measure state of ancilla patch must be Z when initial state is X. Got (measure_state_a={measure_state_dict['a']}). Changed to Z.")
                self.measure_state_dict["a"] = "Z"
        elif self.initial_state_dict["a"] == "Z":
            self.interaction_types = ["XX", "ZZ"]
            if self.measure_state_dict["a"] != "X":
                print(f"Measure state of ancilla patch must be X when initial state is Z. Got (measure_state_a={measure_state_dict['a']}). Changed to X.")
                self.measure_state_dict["a"] = "X"
        else:
            raise ValueError(f"Initial state of ancilla patch must be X or Z. Got (initial_state_a={initial_state_dict['a']})")

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the CNOT Lattice Surgery experiment.
        """
        # 1. Create patches
        # ----------------------------------------------------------------------
        print("Creating patches...")
        patch_c = UnrotatedSurfaceCode(**self.patch_configs["c"])
        patch_t = UnrotatedSurfaceCode(**self.patch_configs["t"])
        patch_a = UnrotatedSurfaceCode(**self.patch_configs["a"])
        
        # Rotate Ancilla patch if requested (to align logical operators with the interacting patches)
        if self.rotate_patches:
            print("Rotating Ancilla and Control patch for alignment...")
            patch_a.rotate_coords(np.pi)
            patch_a.reset_rotation_angle()
            patch_c.transpose_coords()
            patch_c.rotate_coords(np.pi/2)
            patch_c.reset_rotation_angle()
            patch_c.reset_transposition()

        
        # 2. Create system and add patches
        # ----------------------------------------------------------------------
        print("Setting up QEC system...")
        self.system = QECSystem()
        self.system.add_patch(patch_a, name=self.patch_a_name)
        self.system.add_patch(patch_c, name=self.patch_c_name, offset=self.offset_ca)
        self.system.add_patch(patch_t, name=self.patch_t_name, offset=self.offset_ta)
        
        # 3. Register couplers
        # ----------------------------------------------------------------------
        print("Registering couplers...")
        coupler_protocol = UnrotatedTwoPatchCoupler()
        
        # Control-Ancilla coupler for ZZ measurement
        self.system.register_coupler(
            coupler_protocol,
            patch_names=[self.patch_c_name, self.patch_a_name],
            name=self.coupler_ca_name,
            interaction_type="ZZ"
        )
        
        # Target-Ancilla coupler for XX measurement
        self.system.register_coupler(
            coupler_protocol,
            patch_names=[self.patch_a_name, self.patch_t_name],
            name=self.coupler_ta_name,
            interaction_type="XX"
        )

        if self.interaction_types[0] == "ZZ":
            first_coupler = self.coupler_ca_name
            second_coupler = self.coupler_ta_name
        elif self.interaction_types[0] == "XX":
            first_coupler = self.coupler_ta_name
            second_coupler = self.coupler_ca_name
        else:
            raise ValueError(f"Invalid interaction type. Got (interaction_type={self.interaction_types[0]})")
        
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
        init_dict = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch_c_name:
                init_dict[q] = self.initial_state_dict["c"]
            elif owner == self.patch_t_name:
                init_dict[q] = self.initial_state_dict["t"]
            elif owner == self.patch_a_name:
                init_dict[q] = self.initial_state_dict["a"]
            # Skip coupler qubits - they will be initialized later
        self.builder.initialize(init_dict=init_dict, n=num_qubits)
        
        # 7. Syndrome extraction rounds (before any coupler activation)
        # ----------------------------------------------------------------------
        print(f"Building {self.rounds} rounds of syndrome extraction (before coupler activations)...")
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 8. First logical measurement (ZZ or XX depending on initial state of ancilla patch)
        # ----------------------------------------------------------------------
        print(f"Activating {self.interaction_types[0]} coupler for {self.interaction_types[0]} measurement...")
        self.builder.activate_coupler(first_coupler)

        # Initialize first coupler data qubits
        print(f"Initializing {first_coupler} data qubits...")
        coupler_data_indices_local = self.system.coupler_patches[first_coupler].data_indices
        coupler_data_indices = [
            self.system.local_to_global_map[first_coupler][q]
            for q in coupler_data_indices_local
        ]
        if self.interaction_types[0] == "ZZ":
            coupler_init_dict_1 = {q: "X" for q in coupler_data_indices}
        elif self.interaction_types[0] == "XX":
            coupler_init_dict_1 = {q: "Z" for q in coupler_data_indices}

        self.builder.initialize(init_dict=coupler_init_dict_1, n=num_qubits)
        
        # Syndrome extraction after first coupler activation
        print(f"Building {self.rounds} rounds of syndrome extraction (after the first coupler activation)...")
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 9. Second logical measurement (XX or ZZ depending on initial state of ancilla patch)
        # ----------------------------------------------------------------------
        print(f"Deactivating {first_coupler} coupler...")
        self.builder.deactivate_coupler(first_coupler)

        # Measure first coupler data qubits immediately at deactivation.
        # If left until final readout, these qubits sit idle through the entire
        # second-coupler SE phase — the extra idle time corrupts the detector chain.
        print(f"Measuring {first_coupler} data qubits immediately at deactivation...")
        self.builder.apply_data_readout(final_measurements=coupler_init_dict_1)

        print(f"Activating {self.interaction_types[1]} coupler for {self.interaction_types[1]} measurement...")
        self.builder.activate_coupler(second_coupler)

        # Initialize second coupler data qubits in X basis (for XX measurement)
        print(f"Initializing {second_coupler} data qubits...")
        coupler_data_indices_local = self.system.coupler_patches[second_coupler].data_indices
        coupler_data_indices = [
            self.system.local_to_global_map[second_coupler][q]
            for q in coupler_data_indices_local
        ]
        if self.interaction_types[1] == "XX":
            coupler_init_dict_2 = {q: "Z" for q in coupler_data_indices}
        elif self.interaction_types[1] == "ZZ":
            coupler_init_dict_2 = {q: "X" for q in coupler_data_indices}

        self.builder.initialize(init_dict=coupler_init_dict_2, n=num_qubits)
        
        # Syndrome extraction after second coupler activation
        print(f"Building {self.rounds} rounds of syndrome extraction (after the second coupler activation)...")
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 10. Final readout
        # ----------------------------------------------------------------------
        print("Measuring data qubits...")
        # Measure code patch qubits in the specified basis
        measure_dict = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch_c_name:
                measure_dict[q] = self.measure_state_dict["c"]
            elif owner == self.patch_t_name:
                measure_dict[q] = self.measure_state_dict["t"]
            elif owner == self.patch_a_name:
                measure_dict[q] = self.measure_state_dict["a"]
        # Add second coupler measurements (first coupler already measured at deactivation)
        measure_dict.update(coupler_init_dict_2)
        
        self.builder.apply_data_readout(final_measurements=measure_dict)
        
        # 11. Noise injection
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
