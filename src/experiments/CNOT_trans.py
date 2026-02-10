# src/experiments/CNOT_trans.py

import stim
from typing import Type, Literal, Optional, Any, Tuple, Union

from ..ir.experiment import QECExperiment
from ..ir.qec_system import QECSystem
from ..noise.config import NoiseConfig
from ..ir.operation import CSSLogicalOpSet
from ..qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)


class CNOTTransExperiment(QECExperiment):
    """
    Orchestrates a Transversal CNOT Gate Experiment between two Surface Code patches.
    
    This class implements a transversal CNOT gate between two unrotated surface code patches:
    1. Creates and configures two Unrotated Surface Code patches.
    2. Sets up their relative positions via offset.
    3. Initializes data qubits in specified bases.
    4. Applies syndrome extraction rounds before the CNOT.
    5. Applies transversal CNOT gate via LogicalExecutor.
    6. Applies syndrome extraction rounds after the CNOT.
    7. Measures data qubits in specified bases.
    8. Injects Noise using the configured strategy.
    """

    def __init__(self,
                 distance: Union[int, Tuple[int, int]] = 3,
                 offset_target: Tuple[float, float] = (6, 0),
                 initial_basis_control: Literal["X", "Z"] = "Z",
                 initial_basis_target: Literal["X", "Z"] = "Z",
                 measure_basis_control: Literal["X", "Z"] = "Z",
                 measure_basis_target: Literal["X", "Z"] = "Z",
                 rounds_before: int = 2,
                 rounds_after: int = 2,
                 extraction_block_class: Type = UnrotatedSurfaceCodeExtractionBlock,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True):
        """
        Args:
            distance: Distance for both patches (int) or (distance_control, distance_target) tuple.
            offset_target: (dx, dy) offset for target patch relative to control patch.
            initial_basis_control: Initial state basis for control patch data qubits ("X" or "Z").
            initial_basis_target: Initial state basis for target patch data qubits ("X" or "Z").
            measure_basis_control: Measurement basis for control patch data qubits ("X" or "Z").
            measure_basis_target: Measurement basis for target patch data qubits ("X" or "Z").
            rounds_before: Number of QEC rounds before transversal CNOT.
            rounds_after: Number of QEC rounds after transversal CNOT.
            extraction_block_class: Class for syndrome extraction block. Defaults to UnrotatedSurfaceCodeExtractionBlock.
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('circuit_level', 'code_capacity', etc.).
            if_detector: Whether to generate detectors.
        """
        super().__init__(
            extraction_block_class=extraction_block_class,
            rounds=rounds_before,  # Base class uses single 'rounds', but we use rounds_before/after separately
            noise_params=noise_params,
            noise_model=noise_model,
            if_detector=if_detector
        )
        
        # Handle distance parameter
        if isinstance(distance, int):
            self.distance_control = distance
            self.distance_target = distance
        elif isinstance(distance, tuple) and len(distance) == 2:
            self.distance_control, self.distance_target = distance
        else:
            raise ValueError(f"distance must be int or (int, int) tuple, got {type(distance)}")
        
        self.offset_target = offset_target
        self.initial_basis_control = initial_basis_control.upper()
        self.initial_basis_target = initial_basis_target.upper()
        self.measure_basis_control = measure_basis_control.upper()
        self.measure_basis_target = measure_basis_target.upper()
        self.rounds_before = rounds_before
        self.rounds_after = rounds_after
        
        # Internal state
        self.patch_control_name = "control"
        self.patch_target_name = "target"

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the transversal CNOT experiment.
        """
        # 1. Create patches
        # ----------------------------------------------------------------------
        print("Creating patches...")
        control_patch_local = UnrotatedSurfaceCode(distance=self.distance_control)
        target_patch_local = UnrotatedSurfaceCode(distance=self.distance_target)
        
        # 2. Create system and add patches
        # ----------------------------------------------------------------------
        print("Setting up QEC system...")
        self.system = QECSystem()
        # add_patch returns global patch view (with global indices)
        control_patch_global = self.system.add_patch(control_patch_local, name=self.patch_control_name)
        target_patch_global = self.system.add_patch(target_patch_local, name=self.patch_target_name, offset=self.offset_target)
        
        # 3. Setup tracker, builder, and logical executor
        # ----------------------------------------------------------------------
        self._setup_experiment()
        
        # Register LogicalOpSet for transversal CNOT
        self.logical_executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())
        
        # 4. Write coordinates
        # ----------------------------------------------------------------------
        print("Writing coordinates...")
        self.builder.write_coordinates()
        
        # 5. Initialize data qubits (basis selectable per patch)
        # ----------------------------------------------------------------------
        print("Initializing data qubits...")
        init_dict = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch_control_name:
                init_dict[q] = self.initial_basis_control
            elif owner == self.patch_target_name:
                init_dict[q] = self.initial_basis_target
        
        self.builder.initialize(init_dict=init_dict, n=self.system.num_qubits)
        
        # 6. Syndrome extraction (before transversal CNOT)
        # ----------------------------------------------------------------------
        print(f"Building {self.rounds_before} rounds of syndrome extraction (before CNOT)...")
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds_before
        )
        
        # 7. Transversal CNOT via LogicalExecutor
        # ----------------------------------------------------------------------
        print("Applying transversal CNOT gate...")
        self.logical_executor.apply_logical_operation(
            op_name="transversal_cnot",
            patches=[control_patch_global, target_patch_global],
        )
        
        # 8. Syndrome extraction (after transversal CNOT)
        # ----------------------------------------------------------------------
        print(f"Building {self.rounds_after} rounds of syndrome extraction (after CNOT)...")
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds_after
        )
        
        # 9. Final data readout (basis selectable per patch)
        # ----------------------------------------------------------------------
        print("Measuring data qubits...")
        measurements = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch_control_name:
                measurements[q] = self.measure_basis_control
            elif owner == self.patch_target_name:
                measurements[q] = self.measure_basis_target
        
        self.builder.apply_data_readout(final_measurements=measurements)
        
        # 10. Noise injection
        # ----------------------------------------------------------------------
        return self._inject_noise(self.builder.circuit)
