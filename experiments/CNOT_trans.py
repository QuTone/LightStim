# experiments/CNOT_trans.py

import stim
from typing import Type, Literal, Optional, Any, Tuple, Union, Dict

from src.ir.experiment import QECExperiment
from src.ir.qec_system import QECSystem
from src.ir.qec_patch import QECPatch
from src.noise.config import NoiseConfig
from src.ir.operation import CSSLogicalOpSet


class CNOTTransExperiment(QECExperiment):
    """
    Orchestrates a Transversal CNOT Gate Experiment between two CSS code patches.
    
    Unified interface for any CSS code:
    1. User specifies code_patch_class, code_params, and extraction_block_class.
    2. Creates patches internally and adds them to QECSystem.
    3. Applies syndrome extraction, transversal CNOT, and readout.
    
    Works with UnrotatedSurfaceCode, RotatedSurfaceCode, ToricCode, etc.
    """

    def __init__(self,
                 code_patch_class: Type[QECPatch],
                 extraction_block_class: Type,
                 code_params_control: Dict[str, Any],
                 code_params_target: Optional[Dict[str, Any]] = None,
                 offset_target: Tuple[float, float] = (6, 0),
                 initial_basis_control: Literal["X", "Z"] = "Z",
                 initial_basis_target: Literal["X", "Z"] = "Z",
                 measure_basis_control: Literal["X", "Z"] = "Z",
                 measure_basis_target: Literal["X", "Z"] = "Z",
                 rounds_before: int = 2,
                 rounds_after: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True):
        """
        Args:
            code_patch_class: Patch class (e.g. UnrotatedSurfaceCode, RotatedSurfaceCode, ToricCode).
            extraction_block_class: Syndrome extraction block class (takes system, has .circuit).
            code_params_control: Dict of kwargs for control patch (e.g. {"distance": 3} or {"l_z": 3, "l_x": 3}).
            code_params_target: Dict of kwargs for target patch. Defaults to same as control.
            offset_target: (dx, dy) offset for target patch relative to control patch.
            initial_basis_control/target: Initial state basis for data qubits ("X" or "Z").
            measure_basis_control/target: Measurement basis for data qubits ("X" or "Z").
            rounds_before/after: Number of QEC rounds before/after transversal CNOT.
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('circuit_level', 'code_capacity', etc.).
            if_detector: Whether to generate detectors.
        """
        super().__init__(
            extraction_block_class=extraction_block_class,
            rounds=rounds_before,
            noise_params=noise_params,
            noise_model=noise_model,
            if_detector=if_detector
        )
        
        self.code_patch_class = code_patch_class
        self.code_params_control = dict(code_params_control)
        self.code_params_target = dict(code_params_target) if code_params_target is not None else dict(code_params_control)
        self.offset_target = offset_target
        self.initial_basis_control = initial_basis_control.upper()
        self.initial_basis_target = initial_basis_target.upper()
        self.measure_basis_control = measure_basis_control.upper()
        self.measure_basis_target = measure_basis_target.upper()
        self.rounds_before = rounds_before
        self.rounds_after = rounds_after
        
        self.patch_control_name = "control"
        self.patch_target_name = "target"

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the transversal CNOT experiment.
        """
        # 1. Create patches from code_patch_class and code_params
        # ----------------------------------------------------------------------
        print("Creating patches...")
        control_patch_local = self.code_patch_class(**self.code_params_control)
        target_patch_local = self.code_patch_class(**self.code_params_target)
        
        # 2. Create system and add patches
        # ----------------------------------------------------------------------
        print("Setting up QEC system...")
        self.system = QECSystem()
        control_patch_global = self.system.add_patch(control_patch_local, name=self.patch_control_name)
        target_patch_global = self.system.add_patch(target_patch_local, name=self.patch_target_name, offset=self.offset_target)
        
        # 3. Setup tracker, builder, logical executor
        # ----------------------------------------------------------------------
        self._setup_experiment()
        self.logical_executor.register_op_set(self.code_patch_class, CSSLogicalOpSet())
        
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
