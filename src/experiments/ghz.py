# src/experiments/ghz.py

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


class GHZExperiment(QECExperiment):
    """
    Orchestrates a GHZ State Preparation Experiment using three Surface Code patches.
    
    This class implements GHZ state preparation using transversal CNOT gates:
    1. Creates and configures three Unrotated Surface Code patches.
    2. Sets up their relative positions via offsets.
    3. Initializes data qubits: |+>, |0>, |0> (patch1, patch2, patch3).
    4. Applies syndrome extraction rounds before CNOT gates.
    5. Applies CNOT(1,2) and CNOT(1,3) via LogicalExecutor.
    6. Applies syndrome extraction rounds after CNOT gates.
    7. Measures data qubits in specified bases (per patch).
    8. Injects Noise using the configured strategy.
    """

    def __init__(self,
                 distance: Union[int, Tuple[int, int, int]] = 3,
                 offset_patch2: Tuple[float, float] = (6, 0),
                 offset_patch3: Tuple[float, float] = (12, 0),
                 initial_basis_patch1: Literal["X", "Z"] = "X",
                 initial_basis_patch2: Literal["X", "Z"] = "Z",
                 initial_basis_patch3: Literal["X", "Z"] = "Z",
                 measure_basis_patch1: Literal["X", "Z"] = "Z",
                 measure_basis_patch2: Literal["X", "Z"] = "Z",
                 measure_basis_patch3: Literal["X", "Z"] = "Z",
                 rounds_before: int = 2,
                 rounds_after: int = 2,
                 extraction_block_class: Type = UnrotatedSurfaceCodeExtractionBlock,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True):
        """
        Args:
            distance: Distance for all patches (int) or (distance_patch1, distance_patch2, distance_patch3) tuple.
            offset_patch2: (dx, dy) offset for patch2 relative to patch1.
            offset_patch3: (dx, dy) offset for patch3 relative to patch1.
            initial_basis_patch1: Initial state basis for patch1 data qubits ("X" or "Z"). Default "X" for |+>.
            initial_basis_patch2: Initial state basis for patch2 data qubits ("X" or "Z"). Default "Z" for |0>.
            initial_basis_patch3: Initial state basis for patch3 data qubits ("X" or "Z"). Default "Z" for |0>.
            measure_basis_patch1: Measurement basis for patch1 data qubits ("X" or "Z").
            measure_basis_patch2: Measurement basis for patch2 data qubits ("X" or "Z").
            measure_basis_patch3: Measurement basis for patch3 data qubits ("X" or "Z").
            rounds_before: Number of QEC rounds before CNOT gates.
            rounds_after: Number of QEC rounds after CNOT gates.
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
            self.distance_patch1 = distance
            self.distance_patch2 = distance
            self.distance_patch3 = distance
        elif isinstance(distance, tuple) and len(distance) == 3:
            self.distance_patch1, self.distance_patch2, self.distance_patch3 = distance
        else:
            raise ValueError(f"distance must be int or (int, int, int) tuple, got {type(distance)}")
        
        self.offset_patch2 = offset_patch2
        self.offset_patch3 = offset_patch3
        self.initial_basis_patch1 = initial_basis_patch1.upper()
        self.initial_basis_patch2 = initial_basis_patch2.upper()
        self.initial_basis_patch3 = initial_basis_patch3.upper()
        self.measure_basis_patch1 = measure_basis_patch1.upper()
        self.measure_basis_patch2 = measure_basis_patch2.upper()
        self.measure_basis_patch3 = measure_basis_patch3.upper()
        self.rounds_before = rounds_before
        self.rounds_after = rounds_after
        
        # Internal state
        self.patch1_name = "patch1"
        self.patch2_name = "patch2"
        self.patch3_name = "patch3"

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the GHZ state preparation experiment.
        """
        # 1. Create patches
        # ----------------------------------------------------------------------
        print("Creating patches...")
        patch1_local = UnrotatedSurfaceCode(distance=self.distance_patch1)
        patch2_local = UnrotatedSurfaceCode(distance=self.distance_patch2)
        patch3_local = UnrotatedSurfaceCode(distance=self.distance_patch3)
        
        # 2. Create system and add patches
        # ----------------------------------------------------------------------
        print("Setting up QEC system...")
        self.system = QECSystem()
        # add_patch returns global patch view (with global indices)
        patch1_global = self.system.add_patch(patch1_local, name=self.patch1_name)
        patch2_global = self.system.add_patch(patch2_local, name=self.patch2_name, offset=self.offset_patch2)
        patch3_global = self.system.add_patch(patch3_local, name=self.patch3_name, offset=self.offset_patch3)
        
        # 3. Setup tracker, builder, and logical executor
        # ----------------------------------------------------------------------
        self._setup_experiment()
        
        # Register LogicalOpSet for transversal CNOT
        self.logical_executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())
        
        # 4. Write coordinates
        # ----------------------------------------------------------------------
        print("Writing coordinates...")
        self.builder.write_coordinates()
        
        # 5. Initialize data qubits: |+>, |0>, |0>
        # ----------------------------------------------------------------------
        print("Initializing data qubits...")
        init_dict = {}
        for q in self.system.data_indices:
            owner = self.system.index_to_owner_map[q]
            if owner == self.patch1_name:
                init_dict[q] = self.initial_basis_patch1
            elif owner == self.patch2_name:
                init_dict[q] = self.initial_basis_patch2
            elif owner == self.patch3_name:
                init_dict[q] = self.initial_basis_patch3
        
        self.builder.initialize(init_dict=init_dict, n=self.system.num_qubits)
        
        # 6. Syndrome extraction (before CNOT gates)
        # ----------------------------------------------------------------------
        print(f"Building {self.rounds_before} rounds of syndrome extraction (before CNOT)...")
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds_before
        )
        
        # 7. Transversal CNOT gates: CNOT(1,2) and CNOT(1,3)
        # ----------------------------------------------------------------------
        # CNOT(1,2): patch1 controls patch2
        print("Applying CNOT(1,2)...")
        self.logical_executor.apply_logical_operation(
            op_name="transversal_cnot",
            patches=[patch1_global, patch2_global],
        )
        
        # Syndrome extraction between CNOT gates (as in notebook)
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds_before
        )
        
        # CNOT(1,3): patch1 controls patch3
        print("Applying CNOT(1,3)...")
        self.logical_executor.apply_logical_operation(
            op_name="transversal_cnot",
            patches=[patch1_global, patch3_global],
        )
        
        # 8. Syndrome extraction (after CNOT gates)
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
            if owner == self.patch1_name:
                measurements[q] = self.measure_basis_patch1
            elif owner == self.patch2_name:
                measurements[q] = self.measure_basis_patch2
            elif owner == self.patch3_name:
                measurements[q] = self.measure_basis_patch3
        
        self.builder.apply_data_readout(final_measurements=measurements)
        
        # 10. Noise injection
        # ----------------------------------------------------------------------
        return self._inject_noise(self.builder.circuit)
