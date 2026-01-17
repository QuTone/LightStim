# src/experiments/memory.py

import stim
from typing import Type, Literal, Optional, Any

from ..circuit.builder import CircuitBuilder
from ..ir.tracker import SyndromeTracker
from ..noise.config import NoiseConfig

class MemoryExperiment:
    """
    Orchestrates a Quantum Memory Experiment.
    
    This class acts as the 'Director':
    1. Initializes the System and Tracker.
    2. Uses CircuitBuilder to layout the circuit (Init -> SE Loops -> Readout).
    3. Injects Noise using the configured strategy.
    """

    def __init__(self, 
                 qec_patch: Any,  # The System/Geometry object
                 extraction_block_class: Type, # Class ref, e.g. RotatedSurfaceCodeExtractionBlock
                 rounds: int,
                 noise_params: NoiseConfig,
                 noise_model: str = 'circuit_level',
                 basis: Literal['X', 'Z'] = 'Z',
                 if_detector: bool = True):
        """
        Args:
            qec_patch: The system configuration object (contains coords, indices, map).
            extraction_block_class: The class used to generate the SE circuit chunk.
            rounds: Number of QEC rounds (d).
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('code_capacity', 'phenomenological', etc.)
            basis: Memory basis to preserve ('X' or 'Z').
        """
        self.patch = qec_patch
        self.block_class = extraction_block_class
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.basis = basis.upper()
        
        # Internal state
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None
        self.if_detector = if_detector

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the experiment.
        """
        # 1. Setup
        num_qubits = len(self.patch.qubit_coords)
        num_logicals = self.patch.num_logicals
        
        self.tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=num_logicals)
        self.builder = CircuitBuilder(tracker=self.tracker, system_config=self.patch, if_detector=self.if_detector)

        # 2. Coordinate Definitions
        # ----------------------------------------------------------------------
        print("Writing coordinates...")
        self.builder.write_coordinates()

        # 3. Initialization
        # ----------------------------------------------------------------------
        # Initialize Data Qubits in the target memory basis.
        # The Tracker will automatically register the initial stabilizers.
        print("Initializing...")
        data_indices = [self.patch.index_map[coord] for coord in self.patch.data_coords]
        init_dict = {q: self.basis for q in data_indices}
        self.builder.initialize(init_dict=init_dict, n=num_qubits)

        # 4. Syndrome Extraction Loop
        # ----------------------------------------------------------------------
        # Instantiate the block to get the unit-cell circuit (One Round)
        # We pass self.patch because the Block needs coordinate info
        print("Building syndrome extraction rounds...")
        se_block = self.block_class(self.patch)
        se_round = se_block.circuit

        # Apply the loop using Builder, construct detectors
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_round, 
            rounds=self.rounds
        )

        # 5. Final Readout
        # ----------------------------------------------------------------------
        # Measure data qubits in the memory basis. Construct detectors and logical observables.
        print("Measuring data qubits...")
        measurements = {q: self.basis for q in data_indices}
        self.builder.apply_final_readout(final_measurements=measurements)

        # 6. Noise Injection
        # ----------------------------------------------------------------------
        # Finally, wrap the clean topological circuit with the requested noise model.
        print("Injecting noise...")
        noisy_circuit = self.builder.build_noisy_circuit(
            noise_params=self.noise_params,
            noise_model=self.noise_model
        )

        return noisy_circuit


    # Log Helper Functions
