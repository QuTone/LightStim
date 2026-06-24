import logging
import stim
from typing import Type, Literal, Optional, Any, Dict

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig

class MemoryExperiment:
    """
    Orchestrates a Quantum Memory Experiment.
    
    This class acts as the 'Director':
    1. Initializes the System and Tracker.
    2. Uses CircuitBuilder to layout the circuit (Init -> SE Loops -> Readout).
    3. Injects Noise using the configured strategy.
    """

    def __init__(self,
                 qec_system: Any,  # The System/Geometry object
                 extraction_block_class: Type, # Class ref, e.g. RotatedSurfaceCodeExtractionBlock
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 basis: Literal['X', 'Z'] = 'Z',
                 if_detector: bool = True,
                 se_block_kwargs: Optional[dict] = None,
                 z_only: bool = False,
                 data_basis_map: Optional[Dict[int, str]] = None):
        """
        Args:
            qec_patch: The system configuration object (contains coords, indices, map).
            extraction_block_class: The class used to generate the SE circuit chunk.
            rounds: Number of QEC rounds (d).
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('code_capacity', 'phenomenological', etc.)
            basis: Memory basis to preserve ('X' or 'Z'). Ignored when data_basis_map
                is provided — the per-qubit map then fully determines both the
                initialization and readout bases.
            if_detector: If True, emit DETECTOR and OBSERVABLE_INCLUDE instructions.
            se_block_kwargs: Extra keyword arguments passed to extraction_block_class constructor.
                e.g. {'scheduling': 'parallel'} for RotatedSurfaceCodeExtractionBlock.
            z_only: If True, only Z-ancilla measurements emit DETECTOR instructions.
                Produces a smaller DEM matching Z-basis-only decoding (e.g. gong_circuit style).
            data_basis_map: Optional per-qubit init/readout basis override, mapping
                GLOBAL data qubit index -> 'X' | 'Y' | 'Z'. Keys must be exactly the
                system's data qubit indices, so generate it AFTER system.add_patch()
                (e.g. xzzx_memory_basis(system, basis) for the XZZX surface code,
                whose mixed X/Z checks need a checkerboard of bases to be
                deterministic). The same map is used for initialization and final
                readout. When None, all data qubits use the uniform `basis`.
                Incompatible with z_only=True.
        """
        self._log = logging.getLogger(__name__)
        if data_basis_map is not None:
            if z_only:
                raise ValueError(
                    "data_basis_map is incompatible with z_only=True: the z_only "
                    "readout path reconstructs detectors assuming a uniform Z-basis "
                    "data measurement."
                )
            data_basis_map = {q: str(b).upper() for q, b in data_basis_map.items()}
            bad = sorted({b for b in data_basis_map.values() if b not in ('X', 'Y', 'Z')})
            if bad:
                raise ValueError(f"data_basis_map values must be 'X', 'Y' or 'Z'; got {bad}")
        self.data_basis_map = data_basis_map
        self.system = qec_system
        self.block_class = extraction_block_class
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.basis = basis.upper()
        self.se_block_kwargs = se_block_kwargs or {}
        self.z_only = z_only

        # Internal state
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None
        self.if_detector = if_detector

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the experiment.
        """
        # 1. Setup
        num_qubits = len(self.system.qubit_coords)
        num_logicals = self.system.num_logicals
        
        self.tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=num_logicals)
        self.builder = CircuitBuilder(tracker=self.tracker, system_config=self.system, if_detector=self.if_detector)

        # 2. Coordinate Definitions
        # ----------------------------------------------------------------------
        self._log.debug("Writing coordinates...")
        self.builder.write_coordinates()

        # 3. Initialization
        # ----------------------------------------------------------------------
        # Initialize Data Qubits in the target memory basis.
        # The Tracker will automatically register the initial stabilizers.
        self._log.debug("Initializing...")
        data_indices = [self.system.index_map[coord] for coord in self.system.data_coords]
        if self.data_basis_map is not None:
            missing = set(data_indices) - self.data_basis_map.keys()
            extra = self.data_basis_map.keys() - set(data_indices)
            if missing or extra:
                raise ValueError(
                    f"data_basis_map keys must be exactly the system's data qubit indices "
                    f"(missing: {sorted(missing)}, unexpected: {sorted(extra)}). "
                    "Generate the map AFTER system.add_patch(), e.g. "
                    "xzzx_memory_basis(system, basis)."
                )
            data_basis = {q: self.data_basis_map[q] for q in data_indices}
        else:
            data_basis = {q: self.basis for q in data_indices}
        init_dict = dict(data_basis)
        self.builder.initialize(init_dict=init_dict, n=num_qubits)

        # 4. Syndrome Extraction Loop
        # ----------------------------------------------------------------------
        # Instantiate the block to get the unit-cell circuit (One Round)
        # We pass self.patch because the Block needs coordinate info
        self._log.debug("Building syndrome extraction rounds...")
        se_block = self.block_class(self.system, **self.se_block_kwargs)

        # Apply the loop using Builder, construct detectors
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds,
            z_only=self.z_only,
        )

        # 5. Final Readout
        # ----------------------------------------------------------------------
        # Measure data qubits in the memory basis. Construct detectors and logical observables.
        # Same per-qubit map as initialization: both time boundaries must agree for the
        # final stabilizer reconstruction (detectors + observable) to be deterministic.
        self._log.debug("Measuring data qubits...")
        measurements = dict(data_basis)
        self.builder.apply_data_readout(final_measurements=measurements, z_only=self.z_only)

        # 6. Noise Injection
        # ----------------------------------------------------------------------
        # Finally, wrap the clean topological circuit with the requested noise model.
        if self.noise_params is not None:
            self._log.debug("Injecting noise...")
            noisy_circuit = self.builder.build_noisy_circuit(
                noise_params=self.noise_params,
                noise_model=self.noise_model
            )
            return noisy_circuit
        else:
            return self.builder.circuit


    # Log Helper Functions
