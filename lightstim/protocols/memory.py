import logging
import stim
from typing import Type, Literal, Optional, Any, Dict

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
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
                 qec_system: Optional[Any] = None,
                 extraction_block_class: Optional[Type] = None,
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 basis: Literal['X', 'Y', 'Z'] = 'Z',
                 if_detector: bool = True,
                 se_block_kwargs: Optional[dict] = None,
                 z_only: bool = False,
                 data_basis_map: Optional[Dict[int, str]] = None,
                 qec_patch: Optional[Any] = None,
                 patch_name: str = 'memory'):
        """
        Args:
            qec_system: An existing system configuration. Mutually exclusive with
                `qec_patch`.
            qec_patch: A single patch to place into a new QECSystem. This is the
                concise interface for one-patch memory experiments.
            patch_name: Name used when placing `qec_patch` into the new system.
            extraction_block_class: The class used to generate the SE circuit chunk.
                If omitted, a patch-provided default is used when present,
                otherwise GenericCSSColorationExtractionBlock is used.
            rounds: Number of QEC rounds (d).
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('code_capacity', 'phenomenological', etc.)
            basis: Memory basis to preserve ('X', 'Y', or 'Z'). Ignored when data_basis_map
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
        if (qec_system is None) == (qec_patch is None):
            raise ValueError("Provide exactly one of 'qec_system' or 'qec_patch'.")

        self.basis = str(basis).upper()
        if self.basis not in ('X', 'Y', 'Z'):
            raise ValueError(f"basis must be 'X', 'Y', or 'Z'; got {basis!r}")

        self.patch = None
        self.patch_name = None
        if qec_patch is not None:
            qec_system = QECSystem()
            qec_system.add_patch(qec_patch, name=patch_name)
            self.patch = qec_system.patches[patch_name][0]
            self.patch_name = patch_name

        self.system = qec_system
        self.block_class = (
            extraction_block_class or self._infer_extraction_block_class()
        )

        if qec_patch is not None:
            if data_basis_map is None:
                basis_map_factory = getattr(
                    self.block_class,
                    'memory_data_basis_map',
                    None,
                )
                if basis_map_factory is not None:
                    local_basis_map = basis_map_factory(self.patch, self.basis)
                    local_to_global = qec_system.local_to_global_map[patch_name]
                    data_basis_map = {
                        local_to_global[local_idx]: qubit_basis
                        for local_idx, qubit_basis in local_basis_map.items()
                    }

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
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
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

        # 3. Syndrome-Extraction Unit Cell
        # ----------------------------------------------------------------------
        # Build the physical unit cell before initialization because some
        # spacetime protocols prepare a subset of data qubits in the first
        # reset layer of the extraction block itself.
        self._log.debug("Building syndrome extraction unit cell...")
        se_block = self.block_class(self.system, **self.se_block_kwargs)

        # 4. Initialization
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
            data_basis = dict(self.data_basis_map)
        else:
            data_basis = {q: self.basis for q in data_indices}
        block_initialized_data = set(
            getattr(se_block, "data_qubits_initialized_by_block", ())
        )
        unexpected_block_initialized_data = (
            block_initialized_data - set(data_indices)
        )
        if unexpected_block_initialized_data:
            raise ValueError(
                "An extraction block may initialize only data qubits in this "
                "memory experiment; got unexpected indices "
                f"{sorted(unexpected_block_initialized_data)}."
            )
        init_dict = {
            qubit: qubit_basis
            for qubit, qubit_basis in data_basis.items()
            if qubit not in block_initialized_data
        }
        self.builder.initialize(init_dict=init_dict, n=num_qubits)
        self.system.active_qubit_indices.update(data_indices)

        # 5. Syndrome Extraction Loop
        # ----------------------------------------------------------------------
        self._log.debug("Building syndrome extraction rounds...")
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds,
            z_only=self.z_only,
            measurement_blocks=getattr(se_block, "measurement_blocks", None),
        )

        # 6. Final Readout
        # ----------------------------------------------------------------------
        # Measure data qubits in the memory basis. Construct detectors and logical observables.
        # Same per-qubit map as initialization: both time boundaries must agree for the
        # final stabilizer reconstruction (detectors + observable) to be deterministic.
        self._log.debug("Measuring data qubits...")
        measurements = dict(data_basis)
        self.builder.apply_data_readout(final_measurements=measurements, z_only=self.z_only)

        # 7. Noise Injection
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

    def _infer_extraction_block_class(self) -> Type:
        defaults = set()
        for patch, _ in self.system.patches.values():
            block_class = getattr(patch, "default_extraction_block_class", None)
            if block_class is not None:
                defaults.add(block_class)

        if len(defaults) == 1:
            return defaults.pop()
        if len(defaults) > 1:
            raise ValueError(
                "Multiple patch default extraction blocks are present. "
                "Pass extraction_block_class explicitly."
            )

        from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock
        return GenericCSSColorationExtractionBlock

    # Log Helper Functions
