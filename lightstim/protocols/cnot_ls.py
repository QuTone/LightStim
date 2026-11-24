# experiments/CNOT_LS.py

import logging

import stim
from typing import Type, Literal, Optional, Any, Dict, Tuple, Mapping, Sequence

from lightstim.ir.builder import CircuitBuilder, DataReadoutStage
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler
)


_log = logging.getLogger(__name__)


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
                 offset_ta: Optional[Tuple[float, float]] = None,
                 offset_ca: Optional[Tuple[float, float]] = None,
                 initial_state_dict: Optional[Dict[str, Literal["X", "Z"]]] = None,
                 measure_state_dict: Optional[Dict[str, Literal["X", "Z"]]] = None,
                 code_patch_class: Type = UnrotatedSurfaceCode,
                 coupler_class: Type = UnrotatedTwoPatchCoupler,
                 extraction_block_class: Type = UnrotatedSurfaceCodeExtractionBlock,
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True,
                 rotate_patches: bool = True,
                 logical_tracking: Literal["legacy", "slabs"] = "slabs",
                 *,
                 offset_ancilla: Tuple[float, float] = (0, 0),
                 patch_transforms: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None):
        """
        Args:
            logical_tracking: Logical tracking model used by the syndrome tracker.
            patch_configs: Patch constructor arguments keyed by a, c and t.
            offset_ta: Target displacement from ancilla. Defaults to (2d, 0)
                for a square ancilla of distance d.
            offset_ca: Control displacement from ancilla. Defaults to (0, 2d).
            offset_ancilla: Ancilla position on the canvas.
            patch_transforms: Ordered canvas transform steps keyed by patch role
                (a, c, t). An explicit map replaces the rotate_patches preset.
            initial_state_dict: X or Z preparation basis keyed by patch role.
            measure_state_dict: X or Z readout basis keyed by patch role.
            extraction_block_class: Class for syndrome extraction block. Defaults to UnrotatedSurfaceCodeExtractionBlock.
            rounds: Number of QEC rounds before and after coupler activations.
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('circuit_level', 'code_capacity', etc.).
            if_detector: Whether to generate detectors.
            rotate_patches: Whether to rotate ancilla patches by pi and reset rotation angle (default True).
        """
        self.patch_configs = patch_configs
        distance = patch_configs["a"].get("distance")
        if distance is None and (offset_ta is None or offset_ca is None):
            raise ValueError("Specify offset_ta and offset_ca for a non-square ancilla.")
        self.offset_ta = (2 * distance, 0) if offset_ta is None else offset_ta
        self.offset_ca = (0, 2 * distance) if offset_ca is None else offset_ca
        self.offset_ancilla = offset_ancilla
        if patch_transforms is not None and set(patch_transforms) - {"a", "c", "t"}:
            raise ValueError("patch_transforms keys must be a, c or t.")
        self.patch_transforms = patch_transforms
        self.initial_state_dict = ({"a": "X", "c": "X", "t": "X"} if initial_state_dict is None
                                   else dict(initial_state_dict))
        self.measure_state_dict = ({"a": "Z", "c": "X", "t": "X"} if measure_state_dict is None
                                   else dict(measure_state_dict))
        self.code_patch_class = code_patch_class
        self.coupler_class = coupler_class
        self.extraction_block_class = extraction_block_class
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.if_detector = if_detector
        self.rotate_patches = rotate_patches
        self.logical_tracking = logical_tracking
        
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
                _log.warning("Changing ancilla readout from %s to Z for an X preparation.",
                             self.measure_state_dict['a'])
                self.measure_state_dict["a"] = "Z"
        elif self.initial_state_dict["a"] == "Z":
            self.interaction_types = ["XX", "ZZ"]
            if self.measure_state_dict["a"] != "X":
                _log.warning("Changing ancilla readout from %s to X for a Z preparation.",
                             self.measure_state_dict['a'])
                self.measure_state_dict["a"] = "X"
        else:
            raise ValueError(f"Initial state of ancilla patch must be X or Z. Got (initial_state_a={self.initial_state_dict['a']})")

    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the CNOT Lattice Surgery experiment.
        """
        # 1. Create patches
        # ----------------------------------------------------------------------
        _log.debug("Creating patches...")
        patch_c = self.code_patch_class(**self.patch_configs["c"])
        patch_t = self.code_patch_class(**self.patch_configs["t"])
        patch_a = self.code_patch_class(**self.patch_configs["a"])
        
        transforms = self.patch_transforms
        if transforms is None:
            transforms = ({
                "a": [{"rotate_deg": 180}],
                "c": [{"transpose": True}, {"rotate_deg": 90}],
            } if self.rotate_patches else {})
        for role, patch in (("a", patch_a), ("c", patch_c), ("t", patch_t)):
            patch.apply_canvas_transform(transforms.get(role, ()))

        # Register in protocol order so the qubit and stabilizer IDs are stable.
        self.system = QECSystem()
        ax, ay = self.offset_ancilla
        self.system.add_patch(patch_a, name=self.patch_a_name, offset=(ax, ay))
        self.system.add_patch(patch_c, name=self.patch_c_name,
                              offset=(ax + self.offset_ca[0], ay + self.offset_ca[1]))
        self.system.add_patch(patch_t, name=self.patch_t_name,
                              offset=(ax + self.offset_ta[0], ay + self.offset_ta[1]))

        # 3. Register couplers
        # ----------------------------------------------------------------------
        _log.debug("Registering couplers...")
        coupler_protocol = self.coupler_class()
        
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
        
        self.tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=num_logicals,
                                      logical_tracking=self.logical_tracking)
        self.builder = CircuitBuilder(
            tracker=self.tracker,
            system_config=self.system,
            if_detector=self.if_detector
        )
        
        # 5. Write coordinates
        # ----------------------------------------------------------------------
        _log.debug("Writing coordinates...")
        self.builder.write_coordinates()
        
        # 6. Initialize code patch data qubits
        # ----------------------------------------------------------------------
        _log.debug("Initializing code patch data qubits...")
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
        _log.debug("Building %s rounds of syndrome extraction (before coupler activations)...", self.rounds)
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 8. First logical measurement (ZZ or XX depending on initial state of ancilla patch)
        # ----------------------------------------------------------------------
        _log.debug("Activating %s coupler for %s measurement...", self.interaction_types[0], self.interaction_types[0])
        self.builder.activate_coupler(first_coupler)

        # Initialize first coupler data qubits
        _log.debug("Initializing %s data qubits...", first_coupler)
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
        _log.debug("Building %s rounds of syndrome extraction (after the first coupler activation)...", self.rounds)
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 9. Second logical measurement (XX or ZZ depending on initial state of ancilla patch)
        # ----------------------------------------------------------------------
        _log.debug("Deactivating %s coupler...", first_coupler)
        self.builder.deactivate_coupler(first_coupler)

        # Measure first coupler data qubits immediately at deactivation.
        # If left until final readout, these qubits sit idle through the entire
        # second-coupler SE phase — the extra idle time corrupts the detector chain.
        _log.debug("Measuring %s data qubits immediately at deactivation...", first_coupler)
        self.builder.apply_data_readout(final_measurements=coupler_init_dict_1)

        _log.debug("Activating %s coupler for %s measurement...", self.interaction_types[1], self.interaction_types[1])
        self.builder.activate_coupler(second_coupler)

        # Initialize second coupler data qubits in X basis (for XX measurement)
        _log.debug("Initializing %s data qubits...", second_coupler)
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
        _log.debug("Building %s rounds of syndrome extraction (after the second coupler activation)...", self.rounds)
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds
        )
        
        # 10. Final readout
        # ----------------------------------------------------------------------
        _log.debug("Measuring data qubits...")
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
        
        self.builder.apply_staged_data_readout(measure_dict, stages=(
            DataReadoutStage(tuple(coupler_init_dict_2), (second_coupler,)),
            DataReadoutStage(tuple(q for q in measure_dict if q not in coupler_init_dict_2)),
        ))
        
        # 11. Noise injection
        # ----------------------------------------------------------------------
        if self.noise_params is not None:
            _log.debug("Injecting noise...")
            noisy_circuit = self.builder.build_noisy_circuit(
                noise_params=self.noise_params,
                noise_model=self.noise_model
            )
            return noisy_circuit
        else:
            return self.builder.to_stim_circuit()
