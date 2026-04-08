# experiments/state_injection.py

"""
State Injection Experiment for Surface Codes.

Generic single-patch injection workflow via LogicalExecutor:
1. state_injection — initialize + SE rounds (self-contained)
2. logical_unencode — measure all but injection site (auto detectors)
3. Noiseless physical measurement (auto logical observable)

Supports any surface code (rotated, unrotated) by parameterizing the code
patch class, extraction block class, and logical operation set class.

Post-selection modes: full_postselection, full_qec, hybrid (rotated SC only).
"""

import stim
from typing import Type, Literal, Optional, Set, Tuple, Dict, Any

from src.ir.experiment import QECExperiment
from src.ir.qec_system import QECSystem
from src.ir.qec_patch import QECPatch
from src.ir.operation import LogicalOpSet
from src.noise.config import NoiseConfig
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)


class StateInjectionExperiment(QECExperiment):
    """
    Single-patch state injection experiment with configurable post-selection.

    Works with any surface code by specifying code_patch_class, extraction_block_class,
    and op_set_class. Defaults to rotated surface code for backward compatibility.
    """

    def __init__(
        self,
        code_patch_class: Type[QECPatch] = RotatedSurfaceCode,
        extraction_block_class: Type = RotatedSurfaceCodeExtractionBlock,
        op_set_class: Type[LogicalOpSet] = RotatedSurfaceCodeLogicalOpSet,
        code_params: Optional[Dict[str, Any]] = None,
        distance: int = 3,
        rounds: int = 2,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner", "middle"] = "corner",
        post_select_mode: Literal["full_postselection", "full_qec", "hybrid"] = "full_postselection",
        noise_params: Optional[NoiseConfig] = None,
        noise_model: Optional[str] = "circuit_level",
        if_detector: bool = True,
    ):
        """
        Args:
            code_patch_class: Patch class (e.g. RotatedSurfaceCode, UnrotatedSurfaceCode).
            extraction_block_class: SE block class for this code.
            op_set_class: LogicalOpSet class with state_injection/logical_unencode methods.
            code_params: Dict of kwargs for patch constructor. If None, uses {"distance": distance}.
            distance: Code distance (convenience shortcut, used when code_params is None).
            rounds: Number of SE rounds.
            inject_state: Target logical state ('Z', 'X', or 'Y').
            protocol: Injection protocol ('corner' or 'middle').
            post_select_mode: 'full_postselection', 'full_qec', or 'hybrid'.
            noise_params: Optional noise configuration.
            noise_model: Noise model string.
            if_detector: Whether to generate detectors.
        """
        super().__init__(
            extraction_block_class=extraction_block_class,
            rounds=rounds,
            noise_params=noise_params,
            noise_model=noise_model,
            if_detector=if_detector,
        )
        self.code_patch_class = code_patch_class
        self.extraction_block_class = extraction_block_class
        self.op_set_class = op_set_class
        self.code_params = code_params or {"distance": distance}
        self.distance = distance
        self.inject_state = inject_state.upper()
        self.protocol = protocol.lower()
        self.post_select_mode = post_select_mode.lower()

    def _compute_post_select_coords(self, patch, system) -> Set[Tuple[float, ...]]:
        """Compute post-selection detector coords based on mode."""
        all_syn_coords_2d = set()
        for stab in patch.stabilizers:
            syn_idx = stab.get("syn_idx")
            if syn_idx is not None and syn_idx in system.qubit_coords:
                coord = system.qubit_coords[syn_idx]
                all_syn_coords_2d.add((int(coord[0]), int(coord[1])))

        if self.post_select_mode == "full_qec":
            selected_2d = set()
        elif self.post_select_mode == "full_postselection":
            selected_2d = all_syn_coords_2d
        else:
            # hybrid: logical strip post-selection (rotated SC only)
            selected_2d = self._logical_strip_coords(all_syn_coords_2d)

        selected_2d = selected_2d & all_syn_coords_2d
        return {(float(x), float(y), 0.0) for x, y in selected_2d}

    def _logical_strip_coords(self, all_syn_coords_2d) -> Set[Tuple[int, int]]:
        """
        Logical-strip hybrid post-selection.
        Tag syndrome coords adjacent to the logical operator strips.

        Rotated SC (spacing=2):  corner Z → y∈{0,2}, X → x∈{0,2}
        Unrotated SC (spacing=1): corner Z → y∈{0,1}, X → x∈{0,1}
        Y state: union of Z and X strips.
        """
        from src.qec_code.surface_code.unrotated import UnrotatedSurfaceCode

        # Determine strip width based on code type
        if self.code_patch_class == RotatedSurfaceCode:
            strip_vals = (0, 2)  # rotated SC coord spacing = 2
        elif self.code_patch_class == UnrotatedSurfaceCode:
            strip_vals = (0, 1)  # unrotated SC coord spacing = 1
        else:
            raise NotImplementedError(
                f"Hybrid post-selection not implemented for {self.code_patch_class.__name__}. "
                f"Use 'full_postselection' or 'full_qec' instead."
            )

        if self.protocol == "corner":
            z_strip = {(x, y) for (x, y) in all_syn_coords_2d if y in strip_vals}
            x_strip = {(x, y) for (x, y) in all_syn_coords_2d if x in strip_vals}
            if self.inject_state == "Z":
                return z_strip
            elif self.inject_state == "X":
                return x_strip
            else:
                return z_strip | x_strip
        else:
            # Middle injection (rotated SC only)
            mid = self.distance // 2 + 1
            if self.code_patch_class == RotatedSurfaceCode:
                center = 2 * (mid - 1) + 1
            else:
                center = mid - 1  # unrotated uses unit spacing
            z_rows = (center - 1, center + 1)
            x_cols = (center - 1, center + 1)
            z_strip = {(x, y) for (x, y) in all_syn_coords_2d if y in z_rows}
            x_strip = {(x, y) for (x, y) in all_syn_coords_2d if x in x_cols}
            if self.inject_state == "Z":
                return z_strip
            elif self.inject_state == "X":
                return x_strip
            else:
                return z_strip | x_strip

    def build(self) -> stim.Circuit:
        """Constructs the full Stim circuit for the state injection experiment."""
        # 1. Create patch and register in QECSystem
        print(f"Creating {self.code_patch_class.__name__} patch...")
        patch_local = self.code_patch_class(**self.code_params)
        self.system = QECSystem()
        patch = self.system.add_patch(patch_local, name="patch")

        # 2. Setup tracker, builder, logical executor
        self._setup_experiment()
        op_set = self.op_set_class(extraction_block_class=self.extraction_block_class)
        self.logical_executor.register_op_set(self.code_patch_class, op_set)

        # 3. Write coordinates
        self.builder.write_coordinates()

        # 4. Compute post-selection coords based on mode
        ps_coords = self._compute_post_select_coords(patch, self.system)

        # 5. State injection (init + SE rounds)
        print(f"State injection: {self.inject_state} ({self.protocol}, {self.post_select_mode})...")
        self.logical_executor.apply_logical_operation(
            op_name="state_injection",
            patches=[patch],
            inject_state=self.inject_state,
            protocol=self.protocol,
            rounds=self.rounds,
            post_select_coords=ps_coords,
        )

        # 6. Readout
        #    Z/X: transversal MZ/MX (full final-round detector coverage)
        #    Y:   noiseless S_DAG → transversal MX if fold_transversal_s_dag available,
        #         otherwise fall back to unencode + noiseless MY
        if self.inject_state == "Y" and hasattr(op_set, "fold_transversal_s_dag"):
            # Noiseless S_DAG rotates |+i⟩→|+⟩, then transversal MX
            print("Noiseless S_DAG + transversal MX readout...")
            self.logical_executor.apply_logical_operation(
                op_name="fold_transversal_s_dag",
                patches=[patch],
                noiseless=True,
            )
            self.builder.apply_data_readout(
                {q: "X" for q in self.system.data_indices}
            )
        elif self.inject_state == "Y":
            # Fallback: unencode + noiseless MY (for codes without S_DAG)
            print("Logical unencode (Y fallback)...")
            phys_q = self.logical_executor.apply_logical_operation(
                op_name="logical_unencode",
                patches=[patch],
                inject_state=self.inject_state,
                protocol=self.protocol,
            )
            print("Noiseless MY...")
            self.builder.apply_data_readout(
                final_measurements={phys_q: "Y"}, noiseless=True,
            )
        else:
            measure_basis = self.inject_state
            print(f"Transversal M{measure_basis} readout...")
            self.builder.apply_data_readout(
                {q: measure_basis for q in self.system.data_indices}
            )

        # 7. Noise injection
        return self._inject_noise(self.builder.circuit)


class YMemoryExperiment(QECExperiment):
    """
    Y-state memory experiment with configurable SE block scheduling.

    Circuit structure:
        1. Product-state Y injection  (noiseless physical init)
        2. 1 noiseless SE round       (injection stabilization — always perpendicular)
        3. `rounds` noisy SE rounds   (configurable scheduling — the variable under test)
        4. Noiseless logical unencode
        5. Noiseless physical MY

    Because the Y logical is a ±1 eigenstate of Y_L = X_L · Z_L, it is
    sensitive to BOTH X and Z hook errors simultaneously, making it ideal
    for comparing SE schedulings that differ in hook-error orientation.
    """

    def __init__(
        self,
        distance: int = 3,
        rounds: int = 2,
        scheduling: str = 'perpendicular',
        protocol: Literal["corner", "middle"] = "corner",
        noise_params: Optional[NoiseConfig] = None,
        noise_model: Optional[str] = "circuit_level",
        if_detector: bool = True,
    ):
        """
        Args:
            distance: Code distance.
            rounds: Number of noisy SE rounds (memory phase only; injection round is separate).
            scheduling: SE block scheduling for memory rounds ('perpendicular' or 'swapped').
            protocol: Injection site ('corner' or 'middle').
            noise_params: Noise configuration (None = noiseless simulation).
            noise_model: Noise model string ('circuit_level', etc.).
            if_detector: Whether to emit DETECTOR instructions.
        """
        super().__init__(
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            rounds=rounds,
            noise_params=noise_params,
            noise_model=noise_model,
            if_detector=if_detector,
        )
        self.distance = distance
        self.scheduling = scheduling
        self.protocol = protocol

    def build(self) -> stim.Circuit:
        # 1. Create patch and system
        patch_local = RotatedSurfaceCode(distance=self.distance)
        self.system = QECSystem()
        patch = self.system.add_patch(patch_local, name="patch")

        self._setup_experiment()
        op_set = RotatedSurfaceCodeLogicalOpSet(
            extraction_block_class=RotatedSurfaceCodeExtractionBlock
        )
        self.logical_executor.register_op_set(RotatedSurfaceCode, op_set)

        self.builder.write_coordinates()

        # 2. Product-state Y injection (rounds=0 → init only, noiseless, no post-selection)
        self.logical_executor.apply_logical_operation(
            op_name="state_injection",
            patches=[patch],
            inject_state="Y",
            protocol=self.protocol,
            rounds=0,
            post_select_coords=set(),
            noiseless_init=True,
        )

        # 3. 1 noiseless SE round (injection stabilization)
        inj_se = RotatedSurfaceCodeExtractionBlock(self.system)
        self.builder.apply_syndrome_extraction(inj_se.circuit, rounds=1, noiseless=True)

        # 4. `rounds` noisy SE rounds (scheduling under test)
        mem_se = RotatedSurfaceCodeExtractionBlock(self.system, scheduling=self.scheduling)
        self.builder.apply_syndrome_extraction(mem_se.circuit, rounds=self.rounds)

        # 5. Noiseless unencode → corner qubit carries logical Y
        phys_q = self.logical_executor.apply_logical_operation(
            op_name="logical_unencode",
            patches=[patch],
            inject_state="Y",
            protocol=self.protocol,
        )

        # 6. Noiseless physical MY
        self.builder.apply_data_readout({phys_q: "Y"}, noiseless=True)

        return self._inject_noise(self.builder.circuit)
