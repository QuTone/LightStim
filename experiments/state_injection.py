# experiments/state_injection.py

"""
State Injection Experiment for Rotated Surface Code.

Implements corner and middle injection protocols to prepare logical |0⟩ or |+⟩
into a rotated surface code patch. The circuit construction follows the protocol
where data qubits are split by a diagonal pattern for initialization, with the
injection site (corner or center) receiving the target state.
"""

import stim
from typing import Type, Literal, Optional, Any, Tuple, List

from src.ir.builder import CircuitBuilder
from src.ir.tracker import SyndromeTracker
from src.ir.qec_system import QECSystem
from src.noise.config import NoiseConfig
from src.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock


def _logical_to_physical(coord: Tuple[int, int]) -> Tuple[int, int]:
    """Map logical coords (1..d) to physical coords used by RotatedSurfaceCode."""
    x, y = coord
    return (2 * (x - 1) + 1, 2 * (y - 1) + 1)


def _get_corner_injection_init(
    system: Any,
    inject_state: Literal["Z", "X"],
) -> dict:
    """
    Build init_dict for corner injection.
    Corner at (1,1). Lower diagonal (y>=x, excluding corner) -> |+⟩, upper (y<x) -> |0⟩.
    Injection site gets inject_state (Z->|0⟩, X->|+⟩).
    """
    data_coords = system.data_coords
    index_map = system.index_map
    corner = (1, 1)

    init_dict = {}
    for coord in data_coords:
        c = (int(coord[0]), int(coord[1]))
        if c == corner:
            init_dict[index_map[c]] = inject_state
        elif c[1] >= c[0]:
            init_dict[index_map[c]] = "X"  # lower diagonal -> |+⟩
        else:
            init_dict[index_map[c]] = "Z"  # upper diagonal -> |0⟩
    return init_dict


def _get_middle_injection_init(
    system: Any,
    inject_state: Literal["Z", "X"],
) -> dict:
    """
    Build init_dict for middle (center) injection.
    Injection at center (mid, mid) in logical coords.
    Split: zero_coords get |0⟩, plus_coords get |+⟩ per the middle-injection diagonal rule.
    """
    # Get distance from the patch (system may be QECSystem - get from first patch)
    patch = list(system.patches.values())[0][0]
    d = patch.distance_z  # assume square
    mid = d // 2 + 1
    injection_logical = (mid, mid)
    index_map = system.index_map

    # Map logical 1..d to physical. Our patch uses (2x-1, 2y-1) for logical (x,y)
    def phys(coord):
        x, y = coord
        return (2 * (x - 1) + 1, 2 * (y - 1) + 1)

    zero_coords_logical: List[Tuple[int, int]] = []
    plus_coords_logical: List[Tuple[int, int]] = []

    for x in range(1, d + 1):
        for y in range(1, d + 1):
            if (x, y) == injection_logical:
                continue
            if (x < y and x + y <= d + 1) or (x > y and x + y >= d + 1):
                zero_coords_logical.append((x, y))
            else:
                plus_coords_logical.append((x, y))

    zero_physical = [phys(c) for c in zero_coords_logical if phys(c) in index_map]
    plus_physical = [phys(c) for c in plus_coords_logical if phys(c) in index_map]
    injection_physical = phys(injection_logical)

    if injection_physical not in index_map:
        raise ValueError(
            f"Injection coordinate {injection_physical} not in layout. "
            f"Middle injection may require odd distance."
        )

    init_dict = {}
    for c in zero_physical:
        init_dict[index_map[c]] = "Z"
    for c in plus_physical:
        init_dict[index_map[c]] = "X"
    init_dict[index_map[injection_physical]] = inject_state
    return init_dict


class StateInjectionExperiment:
    """
    Orchestrates a State Injection Experiment for Rotated Surface Code.

    Prepares logical |0⟩ or |+⟩ via corner or middle injection protocol:
    1. Diagonal-split initialization of data qubits (|0⟩ and |+⟩ regions).
    2. Injection site receives the target state.
    3. Syndrome extraction rounds.
    4. Data qubit measurement (X or Z basis depending on inject_state).

    Circuit only - no noise, detectors, or observables are required for basic validation.
    """

    def __init__(
        self,
        distance: int = 3,
        rounds: int = 2,
        injection_protocol: Literal["corner", "middle"] = "corner",
        inject_state: Literal["Z", "X"] = "Z",
        extraction_block_class: Type = RotatedSurfaceCodeExtractionBlock,
        noise_params: Optional[NoiseConfig] = None,
        noise_model: Optional[str] = "circuit_level",
        if_detector: bool = True,
    ):
        """
        Args:
            distance: Code distance (odd integer).
            rounds: Number of syndrome extraction rounds.
            injection_protocol: 'corner' (inject at (1,1)) or 'middle' (inject at center).
            inject_state: Target logical state ('Z' -> |0⟩, 'X' -> |+⟩).
            extraction_block_class: SE block class (default RotatedSurfaceCodeExtractionBlock).
            noise_params: Optional noise configuration.
            noise_model: Noise model string.
            if_detector: Whether to generate detectors.
        """
        self.distance = distance
        self.rounds = rounds
        self.injection_protocol = injection_protocol.lower()
        self.inject_state = inject_state.upper()
        self.block_class = extraction_block_class
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.if_detector = if_detector

        if self.injection_protocol not in ("corner", "middle"):
            raise ValueError("injection_protocol must be 'corner' or 'middle'")
        if self.inject_state not in ("Z", "X"):
            raise ValueError("inject_state must be 'Z' or 'X'")

        self.system: Optional[Any] = None
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None

    def build(self) -> stim.Circuit:
        """Constructs the full Stim circuit for the state injection experiment."""
        # 1. Create patch and add to QECSystem (SE block expects QECSystem with active_syndrome_indices)
        patch = RotatedSurfaceCode(distance=self.distance)
        self.system = QECSystem()
        self.system.add_patch(patch, name="surface_code")

        # 2. Setup tracker and builder
        num_qubits = self.system.num_qubits
        num_logicals = self.system.num_logicals
        self.tracker = SyndromeTracker(
            num_qubits=num_qubits, expected_num_logicals=num_logicals
        )
        self.builder = CircuitBuilder(
            tracker=self.tracker,
            system_config=self.system,
            if_detector=self.if_detector,
        )

        # 3. Write coordinates
        self.builder.write_coordinates()

        # 4. Injection-specific initialization (system has global indices after add_patch)
        if self.injection_protocol == "corner":
            init_dict = _get_corner_injection_init(
                self.system, self.inject_state
            )
        else:
            init_dict = _get_middle_injection_init(
                self.system, self.inject_state
            )

        self.builder.initialize(init_dict=init_dict, n=num_qubits)

        # 5. Syndrome extraction rounds
        se_block = self.block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=self.rounds,
        )

        # 6. Final readout: measure in X basis if inject_state X, else Z
        data_indices = [self.system.index_map[c] for c in self.system.data_coords]
        final_measurements = {q: self.inject_state for q in data_indices}
        self.builder.apply_data_readout(final_measurements=final_measurements)

        # 7. Optional noise
        if self.noise_params is not None:
            return self.builder.build_noisy_circuit(
                noise_params=self.noise_params,
                noise_model=self.noise_model,
            )
        return self.builder.circuit
