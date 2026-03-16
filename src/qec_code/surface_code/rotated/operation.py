# src/qec_code/surface_code/rotated/operation.py

"""
Logical operation set for Rotated Surface Code.

Provides state_injection and logical_unencode as LogicalOpSet methods,
callable via LogicalExecutor.apply_logical_operation().
"""

from typing import Literal, Tuple, Dict, Any, Type

from src.ir.operation import LogicalOpSet
from src.ir.builder import CircuitBuilder
from src.ir.qec_patch import QECPatch
from src.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
import stim


# ------------------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------------------

def _get_injection_site(
    patch: QECPatch,
    system: Any,
    protocol: Literal["corner", "middle"],
) -> Tuple[Tuple[float, float], int]:
    """
    Return the (coord, global_index) of the injection site for the given protocol.

    Args:
        patch: Global patch (data_indices are global).
        system: QECSystem for coord lookup.
        protocol: 'corner' or 'middle'.

    Returns:
        (injection_coord, injection_global_index)
    """
    data_global_indices = sorted(patch.data_indices)
    coords = [(system.qubit_coords[gidx], gidx) for gidx in data_global_indices]

    if protocol == "corner":
        # Corner = qubit with smallest (x, y)
        corner_coord, corner_gidx = min(coords, key=lambda t: (t[0][0], t[0][1]))
        return corner_coord, corner_gidx
    else:
        # Middle = center qubit in logical grid
        d = patch.distance_z
        mid = d // 2 + 1
        # Find the corner to compute relative offsets
        corner_coord = min(c for c, _ in coords)
        # In rotated surface code, data qubits are at odd coords with spacing 2
        # Logical (mid, mid) maps to corner + (2*(mid-1), 2*(mid-1))
        mid_coord = (corner_coord[0] + 2 * (mid - 1), corner_coord[1] + 2 * (mid - 1))
        for c, gidx in coords:
            if abs(c[0] - mid_coord[0]) < 0.5 and abs(c[1] - mid_coord[1]) < 0.5:
                return c, gidx
        raise ValueError(
            f"Middle injection coordinate {mid_coord} not found in patch layout. "
            f"Middle injection may require odd distance."
        )


def _get_injection_init(
    patch: QECPatch,
    system: Any,
    inject_state: Literal["Z", "X", "Y"],
    protocol: Literal["corner", "middle"],
) -> Tuple[Dict[int, str], int]:
    """
    Build init_dict for state injection.

    For corner protocol:
        Corner qubit gets inject_state. Lower diagonal (rel_y >= rel_x) -> X, upper -> Z.
    For middle protocol:
        Center qubit gets inject_state. Diagonal split per middle-injection rule.

    Args:
        patch: Global patch (data_indices are global).
        system: QECSystem for coord lookup.
        inject_state: Target state ('Z', 'X', or 'Y').
        protocol: 'corner' or 'middle'.

    Returns:
        (init_dict, injection_global_index)
    """
    injection_coord, injection_gidx = _get_injection_site(patch, system, protocol)
    data_global_indices = sorted(patch.data_indices)

    init_dict: Dict[int, str] = {}

    if protocol == "corner":
        # Use injection site as origin for relative coordinates
        ox, oy = injection_coord
        for gidx in data_global_indices:
            cx, cy = system.qubit_coords[gidx]
            rel_x, rel_y = cx - ox, cy - oy
            if gidx == injection_gidx:
                init_dict[gidx] = inject_state
            elif rel_y >= rel_x:
                init_dict[gidx] = "X"   # lower diagonal -> |+>
            else:
                init_dict[gidx] = "Z"   # upper diagonal -> |0>
    else:
        # Middle injection: diagonal split based on relative position
        d = patch.distance_z
        corner_coord = min(
            (system.qubit_coords[gidx] for gidx in data_global_indices),
            key=lambda c: (c[0], c[1]),
        )
        ox, oy = corner_coord

        for gidx in data_global_indices:
            cx, cy = system.qubit_coords[gidx]
            # Convert to logical coords (1..d)
            lx = int(round((cx - ox) / 2)) + 1
            ly = int(round((cy - oy) / 2)) + 1

            if gidx == injection_gidx:
                init_dict[gidx] = inject_state
            elif (lx < ly and lx + ly <= d + 1) or (lx > ly and lx + ly >= d + 1):
                init_dict[gidx] = "Z"
            else:
                init_dict[gidx] = "X"

    return init_dict, injection_gidx


# ------------------------------------------------------------------------------
# LogicalOpSet implementation
# ------------------------------------------------------------------------------

class RotatedSurfaceCodeLogicalOpSet(LogicalOpSet):
    """
    Logical operation set for Rotated Surface Code.

    Implements state injection, logical unencode, and surface-code-specific
    gates as composable operations for use with LogicalExecutor.
    """

    def __init__(self, extraction_block_class: Type):
        """
        Args:
            extraction_block_class: SE block class for this code
                (e.g. RotatedSurfaceCodeExtractionBlock). Takes system, has .circuit.
        """
        super().__init__("RotatedSurfaceCode")
        self.extraction_block_class = extraction_block_class

    # ------------------------------------------------------------------
    # State preparation / teardown
    # ------------------------------------------------------------------

    def state_injection(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner", "middle"] = "corner",
        rounds: int = 0,
        post_select_coords=None,
    ):
        """
        State injection: initialize data qubits and optionally run SE rounds.

        The injection site receives the target state; surrounding qubits are initialized
        in a diagonal pattern (X/Z split). Syndrome detectors are tagged for post-selection.

        When rounds > 0, syndrome extraction is performed immediately (convenient for
        single-patch experiments). For multi-patch experiments where all patches must be
        initialized before SE, call with rounds=0 and run SE at the experiment level.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global QECPatch (returned by system.add_patch).
            inject_state: Target logical state ('Z' -> |0>, 'X' -> |+>, 'Y' -> |+i>).
            protocol: Injection site — 'corner' or 'middle'.
            rounds: Number of SE rounds to run after init (0 = init only).
            post_select_coords: Set of (x, y, t) detector coords to tag for post-selection.
                If None (default), tag ALL syndrome coords (full post-selection).
                Pass empty set() for no post-selection (full QEC).
        """
        if not isinstance(patch, RotatedSurfaceCode):
            raise TypeError(
                f"Expected RotatedSurfaceCode patch, got {type(patch).__name__}"
            )

        system = builder.system

        # Build qubit-initialization dict (global indices)
        init_dict, _ = _get_injection_init(patch, system, inject_state, protocol)

        # Tag syndrome coords for post-selection
        if post_select_coords is None:
            # Default: tag all syndrome coords (full post-selection)
            post_select_coords = set()
            for stab in patch.stabilizers:
                syn_idx = stab.get("syn_idx")
                if syn_idx is not None and syn_idx in system.qubit_coords:
                    coord = system.qubit_coords[syn_idx]
                    post_select_coords.add(tuple(coord) + (0.0,))
        builder.tracker.post_select_detector_coords |= post_select_coords

        # Emit reset instructions and update the tracker tableau
        builder.initialize(init_dict=init_dict, n=system.num_qubits)

        # Optional syndrome extraction rounds
        if rounds > 0:
            se_block = self.extraction_block_class(system)
            builder.apply_syndrome_extraction(
                circuit_chunk=se_block.circuit,
                rounds=rounds,
            )

    def logical_unencode(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner", "middle"] = "corner",
    ) -> int:
        """
        Unencode the logical state back to a single physical qubit (inverse of injection).

        Measures all data qubits of the patch EXCEPT the injection site in their
        initialization basis. The tracker automatically generates final-round detectors.
        The injection site qubit is left unmeasured, carrying the logical state.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global QECPatch (returned by system.add_patch).
            inject_state: The state that was injected ('Z', 'X', or 'Y').
            protocol: Must match the protocol used in state_injection.

        Returns:
            Global qubit index of the unmeasured injection-site qubit.
        """
        if not isinstance(patch, RotatedSurfaceCode):
            raise TypeError(
                f"Expected RotatedSurfaceCode patch, got {type(patch).__name__}"
            )

        system = builder.system

        # Reconstruct the init dict to know each qubit's basis
        init_dict, injection_gidx = _get_injection_init(
            patch, system, inject_state, protocol
        )

        # Measure all data qubits except the injection site, in their init basis
        unencode_measurements = {
            gidx: basis for gidx, basis in init_dict.items()
            if gidx != injection_gidx
        }

        # Use apply_data_readout for automatic detector/observable generation.
        # Noiseless: unencode measurements are deterministic given the stabilizer
        # state — noise here would corrupt the injection fidelity measurement.
        builder.apply_data_readout(final_measurements=unencode_measurements, noiseless=True)

        return injection_gidx

    # ------------------------------------------------------------------
    # Gate operations (stubs for future implementation)
    # ------------------------------------------------------------------

    def transversal_Hadamard(self, builder: CircuitBuilder, patch: QECPatch) -> stim.Circuit:
        """
        Applies a fold-transversal Hadamard gate using H-SWAP gates.
        """
        pass

    def LS_Hadamard(self, builder: CircuitBuilder, patch: QECPatch) -> stim.Circuit:
        """
        Applies a Hadamard gate using transversal H with patch rotation.
        """
        pass
