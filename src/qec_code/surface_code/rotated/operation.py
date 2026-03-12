# src/qec_code/surface_code/rotated/operation.py

"""
Logical operation set for Rotated Surface Code.

Provides state_injection and logical_unencode as LogicalOpSet methods,
callable via LogicalExecutor.apply_logical_operation().
"""

from typing import Literal, List, Tuple, Dict, Any, Optional

from src.ir.operation import LogicalOpSet
from src.ir.builder import CircuitBuilder
from src.ir.qec_patch import QECPatch
import stim


# ------------------------------------------------------------------------------
# Module-level helpers (moved from experiments/state_injection.py)
# ------------------------------------------------------------------------------

def _logical_to_physical(coord: Tuple[int, int]) -> Tuple[int, int]:
    """Map logical coords (1..d) to physical coords used by RotatedSurfaceCode."""
    x, y = coord
    return (2 * (x - 1) + 1, 2 * (y - 1) + 1)


def _get_corner_injection_init(
    system: Any,
    inject_state: Literal["Z", "X", "Y"],
) -> Dict[int, str]:
    """
    Build init_dict for corner injection.

    Corner at (1,1). Lower diagonal (y>=x, excluding corner) -> |+>, upper (y<x) -> |0>.
    Injection site gets inject_state (Z->|0>, X->|+>, Y->|+i>).

    For Y injection the surrounding split mirrors X injection (lower -> X, upper -> Z),
    since |+i> = S|+> and the post-selection round handles stochastic stabilizer outcomes.

    Returns:
        Dict mapping global qubit index -> basis string ('Z', 'X', or 'Y').
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
            init_dict[index_map[c]] = "X"  # lower diagonal -> |+>
        else:
            init_dict[index_map[c]] = "Z"  # upper diagonal -> |0>
    return init_dict


def _get_middle_injection_init(
    system: Any,
    inject_state: Literal["Z", "X", "Y"],
) -> Dict[int, str]:
    """
    Build init_dict for middle (center) injection.

    Injection at center (mid, mid) in logical coords.
    Split: zero_coords get |0>, plus_coords get |+> per the middle-injection diagonal rule.

    Returns:
        Dict mapping global qubit index -> basis string ('Z' or 'X').
    """
    patch = list(system.patches.values())[0][0]
    d = patch.distance_z  # assume square
    mid = d // 2 + 1
    injection_logical = (mid, mid)
    index_map = system.index_map

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

    zero_physical = [_logical_to_physical(c) for c in zero_coords_logical if _logical_to_physical(c) in index_map]
    plus_physical = [_logical_to_physical(c) for c in plus_coords_logical if _logical_to_physical(c) in index_map]
    injection_physical = _logical_to_physical(injection_logical)

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


# ------------------------------------------------------------------------------
# LogicalOpSet implementation
# ------------------------------------------------------------------------------

class RotatedSurfaceCodeLogicalOpSet(LogicalOpSet):
    """
    Logical operation set for Rotated Surface Code.

    Implements state injection, logical unencode, and surface-code-specific
    gates as composable operations for use with LogicalExecutor.
    """

    def __init__(self):
        super().__init__("RotatedSurfaceCode")

    # ------------------------------------------------------------------
    # State preparation / teardown
    # ------------------------------------------------------------------

    def state_injection(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner", "middle"] = "corner",
    ):
        """
        Initialize data qubits for state injection and tag syndrome detectors
        for post-selection.

        Must be called before apply_syndrome_extraction so that
        tracker.post_select_detector_coords is set in time.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: The QECPatch (used for type dispatch by LogicalExecutor).
            inject_state: Target logical state ('Z' -> |0>, 'X' -> |+>, 'Y' -> |+i>).
            protocol: Injection site — 'corner' (1,1) or 'middle' (center qubit).
        """
        system = builder.system

        # Build qubit-initialization dict (global indices)
        if protocol == "corner":
            init_dict = _get_corner_injection_init(system, inject_state)
        else:
            init_dict = _get_middle_injection_init(system, inject_state)

        # Tag all syndrome qubit coords for post-selection in the tracker.
        # Must be set before apply_syndrome_extraction is called.
        post_select_coords = {
            tuple(system.qubit_coords[s["syn_idx"]]) + (0.0,)
            for s in system.stabilizers
        }
        builder.tracker.post_select_detector_coords = post_select_coords

        # Emit reset instructions and update the tracker tableau
        builder.initialize(init_dict=init_dict, n=system.num_qubits)

    def logical_unencode(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
    ):
        """
        Measure all data qubits of the patch in the injection basis
        (corner-shrink unencode / final readout).

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: The QECPatch (used for type dispatch by LogicalExecutor).
            inject_state: Basis in which to measure ('Z', 'X', or 'Y').
        """
        system = builder.system

        data_indices = [system.index_map[c] for c in system.data_coords]
        final_measurements = {q: inject_state for q in data_indices}
        builder.apply_data_readout(final_measurements=final_measurements)

    def logical_shrink(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
    ) -> int:
        """
        Corner-shrink: measure all data qubits except the corner (1,1) in their
        initialization basis (MX for lower diagonal, M for upper diagonal),
        and emit data-only DETECTOR instructions for pure-region stabilizers.

        Lower diagonal (y >= x, excluding corner): measured with MX.
        Upper diagonal (y < x): measured with M.
        Corner qubit (1,1): NOT measured — carries the logical state forward.

        Detectors are data-only (no ancilla history needed):
        - Z-stabs with ancilla ax > ay (pure upper-diagonal support)
        - X-stabs with ancilla ax < ay (pure lower-diagonal support)

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: The QECPatch (for LogicalExecutor dispatch).
            inject_state: Reserved for API consistency; measurement pattern is
                          invariant for corner injection.

        Returns:
            Global qubit index of the corner qubit (1,1).
        """
        system = builder.system
        corner_physical = (1, 1)

        lower_qs: List[int] = []   # y >= x, not corner -> MX
        upper_qs: List[int] = []   # y < x             -> M
        corner_q: Optional[int] = None

        for coord in system.data_coords:
            c = (int(coord[0]), int(coord[1]))
            gidx = system.index_map[coord]
            if c == corner_physical:
                corner_q = gidx
            elif c[1] >= c[0]:
                lower_qs.append(gidx)
            else:
                upper_qs.append(gidx)

        if corner_q is None:
            raise ValueError(
                "Corner qubit (1,1) not found in data layout. "
                "logical_shrink requires a corner-injection rotated surface code."
            )

        n_lower = len(lower_qs)
        n_upper = len(upper_qs)
        total_shrunk = n_lower + n_upper

        # Build: qubit global index -> relative stim record offset
        # After appending MX(lower_qs) then M(upper_qs), records are:
        #   lower_qs[i] -> rec(i - total_shrunk)
        #   upper_qs[j] -> rec(n_lower + j - total_shrunk)
        q_to_rec: Dict[int, int] = {}
        for i, q in enumerate(lower_qs):
            q_to_rec[q] = i - total_shrunk
        for j, q in enumerate(upper_qs):
            q_to_rec[q] = n_lower + j - total_shrunk

        # Emit measurement instructions
        if lower_qs:
            builder.circuit.append("MX", lower_qs)
        if upper_qs:
            builder.circuit.append("M", upper_qs)

        # Emit data-only DETECTOR instructions for pure-region stabilizers
        if builder.if_detector:
            for stab in system.stabilizers:
                syn_idx = stab.get("syn_idx")
                if syn_idx is None:
                    continue
                ax, ay = system.qubit_coords[syn_idx]
                stab_type = stab.get("type")

                # Pure-region: Z-stab fully in upper diagonal, X-stab fully in lower
                is_pure = (stab_type == "Z" and ax > ay) or (stab_type == "X" and ax < ay)
                if not is_pure:
                    continue

                data_indices = stab.get("data_indices", [])
                # Skip if any data qubit is not in the shrunk set (e.g. corner)
                if any(q not in q_to_rec for q in data_indices):
                    continue

                args = [stim.target_rec(q_to_rec[q]) for q in data_indices]
                builder.circuit.append("DETECTOR", args, [ax, ay, 0.0])

        # Sync tracker measurement count (no tableau update needed for final measurements)
        builder.tracker.total_measurements += total_shrunk

        return corner_q

    # ------------------------------------------------------------------
    # Gate operations (stubs for future implementation)
    # ------------------------------------------------------------------

    def transversal_Hadamard(self, patch: QECPatch) -> stim.Circuit:
        """
        Applies a fold-transversal Hadamard gate using H-SWAP gates.
        """
        pass

    def LS_Hadamard(self, patch: QECPatch) -> stim.Circuit:
        """
        Applies a Hadamard gate using transversal H with patch rotation.
        """
        pass
