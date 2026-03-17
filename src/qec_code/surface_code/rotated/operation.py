# src/qec_code/surface_code/rotated/operation.py

"""
Logical operation set for Rotated Surface Code.

Provides state_injection, logical_unencode, fold_transversal_hadamard,
and fold_transversal_s as LogicalOpSet methods, callable via
LogicalExecutor.apply_logical_operation().
"""

from typing import List, Literal, Tuple, Dict, Any, Type

from src.ir.operation import CSSLogicalOpSet
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
# Fold-transversal helpers
# ------------------------------------------------------------------------------

def _get_rotation_cycles(system: Any, patch: QECPatch) -> List[List[int]]:
    """
    Compute the cycle decomposition of the 90° anticlockwise rotation
    (lx, ly) → (2d − ly, lx) acting on data qubits of a square patch.

    Returns:
        cycles: list of lists of global qubit indices; fixed-point (length-1)
                cycles are omitted.

    Raises:
        ValueError if the patch is non-square.
    """
    if patch.distance_z != patch.distance_x:
        raise ValueError(
            "Fold-transversal H requires a square patch "
            f"(distance_z == distance_x). Got {patch.distance_z} != {patch.distance_x}."
        )

    d = patch.distance_z
    sx, sy = patch.shift
    y_max = 2 * d          # rotation: (lx, ly) → (y_max − ly, lx)
    index_map = system.index_map

    visited: set = set()
    cycles: List[List[int]] = []

    for coord in patch.data_coords:
        global_idx = index_map[coord]
        if global_idx in visited:
            continue

        lx = round(coord[0] - sx)
        ly = round(coord[1] - sy)
        cycle: List[int] = []
        cur_lx, cur_ly = lx, ly

        while True:
            cur_global = QECPatch.snap_coord((cur_lx + sx, cur_ly + sy))
            cur_idx = index_map[cur_global]
            if cur_idx in visited:
                break
            cycle.append(cur_idx)
            visited.add(cur_idx)
            cur_lx, cur_ly = y_max - cur_ly, cur_lx   # 90° anticlockwise

        if len(cycle) > 1:
            cycles.append(cycle)

    return cycles


def _get_fold_s_pairs(
    system: Any, patch: QECPatch
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Partition ALL qubits of the patch (data + syndrome) by the y=x diagonal
    reflection, for use in the fold-transversal S gate at the half-cycle
    unrotated state.

    Even-row diagonal qubits (local y even) receive S.
    Odd-row diagonal qubits  (local y odd)  receive S_DAG.
    Mirror pairs ((lx,ly) ↔ (ly,lx) with lx < ly) receive CZ.
    Qubits with no mirror partner (asymmetric boundary) receive no operation.

    Returns:
        even_diag   : global indices of diagonal qubits with even local y → S
        odd_diag    : global indices of diagonal qubits with odd  local y → S_DAG
        mirror_pairs: list of (idx_a, idx_b) where local coord_a = (lx, ly),
                      lx < ly, and coord_b = (ly, lx)

    Raises:
        ValueError if the patch is non-square.
    """
    if patch.distance_z != patch.distance_x:
        raise ValueError(
            "Fold-transversal S requires a square patch "
            f"(distance_z == distance_x). Got {patch.distance_z} != {patch.distance_x}."
        )

    sx, sy = patch.shift
    index_map = system.index_map

    even_diag: List[int] = []
    odd_diag:  List[int] = []
    mirror_pairs: List[Tuple[int, int]] = []

    for local_idx, local_coord in patch.qubit_coords.items():
        lx = round(local_coord[0])
        ly = round(local_coord[1])
        global_coord = QECPatch.snap_coord((lx + sx, ly + sy))
        if global_coord not in index_map:
            continue
        global_idx = index_map[global_coord]

        if lx == ly:
            if ly % 2 == 0:
                even_diag.append(global_idx)
            else:
                odd_diag.append(global_idx)
        elif lx < ly:
            mirror_global = QECPatch.snap_coord((ly + sx, lx + sy))
            if mirror_global in index_map:
                mirror_pairs.append((global_idx, index_map[mirror_global]))

    return even_diag, odd_diag, mirror_pairs


# ------------------------------------------------------------------------------
# LogicalOpSet implementation
# ------------------------------------------------------------------------------

class RotatedSurfaceCodeLogicalOpSet(CSSLogicalOpSet):
    """
    Logical operation set for Rotated Surface Code.

    Implements state injection, logical unencode, and surface-code-specific
    gates as composable operations for use with LogicalExecutor.
    """

    def __init__(self, extraction_block_class: Type = None):
        """
        Args:
            extraction_block_class: SE block class for this code
                (e.g. RotatedSurfaceCodeExtractionBlock). Takes system, has .circuit.
                Required for state_injection; optional otherwise.
        """
        super().__init__()
        self.name = "RotatedSurfaceCode"
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
    # Fold-transversal gates
    # ------------------------------------------------------------------

    def fold_transversal_hadamard(self, builder: CircuitBuilder, patch: QECPatch):
        """
        Implement the logical Hadamard H_L via fold-transversal gate.

        Physical circuit:
          Layer 1 — transversal H : H applied to every data qubit
          Layers 2-N — rotation   : SWAP gates implementing the 90° anticlockwise
                                    permutation (lx, ly) → (2d − ly, lx) on data qubits

        The rotation decomposes into permutation cycles (two 4-cycles + one fixed
        point for square d). Each k-cycle needs k−1 sequential SWAPs; cycles are
        parallelised across TICKs.

        Logical action: X_L ↔ Z_L

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The RotatedSurfaceCode patch (must be square).
        """
        system = builder.system
        cycles = _get_rotation_cycles(system, patch)
        all_data = sorted(patch.data_indices)

        unitary = stim.Circuit()
        unitary.append("H", all_data)

        if cycles:
            max_steps = max(len(c) - 1 for c in cycles)
            for step in range(max_steps):
                flat = []
                for cycle in cycles:
                    if step >= len(cycle) - 1:
                        continue
                    if step == 0:
                        flat.extend([cycle[0], cycle[-1]])
                    else:
                        flat.extend([cycle[-step], cycle[-step - 1]])
                unitary.append("TICK")
                if flat:
                    unitary.append("SWAP", flat)

        builder.apply_unitary_block(unitary)

    def fold_transversal_s(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        se_block=None,
    ):
        """
        Implement the logical phase gate S_L by embedding the fold-transversal
        operation inside a single syndrome-extraction round.

        The rotated surface code morphs into an unrotated surface code at the
        half-cycle point (after CNOT ticks 1-2). The fold-transversal S is
        applied there, then the SE round is completed (CNOT ticks 3-4 + measure).

        Physical circuit (one modified SE round):
          [SE first half]  Reset + H_x + CNOT ticks 1-2
          [fold-S]         S on even-diagonal, S_DAG on odd-diagonal, CZ on pairs
          [SE second half] CNOT ticks 3-4 + H_x + Measure

        Diagonal / mirror pairs are defined on ALL qubits (data + syndrome) of
        the patch, using the y=x fold of the half-cycle unrotated layout.

        Logical action: Z_L → Z_L,  X_L → Y_L

        Args:
            builder:  CircuitBuilder driving the experiment.
            patch:    The RotatedSurfaceCode patch (must be square).
            se_block: RotatedSurfaceCodeExtractionBlock instance (required).
        """
        if se_block is None:
            raise ValueError(
                "fold_transversal_s requires se_block "
                "(RotatedSurfaceCodeExtractionBlock) as a keyword argument."
            )

        system = builder.system
        even_diag, odd_diag, mirror_pairs = _get_fold_s_pairs(system, patch)

        fold_circ = stim.Circuit()
        fold_circ.append("TICK")
        if even_diag:
            fold_circ.append("S",     sorted(even_diag))
        if odd_diag:
            fold_circ.append("S_DAG", sorted(odd_diag))
        for a, b in mirror_pairs:
            fold_circ.append("CZ", [a, b])

        modified_se = se_block.first_half + fold_circ + se_block.second_half
        builder.apply_syndrome_extraction(modified_se)
