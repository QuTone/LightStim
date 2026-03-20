from typing import List, Tuple, Dict, Any, Literal, Type

import stim

from src.ir.operation import CSSLogicalOpSet
from src.ir.builder import CircuitBuilder
from src.ir.qec_patch import QECPatch
from src.qec_code.surface_code.unrotated.code_patch import UnrotatedSurfaceCode


# ------------------------------------------------------------------------------
# Module-level helpers — fold-transversal gates
# ------------------------------------------------------------------------------

def _get_fold_yx_pairs(
    system: Any, patch: QECPatch,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Partition data qubits of a single patch by the y=x diagonal reflection.

    Uses the patch's built-in shift to convert global coords back to local
    coords for the diagonal check, so this works correctly in multi-patch
    systems where each patch has been created with an appropriate shift.

    Returns:
        diag_s_qs    : global qubit indices on diagonal that receive S gate
        diag_sdag_qs : global qubit indices on diagonal that receive S† gate
        mirror_pairs : list of (idx_a, idx_b) where local coord_a=(x,y) with
                       x < y and coord_b is the reflected qubit at local (y, x)

    The S/S† alternation follows the fold-transversal S protocol (ref: arxiv
    2406.17653v1 Extended Data Fig. 1c): diagonal qubits alternate S and S†
    based on their position index along the diagonal.

    Raises:
        ValueError if the patch is non-square or a reflected qubit is missing.
    """
    if patch.distance_z != patch.distance_x:
        raise ValueError(
            "Fold-transversal gates require a square patch "
            f"(distance_z == distance_x). Got {patch.distance_z} != {patch.distance_x}."
        )

    sx, sy = patch.shift          # shift baked in at build() time
    index_map = system.index_map  # global coord → global index

    diagonal_with_local: List[Tuple[int, int]] = []  # (local_coord, global_idx)
    mirror_pairs: List[Tuple[int, int]] = []

    for coord in patch.data_coords:      # global coords (already shifted)
        lx = round(coord[0] - sx)
        ly = round(coord[1] - sy)

        if lx == ly:
            diagonal_with_local.append((lx, index_map[coord]))
        elif lx < ly:
            # reflected partner in global coords
            refl = QECPatch.snap_coord((ly + sx, lx + sy))
            if refl not in index_map:
                raise ValueError(
                    f"Mirror qubit at {refl} not found. "
                    "Ensure the patch is square and data qubits have y=x symmetry."
                )
            mirror_pairs.append((index_map[coord], index_map[refl]))

    # Sort by position along diagonal, then alternate S / S†
    diagonal_with_local.sort(key=lambda t: t[0])
    diag_s_qs: List[int] = []
    diag_sdag_qs: List[int] = []
    for i, (_, gidx) in enumerate(diagonal_with_local):
        if i % 2 == 0:
            diag_s_qs.append(gidx)
        else:
            diag_sdag_qs.append(gidx)

    return diag_s_qs, diag_sdag_qs, mirror_pairs


# ------------------------------------------------------------------------------
# Module-level helpers — corner injection
# ------------------------------------------------------------------------------

def _get_corner_injection_init(
    patch: QECPatch,
    system: Any,
    inject_state: Literal["Z", "X", "Y"],
) -> Tuple[Dict[int, str], int]:
    """
    Build init_dict for corner injection on an unrotated surface code patch.

    Corner = data qubit with smallest (x, y).
    Diagonal split: rel_y >= rel_x -> X (|+>), rel_y < rel_x -> Z (|0>).
    Corner qubit gets inject_state.

    Returns:
        (init_dict, injection_global_index)
    """
    data_global_indices = sorted(patch.data_indices)
    coords = [(system.qubit_coords[gidx], gidx) for gidx in data_global_indices]

    # Corner = min (x, y)
    corner_coord, corner_gidx = min(coords, key=lambda t: (t[0][0], t[0][1]))
    ox, oy = corner_coord

    init_dict: Dict[int, str] = {}
    for gidx in data_global_indices:
        cx, cy = system.qubit_coords[gidx]
        rel_x, rel_y = cx - ox, cy - oy
        if gidx == corner_gidx:
            init_dict[gidx] = inject_state
        elif rel_y >= rel_x:
            init_dict[gidx] = "X"   # lower diagonal -> |+>
        else:
            init_dict[gidx] = "Z"   # upper diagonal -> |0>

    return init_dict, corner_gidx


# ------------------------------------------------------------------------------
# LogicalOpSet implementation
# ------------------------------------------------------------------------------

class UnrotatedSurfaceCodeLogicalOpSet(CSSLogicalOpSet):
    """
    Logical operation set for the Unrotated Surface Code.

    Inherits transversal_cnot from CSSLogicalOpSet and adds:
    - fold-transversal H and S gates (y=x diagonal reflection symmetry)
    - state_injection and logical_unencode (corner injection protocol)

    Available via LogicalExecutor:
        executor.apply_logical_operation("fold_transversal_hadamard", [patch])
        executor.apply_logical_operation("fold_transversal_s",        [patch])
        executor.apply_logical_operation("transversal_cnot", [ctrl, tgt])
        executor.apply_logical_operation("state_injection", [patch], ...)
        executor.apply_logical_operation("logical_unencode", [patch], ...)
    """

    def __init__(self, extraction_block_class: Type = None):
        super().__init__()
        self.name = "UnrotatedSurfaceCode"
        self.extraction_block_class = extraction_block_class

    # ------------------------------------------------------------------
    # Fold-transversal gates
    # ------------------------------------------------------------------

    def fold_transversal_hadamard(self, builder: CircuitBuilder, patch: QECPatch,
                                   noiseless: bool = False):
        """
        Implement the logical Hadamard H_L via fold-transversal gate.

        Physical circuit (two layers):
          Layer 1 — transversal H : H applied to every data qubit (X <-> Z)
          Layer 2 — fold (SWAP)   : SWAP every mirror pair (x,y) <-> (y,x)

        Logical action: X_L <-> Z_L

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The UnrotatedSurfaceCode patch (must be square).
            noiseless: If True, tag gate instructions as noiseless.
        """
        system = builder.system
        diag_s, diag_sdag, mirror_pairs = _get_fold_yx_pairs(system, patch)
        all_data = sorted(system.index_map[c] for c in patch.data_coords)

        unitary = stim.Circuit()
        unitary.append("H", all_data)
        if mirror_pairs:
            unitary.append("TICK")
            flat = [q for a, b in mirror_pairs for q in (a, b)]
            unitary.append("SWAP", flat)

        builder.apply_unitary_block(unitary, noiseless=noiseless)

    def fold_transversal_s(self, builder: CircuitBuilder, patch: QECPatch,
                            noiseless: bool = False):
        """
        Implement the logical phase gate S_L via fold-transversal gate.

        Physical circuit (single layer — S/S† and CZ act on disjoint qubits):
          S   on even-indexed diagonal data qubits
          S†  on odd-indexed diagonal data qubits
          CZ  on every mirror pair  {(x,y) <-> (y,x) : local x < y}

        The S/S† alternation follows arxiv 2406.17653v1 Extended Data Fig. 1c.

        Logical action: Z_L -> Z_L,  X_L -> Y_L

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The UnrotatedSurfaceCode patch (must be square).
            noiseless: If True, tag gate instructions as noiseless.
        """
        system = builder.system
        diag_s, diag_sdag, mirror_pairs = _get_fold_yx_pairs(system, patch)

        unitary = stim.Circuit()
        if diag_s:
            unitary.append("S", sorted(diag_s))
        if diag_sdag:
            unitary.append("S_DAG", sorted(diag_sdag))
        for a, b in mirror_pairs:
            unitary.append("CZ", [a, b])

        builder.apply_unitary_block(unitary, noiseless=noiseless)

    def fold_transversal_s_dag(self, builder: CircuitBuilder, patch: QECPatch,
                               noiseless: bool = False):
        """
        Implement the logical S†_L via fold-transversal gate (inverse of S_L).

        Physical circuit (single layer):
          S†  on even-indexed diagonal data qubits  (swapped vs fold_transversal_s)
          S   on odd-indexed diagonal data qubits   (swapped vs fold_transversal_s)
          CZ  on every mirror pair  (same; CZ is self-inverse)

        Logical action: Z_L -> Z_L,  Y_L -> X_L

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The UnrotatedSurfaceCode patch (must be square).
            noiseless: If True, tag gate instructions as noiseless.
        """
        system = builder.system
        diag_s, diag_sdag, mirror_pairs = _get_fold_yx_pairs(system, patch)

        # Swap S <-> S_DAG compared to fold_transversal_s
        unitary = stim.Circuit()
        if diag_s:
            unitary.append("S_DAG", sorted(diag_s))
        if diag_sdag:
            unitary.append("S", sorted(diag_sdag))
        for a, b in mirror_pairs:
            unitary.append("CZ", [a, b])

        builder.apply_unitary_block(unitary, noiseless=noiseless)

    # ------------------------------------------------------------------
    # State preparation / teardown
    # ------------------------------------------------------------------

    def state_injection(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner"] = "corner",
        rounds: int = 0,
        post_select_coords=None,
    ):
        """
        State injection: initialize data qubits and optionally run SE rounds.

        Corner injection: the corner qubit (min x, y) receives the target state;
        surrounding qubits are initialized in a diagonal X/Z split.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global QECPatch (returned by system.add_patch).
            inject_state: Target logical state ('Z' -> |0>, 'X' -> |+>, 'Y' -> |+i>).
            protocol: Injection site — only 'corner' is supported for unrotated SC.
            rounds: Number of SE rounds to run after init (0 = init only).
            post_select_coords: Set of (x, y, t) detector coords to tag for post-selection.
                If None (default), tag ALL syndrome coords (full post-selection).
                Pass empty set() for no post-selection (full QEC).
        """
        if not isinstance(patch, UnrotatedSurfaceCode):
            raise TypeError(
                f"Expected UnrotatedSurfaceCode patch, got {type(patch).__name__}"
            )
        if protocol != "corner":
            raise NotImplementedError(
                f"Only 'corner' protocol is supported for unrotated SC, got '{protocol}'"
            )

        system = builder.system

        # Build qubit-initialization dict (global indices)
        init_dict, _ = _get_corner_injection_init(patch, system, inject_state)

        # Tag syndrome coords for post-selection
        if post_select_coords is None:
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
            if self.extraction_block_class is None:
                raise ValueError(
                    "extraction_block_class must be set to run SE rounds. "
                    "Pass it to UnrotatedSurfaceCodeLogicalOpSet(extraction_block_class=...)."
                )
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
        protocol: Literal["corner"] = "corner",
    ) -> int:
        """
        Unencode the logical state back to a single physical qubit (inverse of injection).

        Measures all data qubits of the patch EXCEPT the injection site in their
        initialization basis. The tracker automatically generates final-round detectors.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global QECPatch (returned by system.add_patch).
            inject_state: The state that was injected ('Z', 'X', or 'Y').
            protocol: Must match the protocol used in state_injection.

        Returns:
            Global qubit index of the unmeasured injection-site qubit.
        """
        if not isinstance(patch, UnrotatedSurfaceCode):
            raise TypeError(
                f"Expected UnrotatedSurfaceCode patch, got {type(patch).__name__}"
            )
        if protocol != "corner":
            raise NotImplementedError(
                f"Only 'corner' protocol is supported for unrotated SC, got '{protocol}'"
            )

        system = builder.system

        # Reconstruct the init dict to know each qubit's basis
        init_dict, injection_gidx = _get_corner_injection_init(
            patch, system, inject_state
        )

        # Measure all data qubits except the injection site, in their init basis
        unencode_measurements = {
            gidx: basis for gidx, basis in init_dict.items()
            if gidx != injection_gidx
        }

        # Noiseless: unencode measurements are deterministic given the stabilizer state
        builder.apply_data_readout(final_measurements=unencode_measurements, noiseless=True)

        return injection_gidx
