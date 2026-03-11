from typing import List, Tuple, Any

import stim

from src.ir.operation import CSSLogicalOpSet
from src.ir.builder import CircuitBuilder
from src.ir.qec_patch import QECPatch


# ------------------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------------------

def _get_fold_yx_pairs(system: Any, patch: QECPatch) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Partition data qubits of a single patch by the y=x diagonal reflection.

    Uses the patch's built-in shift to convert global coords back to local
    coords for the diagonal check, so this works correctly in multi-patch
    systems where each patch has been created with an appropriate shift.

    Returns:
        diagonal_qs  : global qubit indices where local x == y
        mirror_pairs : list of (idx_a, idx_b) where local coord_a=(x,y) with
                       x < y and coord_b is the reflected qubit at local (y, x)

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

    diagonal_qs: List[int] = []
    mirror_pairs: List[Tuple[int, int]] = []

    for coord in patch.data_coords:      # global coords (already shifted)
        lx = round(coord[0] - sx)
        ly = round(coord[1] - sy)

        if lx == ly:
            diagonal_qs.append(index_map[coord])
        elif lx < ly:
            # reflected partner in global coords
            refl = QECPatch.snap_coord((ly + sx, lx + sy))
            if refl not in index_map:
                raise ValueError(
                    f"Mirror qubit at {refl} not found. "
                    "Ensure the patch is square and data qubits have y=x symmetry."
                )
            mirror_pairs.append((index_map[coord], index_map[refl]))

    return diagonal_qs, mirror_pairs


# ------------------------------------------------------------------------------
# LogicalOpSet implementation
# ------------------------------------------------------------------------------

class UnrotatedSurfaceCodeLogicalOpSet(CSSLogicalOpSet):
    """
    Logical operation set for the Unrotated Surface Code.

    Inherits transversal_cnot from CSSLogicalOpSet and adds fold-transversal
    H and S gates using the y=x diagonal reflection symmetry.

    For a square (d_z == d_x) unrotated surface code the y=x fold maps every
    X-syndrome position to a Z-syndrome position and vice versa, enabling:

        H_L = (transversal H on all data qubits) + (SWAP mirror pairs)
        S_L = (S on diagonal data qubits)        + (CZ  on mirror pairs)

    Available via LogicalExecutor:
        executor.apply_logical_operation("fold_transversal_hadamard", [patch])
        executor.apply_logical_operation("fold_transversal_s",        [patch])
        executor.apply_logical_operation("transversal_cnot", [ctrl, tgt])
    """

    def __init__(self):
        super().__init__()
        self.name = "UnrotatedSurfaceCode"

    def fold_transversal_hadamard(self, builder: CircuitBuilder, patch: QECPatch):
        """
        Implement the logical Hadamard H_L via fold-transversal gate.

        Physical circuit (two layers):
          Layer 1 — transversal H : H applied to every data qubit (X <-> Z)
          Layer 2 — fold (SWAP)   : SWAP every mirror pair (x,y) <-> (y,x)

        Logical action: X_L <-> Z_L

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The UnrotatedSurfaceCode patch (must be square).
        """
        system = builder.system
        diagonal_qs, mirror_pairs = _get_fold_yx_pairs(system, patch)
        all_data = sorted(system.index_map[c] for c in patch.data_coords)

        unitary = stim.Circuit()
        unitary.append("H", all_data)
        if mirror_pairs:
            unitary.append("TICK")
            flat = [q for a, b in mirror_pairs for q in (a, b)]
            unitary.append("SWAP", flat)

        builder.apply_unitary_block(unitary)

    def fold_transversal_s(self, builder: CircuitBuilder, patch: QECPatch):
        """
        Implement the logical phase gate S_L via fold-transversal gate.

        Physical circuit (single layer — S and CZ act on disjoint qubits):
          S   on diagonal data qubits  {(x,y) : local x == y}
          CZ  on every mirror pair     {(x,y) <-> (y,x) : local x < y}

        Logical action: Z_L -> Z_L,  X_L -> Y_L

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The UnrotatedSurfaceCode patch (must be square).
        """
        system = builder.system
        diagonal_qs, mirror_pairs = _get_fold_yx_pairs(system, patch)

        unitary = stim.Circuit()
        if diagonal_qs:
            unitary.append("S", sorted(diagonal_qs))
        for a, b in mirror_pairs:
            unitary.append("CZ", [a, b])

        builder.apply_unitary_block(unitary)
