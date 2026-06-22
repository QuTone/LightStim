from typing import List, Tuple, Dict, Set, Literal, FrozenSet
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.coupler import LogicalCouplerProtocol
from .code_patch import RotatedSurfaceCode


class RotatedTwoPatchCoupler(LogicalCouplerProtocol):
    """
    Lattice-surgery coupler between two standard rotated CSS surface code patches.

    Strategy (constructive / "diff against the merged code"):
    Merging two rotated patches along a shared boundary *is* a single larger rotated
    surface code.  So instead of hand-deriving the seam geometry, we construct the
    reference merged ``RotatedSurfaceCode`` and take the set difference against the two
    input patches.  Whatever ``M`` has that the patches don't (one intermediate data
    row/column + the seam stabilizers + the fused weight-4 boundary stabilizers) becomes
    the coupler.  Correctness of the seam follows from ``M`` being a valid code.

    Layout / handbook §7.2 mapping (verified by Stim peek_observable_expectation; matches the
    unrotated convention). Because the rotated code has X̄ vertical / Z̄ horizontal, measuring
    Z̄₁Z̄₂ requires a VERTICAL stack (the two parallel horizontal Z̄'s have their product
    measured) and X̄₁X̄₂ a SIDE-BY-SIDE placement — i.e. the merge geometry is the transpose of
    the naive "merge along same-type edges" intuition:
      - interaction_type='ZZ'  -> measure Z̄₁Z̄₂, patches stacked VERTICALLY (offset in y),
        merged distance_x = d_xA + d_xB + 1, shared transverse distance_z.
      - interaction_type='XX'  -> measure X̄₁X̄₂, patches SIDE-BY-SIDE (offset in x),
        merged distance_z = d_zA + d_zB + 1, shared transverse distance_x.
    """

    EXPECTED_PATCH_COUNT = 2

    def __init__(self):
        super().__init__(name_prefix="rotated_coupler")

    # ------------------------------------------------------------------ #
    # LogicalCouplerProtocol contract
    # ------------------------------------------------------------------ #
    def _build_coupler_geometry(self, coupler_patch: QECPatch, patches: List[QECPatch], **params):
        interaction_type = params.get("interaction_type")
        if interaction_type not in ("XX", "ZZ"):
            raise ValueError(f"interaction_type must be 'XX' or 'ZZ', got {interaction_type!r}.")

        p_low, p_high = self._order_patches(patches, interaction_type)
        merged = self._build_merged_reference(p_low, p_high, interaction_type)

        # --- diff data qubits: M.data minus the two patches' data = intermediate qubits
        patch_data: Set[Tuple[float, float]] = set(p_low.data_coords) | set(p_high.data_coords)
        merged_data: Set[Tuple[float, float]] = set(merged.data_coords)

        if not patch_data <= merged_data:
            raise ValueError(
                "Patch data qubits are not contained in the merged reference code; "
                "patches are misaligned for this interaction_type "
                "(check offset / same transverse distance)."
            )
        intermediate_data = merged_data - patch_data
        expected = merged.distance_z if interaction_type == "ZZ" else merged.distance_x
        if len(intermediate_data) != expected:
            raise ValueError(
                f"Expected {expected} intermediate data qubits in the seam, "
                f"got {len(intermediate_data)} — patch placement is inconsistent."
            )

        for (x, y) in intermediate_data:
            coupler_patch.add_qubit(x, y, role="data")

        # --- diff syndrome qubits: new seam syndromes are M's syndromes not in either patch
        patch_syn: Set[Tuple[float, float]] = set(p_low.syndrome_coords) | set(p_high.syndrome_coords)
        new_syn = set(merged.syndrome_coords) - patch_syn
        merged_x = set(merged.syndrome_coords_x)
        for (x, y) in new_syn:
            role = "syndrome_x" if (x, y) in merged_x else "syndrome_z"
            coupler_patch.add_qubit(x, y, role=role)

        # --- diff stabilizers: emit M's stabilizers that are new or changed vs the patches
        patch_stabs = self._stab_lookup(p_low)
        patch_stabs.update(self._stab_lookup(p_high))

        coupler_patch.conflicting_stabilizer_coords = set()
        for stab in merged.stabilizers:
            syn_coord = stab["syn_coord"]
            stype = stab["type"]
            coords = frozenset(merged.qubit_coords[i] for i in stab["pauli"])

            if syn_coord in patch_stabs:
                if patch_stabs[syn_coord] == (stype, coords):
                    continue  # identical bulk/outer-boundary stabilizer — kept by the patch
                # fused boundary: weight-2 -> weight-4. Replace the original.
                coupler_patch.conflicting_stabilizer_coords.add(syn_coord)

            coupler_patch.stabilizers.append({
                "pauli": {coord: stype for coord in coords},
                "type": stype,
                "syn_coord": syn_coord,
            })

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _order_patches(patches: List[QECPatch], interaction_type: str) -> Tuple[QECPatch, QECPatch]:
        """Return (low, high) ordered by position along the merge axis
        (y for ZZ = vertical stack, x for XX = side-by-side)."""
        p0, p1 = patches
        b0, b1 = p0._get_bounds(), p1._get_bounds()  # (min_x, max_x, min_y, max_y)
        lo_idx = 2 if interaction_type == "ZZ" else 0  # min_y vs min_x
        return (p0, p1) if b0[lo_idx] <= b1[lo_idx] else (p1, p0)

    @staticmethod
    def _build_merged_reference(p_low: QECPatch, p_high: QECPatch, interaction_type: str) -> RotatedSurfaceCode:
        """Construct the larger rotated code that the merged pair forms, anchored at p_low."""
        dz_low, dx_low = p_low.distance_z, p_low.distance_x
        dz_high, dx_high = p_high.distance_z, p_high.distance_x
        shift = getattr(p_low, "shift", (0, 0))

        if interaction_type == "ZZ":   # ZZ = vertical stack -> grow distance_x
            if dz_low != dz_high:
                raise ValueError(
                    f"ZZ merge (vertical stack) requires equal distance_z; got {dz_low} and {dz_high}."
                )
            return RotatedSurfaceCode(distance_z=dz_low, distance_x=dx_low + dx_high + 1, shift=shift)
        else:  # XX = side-by-side -> grow distance_z
            if dx_low != dx_high:
                raise ValueError(
                    f"XX merge (side-by-side) requires equal distance_x; got {dx_low} and {dx_high}."
                )
            return RotatedSurfaceCode(distance_z=dz_low + dz_high + 1, distance_x=dx_low, shift=shift)

    @staticmethod
    def _stab_lookup(patch: QECPatch) -> Dict[Tuple[float, float], Tuple[str, FrozenSet[Tuple[float, float]]]]:
        """syn_coord -> (type, frozenset(data_coords)) for an index-keyed code patch."""
        out = {}
        for s in patch.stabilizers:
            coords = frozenset(patch.qubit_coords[i] for i in s["pauli"])
            out[s["syn_coord"]] = (s["type"], coords)
        return out
