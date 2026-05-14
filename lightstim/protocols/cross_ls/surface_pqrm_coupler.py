"""
Surface-PQRM coupler for CrossLS: bridges UnrotatedSurfaceCode (left) and PQRMPatch (right).

Middle ancilla column at x=-1. Syndrome at (-1, 2), (-1, 4), ..., (-1, 2*(d_surf, y_range-1));
data at (-1, 3), (-1, 5), ..., (-1, 2*(d_surf, y_range-1)-1).
"""

from typing import List, Tuple, Set, Optional
import math

from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.coupler import LogicalCouplerProtocol

from lightstim.qec_code.PQRM.pqrm_patch import PQRMPatch, LOG_PQRM_LEN_DICT, PQRM_SUPPORTED_PARAMS


def _get_surface_d(patch: QECPatch) -> int:
    """Extract distance from UnrotatedSurfaceCode."""
    return getattr(patch, "distance_z", None) or getattr(patch, "distance", None)


def _get_log_pqrm_len(patch: QECPatch) -> int:
    """Extract log_PQRM_len from PQRMPatch."""
    key = (patch.rx, patch.rz, patch.m)
    return LOG_PQRM_LEN_DICT.get(key, 0)


class SurfacePQRMCoupler(LogicalCouplerProtocol):
    """
    Coupler between UnrotatedSurfaceCode (left, x < 0) and PQRMPatch (right, x >= 0).
    Single ancilla column at x=-1 for ZZ lattice surgery merge.
    """

    EXPECTED_PATCH_COUNT = 2

    def __init__(self):
        super().__init__(name_prefix="surface_pqrm_coupler")

    def _build_coupler_geometry(self, coupler_patch: QECPatch, patches: List[QECPatch], **params):
        surface = patches[0]
        pqrm = patches[1]

        if not isinstance(pqrm, PQRMPatch):
            raise ValueError(
                "SurfacePQRMCoupler expects [UnrotatedSurfaceCode, PQRMPatch]. "
                f"Got {type(pqrm).__name__} as second patch."
            )
        key = (pqrm.rx, pqrm.rz, pqrm.m)
        if key not in PQRM_SUPPORTED_PARAMS:
            raise ValueError(f"Unsupported PQRM params {key}. Supported: {sorted(PQRM_SUPPORTED_PARAMS)}")

        d = _get_surface_d(surface)
        if d is None:
            raise ValueError("Surface patch must have distance_z or distance.")
        log_len = _get_log_pqrm_len(pqrm)
        x_range = pqrm.x_range
        y_range = pqrm.y_range

        self._d = d
        self._log_len = log_len

        # --- Coupling region: middle ancilla column at x=-1 ---
        syndrome_coords_middle = [(-1, y) for y in range(2, 2 * max(d, y_range - 1) + 1, 2)]
        data_coords_ancsys = [(-1, y) for y in range(3, 2 * max(d, y_range - 1), 2)]

        for coord in syndrome_coords_middle:
            coupler_patch.add_qubit(coord[0], coord[1], role="syndrome_z")
        for coord in data_coords_ancsys:
            coupler_patch.add_qubit(coord[0], coord[1], role="data")

        # --- Build Z stabilizers for middle column ---
        self._build_middle_stabilizers(coupler_patch, surface, pqrm)

        # --- Boundary X stabilizers: weight-3 -> weight-4 (add coupler data) ---
        self._build_boundary_x_stabilizers(coupler_patch, surface)

        # --- Conflicting stabilizer coords: Surface Z and X stabs at right boundary ---
        coupler_patch.conflicting_stabilizer_coords = set()
        surf_max_x = max((c[0] for c in surface.qubit_coords.values()), default=-2)
        for syn_coord in surface.syndrome_coords_z:
            if math.isclose(syn_coord[0], surf_max_x, abs_tol=1e-3):
                coupler_patch.conflicting_stabilizer_coords.add(surface.snap_coord(syn_coord))
        for syn_coord in surface.syndrome_coords_x:
            if math.isclose(syn_coord[0], surf_max_x, abs_tol=1e-3):
                coupler_patch.conflicting_stabilizer_coords.add(surface.snap_coord(syn_coord))

    def _is_data_at(self, coupler_patch: QECPatch, surface: QECPatch, pqrm: QECPatch, x: float, y: float) -> bool:
        """Check if a data qubit exists at (x,y) in coupler, surface, or PQRM."""
        coord = (round(x, 6), round(y, 6))
        if coord in coupler_patch.index_map:
            idx = coupler_patch.index_map[coord]
            if idx in coupler_patch.data_indices:
                return True
        if coord in surface.index_map:
            idx = surface.index_map[coord]
            if idx in surface.data_indices:
                return True
        if coord in pqrm.index_map:
            idx = pqrm.index_map[coord]
            if idx in pqrm.data_indices:
                return True
        return False

    def _build_middle_stabilizers(self, coupler_patch: QECPatch, surface: QECPatch, pqrm: QECPatch):
        """Create Z stabilizers for the middle ancilla column."""
        snap = self._snap
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for idx in sorted(coupler_patch.syndrome_indices_z):
            syn = coupler_patch.qubit_coords[idx]
            sx, sy = syn[0], syn[1]
            targets = {}
            for dx, dy in neighbors:
                nc = (sx + dx, sy + dy)
                if self._is_data_at(coupler_patch, surface, pqrm, nc[0], nc[1]):
                    targets[snap(nc)] = "Z"
            if targets:
                coupler_patch.stabilizers.append({
                    "pauli": targets,
                    "type": "Z",
                    "syn_coord": syn,
                })

    def _build_boundary_x_stabilizers(self, coupler_patch: QECPatch, surface: QECPatch):
        """Surface right-boundary X stabs: weight-3 -> weight-4 by adding coupler data."""
        surf_max_x = max((c[0] for c in surface.qubit_coords.values()), default=-2)
        coupler_data_x = surf_max_x + 1
        snap = self._snap

        for stab in surface.stabilizers:
            if stab.get("type") != "X":
                continue
            syn_coord = stab.get("syn_coord")
            if syn_coord is None or not math.isclose(syn_coord[0], surf_max_x, abs_tol=1e-3):
                continue
            data_indices = stab.get("data_indices", [])
            if not data_indices and stab.get("pauli"):
                data_indices = [i for i in stab["pauli"].keys() if isinstance(i, int)]
            data_coords = [surface.qubit_coords[i] for i in data_indices if i in surface.qubit_coords]
            targets = {snap(c): "X" for c in data_coords}
            extra = (coupler_data_x, syn_coord[1])
            extra_snapped = snap(extra)
            if extra_snapped in coupler_patch.index_map:
                idx = coupler_patch.index_map[extra_snapped]
                if idx in coupler_patch.data_indices:
                    targets[extra_snapped] = "X"
            if targets:
                coupler_patch.stabilizers.append({"pauli": targets, "type": "X", "syn_coord": syn_coord})

    @staticmethod
    def _snap(c: Tuple[float, float]) -> Tuple[float, float]:
        return (round(c[0], 6), round(c[1], 6))
