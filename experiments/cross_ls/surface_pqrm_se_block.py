"""
Synchronized SE block for Surface + PQRM + Coupler.

Matches UnrotatedSurfaceCodeExtractionBlock exactly for Surface+coupler: same 6-tick schedule,
X and Z stabilizers interleaved. PQRM Z-only uses the same 6 ticks; CNOTs are batched
per tick to avoid overlap (same qubit used twice).
"""

import stim
from typing import Any, Dict, List, Set, Tuple

from src.ir.qec_patch import QECPatch

from src.qec_code.PQRM.pqrm_patch import _SYNDROME_BOUND_CONFIG
from src.qec_code.PQRM.pqrm_se_config import TICK_DELTA_BULK, TICK_DELTA_BOUNDARY


# Same as UnrotatedSurfaceCodeExtractionBlock
CANONICAL_TICK_DELTAS = [
    ((0, 0), (-1, 0)),   # Tick 0: Z left
    ((0, 0), (+1, 0)),   # Tick 1: Z right
    ((0, +1), (0, +1)),  # Tick 2: X down, Z down
    ((0, -1), (0, -1)),  # Tick 3: X up, Z up
    ((-1, 0), (0, 0)),   # Tick 4: X left
    ((+1, 0), (0, 0)),   # Tick 5: X right
]


def _partition_non_overlapping(pairs: List[Tuple[int, int]]) -> List[List[int]]:
    """Partition CNOT (control, target) pairs into batches with no qubit overlap.
    Returns list of flat [c1,t1,c2,t2,...] per batch."""
    batches: List[List[int]] = []
    used: Set[int] = set()
    current: List[int] = []

    for c, t in pairs:
        if c in used or t in used:
            if current:
                batches.append(current)
            current = []
            used = set()
        current.extend([c, t])
        used.add(c)
        used.add(t)
    if current:
        batches.append(current)
    return batches


class SurfacePQRMSEBlock:
    """
    One round of syndrome extraction: Surface full (X+Z), PQRM Z-only.
    Same 6-tick structure as UnrotatedSurfaceCodeExtractionBlock; Surface+coupler
    logic is identical to the initial Surface-only SE.
    """

    def __init__(self, system: Any):
        self.system = system
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _get_active_x_stabilizers(self) -> List[Dict]:
        if hasattr(self.system, "active_stabilizers_x") and self.system.active_stabilizers_x:
            return self.system.active_stabilizers_x
        return [s for s in self.system.stabilizers if s.get("type") == "X" and s.get("syn_idx") is not None]

    def _get_active_z_stabilizers(self) -> List[Dict]:
        if hasattr(self.system, "active_stabilizers_z") and self.system.active_stabilizers_z:
            return self.system.active_stabilizers_z
        return [s for s in self.system.stabilizers if s.get("type") == "Z" and s.get("syn_idx") is not None]

    def _get_pqrm_z_delta(self, syn_coord: Tuple[float, float], tick_idx: int) -> Tuple[float, float]:
        """PQRM Z-stab delta for this tick (bulk or boundary). Matches pqrm_se_block exactly."""
        x_range = self._get_pqrm_x_range()
        y_range = self._get_pqrm_y_range()
        bound_config = _SYNDROME_BOUND_CONFIG.get((x_range, y_range), [])
        bound_syn_to_region = {s[0]: s[1] for s in bound_config}
        for key in (syn_coord, QECPatch.snap_coord(syn_coord)):
            if key in bound_syn_to_region:
                region_key = bound_syn_to_region[key]
                tick_deltas = TICK_DELTA_BOUNDARY.get(region_key, TICK_DELTA_BULK)
                return tick_deltas[tick_idx]
        return TICK_DELTA_BULK[tick_idx]

    def _get_pqrm_x_range(self) -> int:
        patches = getattr(self.system, "patches", {})
        for name, (patch, _) in patches.items():
            if name == "pqrm" and hasattr(patch, "x_range"):
                return patch.x_range
        for patch, _ in patches.values():
            if hasattr(patch, "x_range"):
                return patch.x_range
        return 4

    def _get_pqrm_y_range(self) -> int:
        patches = getattr(self.system, "patches", {})
        for name, (patch, _) in patches.items():
            if name == "pqrm" and hasattr(patch, "y_range"):
                return patch.y_range
        for patch, _ in patches.values():
            if hasattr(patch, "y_range"):
                return patch.y_range
        return 4

    def _is_surface_or_coupler(self, syn_coord: Tuple[float, float]) -> bool:
        """Surface or middle column (x <= -1) use full SE; PQRM (x > 0) uses Z-only."""
        return syn_coord[0] <= -1

    def _get_owner_patch(self, syn_coord: Tuple[float, float]):
        """Get owner patch for syn_coord; use snap for lookup."""
        snapped = QECPatch.snap_coord(syn_coord)
        name = self.system.coord_to_owner_map.get(snapped) or self.system.coord_to_owner_map.get(syn_coord)
        if not name:
            return None
        t = self.system.patches.get(name, (None,))
        return t[0] if t else None

    def _build_circuit(self):
        x_stabs = self._get_active_x_stabilizers()
        z_stabs = self._get_active_z_stabilizers()

        active_x_syn = sorted(set(s["syn_idx"] for s in x_stabs if s.get("syn_idx") is not None))
        active_z_syn = sorted(set(s["syn_idx"] for s in z_stabs if s.get("syn_idx") is not None))
        all_syn = sorted(
            set(active_x_syn) | set(active_z_syn),
            key=lambda idx: (self.system.qubit_coords[idx][0], self.system.qubit_coords[idx][1])
        )
        if not all_syn:
            return

        grid_map = getattr(self.system, "grid_map", None) or {}
        index_map = getattr(self.system, "index_map", {})

        # --- Step 1: Reset all syndrome qubits ---
        self.circuit.append("R", all_syn)
        self.circuit.append("TICK", tag="SE_start")

        # --- Step 2: H on X-type syndromes ---
        if active_x_syn:
            self.circuit.append("H", active_x_syn)
        self.circuit.append("TICK")

        # --- Step 3: 6-tick CNOT schedule ---
        for tick_idx, (dx_x, dx_z) in enumerate(CANONICAL_TICK_DELTAS):
            pairs: List[Tuple[int, int]] = []

            # X-stabilizers (Syndrome=control -> Data=target)
            for stab in x_stabs:
                syn_coord = stab.get("syn_coord")
                syn_idx = stab.get("syn_idx")
                if syn_coord is None or syn_idx is None:
                    continue
                owner = self._get_owner_patch(syn_coord)
                if owner is None:
                    continue
                dx_global = owner.transform_vector(dx_x)
                raw = (syn_coord[0] + dx_global[0], syn_coord[1] + dx_global[1])
                key = QECPatch.get_grid_key(raw)
                nb = grid_map.get(key)
                if nb is not None and nb in stab.get("data_indices", []):
                    pairs.append((syn_idx, nb))

            # Z-stabilizers (Data=control -> Syndrome=target)
            for stab in z_stabs:
                syn_coord = stab.get("syn_coord")
                syn_idx = stab.get("syn_idx")
                if syn_coord is None or syn_idx is None:
                    continue
                if self._is_surface_or_coupler(syn_coord):
                    owner = self._get_owner_patch(syn_coord)
                    if owner is None:
                        continue
                    dx_global = owner.transform_vector(dx_z)
                    raw = (syn_coord[0] + dx_global[0], syn_coord[1] + dx_global[1])
                else:
                    dx, dy = self._get_pqrm_z_delta(syn_coord, tick_idx)
                    if (dx, dy) == (0, 0):
                        continue
                    raw = (syn_coord[0] + dx, syn_coord[1] + dy)
                raw_snapped = QECPatch.snap_coord(raw)
                key = QECPatch.get_grid_key(raw_snapped)
                nb = grid_map.get(key) if grid_map else index_map.get(raw_snapped)
                if nb is not None and nb in stab.get("data_indices", []):
                    pairs.append((nb, syn_idx))

            for batch in _partition_non_overlapping(pairs):
                if batch:
                    self.circuit.append("CNOT", batch)
            self.circuit.append("TICK")

        # --- Step 4: H on X-type syndromes (basis change back) ---
        if active_x_syn:
            self.circuit.append("H", active_x_syn)
        self.circuit.append("TICK")

        # --- Step 5: Measure all ---
        self.circuit.append("M", all_syn)
