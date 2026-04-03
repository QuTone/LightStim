"""
PQRM Syndrome Extraction Block: Z stabilizers only.

One round of Z-type stabilizer measurement. No X stabilizers (they use post-select).
Includes 6-tick schedule and boundary delta definitions (from former pqrm_se_config).
"""

import stim
from typing import Any, Dict, List, Tuple

from src.ir.qec_patch import QECPatch

from .pqrm_patch import _SYNDROME_BOUND_CONFIG
from .pqrm_se_config import TICK_DELTA_BULK, TICK_DELTA_BOUNDARY, get_boundary_data_deltas


class PQRMExtractionBlock:
    """
    One round of Z-stabilizer syndrome extraction for PQRM.
    No X stabilizers (post-select only).
    """

    def __init__(self, system: Any):
        """
        Args:
            system: QECSystem with one PQRM patch, or PQRMPatch directly.
        """
        self.system = system
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _get_z_stabilizers(self) -> List[Dict]:
        """Z stabilizers with syn_idx (exclude X)."""
        if hasattr(self.system, "active_stabilizers_z") and self.system.active_stabilizers_z:
            return self.system.active_stabilizers_z
        stabs = [s for s in self.system.stabilizers if s.get("type") == "Z"]
        return [s for s in stabs if s.get("syn_idx") is not None]

    def _get_z_syndrome_indices(self) -> List[int]:
        """Z syndrome qubit indices for reset/measure."""
        z_stabs = self._get_z_stabilizers()
        return sorted(set(s["syn_idx"] for s in z_stabs if s.get("syn_idx") is not None))

    def _build_circuit(self):
        z_stabs = self._get_z_stabilizers()
        syn_indices = self._get_z_syndrome_indices()
        if not syn_indices:
            return

        # --- Step 1: Reset Z syndrome qubits ---
        self.circuit.append("R", syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        # --- Step 2: CNOT layers (Data -> Syndrome), 6 ticks matching planar_PQRM ---
        x_range = getattr(self.system, "x_range", None) or self._infer_x_range()
        y_range = getattr(self.system, "y_range", None) or self._infer_y_range()
        bound_config = _SYNDROME_BOUND_CONFIG.get((x_range, y_range), [])
        bound_syn_to_region = {s[0]: s[1] for s in bound_config}
        grid_map = getattr(self.system, "grid_map", None) or {}
        index_map = getattr(self.system, "index_map", {})

        for tick_idx in range(6):
            cnot_targets = []

            for stab in z_stabs:
                syn_coord = stab["syn_coord"]
                syn_idx = stab["syn_idx"]

                if syn_coord in bound_syn_to_region:
                    region_key = bound_syn_to_region[syn_coord]
                    tick_deltas = TICK_DELTA_BOUNDARY.get(region_key, TICK_DELTA_BULK)
                    dx, dy = tick_deltas[tick_idx]
                else:
                    dx, dy = TICK_DELTA_BULK[tick_idx]

                if (dx, dy) == (0, 0):
                    continue

                raw_target = (syn_coord[0] + dx, syn_coord[1] + dy)
                snapped = QECPatch.snap_coord(raw_target)
                target_key = QECPatch.get_grid_key(snapped)
                neighbor_idx = grid_map.get(target_key) if grid_map else index_map.get(snapped)

                if neighbor_idx is not None and neighbor_idx in stab["data_indices"]:
                    cnot_targets.extend([neighbor_idx, syn_idx])

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            self.circuit.append("TICK")

        # --- Step 3: Measurement ---
        self.circuit.append("M", syn_indices)

    def _infer_x_range(self) -> int:
        """Infer x_range from patch."""
        if hasattr(self.system, "x_range"):
            return self.system.x_range
        if hasattr(self.system, "patches"):
            for patch, _ in self.system.patches.values():
                if hasattr(patch, "x_range"):
                    return patch.x_range
        return 4

    def _infer_y_range(self) -> int:
        if hasattr(self.system, "y_range"):
            return self.system.y_range
        if hasattr(self.system, "patches"):
            for patch, _ in self.system.patches.values():
                if hasattr(patch, "y_range"):
                    return patch.y_range
        return 4
