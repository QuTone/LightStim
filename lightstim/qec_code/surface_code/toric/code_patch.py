"""Toric code patch: unrotated surface code with periodic boundary conditions."""
from typing import Tuple, Dict, List, Optional, Literal, Set, Any

from lightstim.ir.qec_patch import QECPatch


class ToricCode(QECPatch):
    """
    Implementation of the Toric Code (Unrotated Surface Code with Periodic Boundaries).

    Geometry:
    - Same parity structure as unrotated surface code
    - Periodic in both x and y directions
    - All stabilizers are weight-4 (no boundaries)
    - Lx = 2 * distance_z, Ly = 2 * distance_x

    Logicals:
    - 2 logical qubits
    - Non-contractible loops in x and y directions

    Parameters (via **kwargs):
        l_z: Code distance along Z (horizontal).
        l_x: Code distance along X (vertical).
        distance: Optional shorthand for square codes (sets both).
        shift: Coordinate offset (dx, dy).
    """

    def _process_params(self):
        self.l_z = self.params.get("l_z")
        self.l_x = self.params.get("l_x")
        if self.l_z is None and "l_z" in self.params:
            self.l_z = self.l_x = self.params["l_z"]
        self.shift = self.params.get("shift", (0, 0))
        if self.l_z is None and "distance" in self.params:
            self.l_z = self.l_x = self.params["distance"]
        if self.l_z is None or self.l_x is None:
            raise ValueError("Both 'l_z' and 'l_x' must be provided.")
        if self.l_z < 2 or self.l_x < 2:
            raise ValueError("Toric code distance must be at least 2.")

    @property
    def Lx(self) -> int:
        return 2 * self.l_z

    @property
    def Ly(self) -> int:
        return 2 * self.l_x

    def _wrap(self, x: float, y: float) -> Tuple[float, float]:
        """Apply periodic boundary conditions."""
        return (x % self.Lx, y % self.Ly)

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        dz = self.l_z
        dx = self.l_x

        # Phase 1: Geometry Registration (Periodic Lattice)
        for y in range(self.Ly):
            for x in range(self.Lx):
                if (x + y) % 2 == 0:
                    self.add_qubit(x, y, role="data")
                elif x % 2 == 0:
                    self.add_qubit(x, y, role="syndrome_x")
                else:
                    self.add_qubit(x, y, role="syndrome_z")

        # Phase 2: Stabilizers (All weight-4 with wrap)
        for syn_idx in self.syndrome_indices_x:
            syn_coord = self.qubit_coords[syn_idx]
            x, y = syn_coord
            neighbors = [
                self._wrap(x - 1, y),
                self._wrap(x + 1, y),
                self._wrap(x, y - 1),
                self._wrap(x, y + 1),
            ]
            targets = {nbr: "X" for nbr in neighbors}
            self.create_stim_stabilizer(targets, syn_coord, "X")

        for syn_idx in self.syndrome_indices_z:
            syn_coord = self.qubit_coords[syn_idx]
            x, y = syn_coord
            neighbors = [
                self._wrap(x - 1, y),
                self._wrap(x + 1, y),
                self._wrap(x, y - 1),
                self._wrap(x, y + 1),
            ]
            targets = {nbr: "Z" for nbr in neighbors}
            self.create_stim_stabilizer(targets, syn_coord, "Z")

        # Phase 3: Logical Operators (2 logical qubits)
        # Logical 0: Z horizontal, X vertical
        z0_targets = {(2 * x, 0): "Z" for x in range(dz)}
        self.create_stim_logical(z0_targets, "Z")
        x0_targets = {(0, 2 * y): "X" for y in range(dx)}
        self.create_stim_logical(x0_targets, "X")

        # Logical 1: Z horizontal shifted, X vertical shifted
        z1_targets = {(2 * x, 2): "Z" for x in range(dz)}
        self.create_stim_logical(z1_targets, "Z")
        x1_targets = {(2, 2 * y): "X" for y in range(dx)}
        self.create_stim_logical(x1_targets, "X")

        self.num_logicals = 2

        # Phase 4: Shift
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "l_z": self.l_z,
            "l_x": self.l_x,
            "num_data_qubits": len(self.data_indices),
            "num_x_syndromes": len(self.syndrome_indices_x),
            "num_z_syndromes": len(self.syndrome_indices_z),
            "num_logicals": self.num_logicals,
            "data_coords": self.data_coords,
            "syndrome_coords": self.syndrome_coords,
            "stabilizers": self.stabilizers,
            "logical_ops": self.logical_ops,
            "index_map": self.index_map,
            "qubit_coords": self.qubit_coords,
        })
        return info
