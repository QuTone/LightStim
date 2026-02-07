# self.num_logicals = 2
# self.num_logicals = 2
from typing import Tuple, List, Optional, Literal, Set
import numpy as np
import math
import stim

from src.ir.qec_patch import QECPatch


class ToricCode(QECPatch):
    """
    Implementation of the Toric Code (Unrotated, Periodic Boundaries).

    Geometry:
    - Same parity structure as unrotated surface code
    - Periodic in both x and y directions
    - No boundaries, no truncated stabilizers

    Logicals:
    - 2 logical qubits
    - Non-contractible loops in x and y directions
    """

    # -------------------------------------------------------------------------
    # Parameter handling
    # -------------------------------------------------------------------------
    def _process_params(self):
        self.distance_z = self.params.get("distance_z")
        self.distance_x = self.params.get("distance_x")

        if self.distance_z is None and "distance" in self.params:
            self.distance_z = self.distance_x = self.params["distance"]

        self.shift = self.params.get("shift", (0, 0))

        if self.distance_z is None or self.distance_x is None:
            raise ValueError("Both 'distance_z' and 'distance_x' must be provided.")

        if self.distance_z < 2 or self.distance_x < 2:
            raise ValueError("Toric code distance must be at least 2.")

    def add_qubit(self, x: float, y: float, role: Literal['data', 'syndrome_x', 'syndrome_z'], uid: Optional[int] = None) -> int:
        uid = super().add_qubit(x, y, role, uid)

        if role == 'syndrome_x':
            self.syndrome_indices_x.add(uid)
        elif role == 'syndrome_z':
            self.syndrome_indices_z.add(uid)

        return uid

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    @property
    def Lx(self) -> int:
        return 2 * self.distance_z

    @property
    def Ly(self) -> int:
        return 2 * self.distance_x

    def _wrap(self, x: float, y: float) -> Tuple[float, float]:
        """Apply periodic boundary conditions."""
        return (x % self.Lx, y % self.Ly)

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------
    def build(self):
        dz = self.distance_z
        dx = self.distance_x

        self.syndrome_indices_x: Set[int] = set()
        self.syndrome_indices_z: Set[int] = set()

        # ---------------------------------------------------------------------
        # Phase 1: Geometry Registration (Periodic Lattice)
        # ---------------------------------------------------------------------
        for y in range(self.Ly):
            for x in range(self.Lx):
                if (x + y) % 2 == 0:
                    self.add_qubit(x, y, role="data")
                elif (x % 2) == 0:
                    self.add_qubit(x, y, role="syndrome_x")
                else:
                    self.add_qubit(x, y, role="syndrome_z")

        # ---------------------------------------------------------------------
        # Phase 2: Stabilizers (Always Weight-4)
        # ---------------------------------------------------------------------
        self.stabilizers = []         # for tracker / MemoryExperiment
        self.stabilizers_list = []    # for extraction block

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
            self.stabilizers_list.append({"coord": syn_coord, "targets": targets})

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
            self.stabilizers_list.append({"coord": syn_coord, "targets": targets})

            
        print("stabilizers_list:", self.stabilizers_list)

        # ---------------------------------------------------------------------
        # Phase 3: Logical Operators (2 Logical Qubits)
        # ---------------------------------------------------------------------
        # Logical qubit 0
        # Z: horizontal non-contractible loop
        z0_coords = {(2 * x, 0):"Z" for x in range(dz)}
        self.create_stim_logical(z0_coords, "Z")

        # X: vertical non-contractible loop
        x0_coords = {(0, 2 * y):"X" for y in range(dx)}
        self.create_stim_logical(x0_coords, "X")

        # Logical qubit 1

        # Z1: horizontal loop, shifted but still even
        z1_coords = {(2 * x, 2):"Z" for x in range(dz)}
        self.create_stim_logical(z1_coords, "Z")

        # X1: vertical loop, shifted but still even
        x1_coords = {(2, 2 * y):"X" for y in range(dx)}
        self.create_stim_logical(x1_coords, "X")

        self.num_logicals = 2

        # ---------------------------------------------------------------------
        # Phase 4: Shift
        # ---------------------------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)
        
        # ---------------------------------------------------------------------
        # Phase 5: Composite compatibility (single patch wrapper)
        # ---------------------------------------------------------------------
        # Treat the toric code as a single patch composite
        self.patches = [self]

        # Every coordinate belongs to patch 0
        self.spatial_map = {
            coord: 0 for coord in self.qubit_coords
        }

        # ---------------------------------------------------------------------
        # Phase X: Surface-code compatibility layer
        # ---------------------------------------------------------------------
        # Explicitly expose syndrome coordinate sets
        self.syndrome_coords_x = [
            self.qubit_coords[i] for i in self.syndrome_indices_x
        ]

        self.syndrome_coords_z = [
            self.qubit_coords[i] for i in self.syndrome_indices_z
        ]


    # -------------------------------------------------------------------------
    def transform_vector(self, v):
        """
        Identity transform.
        Required by ExtractionBlock interface.
        """
        return v
    
    def get_stabilizer_matrix(self):
        """
        Returns stabilizers in symplectic form.
        Shape: (num_stabilizers, 2*num_qubits)
        """
        n = len(self.qubit_coords)
        rows = []

        for stab in self.stabilizers:
            x = np.zeros(n, dtype=int)
            z = np.zeros(n, dtype=int)

            for coord, pauli in stab["targets"].items():
                q = self.index_map[coord]
                if pauli == "X":
                    x[q] = 1
                elif pauli == "Z":
                    z[q] = 1

            rows.append(np.hstack([x, z]))

        return np.array(rows, dtype=int)



    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------
    def get_info(self):
        info = super().get_info()
        info.update({
            "distance_z": self.distance_z,
            "distance_x": self.distance_x,
            "num_data_qubits": len(self.data_coords),
            "num_x_syndromes": len(self.syndrome_indices_x),
            "num_z_syndromes": len(self.syndrome_indices_z),
            "num_logicals": self.num_logicals,
            "data_coords": self.data_coords,
            "syndrome_coords": self.syndrome_coords,
            "logical_ops": self.logical_ops,
            "index_map": self.index_map,
            "qubit_coords": self.qubit_coords,
        })
        return info