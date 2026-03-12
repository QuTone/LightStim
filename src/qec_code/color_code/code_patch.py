"""Triangular Color Code on a hexagonal lattice.

Implements the 6-6-6 (hexagonal) color code with a triangular patch
boundary. Each hexagonal face has two ancilla qubits (X-type and Z-type)
for space-multiplexed syndrome extraction.

Coordinate convention follows color-code-stim (Lee et al.):
- Integer coordinates (x, y)
- Data qubits at vertex positions, ancilla pairs at face centers
- Three colored boundaries: Red (top), Green (left), Blue (right)
"""

from typing import Tuple, Dict, List, Optional, Any
from src.ir.qec_patch import QECPatch


class ColorCode(QECPatch):
    """
    Triangular Color Code (6-6-6 hexagonal lattice).

    Encodes 1 logical qubit with code distance d.
    CSS code: each hexagonal face defines both an X-stabilizer and a Z-stabilizer.

    Parameters (via **kwargs):
    -------------------------
    distance : int
        Code distance (must be odd, >= 3).
    shift : tuple, optional (default: (0, 0))
        Global coordinate offset.

    Examples:
    ---------
    >>> code = ColorCode(distance=3)   # [[7, 1, 3]]
    >>> code = ColorCode(distance=5)   # [[19, 1, 5]]
    >>> code = ColorCode(distance=7)   # [[37, 1, 7]]
    """

    # 6 offsets from face center (x, y) to neighboring data qubit vertices.
    # Ordered counterclockwise starting from upper-left.
    FACE_OFFSETS = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]

    def _process_params(self):
        self.distance = self.params.get('distance')
        self.shift = self.params.get('shift', (0, 0))

        if self.distance is None:
            raise ValueError("'distance' must be provided.")
        if self.distance < 3:
            raise ValueError(f"distance must be >= 3, got {self.distance}")
        if self.distance % 2 == 0:
            raise ValueError(f"distance must be odd, got {self.distance}")
        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two numbers.")

        self.L = round(3 * (self.distance - 1) / 2)

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        L = self.L

        # -------------------------------------------------------------------
        # Phase 1: Geometry Registration
        # -------------------------------------------------------------------
        # Store face metadata for SE_block and stabilizer construction.
        # Each face: {center, color, z_ancilla_coord, x_ancilla_coord, boundary}
        self.faces: List[Dict[str, Any]] = []

        for y in range(L + 1):
            # Determine ancilla color and classification index for this row
            if y % 3 == 0:
                anc_color = 'g'   # green
                anc_pos = 2
            elif y % 3 == 1:
                anc_color = 'b'   # blue
                anc_pos = 0
            else:  # y % 3 == 2
                anc_color = 'r'   # red
                anc_pos = 1

            for x in range(2 * y, 4 * L - 2 * y + 1, 4):
                # Boundary classification
                boundary = []
                if y == 0:
                    boundary.append('r')   # red boundary (top)
                if x == 2 * y:
                    boundary.append('g')   # green boundary (left)
                if x == 4 * L - 2 * y:
                    boundary.append('b')   # blue boundary (right)
                boundary_str = ''.join(boundary) if boundary else None

                # Data qubit vs ancilla pair classification
                if round((x / 2 - y) / 2) % 3 != anc_pos:
                    # This is a DATA QUBIT
                    self.add_qubit(x, y, role='data')
                else:
                    # This is a FACE CENTER — place ancilla pair
                    z_anc_x = x - 1
                    x_anc_x = x + 1

                    z_idx = self.add_qubit(z_anc_x, y, role='syndrome_z')
                    x_idx = self.add_qubit(x_anc_x, y, role='syndrome_x')

                    face = {
                        'center': (x, y),
                        'color': anc_color,
                        'z_ancilla_coord': self.snap_coord((z_anc_x, y)),
                        'x_ancilla_coord': self.snap_coord((x_anc_x, y)),
                        'z_ancilla_idx': z_idx,
                        'x_ancilla_idx': x_idx,
                        'boundary': boundary_str,
                    }
                    self.faces.append(face)

        # -------------------------------------------------------------------
        # Phase 2: Stabilizer Construction
        # -------------------------------------------------------------------
        for face in self.faces:
            cx, cy = face['center']

            # Find data qubit vertices for this face
            data_coords = []
            for dx, dy in self.FACE_OFFSETS:
                data_coord = self.snap_coord((cx + dx, cy + dy))
                if data_coord in self.index_map:
                    idx = self.index_map[data_coord]
                    if idx in self.data_indices:
                        data_coords.append(data_coord)

            # X-stabilizer
            x_targets = {coord: 'X' for coord in data_coords}
            self.create_stim_stabilizer(
                x_targets,
                syn_coord=face['x_ancilla_coord'],
                type='X'
            )

            # Z-stabilizer
            z_targets = {coord: 'Z' for coord in data_coords}
            self.create_stim_stabilizer(
                z_targets,
                syn_coord=face['z_ancilla_coord'],
                type='Z'
            )

        # -------------------------------------------------------------------
        # Phase 3: Logical Operators
        # -------------------------------------------------------------------
        # Logical X and Z on the red boundary (y == 0 top row data qubits).
        # The red boundary has d data qubits. Since d is odd,
        # X^d and Z^d on the same qubits anti-commute.
        red_boundary_data = []
        for idx in sorted(self.data_indices):
            coord = self.qubit_coords[idx]
            if coord[1] == 0.0:  # y == 0 is the red boundary
                red_boundary_data.append(coord)

        lx_targets = {coord: 'X' for coord in red_boundary_data}
        self.create_stim_logical(lx_targets, 'X')

        lz_targets = {coord: 'Z' for coord in red_boundary_data}
        self.create_stim_logical(lz_targets, 'Z')

        self.num_logicals = 1

        # -------------------------------------------------------------------
        # Phase 4: Shift
        # -------------------------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def get_info(self):
        info = super().get_info()
        info.update({
            'distance': self.distance,
            'L': self.L,
            'num_faces': len(self.faces),
            'n_data': len(self.data_indices),
            'num_x_syndromes': len(self.syndrome_indices_x),
            'num_z_syndromes': len(self.syndrome_indices_z),
            'num_logicals': self.num_logicals,
        })
        return info
