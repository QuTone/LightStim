"""Triangular Color Code on a hexagonal lattice.

Implements the 6-6-6 (hexagonal) color code with a triangular patch
boundary. Each hexagonal face has two ancilla qubits (X-type and Z-type)
for space-multiplexed syndrome extraction.

Compact integer coordinate system:
- Data qubits and syndrome pairs placed on an integer grid
- Each face center splits into (sZ, sX) at consecutive x positions
- Triangle points downward with red boundary on top
"""

from typing import Tuple, Dict, List, Optional, Any
from lightstim.ir.qec_patch import QECPatch


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

    # Neighbor offsets in (row, position) space.
    # Maps the 6 hex neighbors of a face, counterclockwise from lower-left.
    # Index 0-5 matches the CNOT schedule positions in SE_block.
    NEIGHBOR_OFFSETS = [
        (1, -1),   # pos 0: one row down, one position left
        (1,  0),   # pos 1: one row down, same position
        (0,  1),   # pos 2: same row, one position right
        (-1, 1),   # pos 3: one row up, one position right
        (-1, 0),   # pos 4: one row up, same position
        (0, -1),   # pos 5: same row, one position left
    ]

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
        # Phase 1: Enumerate positions and assign compact coordinates
        # -------------------------------------------------------------------
        # Face classification cycle: determines which position index is a face
        ANC_POS_CYCLE = [1, 2, 0]

        # Compute total slots in row 0 for centering
        anc_pos_row0 = ANC_POS_CYCLE[0]
        num_faces_row0 = sum(1 for p in range(L + 1) if p % 3 == anc_pos_row0)
        total_slots_row0 = (L + 1) + num_faces_row0

        # position_map: (y, p) -> position info
        position_map = {}

        self.faces: List[Dict[str, Any]] = []

        for y in range(L + 1):
            anc_pos = ANC_POS_CYCLE[y % 3]
            num_positions = L - y + 1

            # Count faces and compute compact x start
            num_faces = sum(1 for p in range(num_positions) if p % 3 == anc_pos)
            num_slots = num_positions + num_faces
            x_start = (total_slots_row0 - num_slots + 1) // 2

            cx = x_start
            for p in range(num_positions):
                if p % 3 == anc_pos:
                    # Face center: place sZ and sX syndrome pair
                    z_idx = self.add_qubit(cx, y, role='syndrome_z')
                    x_idx = self.add_qubit(cx + 1, y, role='syndrome_x')

                    position_map[(y, p)] = {
                        'type': 'face',
                        'z_coord': self.snap_coord((cx, y)),
                        'x_coord': self.snap_coord((cx + 1, y)),
                        'z_idx': z_idx,
                        'x_idx': x_idx,
                    }
                    cx += 2
                else:
                    # Data qubit
                    d_idx = self.add_qubit(cx, y, role='data')
                    position_map[(y, p)] = {
                        'type': 'data',
                        'coord': self.snap_coord((cx, y)),
                        'idx': d_idx,
                    }
                    cx += 1

        # -------------------------------------------------------------------
        # Phase 2: Build face metadata with data neighbors
        # -------------------------------------------------------------------
        # Face color cycle
        COLOR_CYCLE = ['r', 'g', 'b']

        for (y, p), info in sorted(position_map.items()):
            if info['type'] != 'face':
                continue

            # Find 6 data neighbors using hex connectivity
            data_neighbors = []
            for dy, dp in self.NEIGHBOR_OFFSETS:
                ny, np_ = y + dy, p + dp
                neighbor = position_map.get((ny, np_))
                if neighbor and neighbor['type'] == 'data':
                    data_neighbors.append((neighbor['coord'], neighbor['idx']))
                else:
                    data_neighbors.append(None)

            # Boundary classification
            num_positions = L - y + 1
            boundary = []
            if y == 0:
                boundary.append('r')       # red boundary (top)
            if p == 0:
                boundary.append('g')       # green boundary (left)
            if p == num_positions - 1:
                boundary.append('b')       # blue boundary (right)

            face = {
                'center': ((info['z_coord'][0] + info['x_coord'][0]) / 2, float(y)),
                'color': COLOR_CYCLE[y % 3],
                'z_ancilla_coord': info['z_coord'],
                'x_ancilla_coord': info['x_coord'],
                'z_ancilla_idx': info['z_idx'],
                'x_ancilla_idx': info['x_idx'],
                'boundary': ''.join(boundary) if boundary else None,
                'data_neighbors': data_neighbors,
            }
            self.faces.append(face)

        # -------------------------------------------------------------------
        # Phase 3: Stabilizer Construction
        # -------------------------------------------------------------------
        for face in self.faces:
            data_coords = [n[0] for n in face['data_neighbors'] if n is not None]

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
        # Phase 4: Logical Operators
        # -------------------------------------------------------------------
        # Logical X and Z on the red boundary (y == 0 top row data qubits).
        red_boundary_data = []
        for (y, p), info in sorted(position_map.items()):
            if info['type'] == 'data' and y == 0:
                red_boundary_data.append(info['coord'])

        lx_targets = {coord: 'X' for coord in red_boundary_data}
        self.create_stim_logical(lx_targets, 'X')

        lz_targets = {coord: 'Z' for coord in red_boundary_data}
        self.create_stim_logical(lz_targets, 'Z')

        self.num_logicals = 1

        # -------------------------------------------------------------------
        # Phase 5: Shift
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
