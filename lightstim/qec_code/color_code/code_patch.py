"""Color-code QECPatch builders.

``layout_tiles.py`` defines only local tile geometry.  This file owns the
distance-dependent lattice ranges, boundary membership filtering, and final
``QECPatch`` registration.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Any, Iterable
from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.color_code.layout_tiles import (
    Color,
    Coord,
    ColorCodeTile,
    HEX_ROW_HEIGHT,
    MIDDLE_OUT_FACE_XY_OFFSETS,
    SUPERDENSE_DATA_XY_OFFSETS,
    RAW_FACE_X_BY_Y,
    RAW_FACE_XY_OFFSETS,
    add_coord,
    make_xz_tiles,
    sorted_tiles,
)


@dataclass(frozen=True)
class ColorCodeLatticeFace:
    """One geometric face in final QECPatch/Stim ``(x, y)`` coordinates."""

    center: Coord
    ordered_data_coords: Tuple[Coord | None, ...]
    color: Color
    x_measurement_coord: Coord | None = None
    z_measurement_coord: Coord | None = None

    @property
    def data_coords(self) -> Tuple[Coord, ...]:
        return tuple(coord for coord in self.ordered_data_coords if coord is not None)


@dataclass(frozen=True)
class ColorCodeLattice:
    """A distance-dependent color-code lattice in final coordinates."""

    positions: Tuple[Coord, ...]
    data_coords: Tuple[Coord, ...]
    candidate_face_coords: Tuple[Coord, ...]
    faces: Tuple[ColorCodeLatticeFace, ...]


class ColorCode(QECPatch):
    """
    2D Color Code.

    Encodes 1 logical qubit with code distance d.
    CSS code: each 6-weight or 4-weight face defines both an X-stabilizer
    and a Z-stabilizer.
    Code parameter: [[n, k, d]] = [[3/4 * d^2 + 1/4, 1, d]] (without spurs).

    Parameters (via **kwargs):
    -------------------------
    distance : int
        Code distance (must be odd, >= 3).
    layout : str, optional (default: "superdense")
        One of "superdense", "raw", "triangular", or "rectangle".
    shift : tuple, optional (default: (0, 0))
        Global coordinate offset.

    Examples:
    ---------
    >>> code = ColorCode(distance=3)   # [[7, 1, 3]]
    >>> code = ColorCode(distance=5)   # [[19, 1, 5]]
    >>> code = ColorCode(distance=7)   # [[37, 1, 7]]
    """

    SUPPORTED_LAYOUTS = frozenset({'superdense', 'raw', 'triangular', 'rectangle'})

    def _process_params(self):
        self.distance = self.params.get('distance')
        self.shift = self.params.get('shift', (0, 0))
        self.layout = self.params.get('layout', 'superdense')

        if self.distance is None:
            raise ValueError("'distance' must be provided.")
        if self.distance < 3:
            raise ValueError(f"distance must be >= 3, got {self.distance}")
        if self.distance % 2 == 0:
            raise ValueError(f"distance must be odd, got {self.distance}")
        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two numbers.")
        if self.layout not in self.SUPPORTED_LAYOUTS:
            raise ValueError(
                f"Unknown color-code layout {self.layout!r}. "
                f"Supported layouts: {sorted(self.SUPPORTED_LAYOUTS)}"
            )
        self.L = round(3 * (self.distance - 1) / 2)  # Kept for compatibility/reporting.
        self.has_syndrome_qubits = self.layout != 'rectangle'
        self.shared_syndrome_qubit = self.layout in {'raw', 'triangular'}
        self.tiles: List[ColorCodeTile] = []
        self.lattice = None

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        self.lattice = self._make_lattice()
        self._build_layout_from_tiles(self._make_tiles(self.lattice))
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    @staticmethod
    def _sorted_coords(coords):
        return sorted(coords, key=lambda c: (c[0], c[1]))

    @staticmethod
    def _tile_group_key(tile: ColorCodeTile):
        return tuple(tile.ordered_data_coords)

    @staticmethod
    def _face_center(x_coord, z_coord):
        return (
            (x_coord[0] + z_coord[0]) / 2,
            (x_coord[1] + z_coord[1]) / 2,
        )

    def _register_data_qubits_from_tiles(self, tiles: Iterable[ColorCodeTile]):
        data_coords = self._sorted_coords(
            {
                coord
                for tile in tiles
                for coord in tile.ordered_data_coords
                if coord is not None
            }
        )
        for coord in data_coords:
            self.add_qubit(*coord, role='data')
        return data_coords

    def _make_lattice(self) -> ColorCodeLattice:
        if self.layout in {'raw', 'triangular'}:
            return self._make_raw_grid_lattice()
        if self.layout == 'superdense':
            return self._make_superdense_lattice()
        if self.layout == 'rectangle':
            return self._make_middle_out_lattice()
        raise AssertionError(f"Unexpected color-code layout {self.layout!r}")

    def _raw_grid_to_layout_coord(self, x: int, y: int) -> Coord:
        if self.layout == 'raw':
            return QECPatch.snap_coord((x, y))
        return QECPatch.snap_coord((x + y / 2, y * HEX_ROW_HEIGHT))

    def _make_raw_grid_lattice(self) -> ColorCodeLattice:
        """Build the raw grid, then optionally transform it to triangular."""
        side = self.L + 1
        raw_grid_points = {
            (x, y)
            for y in range(side)
            for x in range(side - y)
        }
        raw_face_points = {
            (x, y)
            for x, y in raw_grid_points
            if x % 3 == RAW_FACE_X_BY_Y[y % 3]
        }
        raw_data_points = raw_grid_points - raw_face_points

        def layout_coord(point):
            x, y = point
            return self._raw_grid_to_layout_coord(x, y)

        faces = []
        for x, y in sorted(raw_face_points, key=lambda point: (point[1], point[0])):
            ordered_data_coords = tuple(
                layout_coord(candidate) if candidate in raw_data_points else None
                for candidate in (
                    (x + dx, y + dy)
                    for dx, dy in RAW_FACE_XY_OFFSETS
                )
            )
            if sum(data_coord is not None for data_coord in ordered_data_coords) < 4:
                continue
            faces.append(
                ColorCodeLatticeFace(
                    center=layout_coord((x, y)),
                    ordered_data_coords=ordered_data_coords,
                    color="rgb"[y % 3],
                )
            )

        return ColorCodeLattice(
            positions=tuple(
                layout_coord(point)
                for point in sorted(raw_grid_points, key=lambda point: (point[1], point[0]))
            ),
            data_coords=tuple(
                layout_coord(point)
                for point in sorted(raw_data_points, key=lambda point: (point[1], point[0]))
            ),
            candidate_face_coords=tuple(
                layout_coord(point)
                for point in sorted(raw_face_points, key=lambda point: (point[1], point[0]))
            ),
            faces=tuple(faces),
        )

    def _make_superdense_lattice(self) -> ColorCodeLattice:
        """Sweep compact superdense face origins and filter by patch boundary."""
        width = 2 * self.distance - 1

        def in_bounds(coord: Coord) -> bool:
            x, y = coord
            return (
                0 <= x < width
                and 0 <= y < width
                and 2 * y <= 3 * x
                and 2 * y <= 3 * (width - x)
            )

        faces = []
        data_coord_set = set()
        candidate_face_coords = []
        for face_x in range(-1, width, 2):
            first_y = (face_x // 2) % 2
            for face_y in range(first_y, width, 2):
                face_origin = (face_x, face_y)
                ordered_data_coords = tuple(
                    candidate if in_bounds(candidate) else None
                    for candidate in (
                        add_coord(face_origin, offset)
                        for offset in SUPERDENSE_DATA_XY_OFFSETS
                    )
                )
                if sum(coord is not None for coord in ordered_data_coords) < 4:
                    continue

                x_measurement_coord = face_origin
                z_measurement_coord = (face_x + 1, face_y)
                candidate_face_coords.append(self._face_center(x_measurement_coord, z_measurement_coord))
                data_coord_set.update(coord for coord in ordered_data_coords if coord is not None)
                faces.append(
                    ColorCodeLatticeFace(
                        center=self._face_center(x_measurement_coord, z_measurement_coord),
                        ordered_data_coords=ordered_data_coords,
                        color="rgb"[face_y % 3],
                        x_measurement_coord=x_measurement_coord,
                        z_measurement_coord=z_measurement_coord,
                    )
                )

        return ColorCodeLattice(
            positions=tuple(self._sorted_coords(data_coord_set | set(candidate_face_coords))),
            data_coords=tuple(self._sorted_coords(data_coord_set)),
            candidate_face_coords=tuple(self._sorted_coords(candidate_face_coords)),
            faces=tuple(faces),
        )

    @staticmethod
    def _middle_out_column_heights(distance: int) -> Tuple[int, ...]:
        heights = [1]

        if distance % 4 == 1:
            peak = (distance - 1) // 4
            for step in range(1, peak + 1):
                heights.extend([6 * step, 6 * step + 1])
            for step in range(peak, 0, -1):
                heights.extend([6 * step - 1, 6 * step - 2])
        else:
            bend = (distance + 1) // 4
            for step in range(1, bend):
                heights.extend([6 * step, 6 * step + 1])
            heights.extend([6 * bend - 2, 6 * bend - 2])
            for step in range(bend - 1, 0, -1):
                heights.extend([6 * step - 1, 6 * step - 2])

        return tuple(heights)

    @staticmethod
    def _middle_out_odd_y_color(y: int) -> Color:
        return ("g", "b", "r")[((y - 1) // 2) % 3]

    @staticmethod
    def _middle_out_even_y_color(y: int) -> Color:
        return ("b", "r", "g")[((y - 2) // 2) % 3]

    def _make_middle_out_lattice(self) -> ColorCodeLattice:
        """Build rectangle/middle-out supports from positions plus one offset."""
        column_heights = self._middle_out_column_heights(self.distance)
        positions = {
            (x, y)
            for x, height in enumerate(column_heights)
            for y in range(height)
        }
        y_max = max(column_heights) - 1
        faces = []

        def data_from_offsets(origin: Coord) -> Tuple[Coord, ...]:
            return tuple(
                coord
                for coord in (
                    add_coord(origin, offset)
                    for offset in MIDDLE_OUT_FACE_XY_OFFSETS
                )
                if coord in positions
            )

        def is_boundary_corner_pair(origin: Coord, data_coords: Tuple[Coord, ...]) -> bool:
            if len(data_coords) != 2:
                return False
            columns = {x for x, _ in data_coords}
            rows = sorted(y for _, y in data_coords)
            if len(columns) != 1 or rows[1] != rows[0] + 1:
                return False

            column = next(iter(columns))
            top_y = rows[0]
            origin_x, _ = origin
            if column == origin_x + 1:
                return (column + 1, top_y + 2) in positions
            if column == origin_x:
                return (column - 1, top_y + 1) in positions
            return False

        def is_valid_support(origin: Coord, data_coords: Tuple[Coord, ...]) -> bool:
            return len(data_coords) in (4, 6) or is_boundary_corner_pair(origin, data_coords)

        def measurement_coord(origin: Coord, data_coords: Tuple[Coord, ...]) -> Coord:
            if len(data_coords) == 2:
                column = next(iter({x for x, _ in data_coords}))
                return (column, min(y for _, y in data_coords))

            x, y = origin
            if x % 2 == 0:
                return (x + 1, y)
            return (x, y + 2)

        def tile_color(origin: Coord, data_coords: Tuple[Coord, ...]) -> Color:
            x, y = origin
            if len(data_coords) == 2 or x % 2 == 0:
                return self._middle_out_odd_y_color(y + 1)
            return self._middle_out_even_y_color(y + 3)

        # One 2x3 rectangle offset generates both checkerboard orientations.
        # Even columns start at y=0; odd columns start one step lower at y=-1.
        for x in range(self.distance):
            y_start = 0 if x % 2 == 0 else -1
            for y in range(y_start, y_max + 1, 2):
                origin = (x, y)
                data_coords = data_from_offsets(origin)
                if not is_valid_support(origin, data_coords):
                    continue

                faces.append(
                    ColorCodeLatticeFace(
                        center=measurement_coord(origin, data_coords),
                        ordered_data_coords=data_coords,
                        color=tile_color(origin, data_coords),
                    )
                )

        return ColorCodeLattice(
            positions=tuple(self._sorted_coords(positions)),
            data_coords=tuple(self._sorted_coords(positions)),
            candidate_face_coords=tuple(self._sorted_coords(face.center for face in faces)),
            faces=tuple(faces),
        )

    def _make_tiles(self, lattice: ColorCodeLattice):
        tiles = []
        for face in lattice.faces:
            if self.layout == 'rectangle':
                tiles.append(
                    ColorCodeTile(
                        basis=None,
                        measurement_coord=face.center,
                        ordered_data_coords=face.ordered_data_coords,
                        color=face.color,
                    )
                )
                continue

            tiles.extend(
                make_xz_tiles(
                    x_measurement_coord=face.x_measurement_coord or face.center,
                    z_measurement_coord=face.z_measurement_coord or face.center,
                    data_coords=face.ordered_data_coords,
                    color=face.color,
                )
            )
        return sorted_tiles(tiles)

    def _build_layout_from_tiles(self, tiles):
        self.tiles = list(tiles)
        self.faces = []
        data_coords = self._register_data_qubits_from_tiles(self.tiles)

        if not self.has_syndrome_qubits:
            for tile in self.tiles:
                data_neighbors = [
                    (coord, self.index_map[coord])
                    for coord in tile.data_coords
                ]
                self.faces.append({
                    'center': tile.measurement_coord,
                    'color': tile.color,
                    'data_neighbors': data_neighbors,
                    'layout': self.layout,
                    'tile': tile,
                })
                for basis in ('X', 'Z'):
                    self.create_stim_stabilizer(
                        {coord: basis for coord in tile.data_coords},
                        type=basis,
                    )

            self.create_stim_logical(
                {coord: 'X' for coord in data_coords},
                'X',
            )
            self.create_stim_logical(
                {coord: 'Z' for coord in data_coords},
                'Z',
            )
            self.num_logicals = 1
            return

        face_groups: Dict[Any, Dict[str, ColorCodeTile]] = {}
        for tile in self.tiles:
            face_groups.setdefault(self._tile_group_key(tile), {})[tile.basis] = tile

        for key, group in sorted(
            face_groups.items(),
            key=lambda item: (
                min(c[0] for c in item[0] if c is not None),
                min(c[1] for c in item[0] if c is not None),
                item[1]['X'].measurement_coord,
            ),
        ):
            x_tile = group['X']
            z_tile = group['Z']

            if self.layout == 'superdense':
                x_coord = x_tile.measurement_coord
                z_coord = z_tile.measurement_coord
                x_idx = self.add_qubit(*x_coord, role='syndrome_x')
                z_idx = self.add_qubit(*z_coord, role='syndrome_z')
            elif self.layout in {'raw', 'triangular'}:
                center = self._face_center(
                    x_tile.measurement_coord,
                    z_tile.measurement_coord,
                )
                x_coord = z_coord = center
                shared_idx = self.add_qubit(*center, role='syndrome')
                self.syndrome_indices_x.add(shared_idx)
                self.syndrome_indices_z.add(shared_idx)
                x_idx = z_idx = shared_idx
            else:
                raise AssertionError(f"Unexpected face layout {self.layout!r}")

            data_neighbors = [
                None if coord is None else (coord, self.index_map[coord])
                for coord in x_tile.ordered_data_coords
            ]
            face_data_coords = [coord for coord in key if coord is not None]
            face = {
                'center': self._face_center(x_coord, z_coord),
                'color': x_tile.color,
                'z_ancilla_coord': z_coord,
                'x_ancilla_coord': x_coord,
                'z_ancilla_idx': z_idx,
                'x_ancilla_idx': x_idx,
                'data_neighbors': data_neighbors,
                'layout': self.layout,
                'x_tile': x_tile,
                'z_tile': z_tile,
            }
            self.faces.append(face)

            self.create_stim_stabilizer(
                {coord: 'X' for coord in face_data_coords},
                syn_coord=x_coord,
                type='X',
            )
            self.create_stim_stabilizer(
                {coord: 'Z' for coord in face_data_coords},
                syn_coord=z_coord,
                type='Z',
            )

        boundary_data = [coord for coord in data_coords if coord[1] == 0]
        self.create_stim_logical({coord: 'X' for coord in boundary_data}, 'X')
        self.create_stim_logical({coord: 'Z' for coord in boundary_data}, 'Z')
        self.num_logicals = 1

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
            'layout': self.layout,
            'has_syndrome_qubits': self.has_syndrome_qubits,
            'shared_syndrome_qubit': self.shared_syndrome_qubit,
        })
        return info
