"""Minimal geometry pieces used by color-code patch builders.

This module intentionally does not decide the lattice range or boundary.  It
only defines reusable local tile geometry and the concrete ``ColorCodeTile``
container.  ``code_patch.py`` owns the distance-dependent lattice sweep,
membership filtering, and optional triangular embedding.
"""

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

Coord = Tuple[float, float]
Basis = Literal["X", "Z"]
Color = Literal["r", "g", "b"]


@dataclass(frozen=True)
class ColorCodeTile:
    """One stabilizer footprint in a concrete color-code layout."""

    basis: Basis | None
    measurement_coord: Coord
    ordered_data_coords: Tuple[Coord | None, ...]
    color: Color

    # ``ordered_data_coords`` preserves schedule slots around the tile.  This
    # matters for superdense/triangular SE blocks.  Rectangle/middle-out uses
    # tiles only as geometric supports; its schedule comes from
    # ``RectangleMiddleOutPlan``.

    @property
    def data_coords(self) -> Tuple[Coord, ...]:
        return tuple(coord for coord in self.ordered_data_coords if coord is not None)

    @property
    def weight(self) -> int:
        return len(self.data_coords)


# Raw grid used by the raw layout.  These offsets are written as ``(dx, dy)``
# around a face center ``(x, y)``.  The triangular layout transforms this same
# grid into an equilateral-triangle/regular-hexagon embedding.
RAW_FACE_X_BY_Y = (1, 2, 0)
RAW_FACE_XY_OFFSETS = (
    (-1, 1),
    (0, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
)
# Around a raw-grid face center ``(x, y)``:
#
#             4(x,y-1) - 3(x+1,y-1)
#                |            |
# 5(x-1,y)  -  center  -  2(x+1,y)
#     |            |
# 0(x-1,y+1) - 1(x,y+1)


# Row height used only by triangular layout's equilateral embedding.
HEX_ROW_HEIGHT = math.sqrt(3) / 2


# Superdense compact-coordinate footprint.  These ARE QECPatch/Stim ``(x, y)``
# offsets from the X-check measurement coordinate ``(face_x, face_y)``.  This
# diagram is drawn in the usual screen/Stim visualization frame: x increases to
# the right, y increases downward.
SUPERDENSE_DATA_XY_OFFSETS = (
    (-1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (1, -1),
    (0, -1),
)
# Around a superdense X-check coordinate ``(x, y)``:
#
#             5(x,y-1) - 4(x+1,y-1)
#                |            |
# 0(x-1,y)  -  X-check  -  Z-check (x+1,y) - 3(x+2,y)
#                |            |
#             1(x,y+1) - 2(x+1,y+1)


# Rectangle/middle-out geometry.  These ARE QECPatch/Stim ``(x, y)`` offsets
# from the upper-left corner of a 2x3 data-qubit footprint.  Boundary
# membership filtering turns this into 6-, 4-, and 2-qubit supports.  This
# diagram is drawn in the usual screen/Stim visualization frame: x increases to
# the right, y increases downward.
MIDDLE_OUT_FACE_XY_OFFSETS = (
    (1, 0),
    (1, 1),
    (1, 2),
    (0, 2),
    (0, 1),
    (0, 0),
)
# Around a middle-out rectangle origin ``(x, y)``:
#
# 5(x,y)   - 0(x+1,y)
#   |           |
# 4(x,y+1) - 1(x+1,y+1)
#   |           |
# 3(x,y+2) - 2(x+1,y+2)


def add_coord(coord: Coord, offset: Coord) -> Coord:
    return (coord[0] + offset[0], coord[1] + offset[1])


def make_xz_tiles(
    *,
    x_measurement_coord: Coord,
    z_measurement_coord: Coord,
    data_coords: Iterable[Coord | None],
    color: Color,
) -> Tuple[ColorCodeTile, ColorCodeTile]:
    data_coords = tuple(data_coords)
    return (
        ColorCodeTile(
            basis="X",
            measurement_coord=x_measurement_coord,
            ordered_data_coords=data_coords,
            color=color,
        ),
        ColorCodeTile(
            basis="Z",
            measurement_coord=z_measurement_coord,
            ordered_data_coords=data_coords,
            color=color,
        ),
    )


def sorted_tiles(tiles: Iterable[ColorCodeTile]) -> Tuple[ColorCodeTile, ...]:
    def sort_key(tile: ColorCodeTile):
        return (*tile.measurement_coord, "" if tile.basis is None else tile.basis)

    return tuple(sorted(tiles, key=sort_key))
