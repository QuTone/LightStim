"""Degenerate XXZZ "Y boundary" rotated surface-code patch.

This is a line-by-line port of Craig Gidney's ``make_ztop_yboundary_patch``
from ``code/src/midout/circuits/steps/_patches.py`` (zenodo record 7487893,
*Inplace Access to the Surface Code Y Basis*), expressed in LightStim's
:class:`~lightstim.ir.qec_patch.QECPatch` record format.

What the patch is
-----------------
A ``distance x distance`` rectangular surface-code patch whose four boundaries
are ``top=Z  left=Z  bottom=X  right=X``.  Because the top-left corner sits at
the meeting point of two *same-basis* (Z) boundaries, the generic construction
in :func:`_surface_code_tiles` keeps no measure qubit that would hold it, so
the corner data qubit ``g = 0`` drops out of the code.  What is left is a
patch on ``d*d - 1`` data qubits carrying ``d*d - 1`` independent checks —
i.e. **zero logical qubits**.  It is the "degenerate" half of Gidney's Y-basis
trick: the qubit patch's ``Y_L`` is transferred onto this stabilizer group by
the transition round in :mod:`.y_transition_round`.

Exactly one check has weight 3 (the Z check whose measure qubit is Gidney's
``0.5+0.5j``, LightStim ``(2, 2)``): it is the weight-4 corner check with the
deleted corner data qubit removed.  Every other check has weight 2 or 4.

Coordinate bridge
-----------------
Gidney's data qubits live at integer complex coordinates ``gx + gy*1j`` with
``gx, gy in [0, distance)``, his measure qubits at half-integers.  LightStim
uses ``(2*gx + 1, 2*gy + 1)``, which sends data onto odd integers and measure
qubits onto even integers.  Under this map ``RotatedSurfaceCode(distance=d)``
is tile-for-tile equal to Gidney's ``make_xtop_qubit_patch(distance=d)``, so
the two patches share one global lattice.

Faithfulness notes
------------------
* ``DIRS`` / ``checkerboard_basis`` / the ``is_boundary`` predicate / the
  ``measure_qubits`` and ``data_qubits`` filters are transcribed unchanged.
* ``order_func`` is transcribed including the ``and False`` branch that Gidney
  left disabled, so the surviving branch (and therefore the interaction order
  stored in each record's ``data_indices``) is his.
* Gidney's ``ordered_data_qubits`` may contain ``None`` placeholders for the
  interaction layers in which a measure qubit idles.  LightStim's record format
  has no slot for an idle layer, so the ``None`` entries are dropped; the
  surviving order is unchanged.
"""
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from lightstim.ir.qec_patch import QECPatch

__all__ = [
    "DIRS", "DR", "DL", "UL", "UR",
    "checkerboard_basis",
    "make_ztop_yboundary_tiles",
    "DegenerateYBoundaryPatch",
    "make_degenerate_y_boundary_patch",
]

# --- Gidney _patches.py, verbatim -------------------------------------------
DIRS = [(0.5 + 0.5j) * 1j ** d for d in range(4)]
DR, DL, UL, UR = DIRS


def checkerboard_basis(q: complex) -> str:
    """Gidney ``gen/_surface_code.py``: classify a coordinate X- or Z-type."""
    is_x = int(q.real + q.imag) & 1 == 0
    return 'X' if is_x else 'Z'


# A tile as produced here: (basis, measurement qubit, ordered data qubits).
Tile = Tuple[str, complex, Tuple[complex, ...]]


def _surface_code_tiles(
        *,
        possible_data_qubits: Iterable[complex],
        basis: Callable[[complex], str],
        is_boundary_x: Callable[[complex], bool],
        is_boundary_z: Callable[[complex], bool],
        order_func: Callable[[complex], Iterable[Optional[complex]]],
) -> List[Tile]:
    """Port of Gidney's ``surface_code_patch``."""
    possible_data_qubits = set(possible_data_qubits)
    possible_measure_qubits = {
        q + d
        for q in possible_data_qubits
        for d in DIRS
    }
    measure_qubits = {
        m
        for m in possible_measure_qubits
        if sum(m + d in possible_data_qubits for d in DIRS) > 1
        if is_boundary_x(m) <= (basis(m) == 'X')
        if is_boundary_z(m) <= (basis(m) == 'Z')
    }
    data_qubits = {
        q
        for q in possible_data_qubits
        if sum(q + d in measure_qubits for d in DIRS) > 1
    }

    tiles: List[Tile] = []
    for m in measure_qubits:
        ordered = tuple(
            m + d if d is not None and m + d in data_qubits else None
            for d in order_func(m)
        )
        tiles.append((basis(m), m, ordered))
    return tiles


def _rectangular_surface_code_tiles(
        *,
        width: int,
        height: int,
        top_basis: str,
        bot_basis: str,
        left_basis: str,
        right_basis: str,
        order_func: Callable[[complex], Iterable[Optional[complex]]],
) -> List[Tile]:
    """Port of Gidney's ``rectangular_surface_code_patch``."""
    def is_boundary(m: complex, *, b: str) -> bool:
        if top_basis == b and m.imag == -0.5:
            return True
        if left_basis == b and m.real == -0.5:
            return True
        if bot_basis == b and m.imag == height - 0.5:
            return True
        if right_basis == b and m.real == width - 0.5:
            return True
        return False

    return _surface_code_tiles(
        possible_data_qubits=[
            x + 1j * y
            for x in range(width)
            for y in range(height)
        ],
        basis=checkerboard_basis,
        is_boundary_x=lambda m: is_boundary(m, b='X'),
        is_boundary_z=lambda m: is_boundary(m, b='Z'),
        order_func=order_func,
    )


def make_ztop_yboundary_tiles(distance: int) -> List[Tile]:
    """Port of Gidney's ``make_ztop_yboundary_patch``, in HIS coordinates.

    Returned tiles are sorted by measure qubit so the result is deterministic
    (Gidney iterates a Python ``set``).
    """
    def order_func(m: complex) -> List[complex]:
        order_S = [UR, UL, DR, DL]
        order_N = [UR, DR, UL, DL]
        if m.real > m.imag and False:
            if checkerboard_basis(m) == 'X':
                return order_N
            else:
                return order_S
        else:
            if checkerboard_basis(m) == 'X':
                return order_S
            else:
                return order_N

    tiles = _rectangular_surface_code_tiles(
        width=distance,
        height=distance,
        top_basis='Z',
        right_basis='X',
        bot_basis='X',
        left_basis='Z',
        order_func=order_func,
    )
    return sorted(tiles, key=lambda t: (t[1].imag, t[1].real))


# --- LightStim wrapper -------------------------------------------------------

def _to_lightstim(g: complex) -> Tuple[int, int]:
    """Gidney complex coordinate -> LightStim integer coordinate."""
    x = 2 * g.real + 1
    y = 2 * g.imag + 1
    assert x == int(x) and y == int(y), g
    return (int(x), int(y))


class DegenerateYBoundaryPatch(QECPatch):
    """The ``top=Z left=Z bottom=X right=X`` patch with the corner deleted.

    Parameters (via ``**kwargs``)
    -----------------------------
    distance : int
        Odd code distance of the *parent* qubit patch.  The degenerate patch
        itself holds ``distance**2 - 1`` data qubits and no logical qubit.
    shift : Tuple[float, float], optional
        Global offset applied after construction, as in
        :class:`~lightstim.qec_code.surface_code.rotated.code_patch.RotatedSurfaceCode`.
    """

    def _process_params(self):
        self.distance = self.params.get("distance")
        if self.distance is None:
            raise ValueError("'distance' must be provided.")
        if not isinstance(self.distance, int):
            raise ValueError("'distance' must be an integer.")
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("'distance' must be an odd integer >= 3.")
        self.shift = self.params.get("shift", (0, 0))
        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two numbers.")

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    def build(self):
        tiles = make_ztop_yboundary_tiles(self.distance)

        # ------------------------------------------------------------------
        # Phase 1: geometry.  Row-major (y then x) over data + measure qubits,
        # matching RotatedSurfaceCode's reading order.  uid is passed
        # explicitly — QECPatch.add_qubit's default ``uid = len(qubit_coords)``
        # collides with a live index as soon as anything is deleted.
        # ------------------------------------------------------------------
        roles: Dict[Tuple[int, int], str] = {}
        for basis, m, ordered in tiles:
            roles[_to_lightstim(m)] = 'syndrome_x' if basis == 'X' else 'syndrome_z'
            for q in ordered:
                if q is not None:
                    roles.setdefault(_to_lightstim(q), 'data')

        for uid, coord in enumerate(sorted(roles, key=lambda c: (c[1], c[0]))):
            self.add_qubit(coord[0], coord[1], role=roles[coord], uid=uid)

        # ------------------------------------------------------------------
        # Phase 2: physics.  Gidney's interaction order is preserved in each
        # record's ``data_indices`` (None placeholders dropped).
        # ------------------------------------------------------------------
        for basis, m, ordered in tiles:
            data = [_to_lightstim(q) for q in ordered if q is not None]
            syn_coord = _to_lightstim(m)
            targets = {c: basis for c in data}
            assert len(targets) == len(data), f"duplicate data qubit in tile at {m}"
            self.create_stim_stabilizer(targets, syn_coord, basis)
            # create_stim_stabilizer SILENTLY drops coordinates that are not in
            # index_map, so the weight must be checked afterwards.
            record = self.stabilizers[-1]
            if len(record['data_indices']) != len(data):
                raise AssertionError(
                    f"stabilizer at {syn_coord} lost data qubits: "
                    f"wanted {data}, kept "
                    f"{[self.qubit_coords[i] for i in record['data_indices']]}"
                )
            if record['syn_idx'] is None:
                raise AssertionError(f"syndrome qubit {syn_coord} was not registered")

        # ------------------------------------------------------------------
        # Phase 3: no logical qubit lives here.
        # ------------------------------------------------------------------
        self.logical_ops = []
        self.num_logicals = 0

        # ------------------------------------------------------------------
        # Phase 4: shift
        # ------------------------------------------------------------------
        if tuple(self.shift) != (0, 0):
            self.shift_coords(*self.shift)

    def get_info(self):
        info = super().get_info()
        info.update({
            'distance': self.distance,
            'num_data_qubits': len(self.data_indices),
            'num_x_syndromes': len(self.syndrome_indices_x),
            'num_z_syndromes': len(self.syndrome_indices_z),
            'data_coords': self.data_coords,
            'syndrome_coords_x': self.syndrome_coords_x,
            'syndrome_coords_z': self.syndrome_coords_z,
            'stabilizers': self.stabilizers,
            'num_logicals': self.num_logicals,
        })
        return info


def make_degenerate_y_boundary_patch(distance: int,
                                     shift: Tuple[float, float] = (0, 0)
                                     ) -> DegenerateYBoundaryPatch:
    """Build the degenerate XXZZ Y-boundary patch of the given distance.

    >>> p = make_degenerate_y_boundary_patch(3)
    >>> len(p.data_indices), len(p.stabilizers), p.num_logicals
    (8, 8, 0)
    >>> (1, 1) in p.index_map          # the corner data qubit is gone
    False
    """
    return DegenerateYBoundaryPatch(distance=distance, shift=shift)
