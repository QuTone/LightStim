"""Middle-out extraction plan for the rectangle color-code layout."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class RectangleMiddleOutPlan:
    """Schedule data for the no-ancilla rectangle middle-out circuit.

    The rectangle patch supplies only data-qubit coordinates.  This plan
    describes how the inline-folding extraction circuit uses those data qubits:
    which CNOT layers fold/unfold the checks, which qubits are measured during
    each half-cycle, and which mixed X/Z initialization basis matches the
    paper/sample circuit frame.
    """

    cnot_layers: List[List[Tuple[int, int]]]
    # These two sets are named by their FIRST half-cycle measurement basis.
    # The middle-out block alternates basis within one extraction cycle:
    #   first half:  MX(x_measure_indices), MZ(z_measure_indices)
    #   second half: MX(z_measure_indices), MZ(x_measure_indices)
    # Thus these are not permanent qubit roles; they are schedule slots in an
    # alternating X/Z inline-folding circuit.
    x_measure_indices: List[int]
    z_measure_indices: List[int]
    initial_z_indices: List[int]
    initial_x_indices: List[int]

    @classmethod
    def from_patch(cls, patch: Any) -> "RectangleMiddleOutPlan":
        x_measure_indices = cls._x_measure_indices(patch)
        z_measure_indices = cls._z_measure_indices(patch)
        initial_z_indices = list(x_measure_indices)
        initial_x_indices = sorted(patch.data_indices - set(initial_z_indices))
        return cls(
            cnot_layers=cls._cnot_layers(patch),
            x_measure_indices=x_measure_indices,
            z_measure_indices=z_measure_indices,
            initial_z_indices=initial_z_indices,
            initial_x_indices=initial_x_indices,
        )

    def initial_basis_map(
        self,
        data_indices,
        local_to_global_map=None,
        *,
        memory_basis: str,
    ) -> Dict[int, str]:
        """Return the mixed initialization basis for X- or Z-memory."""
        memory_basis = memory_basis.upper()
        if memory_basis not in ("X", "Z"):
            raise ValueError(
                f"Middle-out memory basis must be 'X' or 'Z'; got {memory_basis!r}"
            )

        local_to_global_map = local_to_global_map or {}
        result = {}
        z_basis = set(self.initial_z_indices)
        z_measure = set(self.z_measure_indices)

        # The representative qubits always use the mixed X/Z frame required by
        # inline folding. Only passive data qubits follow the logical memory basis.
        for idx in self.z_measure_indices:
            result[local_to_global_map.get(idx, idx)] = 'X'
        for idx in sorted(data_indices - z_basis - z_measure):
            result[local_to_global_map.get(idx, idx)] = memory_basis
        for idx in self.initial_z_indices:
            result[local_to_global_map.get(idx, idx)] = 'Z'
        return result

    @staticmethod
    def _cnot_layers(patch: Any) -> List[List[Tuple[int, int]]]:
        coord_to_idx = {coord: idx for idx, coord in patch.qubit_coords.items()}
        coords = sorted((idx, *coord) for idx, coord in patch.qubit_coords.items())

        rung_m = []
        rail_b = []
        rail_a = []
        for idx, x, y in coords:
            if (x + 1, y) in coord_to_idx and int(x + y) % 2 == 0:
                rung_m.append((idx, coord_to_idx[(x + 1, y)]))
            if int(y) % 2 == 1 and (x, y - 1) in coord_to_idx:
                rail_b.append((idx, coord_to_idx[(x, y - 1)]))
            if int(y) % 2 == 0 and (x, y - 1) in coord_to_idx:
                rail_a.append((idx, coord_to_idx[(x, y - 1)]))

        return [
            rung_m,
            rail_b,
            rail_a,
            [(b, a) for a, b in rail_a],
            [(b, a) for a, b in rail_b],
            [(b, a) for a, b in rung_m],
        ]

    @staticmethod
    def _z_measure_indices(patch: Any) -> List[int]:
        # First half-cycle Z-basis representatives. These same qubits are
        # measured in X basis during the second half-cycle.
        return sorted(
            idx for idx, (x, y) in patch.qubit_coords.items()
            if int(x) % 2 == 1 and int(y) % 2 == 1
        )

    @staticmethod
    def _x_measure_indices(patch: Any) -> List[int]:
        # First half-cycle X-basis representatives. These same qubits are
        # measured in Z basis during the second half-cycle.
        # The excluded left-diagonal holes and included right-boundary spurs
        # both occur at y=6j+2 in the zero-based rectangle coordinates.
        bend_count = (patch.distance + 1) // 4
        hole_coords = {(2 * j + 1, 6 * j + 2) for j in range(bend_count)}
        spur_coords = {
            (patch.distance - 1 - 2 * j, 6 * j + 2)
            for j in range(bend_count)
        }

        result = []
        for idx, (x, y) in patch.qubit_coords.items():
            xi = int(x)
            yi = int(y)
            if (xi, yi) in spur_coords:
                # Boundary spurs keep the paper circuit's folding wavefront on
                # the same six-layer cadence as the bulk.
                result.append(idx)
            elif xi % 2 == 1 and yi % 2 == 0 and (xi, yi) not in hole_coords:
                result.append(idx)
        return sorted(result)
