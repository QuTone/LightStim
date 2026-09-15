"""Jones's even-length H family, preserving the original Magic-H6 convention."""

from math import isfinite
from numbers import Integral, Real
from typing import List, Sequence, Tuple

from lightstim.ir.qec_patch import QECPatch


def _validate_n(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 6 or n % 2:
        raise ValueError("'n' must be an even integer >= 6.")
    return int(n)


class HCode(QECPatch):
    """The self-dual CSS ``[[n, n-4, 2]]`` H family, for even ``n >= 6``.

    Checks of both Pauli types have supports A=(0,1,2,3), B=(2,3,...,n-1).
    Logical slot j=q-4 uses (0,2,q) for even q and (1,3,q) for odd q, for
    both X and Z. Logical pairs are registered as (X0,Z0,X1,Z1,...).
    This is a relabeling/change of logical basis of Jones, arXiv:1210.3388,
    Section II; n=6 reproduces Maggie Bao's LightStim H6 patch (PR #98).

    Data i sits at (2*i,0). Check ancillas sit above/below their support
    centroids: (3,+/-1), (n+1,+/-1). Coordinates describe a Tanner layout,
    without assuming local hardware connectivity. ``shift=(dx,dy)`` moves
    the whole patch. Only the four syndrome ancillas belong to the patch.
    """

    def __init__(self, n: int = 6, **kwargs):
        super().__init__(n=n, **kwargs)

    def _process_params(self):
        unknown = self.params.keys() - {"n", "shift"}
        if unknown:
            raise ValueError(f"Unknown HCode parameters: {sorted(unknown)}")
        self.n = _validate_n(self.params["n"])
        shift = self.params.get("shift", (0, 0))
        if not (
            isinstance(shift, (tuple, list)) and len(shift) == 2
            and all(isinstance(v, Real) and isfinite(v) for v in shift)
        ):
            raise ValueError("'shift' must contain two finite real coordinates.")
        self._initial_shift = tuple(shift)
        self.shift = (0, 0)  # QECPatch.shift_coords accumulates this itself.
        self._X_CHECKS = self._Z_CHECKS = ((0, 1, 2, 3), tuple(range(2, self.n)))
        self._X_LOGICALS = self._Z_LOGICALS = tuple(
            (0, 2, q) if q % 2 == 0 else (1, 3, q) for q in range(4, self.n)
        )

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]

    @staticmethod
    def _data_coord(label: int) -> Tuple[float, float]:
        return (2 * label, 0)

    @staticmethod
    def get_stabs(bits: Sequence[int]) -> List[int]:
        """Two same-basis check parities; family size is inferred from len(bits)."""
        _validate_n(len(bits))
        return [sum(bits[:4]) % 2, sum(bits[2:]) % 2]

    @staticmethod
    def get_logicals(bits: Sequence[int]) -> List[int]:
        """Logical parities in canonical order; family size is len(bits)."""
        _validate_n(len(bits))
        return [
            (bits[0] + bits[2] + bits[q]) % 2 if q % 2 == 0
            else (bits[1] + bits[3] + bits[q]) % 2
            for q in range(4, len(bits))
        ]

    def build(self):
        for label in range(self.n):
            self.add_qubit(*self._data_coord(label), role="data")

        for basis, y in (("X", 1), ("Z", -1)):
            for support in self._X_CHECKS:
                coord = (sum(2 * q for q in support) / len(support), y)
                self.add_qubit(*coord, role=f"syndrome_{basis.lower()}")
                self.create_stim_stabilizer(
                    {self._data_coord(q): basis for q in support}, coord, basis
                )

        for support in self._X_LOGICALS:
            for basis in ("X", "Z"):
                self.create_stim_logical(
                    {self._data_coord(q): basis for q in support}, basis
                )
        self.num_logicals = self.n - 4
        if self._initial_shift != (0, 0):
            self.shift_coords(*self._initial_shift)

    def get_info(self):
        info = super().get_info()
        info.update(
            code_distance=2, k=self.num_logicals, n_data=self.n,
            num_x_syndromes=len(self.syndrome_indices_x),
            num_z_syndromes=len(self.syndrome_indices_z),
            data_coords=self.data_coords, syndrome_coords_x=self.syndrome_coords_x,
            syndrome_coords_z=self.syndrome_coords_z, syndrome_coords=self.syndrome_coords,
            stabilizers=self.stabilizers, logical_ops=self.logical_ops,
            index_map=self.index_map, qubit_coords=self.qubit_coords,
            num_logicals=self.num_logicals,
        )
        return info


class HSixCode(HCode):
    """Compatibility constructor for HCode(n=6), with the original logicals."""

    def __init__(self, **kwargs):
        n = _validate_n(kwargs.pop("n", 6))
        if n != 6:
            raise ValueError("HSixCode requires n=6; use HCode for other sizes.")
        super().__init__(n=6, **kwargs)

    @staticmethod
    def get_stabs(bits: Sequence[int]) -> List[int]:
        if len(bits) != 6:
            raise ValueError("HSixCode parity helpers require six bits.")
        return HCode.get_stabs(bits)

    @staticmethod
    def get_logicals(bits: Sequence[int]) -> List[int]:
        if len(bits) != 6:
            raise ValueError("HSixCode parity helpers require six bits.")
        return HCode.get_logicals(bits)
