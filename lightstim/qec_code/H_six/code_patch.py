from typing import List, Sequence, Tuple

from lightstim.ir.qec_patch import QECPatch


class HSixCode(QECPatch):
    """The ``[[6, 2, 2]]`` CSS code used by Quantinuum's Magic-H6 distillation.

    Six data qubits on a line -- data qubit label ``i`` sits at ``(2*i, 0)`` --
    encoding ``k = 2`` logical qubits with distance ``d = 2`` (error-detecting).

    The X and Z checks share the same support, so a transversal Hadamard maps the
    stabilizer group to itself and swaps ``X_L <-> Z_L`` on each logical qubit;
    that self-duality is what makes the code useful for distilling ``|H>`` states.

        S_x1 = X0 X1 X2 X3        S_z1 = Z0 Z1 Z2 Z3
        S_x2 = X2 X3 X4 X5        S_z2 = Z2 Z3 Z4 Z5

        X0_L = X0 X2 X4           Z0_L = Z0 Z2 Z4
        X1_L = X1 X3 X5           Z1_L = Z1 Z3 Z5

    The checks match ``get_stabs`` and the logicals match ``get_logicals`` in the
    raw-stim Magic-H6 port (``Logical-Magic-State-Distillation-Circuits``).

    The code has no 2D locality; qubit coordinates are cosmetic (they only feed
    plotting and syndrome/data role inference). X ancillas are placed above the
    line, Z ancillas below, each centred over its 4-qubit support; any
    ``h_check_ancillas`` sit on a second row above the line.

    Parameters (via ``**kwargs``)
    -----------------------------
    shift : tuple[float, float], optional
        Global ``(dx, dy)`` offset applied to every qubit. Default ``(0, 0)``.
    h_check_ancillas : int, optional
        Number of bare ancillas (role ``"syndrome"``, attached to no stabilizer)
        to add for :class:`HSixLogicalXCheckBlock` -- Magic-H6's "H-check".
        ``HSixExtractionBlock`` ignores them. Default ``0``.

    Examples
    --------
    >>> code = HSixCode()
    >>> code = HSixCode(shift=(20, 0))          # for multi-block layouts
    >>> code = HSixCode.from_config({'shift': (0, 4)})
    """

    # Stabilizer supports and logical-operator supports, in the canonical
    # Magic-H6 qubit ordering (labels 0..5).
    _X_CHECKS = ((0, 1, 2, 3), (2, 3, 4, 5))
    _Z_CHECKS = ((0, 1, 2, 3), (2, 3, 4, 5))
    _X_LOGICALS = ((0, 2, 4), (1, 3, 5))
    _Z_LOGICALS = ((0, 2, 4), (1, 3, 5))

    def _process_params(self):
        self.shift = self.params.get("shift", (0, 0))
        if not (isinstance(self.shift, tuple) and len(self.shift) == 2):
            raise ValueError("'shift' must be a (dx, dy) tuple.")

        self.h_check_ancillas = int(self.params.get("h_check_ancillas", 0))
        if self.h_check_ancillas < 0:
            raise ValueError("'h_check_ancillas' must be a non-negative integer.")

    @staticmethod
    def _data_coord(label: int) -> Tuple[float, float]:
        """Coordinate of data qubit ``label`` (0..5)."""
        return (2 * label, 0)

    @staticmethod
    def get_stabs(bits: Sequence[int]) -> List[int]:
        """The two X- (or Z-) stabilizer parities of a 6-bit data string.

        Port of ``get_stabs`` in CQCL/Magic-H6 ``Code614.py``:
        ``S1 = b0+b1+b2+b3``, ``S2 = b2+b3+b4+b5`` (mod 2).
        """
        return [
            (bits[0] + bits[1] + bits[2] + bits[3]) % 2,
            (bits[2] + bits[3] + bits[4] + bits[5]) % 2,
        ]

    @staticmethod
    def get_logicals(bits: Sequence[int]) -> List[int]:
        """The two logical parities of a 6-bit data string.

        Port of ``get_logicals`` in CQCL/Magic-H6 ``Code614.py``:
        ``L0 = b0+b2+b4``, ``L1 = b1+b3+b5`` (mod 2) -- i.e. Z0_L / Z1_L for a
        Z-basis string, X0_L / X1_L for an X-basis string.
        """
        return [
            (bits[0] + bits[2] + bits[4]) % 2,
            (bits[1] + bits[3] + bits[5]) % 2,
        ]

    def build(self):
        # -- Phase 1: geometry -------------------------------------------------
        for label in range(6):
            self.add_qubit(*self._data_coord(label), role="data")

        x_syn_coords = [
            (sum(2 * q for q in support) / len(support), 1) for support in self._X_CHECKS
        ]
        z_syn_coords = [
            (sum(2 * q for q in support) / len(support), -1) for support in self._Z_CHECKS
        ]
        for coord in x_syn_coords:
            self.add_qubit(*coord, role="syndrome_x")
        for coord in z_syn_coords:
            self.add_qubit(*coord, role="syndrome_z")

        # Bare ancillas for HSixLogicalXCheckBlock (the Magic-H6 "H-check").
        # Attached to no stabilizer, so HSixExtractionBlock skips them.
        for i in range(self.h_check_ancillas):
            self.add_qubit(2 * i, 3, role="syndrome")

        # -- Phase 2: stabilizers -------------------------------------------
        for support, syn_coord in zip(self._X_CHECKS, x_syn_coords):
            self.create_stim_stabilizer(
                {self._data_coord(q): "X" for q in support}, syn_coord, "X"
            )
        for support, syn_coord in zip(self._Z_CHECKS, z_syn_coords):
            self.create_stim_stabilizer(
                {self._data_coord(q): "Z" for q in support}, syn_coord, "Z"
            )

        # -- Phase 3: logical operators -----------------------------------
        # Registered as (X0, Z0, X1, Z1): logical qubit 0's pair, then qubit 1's.
        for x_support, z_support in zip(self._X_LOGICALS, self._Z_LOGICALS):
            self.create_stim_logical(
                {self._data_coord(q): "X" for q in x_support}, "X"
            )
            self.create_stim_logical(
                {self._data_coord(q): "Z" for q in z_support}, "Z"
            )
        self.num_logicals = 2

        # -- Phase 4: shift -------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def get_info(self):
        info = super().get_info()
        info.update(
            {
                "n": 6,
                "k": self.num_logicals,
                "d": 2,
                "num_data_qubits": len(self.data_coords),
                "num_syndrome_qubits": len(self.syndrome_coords),
                "data_coords": self.data_coords,
                "syndrome_coords": self.syndrome_coords,
                "stabilizers": self.stabilizers,
                "logical_ops": self.logical_ops,
                "index_map": self.index_map,
                "qubit_coords": self.qubit_coords,
                "num_logicals": self.num_logicals,
            }
        )
        return info
