"""SHYPS code algebra and a circuit realization with separate gauge ancillas.

Reference: Malcolm et al., *Computing Efficiently in QLDPC Codes*,
https://arxiv.org/html/2502.07150v2, Sections VIII.4--VIII.5.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.HGP.algebra import canonical_kernel_basis


# These two explicit circulants are sufficient for the first two SHYPS sizes.
# r=3 uses the polynomial in Example VIII.7; r=4 uses another primitive
# trinomial in the construction of Section VIII.4. Both classical kernels
# are checked exhaustively against the simplex weight distribution in tests.
_SIMPLEX_CHECK_EXPONENTS = {3: (0, 2, 3), 4: (0, 1, 4)}


class SHYPSCode(QECPatch):
    """A CSS subsystem hypergraph-product simplex code.

    ``r=3`` gives ``[[49, 9, 4]]`` and ``r=4`` gives ``[[225, 16, 8]]``;
    these parameters count data qubits, protected logical qubits and code
    distance. The realization adds one ancilla per weight-three gauge
    generator, for ``3 * (2**r - 1)**2`` physical qubits in total.

    With ``m = 2**r - 1``, ``H`` is the m-by-m circulant simplex parity
    check and ``C`` is a canonical r-by-m generator for its kernel. This
    class declares, separately and once,

    * gauges ``G_X = H kron I`` and ``G_Z = I kron H``;
    * their centre ``S_X = H kron C`` and ``S_Z = C kron H``.

    Redundant rows of the physical gauge checks are retained. Centre
    generators do not have measurement ancillas: their information is
    inferred from the gauge measurement history.

    Physical data index ``row * m + column`` corresponds to coordinate
    ``(column, row)`` before shifting. Logical id ``a * r + b`` has bare
    representatives ``P[a] kron C[b]`` and ``C[a] kron P[b]``, where the
    unit-vector pivot matrix satisfies ``P @ C.T == I`` over GF(2). The
    two representatives intersect only at ``(pivot[a], pivot[b])`` in
    semantic (row, column) coordinates.

    The default extraction is the generic X-then-Z gauge schedule. Its
    display coordinates do not impose local hardware connectivity, and
    this implementation does not reproduce the paper's gate compilation,
    optimized syndrome schedule or logical-error performance.

    Args:
        r: Simplex dimension, currently 3 or 4.
        shift: Translation of the display coordinates.
    """

    def __init__(self, r: int = 3, *, shift: tuple[float, float] = (0, 0)):
        super().__init__(r=r, shift=shift)

    def _process_params(self) -> None:
        raw_r = self.params["r"]
        if isinstance(raw_r, bool) or not isinstance(raw_r, Integral):
            raise ValueError("r must be an integer, either 3 or 4.")
        self.r = int(raw_r)
        if self.r not in _SIMPLEX_CHECK_EXPONENTS:
            raise ValueError("SHYPSCode currently supports r=3 and r=4.")
        self.shift = self.params["shift"]
        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("shift must be a two-number tuple.")

        self.simplex_length = 2**self.r - 1
        self.code_distance = 2 ** (self.r - 1)
        self.num_gauge_qubits = (self.simplex_length - self.r) ** 2
        self.data_qubits: dict[tuple[int, int], int] = {}
        self.x_gauge_ancillas: dict[tuple[int, int], int] = {}
        self.z_gauge_ancillas: dict[tuple[int, int], int] = {}
        self.logical_pairs: list[dict[str, Any]] = []

    @property
    def num_data_qubits(self) -> int:
        return self.simplex_length**2

    @property
    def data_index_order(self) -> tuple[int, ...]:
        return tuple(self.data_qubits[key] for key in sorted(self.data_qubits))

    @property
    def default_extraction_block_class(self):
        from lightstim.qec_code.generic_css.gauge_SE_block import (
            GenericCSSGaugeExtractionBlock,
        )

        return GenericCSSGaugeExtractionBlock

    def build(self) -> None:
        m = self.simplex_length
        first_row = np.zeros(m, dtype=np.uint8)
        first_row[list(_SIMPLEX_CHECK_EXPONENTS[self.r])] = 1
        self.simplex_parity_check = np.stack(
            [np.roll(first_row, shift) for shift in range(m)]
        )
        kernel = canonical_kernel_basis(self.simplex_parity_check)
        if kernel.nullity != self.r:
            raise RuntimeError("The SHYPS seed does not have simplex dimension r.")
        self.simplex_generator = kernel.basis.copy()
        self.simplex_pivots = kernel.pivots
        self.simplex_pivot_matrix = kernel.unit_complement.copy()

        identity = np.eye(m, dtype=np.uint8)
        self.gauge_matrix_x = np.kron(self.simplex_parity_check, identity)
        self.gauge_matrix_z = np.kron(identity, self.simplex_parity_check)
        self.center_matrix_x = np.kron(
            self.simplex_parity_check, self.simplex_generator
        )
        self.center_matrix_z = np.kron(
            self.simplex_generator, self.simplex_parity_check
        )
        self.logical_matrix_x = np.kron(
            self.simplex_pivot_matrix, self.simplex_generator
        )
        self.logical_matrix_z = np.kron(
            self.simplex_generator, self.simplex_pivot_matrix
        )

        self._register_qubits()
        self._register_checks()
        self._register_logicals()
        self.num_logicals = self.r**2

        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    def _register_qubits(self) -> None:
        m = self.simplex_length
        for row in range(m):
            for column in range(m):
                self.data_qubits[(row, column)] = self.add_qubit(
                    column, row, role="data"
                )

        # Display the ancilla sectors below and to the right of the data.
        # Each physical gauge generator has its own reset/readout qubit.
        for check in range(m):
            for column in range(m):
                self.x_gauge_ancillas[(check, column)] = self.add_qubit(
                    column, m + 1 + check, role="syndrome_x"
                )
        for row in range(m):
            for check in range(m):
                self.z_gauge_ancillas[(row, check)] = self.add_qubit(
                    m + 1 + check, row, role="syndrome_z"
                )

    def _targets(self, row: np.ndarray, basis: str) -> dict:
        data_indices = self.data_index_order
        return {
            self.qubit_coords[data_indices[int(column)]]: basis
            for column in np.flatnonzero(row)
        }

    def _register_checks(self) -> None:
        m = self.simplex_length
        for basis, matrix, ancillas in (
            ("X", self.gauge_matrix_x, self.x_gauge_ancillas),
            ("Z", self.gauge_matrix_z, self.z_gauge_ancillas),
        ):
            for index, row in enumerate(matrix):
                pair = divmod(index, m)
                self.create_stim_gauge(
                    self._targets(row, basis),
                    syn_coord=self.qubit_coords[ancillas[pair]],
                    type=basis,
                )
                self.gauges[-1]["product_index"] = pair

        for basis, matrix in (
            ("X", self.center_matrix_x),
            ("Z", self.center_matrix_z),
        ):
            for row in matrix:
                self.create_stim_stabilizer(self._targets(row, basis), type=basis)

    def _register_logicals(self) -> None:
        for logical_id, (x_row, z_row) in enumerate(
            zip(self.logical_matrix_x, self.logical_matrix_z)
        ):
            a, b = divmod(logical_id, self.r)
            pair = {"logical_id": logical_id, "simplex_indices": (a, b)}
            for basis, row in (("X", x_row), ("Z", z_row)):
                self.create_stim_logical(self._targets(row, basis), basis)
                record = self.logical_ops[-1]
                record.update(
                    logical_id=logical_id,
                    simplex_indices=(a, b),
                    logical_kind="bare",
                )
                pair[basis.lower()] = record
            pair["pivot_index"] = self.data_qubits[
                (self.simplex_pivots[a], self.simplex_pivots[b])
            ]
            self.logical_pairs.append(pair)

    def get_info(self):
        info = super().get_info()
        info.update(
            r=self.r,
            num_data_qubits=self.num_data_qubits,
            num_logicals=self.num_logicals,
            num_gauge_qubits=self.num_gauge_qubits,
            code_distance=self.code_distance,
            num_gauge_generators=len(self.gauges),
        )
        return info
