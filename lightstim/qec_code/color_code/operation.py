"""Transversal logical Clifford gates for the triangular color code.

The color code is self-dual (X and Z stabilizers share support), so transversal
Hadamard is logical H. Transversal S is logical S (up to a dagger fixed by
``n mod 4``) **only when every stabilizer has weight divisible by 4** -- true for
the ``d = 3`` code (``[[7, 1, 3]]``, the Steane code; all checks weight 4), not
for the bulk weight-6 checks of ``d >= 5``, which need the 3-coloring S / S_DAG
pattern (not implemented here).

``transversal_x`` / ``transversal_z`` (registered logical support) and
``transversal_cnot`` are inherited from :class:`CSSLogicalOpSet`.
"""

from __future__ import annotations

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_patch import QECPatch


class ColorCodeLogicalOpSet(CSSLogicalOpSet):
    """Transversal ``H`` / ``S`` / ``S_DAG`` for :class:`ColorCode` patches."""

    def __init__(self):
        super().__init__()
        self.name = "ColorCode"

    def _all_data(self, patch: QECPatch) -> list[int]:
        return sorted(patch.data_indices)

    def _apply_all(self, builder: CircuitBuilder, patch: QECPatch, gate: str,
                   noiseless: bool = False) -> None:
        c = stim.Circuit()
        c.append(gate, self._all_data(patch))
        builder.apply_unitary_block(unitary_block=c, noiseless=noiseless)

    def transversal_hadamard(self, builder, patch, noiseless: bool = False):
        """Logical H via transversal physical H (color code is self-dual)."""
        self._apply_all(builder, patch, "H", noiseless)

    def _doubly_even(self, patch: QECPatch) -> bool:
        return all(
            len(s["data_indices"]) % 4 == 0 for s in patch.stabilizers
        )

    def _logical_s_gate(self, patch: QECPatch) -> str:
        """Physical gate whose transversal application is *logical S*.

        Transversal physical S is logical S when ``n = 1 (mod 4)`` and logical
        S_DAG when ``n = 3 (mod 4)``; invert to get logical S.
        """
        if not self._doubly_even(patch):
            raise NotImplementedError(
                "transversal S for the triangular color code is only uniform "
                "when every stabilizer has weight divisible by 4 (d = 3). "
                "d >= 5 needs the 3-coloring S / S_DAG pattern."
            )
        return "S" if len(self._all_data(patch)) % 4 == 1 else "S_DAG"

    def transversal_s(self, builder, patch, noiseless: bool = False):
        """Logical S via transversal physical S (or S_DAG, per ``n mod 4``)."""
        self._apply_all(builder, patch, self._logical_s_gate(patch), noiseless)

    def transversal_s_dag(self, builder, patch, noiseless: bool = False):
        """Logical S_DAG -- the transversal gate opposite :meth:`transversal_s`."""
        opp = "S_DAG" if self._logical_s_gate(patch) == "S" else "S"
        self._apply_all(builder, patch, opp, noiseless)


__all__ = ["ColorCodeLogicalOpSet"]
