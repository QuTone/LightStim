"""Transversal logical operations for the ``[[6, 2, 2]]`` code.

Port of the ``Code6`` helper class in CQCL/Magic-H6
(``Stim/ConcatenatedMSProtocolSim/Code614.py``), which drives the level-2
concatenated Magic-H6 protocol. Every operation ``Code6`` exposes is transversal
(qubit-wise) on the six data qubits:

    Code6.hadamard / s / s_dag / x   ->  single-qubit gate on all 6 data qubits
    Code6.cnot / cz / cy (b2)        ->  the two-qubit gate, qubit i <-> b2.i

Because the ``[[6, 2, 2]]`` code is self-dual, transversal H is logical H
(swapping X_L <-> Z_L on both logical qubits) and transversal S is logical S.

Not ported: ``Code6.measure`` (a raw ``MR`` -- use ``builder.apply_data_readout``)
and ``Code6.rypi2`` (the level-2 Ry(-pi/2)_L teleported through a |+>_L ancilla
block, which needs mid-circuit gauge measurement wiring the builder does not
expose yet).
"""

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_patch import QECPatch


class HSixLogicalOpSet(CSSLogicalOpSet):
    """Transversal logical gate set for :class:`HSixCode` patches.

    Inherits ``transversal_cnot`` from :class:`CSSLogicalOpSet`; adds transversal
    H, S, S_DAG, X (single-patch) and CZ, CY (two-patch).
    """

    def __init__(self):
        super().__init__()
        self.name = "HSixCode"

    # ------------------------------------------------------------------
    # Single-patch transversal gates
    # ------------------------------------------------------------------

    def _transversal_1q(
        self, builder: CircuitBuilder, patch: QECPatch, gate: str,
        noiseless: bool = False,
    ):
        data = sorted(patch.data_indices)
        circuit = stim.Circuit()
        circuit.append(gate, data)
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)

    def transversal_hadamard(self, builder, patch, noiseless: bool = False):
        """Logical H (X_L <-> Z_L on both logical qubits) via transversal H."""
        self._transversal_1q(builder, patch, "H", noiseless)

    def transversal_s(self, builder, patch, noiseless: bool = False):
        """Logical S via transversal physical S."""
        self._transversal_1q(builder, patch, "S", noiseless)

    def transversal_s_dag(self, builder, patch, noiseless: bool = False):
        """Logical S_DAG via transversal physical S_DAG."""
        self._transversal_1q(builder, patch, "S_DAG", noiseless)

    def transversal_x(self, builder, patch, noiseless: bool = False):
        """Transversal physical X (flips X-basis parity of every data qubit)."""
        self._transversal_1q(builder, patch, "X", noiseless)

    # ------------------------------------------------------------------
    # Two-patch transversal gates
    # ------------------------------------------------------------------

    def _transversal_2q(
        self, builder: CircuitBuilder, control_patch: QECPatch,
        target_patch: QECPatch, gate: str, noiseless: bool = False,
    ):
        if type(control_patch) is not type(target_patch):
            raise ValueError(
                f"Type mismatch: {type(control_patch)} vs {type(target_patch)}"
            )
        c_qubits = sorted(control_patch.data_indices)
        t_qubits = sorted(target_patch.data_indices)
        if len(c_qubits) != len(t_qubits):
            raise ValueError(
                f"Size mismatch: {len(c_qubits)} vs {len(t_qubits)} data qubits."
            )
        targets = [q for pair in zip(c_qubits, t_qubits) for q in pair]
        circuit = stim.Circuit()
        circuit.append(gate, targets)
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)

    def transversal_cz(self, builder, control_patch, target_patch, noiseless=False):
        """Transversal CZ between two patches (qubit i <-> qubit i)."""
        self._transversal_2q(builder, control_patch, target_patch, "CZ", noiseless)

    def transversal_cy(self, builder, control_patch, target_patch, noiseless=False):
        """Transversal CY between two patches (qubit i <-> qubit i)."""
        self._transversal_2q(builder, control_patch, target_patch, "CY", noiseless)


__all__ = ["HSixLogicalOpSet"]
