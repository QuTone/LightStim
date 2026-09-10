from abc import ABC, abstractmethod
from typing import List
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.builder import CircuitBuilder
import stim

class LogicalOpSet(ABC):
    """
    Base class for a collection of logical operations for a specific code family.
    Think of this as a 'Driver' for a QEC Code.
    """
    def __init__(self, name: str = 'QECCode'):
        # This name should replaced by the specific code family name in child classes.
        self.name = name


class CSSLogicalOpSet(LogicalOpSet):
    """
    Universal Logic for CSS Codes.
    """

    def __init__(self):
        super().__init__("CSSCode")

    def transversal_cnot(self, builder: CircuitBuilder, control_patch: QECPatch, target_patch: QECPatch):
        """
        Applies a transversal CNOT gate between two CSS code patches.
        """
        # --- 1. Validation (Protocol Rules) ---
        if type(control_patch) != type(target_patch):
            raise ValueError(f"Type mismatch: {type(control_patch)} vs {type(target_patch)}")

        # system.add_patch() remaps data_indices to global indices in the returned patch,
        # so we can use them directly here for any multi-patch configuration.
        c_qubits = sorted(control_patch.data_indices)
        t_qubits = sorted(target_patch.data_indices)

        if len(c_qubits) != len(t_qubits):
            raise ValueError(f"Size mismatch: {len(c_qubits)} vs {len(t_qubits)} data qubits.")

        circuit = stim.Circuit()
        # --- 2. Transversal CNOT (Logic) ---
        cnot_targets = []
        for c, t in zip(c_qubits, t_qubits):
            cnot_targets.extend([c, t])

        # --- 3. Execution ---
        if cnot_targets:
            circuit.append("CNOT", cnot_targets)
            builder.apply_unitary_block(unitary_block=circuit)

    # ------------------------------------------------------------------
    # Transversal logical Pauli (X / Z) -- generic for any CSS patch
    # ------------------------------------------------------------------

    def _logical_support(self, patch: QECPatch, pauli: str, slot: int) -> List[int]:
        """Data-qubit indices of the ``slot``-th registered ``pauli`` logical."""
        ops = [op for op in patch.logical_ops if op.get("type") == pauli]
        if not ops:
            raise ValueError(f"patch registered no {pauli}-type logical operator")
        if not 0 <= slot < len(ops):
            raise ValueError(
                f"slot {slot} out of range: patch has {len(ops)} {pauli} logicals"
            )
        return sorted(ops[slot]["data_indices"])

    def transversal_pauli(self, builder: CircuitBuilder, patch: QECPatch,
                          pauli: str, *, slot: int = 0, noiseless: bool = False):
        """Apply logical ``pauli`` (``"X"`` or ``"Z"``) to logical qubit ``slot``
        by applying the physical Pauli to that logical operator's registered
        data-qubit support. Fault-tolerant (single fault stays single-qubit) and
        correct for ``k > 1`` -- pick the logical qubit with ``slot``."""
        if pauli not in ("X", "Z"):
            raise ValueError("transversal_pauli supports 'X' or 'Z'")
        circuit = stim.Circuit()
        circuit.append(pauli, self._logical_support(patch, pauli, slot))
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)

    def transversal_x(self, builder: CircuitBuilder, patch: QECPatch, *,
                      slot: int = 0, noiseless: bool = False):
        """Logical X on logical qubit ``slot`` (its registered X_L support)."""
        self.transversal_pauli(builder, patch, "X", slot=slot, noiseless=noiseless)

    def transversal_z(self, builder: CircuitBuilder, patch: QECPatch, *,
                      slot: int = 0, noiseless: bool = False):
        """Logical Z on logical qubit ``slot`` (its registered Z_L support)."""
        self.transversal_pauli(builder, patch, "Z", slot=slot, noiseless=noiseless)

    def prepare_logical_z(self, builder: CircuitBuilder, patch: QECPatch, *, slot: int = 0):
        """Prepare ``|1>_L`` on logical qubit ``slot`` from ``|0>_L`` (transversal X)."""
        self.transversal_x(builder, patch, slot=slot)

    def prepare_logical_x(self, builder: CircuitBuilder, patch: QECPatch, *, slot: int = 0):
        """Prepare ``|->_L`` on logical qubit ``slot`` from ``|+>_L`` (transversal Z)."""
        self.transversal_z(builder, patch, slot=slot)


    def prepare_logical_pauli_string(self, builder: CircuitBuilder,patch: QECPatch, pauli_string: str):
        """
        Prepares the logical Pauli string for all logical qubits in a patch.
        This is a general method that can be used to prepare any logical Pauli string, especially useful for code blocks with multiple logical qubits.
        This method will be moved to the specific OpSet for the code family.
        """
        pass
