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


    def logical_pauli(self, builder: CircuitBuilder, patch: QECPatch,
                      pauli: str = "X", logical_index: int = 0,
                      noiseless: bool = False):
        """
        Applies a logical Pauli operator by physically applying its Pauli
        string across the patch.

        With noiseless=True the layer is tagged 'noiseless' so noise-injection
        rules skip it — the circuit-level equivalent of tracking the operator
        classically in a Pauli frame (zero physical overhead). With
        noiseless=False the string qubits pick up gate noise, modeling a
        genuine physical application of the operator.

        Args:
            pauli: 'X' or 'Z' — which logical operator type to apply.
            logical_index: which operator of that type, for patches with k > 1
                logical qubits.
            noiseless: tag the layer so noise injection skips it.
        """
        pauli = pauli.upper()
        candidates = [op for op in patch.logical_ops if op["type"] == pauli]
        if not candidates:
            raise ValueError(f"Patch has no logical '{pauli}' operator record.")
        if not (0 <= logical_index < len(candidates)):
            raise ValueError(
                f"logical_index {logical_index} out of range: patch has "
                f"{len(candidates)} logical '{pauli}' operator(s).")
        record = candidates[logical_index]

        # The record's Pauli string may mix X/Y/Z physical gates even for a
        # 'X'/'Z'-type logical operator; emit one instruction per gate kind.
        block = stim.Circuit()
        for gate in ("X", "Y", "Z"):
            targets = sorted(q for q, p in record["pauli"].items() if p == gate)
            if targets:
                block.append(gate, targets)
        builder.apply_unitary_block(block, noiseless=noiseless)

    def prepare_logical_z(self, builder: CircuitBuilder,patch: QECPatch):
        """
        Prepares the logical Z state for all logical qubits in a CSS code patch.
        """
        pass

    def prepare_logical_x(self, builder: CircuitBuilder,patch: QECPatch):
        """
        Prepares the logical X state for all logical qubits in a CSS code patch.
        """
        pass
    

    def prepare_logical_pauli_string(self, builder: CircuitBuilder,patch: QECPatch, pauli_string: str):
        """
        Prepares the logical Pauli string for all logical qubits in a patch.
        This is a general method that can be used to prepare any logical Pauli string, especially useful for code blocks with multiple logical qubits.
        This method will be moved to the specific OpSet for the code family.
        """
        pass
