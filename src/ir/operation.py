from abc import ABC, abstractmethod
from typing import List
from src.ir.qec_patch import QECPatch
from src.ir.builder import CircuitBuilder
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
        
        c_qubits = sorted(control_patch.data_indices)  # Sort for consistent pairing
        t_qubits = sorted(target_patch.data_indices)   # Sort for consistent pairing

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
