from abc import ABC, abstractmethod
from numbers import Integral
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

    @staticmethod
    def _require_global_patch(builder: CircuitBuilder, patch: QECPatch):
        """Require the global patch view returned by QECSystem.add_patch()."""
        uids = getattr(patch, "_registered_stabilizer_uids", None)
        for name, (registered, _) in builder.system.patches.items():
            if uids is not None and uids == registered._registered_stabilizer_uids:
                mapping = builder.system.local_to_global_map[name]
                expected = {mapping[q] for q in registered.data_indices}
                if patch.data_indices == expected:
                    return
        raise ValueError("Pass the global patch returned by system.add_patch(), not a local patch.")

    @staticmethod
    def _logical_support(patch: QECPatch, pauli: str, slot: int) -> List[int]:
        """Registered logical support, in the patch view's qubit indices."""
        if pauli not in ("X", "Z"):
            raise ValueError("transversal_pauli supports 'X' or 'Z'.")
        ops = [op for op in patch.logical_ops if op.get("type") == pauli]
        if isinstance(slot, bool) or not isinstance(slot, Integral) or not 0 <= slot < len(ops):
            raise ValueError(f"slot {slot!r} out of range: patch has {len(ops)} {pauli} logicals.")
        op = ops[slot]
        support = set(op["pauli"])
        if not support or support != set(op["data_indices"]) or not support <= patch.data_indices:
            raise ValueError("Logical support must be a nonempty subset of patch data qubits.")
        if any(p != pauli for p in op["pauli"].values()):
            raise ValueError(f"Logical {pauli} must have pure {pauli} support for CSS transversal Pauli.")
        return sorted(support)

    def transversal_pauli(self, builder: CircuitBuilder, patch: QECPatch,
                          pauli: str, *, slot: int = 0, noiseless: bool = False):
        """Apply X_L or Z_L to one registered logical slot.

        Adapted from Maggie Bao's PR #98. This is a tensor product of physical
        Paulis on that logical's support, with identity elsewhere; it is not
        necessarily X or Z on every data qubit. It does not prepare a state.
        Pass the global patch returned by system.add_patch().
        """
        self._require_global_patch(builder, patch)
        circuit = stim.Circuit()
        circuit.append(pauli, self._logical_support(patch, pauli, slot))
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)

    def transversal_x(self, builder: CircuitBuilder, patch: QECPatch, *,
                      slot: int = 0, noiseless: bool = False):
        """Apply the registered logical X for one slot (default: slot 0)."""
        self.transversal_pauli(builder, patch, "X", slot=slot, noiseless=noiseless)

    def transversal_z(self, builder: CircuitBuilder, patch: QECPatch, *,
                      slot: int = 0, noiseless: bool = False):
        """Apply the registered logical Z for one slot (default: slot 0)."""
        self.transversal_pauli(builder, patch, "Z", slot=slot, noiseless=noiseless)

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
