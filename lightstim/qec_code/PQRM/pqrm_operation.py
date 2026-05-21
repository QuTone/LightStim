"""
PQRMLogicalOpSet: Hypercube encoding for PQRM code.

Matches processing/pqrm_code.py encode_state logic:
- Diagonal initialization (bin_wt split: X_state vs Z_state vs prepared)
- m CNOT layers with TICK between init and each layer
- CNOT direction: Z -> (control, target)=(i,j); X -> (j,i)
"""

from typing import Literal, Optional
import stim

from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.builder import CircuitBuilder

from .pqrm_patch import int2bin, bin_wt


# -----------------------------------------------------------------------------
# PQRMLogicalOpSet
# -----------------------------------------------------------------------------

class PQRMLogicalOpSet(CSSLogicalOpSet):
    """
    Logical operation set for PQRM code.
    Implements hypercube encoding for |0⟩ (Z) and |+⟩ (X) logical states.
    """

    def __init__(self):
        super().__init__()
        self.name = "PQRM"

    def hypercube_encode(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        target_state: Literal["Z", "X", "Y"],
        patch_name: Optional[str] = None,
    ) -> None:
        """
        Encode logical |0⟩ (Z), |+⟩ (X), or |i⟩ (Y) via hypercube encoding.

        Matches processing/pqrm_code.py encode_state:
        1. Diagonal initialization: bin_wt(i) <= rx -> |+⟩, bin_wt(i) >= m-rz -> |0⟩, else depends on target
        2. TICK after init
        3. m CNOT layers, each with TICK after
        4. CNOT direction: Z uses (i,j), X uses (j,i)

        Args:
            builder: CircuitBuilder to append gates.
            patch: PQRMPatch instance.
            target_state: "Z" for |0⟩, "X" for |+⟩, "Y" for |i⟩ (encode |+⟩ then S on all data).
            patch_name: If patch is in a QECSystem, the patch name for local→global index mapping.
        """
        if target_state == "Y":
            self.hypercube_encode(builder, patch, "X", patch_name)
            self.transversal_s(builder, patch, patch_name)
            return

        m = patch.m
        rx, rz = patch.rx, patch.rz
        N = 2 ** m
        local_data = sorted(patch.data_indices)

        # Resolve to circuit (global) indices
        system = builder.system
        if patch_name and hasattr(system, "local_to_global_map") and patch_name in system.local_to_global_map:
            l2g = system.local_to_global_map[patch_name]
            data_indices = [l2g[i] for i in local_data if i in l2g]
        else:
            l2g = None
            data_indices = local_data

        n = system.num_qubits
        local_valid = set(local_data)

        # --- 1. Diagonal initialization (matches pqrm_code.py 234-268) ---
        X_state_indices: list = []
        Z_state_indices: list = []

        if target_state == "Z":
            for i in local_data:
                if bin_wt(i) <= rx:
                    X_state_indices.append(i)
                else:
                    Z_state_indices.append(i)
        else:  # target_state == "X"
            for i in local_data:
                if bin_wt(i) <= rz:
                    Z_state_indices.append(i)
                else:
                    X_state_indices.append(i)

        # Map to global indices and build init_dict
        init_dict = {}
        if l2g is not None:
            for i in X_state_indices:
                if i in l2g:
                    init_dict[l2g[i]] = "X"
            for i in Z_state_indices:
                if i in l2g:
                    init_dict[l2g[i]] = "Z"
        else:
            for i in X_state_indices:
                init_dict[i] = "X"
            for i in Z_state_indices:
                init_dict[i] = "Z"

        builder.initialize(init_dict=init_dict, n=n)

        # TICK after init (matches pqrm_code.py line 269)
        builder.circuit.append("TICK")

        # --- 2. m CNOT layers, each followed by TICK (matches pqrm_code.py 272-293) ---
        for t in range(m):
            sep = 2 ** t
            cnot_targets: list = []

            for i in local_data:
                j = i + sep
                if int2bin(i, m)[-1 - t] != 0:
                    continue
                if j >= N:
                    continue
                if j not in local_valid:
                    continue

                if l2g is not None:
                    gi, gj = l2g.get(i), l2g.get(j)
                    if gi is None or gj is None:
                        continue
                    if target_state == "Z":
                        cnot_targets.extend([gi, gj])
                    else:
                        cnot_targets.extend([gj, gi])
                else:
                    if target_state == "Z":
                        cnot_targets.extend([i, j])
                    else:
                        cnot_targets.extend([j, i])

            if cnot_targets:
                cnot_layer = stim.Circuit()
                cnot_layer.append("CNOT", cnot_targets)
                builder.circuit += cnot_layer
                builder.tracker.process_unitary_block(cnot_layer)

            builder.circuit.append("TICK")

    def prepare_logical_z(
        self, builder: CircuitBuilder, patch: QECPatch, patch_name: Optional[str] = None
    ) -> None:
        """Prepare logical |0⟩ via hypercube encoding."""
        self.hypercube_encode(builder, patch, target_state="Z", patch_name=patch_name)

    def prepare_logical_x(
        self, builder: CircuitBuilder, patch: QECPatch, patch_name: Optional[str] = None
    ) -> None:
        """Prepare logical |+⟩ via hypercube encoding."""
        self.hypercube_encode(builder, patch, target_state="X", patch_name=patch_name)

    def prepare_logical_y(
        self, builder: CircuitBuilder, patch: QECPatch, patch_name: Optional[str] = None
    ) -> None:
        """Prepare logical |i⟩ via hypercube encoding + S on all data."""
        self.hypercube_encode(builder, patch, target_state="Y", patch_name=patch_name)

    def transversal_s(
        self, builder: CircuitBuilder, patch: QECPatch, patch_name: Optional[str] = None
    ) -> None:
        """Logical S via transversal physical S_DAG on all data qubits."""
        self._apply_transversal_gate(builder, patch, "S_DAG", patch_name)

    def transversal_s_dag(
        self, builder: CircuitBuilder, patch: QECPatch, patch_name: Optional[str] = None
    ) -> None:
        """Logical S_DAG via transversal physical S on all data qubits."""
        self._apply_transversal_gate(builder, patch, "S", patch_name)

    def _apply_transversal_gate(
        self, builder: CircuitBuilder, patch: QECPatch, gate: str,
        patch_name: Optional[str] = None
    ) -> None:
        """Apply a single-qubit gate transversally on all data qubits."""
        system = builder.system
        local_data = sorted(patch.data_indices)
        if patch_name and hasattr(system, "local_to_global_map") and patch_name in system.local_to_global_map:
            l2g = system.local_to_global_map[patch_name]
            data_indices = [l2g[i] for i in local_data if i in l2g]
        else:
            data_indices = local_data
        if data_indices:
            gate_circuit = stim.Circuit()
            gate_circuit.append(gate, data_indices)
            builder.circuit += gate_circuit
            builder.circuit.append("TICK")
            builder.tracker.process_unitary_block(gate_circuit)
