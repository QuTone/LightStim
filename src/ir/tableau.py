import numpy as np
import stim
from typing import List, Optional

class PauliTableau:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        # (M, 2N) Binary Matrix
        self.matrix = np.zeros((0, 2 * num_qubits), dtype=np.uint8)
        # List of lists. records[i] corresponds to matrix[i]
        self.records: List[List[int]] = []

    def add_stabilizers(self, paulis: np.ndarray, new_records: Optional[List[List[int]]] = None):
        """
        Batch add stabilizers.
        Args:
            paulis: (K, 2N) matrix.
            new_records: List of K lists. If None, assumes NO records (empty lists).
        """
        num_new = paulis.shape[0]
        if num_new == 0: 
            return
            
        # 1. Safefy Check: if no records input，automatically generate num_new empty lists
        if new_records is None:
            new_records = [[] for _ in range(num_new)]
            
        # 2. Length Check
        if len(new_records) != num_new:
            raise ValueError(f"Shape mismatch: Adding {num_new} stabilizers but provided {len(new_records)} records.")

        self.matrix = np.vstack([self.matrix, paulis])
        self.records.extend(new_records)

    def update_row(self, target_idx: int, source_idx: int):
        """
        Gottesman-Knill Update: Row[target] ^= Row[source] (XOR)
        Records[target] += Records[source] (Concatenation)
        """
        self.matrix[target_idx] ^= self.matrix[source_idx]
        self.records[target_idx].extend(self.records[source_idx])

    def update_row_from_external(self, target_idx: int, external_pauli: np.ndarray, external_record: List[int]):
        """
        Updates a row using a Pauli string and record from an external source
        (e.g., updating a Logical Operator using a Stabilizer).
            
        Args:
            target_idx: Index of the row in THIS tableau to update.
            external_pauli: The Pauli string (1D array) to add onto the target (XOR).
            external_record: The measurement record list to add to the target.
        """
        self.matrix[target_idx] ^= external_pauli
        self.records[target_idx].extend(external_record)
        
    def replace_row(self, idx: int, new_pauli: np.ndarray, new_record: List[int]):
        """Replaces a stabilizer (e.g. after anti-commutation)."""
        self.matrix[idx] = new_pauli
        self.records[idx] = new_record

    def remove_rows(self, indices: List[int]):
        """Removes rows from the tableau."""
        self.matrix = np.delete(self.matrix, indices, axis=0)
        self.records = [self.records[i] for i in range(len(self.records)) if i not in indices]

    def expand(self, delta: int):
        """
        Expand the tableau to include delta new qubits.
        Layout is [X0..Xn-1 | Z0..Zn-1]. New qubits act as identity on existing rows.
        Must shift old Z block right by delta columns and insert zeros for new qubits' X.
        """
        if delta <= 0:
            return
        n_old = self.num_qubits
        n_new = n_old + delta
        # Old: [X0..X(n_old-1) | Z0..Z(n_old-1)]  shape (M, 2*n_old)
        # New: [X0..X(n_new-1) | Z0..Z(n_new-1)]  shape (M, 2*n_new)
        # Mapping: new[:, 0:n_old] = old[:, 0:n_old] (X for q0..q(n_old-1))
        #         new[:, n_old:n_new] = 0 (X for new qubits)
        #         new[:, n_new:n_new+n_old] = old[:, n_old:2*n_old] (Z for q0..q(n_old-1))
        #         new[:, n_new+n_old:2*n_new] = 0 (Z for new qubits)
        new_matrix = np.zeros((self.matrix.shape[0], 2 * n_new), dtype=np.uint8)
        new_matrix[:, :n_old] = self.matrix[:, :n_old]
        new_matrix[:, n_new : n_new + n_old] = self.matrix[:, n_old : 2 * n_old]
        self.matrix = new_matrix
        self.num_qubits = n_new

    def get_record(self, idx: int) -> List[int]:
        return self.records[idx]

    @property
    def count(self):
        return self.matrix.shape[0]