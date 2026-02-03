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
        self._row_map = {}
    
    def _row_to_key(self, row: np.ndarray) -> bytes:
        """Helper: Convert a numpy row to a hashable bytes object."""
        return row.tobytes()
    
    def _rebuild_map(self):
        """Rebuilds the hash map after row insertion/deletion/reordering."""
        self._row_map = {}
        for i in range(self.matrix.shape[0]):
            key = self._row_to_key(self.matrix[i])
            self._row_map[key] = i

        
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
        self._rebuild_map()

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
        self._rebuild_map()

    def get_record(self, idx: int) -> List[int]:
        return self.records[idx]

    @property
    def count(self):
        return self.matrix.shape[0]