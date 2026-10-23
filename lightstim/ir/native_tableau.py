"""Mutable public views of the persistent native tracking state.

The inverse tableau owns rows until a caller asks for the public NumPy view.
Once exposed, that writable view is reconciled on the next native boundary;
in-place edits therefore keep the historical PauliTableau API semantics.
"""

import numpy as np

from .tableau import PauliTableau


class NativePauliTableau(PauliTableau):
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self._native_state = None
        self._native_indices = ()
        self._matrix = np.zeros((0, 2 * num_qubits), dtype=np.uint8)
        self._records = []

    def attach(self, state, indices):
        self._native_state = state
        self._native_indices = tuple(indices)
        self._matrix = None
        self._records = None

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self._native_state.rows_selected(self._native_indices)
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def records(self):
        if self._records is None:
            self._records = self._native_state.records_selected(self._native_indices)
        return self._records

    @records.setter
    def records(self, value):
        self._records = value

    @property
    def count(self):
        return (len(self._native_indices) if self._matrix is None
                else self._matrix.shape[0])

    def native_is_unexposed(self, state, indices):
        """An unexposed view cannot have received an external in-place edit."""
        return (self._native_state is state
                and self._native_indices == tuple(indices)
                and self._matrix is None and self._records is None)

    def native_matrix_is_unexposed(self, state, indices):
        return (self._native_state is state
                and self._native_indices == tuple(indices) and self._matrix is None)
