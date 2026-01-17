from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Set, Optional
import stim
import numpy as np

class QECPatch(ABC):
    def __init__(self, **kwargs):
        """
        Base class for QEC Patches (Codes).
        
        Core Philosophy:
        1. Physics: Managed by Stim objects (Stabilizers, Logicals) using Integer Indices.
        2. Geometry: Managed by Coordinate Maps (Index <-> (x,y)).
        """
        # --- 1. Geometry Containers (The "Where") ---
        # Master source of truth for coordinates
        self.qubit_coords: Dict[int, Tuple[float, float]] = {} 
        self.index_map: Dict[Tuple[float, float], int] = {}
        
        # Helper lists for visualization/classification (Optional but recommended)
        self.data_coords: List[Tuple[float, float]] = []
        self.syndrome_coords: List[Tuple[float, float]] = []
        
        # --- 2. Physics Containers (The "What") ---
        # Master source of truth for error correction properties
        # Store as stim.PauliString for efficiency
        self.stabilizers: List[stim.PauliString] = [] 
        self.logical_ops: List[stim.PauliString] = [] 
        self.num_logicals: int = 0
        
        # --- 3. Build Process ---
        self.params = kwargs
        self._process_params()
        self.build() # Call the subclass implementation
    
    @property
    def num_qubits(self) -> int:
        """Dynamic property that returns the current number of qubits."""
        return len(self.qubit_coords)

    @classmethod
    def from_config(cls, config: Dict[str, any]) -> 'QECPatch':
        return cls(**config)

    @abstractmethod
    def _process_params(self):
        """Validate input parameters (e.g., check if distance is odd)."""
        pass

    @abstractmethod
    def build(self):
        """
        The main logic. implemented by subclass.
        1. Call self.add_qubit(x, y) to register coordinates.
        2. Construct stim.PauliString and append to self.stabilizers and self.logical_ops.
        """
        pass

    # --- Geometry Helper Methods ---

    def add_qubit(self, x: float, y: float, is_data: bool = True) -> int:
        """
        Register a qubit. Returns its assigned integer index.
        Use this in your subclass build() loop.
        """
        if (x, y) in self.index_map:
            return self.index_map[(x, y)]
        
        idx = len(self.qubit_coords)
        self.qubit_coords[idx] = (x, y)
        self.index_map[(x, y)] = idx
        
        # Optional: categorize automatically if you want, 
        # or let subclass handle data/syndrome lists manually.
        return idx

    @staticmethod
    def _apply_shift_to_list(coords_list: List[Tuple[float, float]], dx: float, dy: float) -> List[Tuple[float, float]]:
        """Helper to shift a list of coordinates."""
        return [(x + dx, y + dy) for (x, y) in coords_list]
    
    @staticmethod
    def _apply_transpose_to_list(coords_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Helper: swaps x and y for a list of coordinates."""
        return [(y, x) for (x, y) in coords_list]


    def shift_coords(self, dx: float, dy: float):
        """
        Base implementation: updates the MASTER records.
        """
        # 1. Update Master Maps (qubit_coords, index_map)
        new_coords = {}
        new_map = {}
        for idx, (x, y) in self.qubit_coords.items():
            nx, ny = x + dx, y + dy
            new_coords[idx] = (nx, ny)
            new_map[(nx, ny)] = idx
        self.qubit_coords = new_coords
        self.index_map = new_map

        # 2. Update Common Lists (Known to Base)
        self.data_coords = self._apply_shift_to_list(self.data_coords, dx, dy)
        self.syndrome_coords = self._apply_shift_to_list(self.syndrome_coords, dx, dy)
        
        # Note: We do NOT need to update self.stabilizers or self.logicals 
        # because they rely on qubit indices, which haven't changed!

    def transpose_coords(self):
        """
        Reflect the patch across the line y=x.
        (x, y) becomes (y, x).
        """
        # 1. Update Master Maps (qubit_coords, index_map)
        new_coords = {}
        new_map = {}
        for idx, (x, y) in self.qubit_coords.items():
            nx, ny = y, x # Swap!
            new_coords[idx] = (nx, ny)
            new_map[(nx, ny)] = idx
        
        self.qubit_coords = new_coords
        self.index_map = new_map

        # 2. Update Common Lists
        self.data_coords = self._apply_transpose_to_list(self.data_coords)
        self.syndrome_coords = self._apply_transpose_to_list(self.syndrome_coords)

    # --- Algebraic Helper Methods (The Bridge) ---

    def get_parity_check_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the internal stim.PauliString stabilizers into binary Hx and Hz matrices.
        Useful for describing qLDPC codes or calculating code distance.
        """
        num_qubits = len(self.qubit_coords)
        xs_rows = []
        zs_rows = []
        
        for stab in self.stabilizers:
            # stim.PauliString.to_numpy() returns (xs, zs) boolean arrays
            # bit_packed=False gives generic boolean array
            xs, zs = stab['pauli'].to_numpy(bit_packed=False)
            
            # Pad or truncate if necessary (though usually matches num_qubits)
            if len(xs) < num_qubits:
                xs = np.pad(xs, (0, num_qubits - len(xs)))
                zs = np.pad(zs, (0, num_qubits - len(zs)))
                
            # Classify as X or Z check (assuming CSS for simplicity here)
            # For non-CSS, you might keep them mixed or handle Y.
            if np.any(xs) and not np.any(zs):
                xs_rows.append(xs.astype(int))
            elif np.any(zs) and not np.any(xs):
                zs_rows.append(zs.astype(int))
            else:
                # Handle Y or mixed stabilizers if needed, for CSS codes, this should never happen
                pass

        Hx = np.array(xs_rows) if xs_rows else np.zeros((0, num_qubits), dtype=int)
        Hz = np.array(zs_rows) if zs_rows else np.zeros((0, num_qubits), dtype=int)
        return Hx, Hz

    def get_info(self):
        return {
            'code_name': self.__class__.__name__,
            'num_qubits': len(self.qubit_coords),
            'num_stabilizers': len(self.stabilizers),
            'params': self.params
        }
    
    # --- Stim Helper Methods for Pauli Strings ---
    def create_stim_stabilizer(self, target_dict: Dict[Tuple[int, int], str], syn_coord: Optional[Tuple[int, int]] = None, type: Optional[str] = None):
        """Helper to convert dictionary definition to stim.PauliString"""
        ps = stim.PauliString(self.num_qubits)
        data_indices = []
        
        for coord, pauli_type in target_dict.items():
            if coord in self.index_map:
                idx = self.index_map[coord]
                ps[idx] = pauli_type
                data_indices.append(idx)
        
        syn_idx = self.index_map.get(syn_coord) if syn_coord else None
        
        stabilizer_record = {
        "pauli": ps,                  
        "type": type, # "X", "Z", or "Mixed"
        "data_indices": data_indices, 
        "syn_coord": syn_coord,
        "syn_idx": syn_idx,          
    }
        
        self.stabilizers.append(stabilizer_record)

    def create_stim_logical(self, coords: List[Tuple[int, int]], pauli_type: str):
        """Helper to convert list of coords to stim.PauliString"""
        ps = stim.PauliString(self.num_qubits)
        data_indices = []

        for coord in coords:
            if coord in self.index_map:
                idx = self.index_map[coord]
                ps[self.index_map[coord]] = pauli_type
                data_indices.append(idx)
        
        logical_op_record  = {
            "pauli": ps,
            "type": pauli_type, # "X", "Z"
            "indices": data_indices
        }
        
        self.logical_ops.append(logical_op_record)

