from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Set, Optional, Any, Literal
import stim
import numpy as np
import math


class QECPatch(ABC):
    
    # Constants 
    STORAGE_PRECISION = 6 # Float Storage, for accurate visualization
    GRID_SCALE = 1000 # Logic Lookup, convert the coordinates to integers for robustness

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

        self.data_indices: Set[int] = set()
        self.syndrome_indices: Set[int] = set()
        self.syndrome_indices_x: Set[int] = set()
        self.syndrome_indices_z: Set[int] = set()

        self.index_map: Dict[Tuple[float, float], int] = {}
        self.grid_map: Dict[Tuple[int, int], int] = {}
        
        # --- 2. Physics Containers (The "What") ---
        # Master source of truth for error correction properties
        # Store as stim.PauliString for efficiency
        self.stabilizers: List[Dict[str, Any]] = [] 
        self.logical_ops: List[Dict[str, Any]] = [] 
        self.num_logicals: int = 0
        self.rotation_angle: float = 0.0
        self.is_transposed: bool = False
        
        # --- 3. Build Process ---
        self.params = kwargs
        self._process_params()

        self.build() # Call the subclass implementation
    
    @property
    def num_qubits(self) -> int:
        """Dynamic property that returns the current number of qubits."""
        return len(self.qubit_coords)

    @property
    def data_coords(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.data_indices)]
    
    @property
    def syndrome_coords(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices)]

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
        raise NotImplementedError("Subclass must implement this method")

    def _rebuild_grid_map(self):
        """
        Refreshes self.grid_map based on current self.qubit_coords.
        Call this whenever coordinates change!
        """
        self.grid_map = {}
        for idx, coord in self.qubit_coords.items():
            key = self.get_grid_key(coord)
            self.grid_map[key] = idx
    
    # --- Geometry Helper Methods ---

    def add_qubit(self, x: float, y: float, role: Literal['data', 'syndrome', 'syndrome_x', 'syndrome_z'], uid: Optional[int] = None) -> int:
        """
        Register a qubit. Returns its assigned integer index (uid).
        Update the qubit coords and 
        Use this in your subclass build() loop.
        """
        pos = self.snap_coord((x, y))

        if pos in self.index_map:
            return self.index_map[pos]
        
        if uid is None:
            uid = len(self.qubit_coords)

        self.qubit_coords[uid] = pos
        self.index_map[pos] = uid
        grid_key = self.get_grid_key(pos)
        self.grid_map[grid_key] = uid
        
        if role == 'data':
            self.data_indices.add(uid)
        elif role.startswith('syndrome'):
            self.syndrome_indices.add(uid)
            if role == 'syndrome_x':
                self.syndrome_indices_x.add(uid)
            elif role == 'syndrome_z':
                self.syndrome_indices_z.add(uid)
            else:
                raise ValueError(f"Invalid role: {role}")
        else:
            raise ValueError(f"Invalid role: {role}")

        return uid
    
    @staticmethod
    def _rotate_transform(pos: Tuple[float, float], theta: float, center: Tuple[float, float]) -> Tuple[float, float]:
        """Helper to transform a single point."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cx, cy = center
        x, y = pos
        dx, dy = x - cx, y - cy
        
        # Counter-Clockwise Rotation Matrix application
        nx = dx * cos_t - dy * sin_t
        ny = dx * sin_t + dy * cos_t
        
        return QECPatch.snap_coord((nx + cx, ny + cy))

    @staticmethod
    def snap_coord(pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Cleans floating point noise for storage and visualization.
        Retains high precision (e.g., 6 decimals).
        """
        x, y = pos
        nx = round(x, QECPatch.STORAGE_PRECISION)
        ny = round(y, QECPatch.STORAGE_PRECISION)
        # eliminate -0.0
        if nx == -0.0: nx = 0.0
        if ny == -0.0: ny = 0.0
        return (nx, ny)
    
    @staticmethod
    def get_grid_key(pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Converts a float coordinate to a robust INTEGER key.
        Includes epsilon bias to fix rounding cliffs (e.g. .5 cases).
        """
        x, y = pos
        epsilon = 1e-7
        
        ix = int(round((x + epsilon) * QECPatch.GRID_SCALE))
        iy = int(round((y + epsilon) * QECPatch.GRID_SCALE))
        return (ix, iy)

    def transform_vector(self, local_vec: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transforms a LOCAL interaction vector (delta) into a GLOBAL vector
        based on the patch's current orientation (transpose + rotation).
        
        Args:
            local_vec: (dx, dy) in the canonical local frame.
        
        Returns:
            (gx, gy) in the global frame.
        """
        dx, dy = local_vec

        # 1. Handle Transpose (Reflection across y=x)
        # If transposed, local x becomes global y, local y becomes global x
        if getattr(self, 'is_transposed', False):
            dx, dy = dy, dx
        
        # 2. Handle Rotation
        # Use the static helper, BUT use (0,0) as center because vectors verify direction, not position.
        theta = getattr(self, 'rotation_angle', 0.0)
        
        if abs(theta) < 1e-6:
            return (dx, dy)
        
        # Reuse your existing static method
        return self._rotate_transform((dx, dy), center=(0, 0), theta=theta)

    # --- Coordinate Transformation Methods ---
    def shift_coords(self, dx: float, dy: float):
        """
        Base implementation: updates the MASTER records.
        """
        # 1. Update Master Maps (qubit_coords, index_map)
        new_coords = {}

        for idx, (x, y) in self.qubit_coords.items():
            nx, ny = x + dx, y + dy
            new_coords[idx] = self.snap_coord((nx, ny))

        self.qubit_coords = new_coords

        self.index_map = {pos: idx for idx, pos in self.qubit_coords.items()}
        self._rebuild_grid_map()

        # 3. Update syndrome coords for the stabilizers
        for stab in self.stabilizers:
            if 'syn_idx' in stab:
                idx = stab['syn_idx']
                if idx in self.qubit_coords:
                    stab['syn_coord'] = self.qubit_coords[idx]
        
        # 4. Track accumulated shift (used by fold-transversal gates to recover local coords)
        if hasattr(self, 'shift'):
            self.shift = (self.shift[0] + dx, self.shift[1] + dy)

        # Note: We do NOT need to update other keys in self.stabilizers or self.logicals
        # because they rely on qubit indices, which haven't changed!

    def transpose_coords(self):
        """
        Reflect the patch across the line y=x.
        (x, y) becomes (y, x).
        """
        # 1. Update Master Maps (qubit_coords, index_map)
        new_coords = {}
    
        for idx, (x, y) in self.qubit_coords.items():
            nx, ny = y, x # Swap!
            new_coords[idx] = self.snap_coord((nx, ny))
        
        self.qubit_coords = new_coords

        # 2. Rebuild the lookup maps
        self.index_map = {pos: idx for idx, pos in self.qubit_coords.items()}
        self._rebuild_grid_map()
        
        # 3. Update syndrome coords for the stabilizers
        for stab in self.stabilizers:
            if 'syn_idx' in stab:
                idx = stab['syn_idx']
                if idx in self.qubit_coords:
                    stab['syn_coord'] = self.qubit_coords[idx]
        
        self.is_transposed = not self.is_transposed
    
    def rotate_coords(self, theta: float, center: Optional[Tuple[float, float]] = None):
        """
        Rotates the entire patch around a 'center' point by 'theta' (radians, Counter-Clockwise, from x-axis to y-axis).
        
        Math:
        To rotate clockwise by theta:
        x' = cx + (x-cx)cos(theta) + (y-cy)sin(theta)
        y' = cy - (x-cx)sin(theta) + (y-cy)cos(theta)
        """
        # 1. Determine Center
        if center is None:
            # Calculate geometric center (centroid of the bounding box)
            all_x = [c[0] for c in self.qubit_coords.values()]
            all_y = [c[1] for c in self.qubit_coords.values()]
            
            if not all_x: # Empty patch
                center = (0.0, 0.0)
            else:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

        # 2. Update Master Maps (qubit_coords, index_map)
        new_coords = {}

        for idx, coord in self.qubit_coords.items():
            new_coord = self._rotate_transform(coord, theta, center)
            new_coords[idx] = new_coord
            
        self.qubit_coords = new_coords

        # 3. Rebuild the lookup maps
        self.index_map = {pos: idx for idx, pos in self.qubit_coords.items()}
        self._rebuild_grid_map()

        # 4. Update Stabilizers (Coordinate Metadata)
        for stab in self.stabilizers:
            if 'syn_idx' in stab:
                idx = stab['syn_idx']
                if idx in self.qubit_coords:
                     stab['syn_coord'] = self.qubit_coords[idx] # Avoid double calculation
                else:
                    raise ValueError(f"Synaptic index {idx} not found in qubit coordinates.")
            else:
                raise ValueError(f"Stabilizer {stab} does not have a syndrome qubit index.")

        # 5. Update accumulated rotation angle and rebuild grid map
        self.rotation_angle = (self.rotation_angle + theta) % (2 * np.pi)
    
    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y)"""
        xs = [c[0] for c in self.qubit_coords.values()]
        ys = [c[1] for c in self.qubit_coords.values()]
        return min(xs), max(xs), min(ys), max(ys)

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
            # Handle both dict and stim.PauliString formats
            pauli = stab['pauli']
            if isinstance(pauli, dict):
                # Convert dict to stim.PauliString
                pauli_str = stim.PauliString(num_qubits)
                for idx, pauli_type in pauli.items():
                    pauli_str[idx] = pauli_type
            else:
                pauli_str = pauli
            
            # stim.PauliString.to_numpy() returns (xs, zs) boolean arrays
            # bit_packed=False gives generic boolean array
            xs, zs = pauli_str.to_numpy(bit_packed=False)
            
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
            'params': self.params,
            'is_transposed': self.is_transposed,
            'rotation_angle': self.rotation_angle,
        }
    
    # --- Stim Helper Methods for Pauli Strings ---
    # The input are coordinates, which are converted to indices
    def create_stim_stabilizer(self, target_dict: Dict[Tuple[int, int], str], syn_coord: Optional[Tuple[int, int]] = None, type: Optional[str] = None):
        """Helper to convert dictionary definition to stim.PauliString"""
        ps = {}
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

    def create_stim_logical(self, target_dict: Dict[Tuple[int, int], str], op_type: str):
        """Helper to convert list of coords to stim.PauliString"""
        ps = {}
        data_indices = []

        for coord, pauli_type in target_dict.items():
            if coord in self.index_map:
                idx = self.index_map[coord]
                ps[idx] = pauli_type
                data_indices.append(idx)
        
        logical_op_record  = {
            "pauli": ps,
            "type": op_type,
            "data_indices": data_indices
        }
        # Note: op_type = "Z" or "X" logical operators, but the corresponding Pauli String can contain "X", "Y", "Z", "I" (pauli_type).
        
        self.logical_ops.append(logical_op_record)

    def reset_rotation_angle(self):
        self.rotation_angle = 0.0
    
    def reset_transposition(self):
        self.is_transposed = False

