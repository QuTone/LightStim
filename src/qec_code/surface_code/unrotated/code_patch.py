from typing import Tuple, Dict, List, Optional, Literal, Set, Any
from typing_extensions import ParamSpecKwargs
import stim
from src.ir.qec_patch import QECPatch
from src.ir.coupler import LogicalCouplerProtocol
import math
import numpy as np

# -----------------------------------------------------------------------------
# Part 1. Code Patch Class Definition
# -----------------------------------------------------------------------------
class UnrotatedSurfaceCode(QECPatch):
    """
    Implementation of the Unrotated Surface Code.

    This class constructs an unrotated surface code patch defined by its code distances.
    It generates the physical qubit layout (coordinates) and defines the stabilizer 
    generators (stim.PauliString) based on the standard planar lattice geometry.
    
    In this layout:
    - Data qubits are located on vertices (or edges, depending on perspective).
    - X-type syndromes (stabilizers) check data qubits mostly left/right.
    - Z-type syndromes (stabilizers) check data qubits mostly up/down.

    Parameters (passed via **kwargs):
    ---------------------------------
    distance : int, optional
        Sets the uniform code distance for both Z and X directions (creates a square patch).
        *Note: Provide either this OR both 'distance_z' and 'distance_x'.*

    distance_z : int, optional
        The code distance along the logical Z direction (width of the patch).
        Determines the number of data qubit columns.
        
    distance_x : int, optional
        The code distance along the logical X direction (height of the patch).
        Determines the number of data qubit rows.

    shift : Tuple[float, float], optional (default: (0, 0))
        A global coordinate offset (dx, dy) applied to all qubits in the patch.

    Examples:
    ---------
    1. Create a standard square d=3 unrotated surface code:
    >>> code = UnrotatedSurfaceCode(distance=3)

    2. Create a rectangular code (dz=5, dx=3):
    >>> code = UnrotatedSurfaceCode(distance_z=5, distance_x=3)

    3. Create a rectangular code (dz=5, dx=3) placed at coordinates (10, 0)
    >>> config = {'distance_z': 5, 'distance_x': 3, 'shift': (10, 0)}
    >>> code = UnrotatedSurfaceCode.from_config(config)
    """

    def _process_params(self):
        self.distance_z = self.params.get("distance_z")
        self.distance_x = self.params.get("distance_x")
        
        # Support generic 'distance' for square codes
        if self.distance_z is None and "distance" in self.params:
             self.distance_z = self.distance_x = self.params["distance"]

        self.shift = self.params.get("shift", (0, 0))

        if self.distance_z is None or self.distance_x is None:
            raise ValueError("Both 'distance_z' and 'distance_x' must be provided.")
        
        # Planar code typically works with integer distances >= 2
        if self.distance_z < 2 or self.distance_x < 2:
            raise ValueError("Code distance must be at least 2.")

    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]
    
    @property
    def logical_ops_x(self) -> Dict[str, Any]:
        log_op_x = [log_op for log_op in self.logical_ops if log_op["type"] == "X"]
        return log_op_x[0]
    
    @property
    def logical_ops_z(self) -> Dict[str, Any]:
        log_op_z = [log_op for log_op in self.logical_ops if log_op["type"] == "Z"]
        return log_op_z[0]


    def build(self):
        dz = self.distance_z
        dx = self.distance_x

        # -----------------------------------------------------------------------
        # Phase 1: Geometry Registration
        # -----------------------------------------------------------------------
        # The loop runs for 2*dx - 1 rows (from y=0 to y=2*dx-2)
        for y in range(2 * dx - 1):
            if y % 2 == 0: 
                # Even rows: Contains X-syndromes and Data qubits
                # Z-syndromes at (2x + 1, y)
                for x in range(dz - 1):
                    coord = (2 * x + 1, y)
                    self.add_qubit(*coord, role='syndrome_x')
                
                # Data qubits at (2x, y)
                for x in range(dz):
                    coord = (2 * x, y)
                    self.add_qubit(*coord, role='data')
            else:
                # Odd rows: Contains Z-syndromes and Data qubits
                # Z-syndromes at (2x, y)
                for x in range(dz):
                    coord = (2 * x, y)
                    self.add_qubit(*coord, role='syndrome_z')
                
                # Data qubits at (2x + 1, y)
                for x in range(dz - 1):
                    coord = (2 * x + 1, y)
                    self.add_qubit(*coord, role='data')

        # -----------------------------------------------------------------------
        # Phase 2: Physics Construction (Stabilizers)
        # -----------------------------------------------------------------------
        for y in range(2 * dx - 1):
            if y % 2 == 0: 
                # Even rows: X-Stabilizers
                for x in range(dz - 1):
                    syn_coord = (2 * x + 1, y) 
                    targets = {}
                    
                    # Left neighbor
                    targets[(2 * x, y)] = 'X'
                    # Right neighbor
                    targets[(2 * x + 2, y)] = 'X'
                    # Top neighbor (if not top row)
                    if y > 0:
                        targets[(2 * x + 1, y - 1)] = 'X'
                    # Bottom neighbor (if not bottom row)
                    if y < 2 * dx - 2:
                        targets[(2 * x + 1, y + 1)] = 'X'
                        
                    self.create_stim_stabilizer(targets, syn_coord, 'X')
            else:
                # Odd rows: Z-Stabilizers
                for x in range(dz):
                    syn_coord = (2 * x, y)
                    targets = {}
                    
                    # Top neighbor
                    targets[(2 * x, y - 1)] = 'Z'
                    # Bottom neighbor
                    targets[(2 * x, y + 1)] = 'Z'
                    # Left neighbor (if not left edge)
                    if x > 0:
                        targets[(2 * x - 1, y)] = 'Z'
                    # Right neighbor (if not right edge)
                    if x < dz - 1:
                        targets[(2 * x + 1, y)] = 'Z'

                    self.create_stim_stabilizer(targets, syn_coord, 'Z')

        # -----------------------------------------------------------------------
        # Phase 3: Logical Operators
        # -----------------------------------------------------------------------
        # Logical X: Vertical line along x=0
        lx_targets = {(0, 2 * y): "X" for y in range(dx)}
        self.create_stim_logical(lx_targets, 'X')
        
        # Logical Z: Horizontal line along y=0
        lz_targets = {(2 * x, 0): "Z" for x in range(dz)}
        self.create_stim_logical(lz_targets, 'Z')

        self.num_logicals = 1
        
        # -----------------------------------------------------------------------
        # Phase 4: Shift
        # -----------------------------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)
        

    def get_info(self):
        info = super().get_info()
        info.update({
            'distance_z': self.distance_z,
            'distance_x': self.distance_x,
            'num_data_qubits': len(self.data_coords),
            'num_x_syndromes': len(self.syndrome_coords_x),
            'num_z_syndromes': len(self.syndrome_coords_z),
            'distance_z': self.distance_z,
            'data_coords': self.data_coords,
            'syndrome_coords_z': self.syndrome_coords_z,
            'syndrome_coords_x': self.syndrome_coords_x,
            'syndrome_coords': self.syndrome_coords,
            'stabilizers': self.stabilizers,
            'logical_ops': self.logical_ops,
            'index_map': self.index_map,
            'qubit_coords': self.qubit_coords,
            'num_logicals': self.num_logicals,
        })
        return info


    def shift_logical_operators(self, op_type: Literal["X", "Z"], offset: float):
        """
        Shift the existing logical operators of a specific type in-place.
        
        Args:
            op_type: "X" (Vertical logical op) -> shifts along x-axis.
                    "Z" (Horizontal logical op) -> shifts along y-axis.
            offset: The distance to shift.
        """
        # Find the logical operator of the specified type
        target_log_op = None
        for log_op in self.logical_ops:
            if log_op["type"] == op_type:
                target_log_op = log_op
                break
        
        if target_log_op is None:
            raise ValueError(f"No logical operator of type '{op_type}' found.")
        
        # Build new pauli dict and data_indices list
        new_pauli: Dict[int, str] = {}
        new_data_indices: List[int] = []
        
        # Determine shift direction
        if op_type == "X":
            # Shift along x-axis (dx = offset, dy = 0)
            dx, dy = offset, 0.0
        else:  # op_type == "Z"
            # Shift along y-axis (dx = 0, dy = offset)
            dx, dy = 0.0, offset
        
        # Process each qubit in the logical operator
        for old_idx, pauli_type in target_log_op["pauli"].items():
            # Get current coordinates
            if old_idx not in self.qubit_coords:
                raise ValueError(f"Qubit index {old_idx} not found in qubit_coords.")
            
            old_coord = self.qubit_coords[old_idx]
            
            # Calculate new coordinates
            new_x = old_coord[0] + dx
            new_y = old_coord[1] + dy
            new_coord = self.snap_coord((new_x, new_y))
            
            # Find the qubit index at the new coordinate
            if new_coord not in self.index_map:
                raise ValueError(f"New coordinate {new_coord} does not have a corresponding qubit index. "
                               f"Make sure the patch has qubits at the shifted location.")
            
            new_idx = self.index_map[new_coord]
            
            # Update pauli dict and data_indices
            new_pauli[new_idx] = pauli_type
            # Only include data qubits in data_indices
            if new_idx in self.data_indices:
                new_data_indices.append(new_idx)
        
        # Update the logical operator in-place
        target_log_op["pauli"] = new_pauli
        target_log_op["data_indices"] = new_data_indices
        