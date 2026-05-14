from typing import Tuple, Dict, List, Optional, Literal, Set
import stim
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.coupler import LogicalCouplerProtocol, LogicalCouplerPatch
import math
import numpy as np

# -----------------------------------------------------------------------------
# Part 1. Code Patch Class Definition
# -----------------------------------------------------------------------------
class RotatedSurfaceCode(QECPatch):
    """
    Implementation of the Rotated Surface Code.

    This class constructs a rotated surface code patch defined by its code distances.
    It generates the physical qubit layout (coordinates) and defines the stabilizer 
    generators (stim.PauliString) based on the standard rotated lattice geometry.

    Parameters (passed via **kwargs):
    ---------------------------------
    distance : int, optional
        Sets the uniform code distance for both Z and X directions (creates a square patch).
        Must be an odd integer (e.g., 3, 5, 7).
        *Note: Provide either this OR both 'distance_z' and 'distance_x'.*

    distance_z : int, optional
        The code distance along the logical Z direction (width of the patch).
        Must be an odd integer.
        
    distance_x : int, optional
        The code distance along the logical X direction (height of the patch).
        Must be an odd integer.

    shift : Tuple[float, float], optional (default: (0, 0))
        A global coordinate offset (dx, dy) applied to all qubits in the patch.
        Useful for placing this patch at a specific location in a multi-patch system 
        (e.g., for lattice surgery alignment).

    Examples:
    ---------
    1. Create a standard square d=3 code (most common usage):
    >>> code = RotatedSurfaceCode(distance=3)

    2. Create a rectangular code (dz=5, dx=3) placed at coordinates (10, 0):
    >>> code = RotatedSurfaceCode(distance_z=5, distance_x=3, shift=(10, 0))

    3. Construct from a configuration dictionary (useful for automation):
    >>> config = {'distance': 5, 'shift': (20, 10)}
    >>> code = RotatedSurfaceCode.from_config(config)
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
        if not isinstance(self.shift, tuple) or len(self.shift) != 2:
            raise ValueError("'shift' must be a tuple of two integers.")
        if self.distance_z % 2 == 0 or self.distance_x % 2 == 0:
            raise ValueError("Both 'distance_z' and 'distance_x' must be odd integers.")
    
    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_x)]

    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[i] for i in sorted(self.syndrome_indices_z)]


    def build(self):
        d_z = self.distance_z
        d_x = self.distance_x
        
        # -----------------------------------------------------------------------
        # Phase 1: Geometry Registration
        # Run the loop to register all qubits and populate index_map
        # -----------------------------------------------------------------------
        for y in range(2 * d_x + 1):
            if y % 2 == 1: 
                # Odd rows: Data Qubits
                for x in range(d_z):
                    coord = (2 * x + 1, y)
                    self.add_qubit(*coord, role='data')
            else:
                # Even rows: syndrome Qubits
                # Logic copied from original to preserve exact layout
                if y == 0: # Top row
                    for x in range((d_z - 1) // 2):
                        coord = (4 * x + 2, 0)
                        self.add_qubit(*coord, role='syndrome_x')
                elif y == 2 * d_x: # Bottom row
                    for x in range((d_z - 1) // 2):
                        coord = (4 * x + 4, 2 * d_x)
                        self.add_qubit(*coord, role='syndrome_x')
                elif y % 4 == 2: # Middle rows type A
                    for x in range((d_z + 1) // 2):
                        coord = (4 * x + 2, y) # Z syndrome
                        self.add_qubit(*coord, role='syndrome_z')
                    for x in range((d_z - 1) // 2):
                        coord = (4 * x + 4, y) # X syndrome
                        self.add_qubit(*coord, role='syndrome_x')
                elif y % 4 == 0: # Middle rows type B
                    for x in range((d_z + 1) // 2):
                        coord = (4 * x, y) # Z syndrome
                        self.add_qubit(*coord, role='syndrome_z')
                    for x in range((d_z - 1) // 2):
                        coord = (4 * x + 2, y) # X syndrome
                        self.add_qubit(*coord, role='syndrome_x')

        # -----------------------------------------------------------------------
        # Phase 2: Physics Construction (Stabilizers)
        # Run the loop again to build Stim objects using indices from Phase 1
        # -----------------------------------------------------------------------
        for y in range(2 * d_x + 1):
            # We only care about syndrome rows for stabilizers
            if y % 2 != 0:
                continue

            if y == 0: # Top row X-stabilizers
                for x in range((d_z - 1) // 2):
                    syn_coord = (4 * x + 2, 0)
                    targets = {(4 * x + 1, 1): 'X', (4 * x + 3, 1): 'X'}
                    self.create_stim_stabilizer(targets, syn_coord, 'X')
                    
            elif y == 2 * d_x: # Bottom row X-stabilizers
                for x in range((d_z - 1) // 2):
                    syn_coord = (4 * x + 4, 2 * d_x)
                    targets = {(4 * x + 3, 2 * d_x - 1): 'X', (4 * x + 5, 2 * d_x - 1): 'X'}
                    self.create_stim_stabilizer(targets, syn_coord, 'X')

            elif y % 4 == 2: # Middle rows
                for x in range((d_z + 1) // 2): # Z-stabilizers
                    syn_coord = (4 * x + 2, y)
                    if x == (d_z + 1) // 2 - 1: # Rightmost
                        targets = {(4 * x + 1, y - 1): 'Z', (4 * x + 1, y + 1): 'Z'}
                    else:
                        targets = {(4 * x + 1, y - 1): 'Z', (4 * x + 1, y + 1): 'Z', 
                                   (4 * x + 3, y - 1): 'Z', (4 * x + 3, y + 1): 'Z'}
                    self.create_stim_stabilizer(targets, syn_coord, 'Z')
                    
                for x in range((d_z - 1) // 2): # X-stabilizers
                    syn_coord = (4 * x + 4, y)
                    targets = {(4 * x + 3, y - 1): "X", (4 * x + 5, y - 1): "X", 
                               (4 * x + 3, y + 1): "X", (4 * x + 5, y + 1): "X"}
                    self.create_stim_stabilizer(targets, syn_coord, 'X')

            elif y % 4 == 0: # Middle rows
                for x in range((d_z + 1) // 2): # Z-stabilizers
                    syn_coord = (4 * x, y)
                    if x == 0: # Leftmost
                        targets = {(4 * x + 1, y - 1): "Z", (4 * x + 1, y + 1): "Z"}
                    else:
                        targets = {(4 * x - 1, y - 1): "Z", (4 * x - 1, y + 1): "Z", 
                                   (4 * x + 1, y - 1): "Z", (4 * x + 1, y + 1): "Z"}
                    self.create_stim_stabilizer(targets, syn_coord, 'Z')
                    
                for x in range((d_z - 1) // 2): # X-stabilizers
                    syn_coord = (4 * x + 2, y)
                    targets = {(4 * x + 1, y - 1): "X", (4 * x + 1, y + 1): "X", 
                               (4 * x + 3, y - 1): "X", (4 * x + 3, y + 1): "X"}
                    self.create_stim_stabilizer(targets, syn_coord, 'X')

        # -----------------------------------------------------------------------
        # Phase 3: Logical Operators
        # -----------------------------------------------------------------------
        # Logical Z: Vertical chain (x=1)
        # Original: [(2*x+1, 1) for x in range(d_z)] -> Wait, vertical? 
        # Your original code logic: logical_Z coords seem to be Row 1 data qubits?
        # Let's preserve your exact logic: logical_Z = [(2*x+1, 1) for x in range(d_z)]
        # This looks like a Horizontal Logical Z operator at y=1.
        lz_targets = {(2 * x + 1, 1): "Z" for x in range(d_z)}
        self.create_stim_logical(lz_targets, 'Z')
        
        # Logical X: Vertical chain?
        # Original: [(1, 2*y+1) for y in range(d_x)]
        lx_targets = {(1, 2 * y + 1): "X" for y in range(d_x)}
        self.create_stim_logical(lx_targets, 'X')

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
            'num_data_qubits': len(self.data_indices),
            'num_x_syndromes': len(self.syndrome_indices_x),
            'num_z_syndromes': len(self.syndrome_indices_z),
            'data_coords': self.data_coords,
            'syndrome_coords_z': self.syndrome_coords_z,
            'syndrome_coords_x': self.syndrome_coords_x,
            'syndrome_coords': self.syndrome_coords,
            'stabilizers': self.stabilizers,
            'logical_ops': self.logical_ops,
            'index_map': self.index_map,
            'qubit_coords': self.qubit_coords,
            'num_logicals': self.num_logicals
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