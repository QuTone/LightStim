from typing import Tuple, Dict, List, Optional, Literal, Set
import stim
from src.ir.qec_patch import QECPatch
from src.ir.coupler import BaseCoupler

# -----------------------------------------------------------------------------
# Code Patch Class Definition
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


    def add_qubit(self, x: float, y: float, role: Literal['data', 'syndrome_x', 'syndrome_z'], uid: Optional[int] = None) -> int:
        uid = super().add_qubit(x, y, role, uid)

        if role == 'data': # Handled by superclass already
            pass
        elif role == 'syndrome_x':
            self.syndrome_indices_x.add(uid)
        elif role == 'syndrome_z':
            self.syndrome_indices_z.add(uid)
        else:
            raise ValueError(f"Invalid role: {role}")

        return uid

    def build(self):
        d_z = self.distance_z
        d_x = self.distance_x
        
        self.syndrome_indices_x: Set[int] = set()
        self.syndrome_indices_z: Set[int] = set()
        
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
        lz_coords = [(2 * x + 1, 1) for x in range(d_z)]
        self.create_stim_logical(lz_coords, 'Z')
        
        # Logical X: Vertical chain?
        # Original: [(1, 2*y+1) for y in range(d_x)]
        lx_coords = [(1, 2 * y + 1) for y in range(d_x)]
        self.create_stim_logical(lx_coords, 'X')

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
            'num_data_qubits': len(self.data_qubit_indices),
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


# -----------------------------------------------------------------------------
# Syndrome Extraction Block
# -----------------------------------------------------------------------------
class RotatedSurfaceCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Rotated Surface Code.
    
    This block represents ONE cycle of stabilizer measurements:
    1. Reset syndrome qubits.
    2. Entangling gates (H, CNOTs) following the "Z" (or "N") scheduling pattern.
    3. Measure syndrome qubits.
    
    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.
    """

    def __init__(self, system):
        """
        Args:
            system: A rotated surface code patch.
        """
        self.system = system
        self.circuit = stim.Circuit()
        
        # Build the circuit immediately upon instantiation
        self._build_circuit()

    def _build_circuit(self):
        # --- Step 1: Reset Syndrome Qubits ---
        # Reset all syndrome qubits to |0> (Z basis)
        syn_indices = [self.system.index_map[c] for c in self.system.syndrome_coords]
        self.circuit.append("R", syn_indices)
        self.circuit.append("TICK", tag="SE_start") # DEPOLARIZE1 will be injected here on data qubits

        # --- Step 2: Preparation (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits to |+> state to measure X operators
        x_syn_indices = [self.system.index_map[c] for c in self.system.syndrome_coords_x]
        self.circuit.append("H", x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 3: Entangling Gates (CNOT Scheduling) ---
        # Standard Rotated Surface Code scheduling (4 ticks)
        # Format: ((dx_x, dy_x), (dx_z, dy_z))
        # dx_x/dy_x is the offset for X-stabilizer checks
        # dx_z/dy_z is the offset for Z-stabilizer checks
        canonical_tick_deltas = [
            ((+1, +1), (+1, +1)), # Tick 1: NE interaction
            ((-1, +1), (+1, -1)), # Tick 2: NW / SE interaction
            ((+1, -1), (-1, +1)), # Tick 3: SE / NW interaction
            ((-1, -1), (-1, -1))  # Tick 4: SW interaction
        ]

        current_tick_deltas = []
        for vec_z, vec_x in canonical_tick_deltas:
            global_vec_z = self.system.transform_vector(vec_z)
            global_vec_x = self.system.transform_vector(vec_x)
            current_tick_deltas.append((global_vec_z, global_vec_x))

        for dx_x, dx_z in current_tick_deltas:
            cnot_targets = []
            
            # 3.1 Handle X-Stabilizers (Syndrome is Control, Data is Target)
            for syn_coord in self.system.syndrome_coords_x:
                raw_target = (
                    syn_coord[0] + dx_x[0], 
                    syn_coord[1] + dx_x[1]
                )
                target_key = self.system.get_grid_key(raw_target)

                if target_key in self.system.grid_map:
                    neighbor_idx = self.system.grid_map[target_key]
                    if neighbor_idx in self.system.data_qubit_indices:
                        syn_idx = self.system.index_map[syn_coord]
                        data_idx = neighbor_idx
                        cnot_targets.extend([syn_idx, data_idx]) # Syndrome -> Data

            # 3.2 Handle Z-Stabilizers (Data is Control, Syndrome is Target)
            for syn_coord in self.system.syndrome_coords_z:
                raw_target = (
                    syn_coord[0] + dx_z[0], 
                    syn_coord[1] + dx_z[1]
                )
                target_key = self.system.get_grid_key(raw_target)
                
                if target_key in self.system.grid_map:
                    neighbor_idx = self.system.grid_map[target_key]
                    if neighbor_idx in self.system.data_qubit_indices:
                        syn_idx = self.system.index_map[syn_coord]
                        data_idx = neighbor_idx
                        cnot_targets.extend([data_idx, syn_idx]) # Data -> Syndrome

            # Apply CNOTs if any pairs exist in this tick
            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            
            self.circuit.append("TICK")

        # --- Step 4: Basis Change (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits back to Z basis for measurement
        self.circuit.append("H", x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 5: Measurement ---
        # Measure all syndrome qubits in Z basis
        self.circuit.append("M", syn_indices)
        
        # Note: No final TICK here. CircuitBuilder controls the flow.


# -----------------------------------------------------------------------------
# ZZ Coupler
# -----------------------------------------------------------------------------
class RotatedZZCoupler(BaseCoupler):
    """
    Connects two RotatedSurfaceCode patches horizontally to measure Logical ZZ.
    It stitches the rightmost Z-boundary of the left patch to the leftmost Z-boundary of the right patch.
    It accommodates the case where two patches have different code distances.
    """
    def __init__(self, name: str, left_patch: RotatedSurfaceCode, right_patch: RotatedSurfaceCode):
        # 注意：这里我们不需要额外的 offset 参数，因为我们假设 left_patch 和 right_patch
        # 已经在 System 里有了确定的 Global Coordinates (通过它們各自的 shift 参数)。
        # Coupler 将直接读取它们的 absolute coordinates 来计算缝隙。
        
        self.left_patch = left_patch
        self.right_patch = right_patch
        
        # 初始化基类
        super().__init__(name)
        
        # 立即构建
        self.build()

    def build(self):
        # 这里将是你实现几何计算的地方
        pass