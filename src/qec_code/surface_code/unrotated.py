from typing import Tuple, Dict, List, Optional
import stim
from src.ir.qec_patch import QECPatch

# -----------------------------------------------------------------------------
# Code Patch Class Definition
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

    def build(self):
        dz = self.distance_z
        dx = self.distance_x
        
        # Initialize helper lists (Terminology: Syndrome instead of Ancilla)
        self.data_coords = []
        self.syndrome_coords_x = []
        self.syndrome_coords_z = []
        
        # Populate the base class's main list (assuming base class now uses syndrome_coords)
        self.syndrome_coords = [] 

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
                    self.add_qubit(*coord)
                    self.syndrome_coords_x.append(coord)
                    self.syndrome_coords.append(coord)
                
                # Data qubits at (2x, y)
                for x in range(dz):
                    coord = (2 * x, y)
                    self.add_qubit(*coord)
                    self.data_coords.append(coord)
            else:
                # Odd rows: Contains Z-syndromes and Data qubits
                # Z-syndromes at (2x, y)
                for x in range(dz):
                    coord = (2 * x, y)
                    self.add_qubit(*coord)
                    self.syndrome_coords_z.append(coord)
                    self.syndrome_coords.append(coord)
                
                # Data qubits at (2x + 1, y)
                for x in range(dz - 1):
                    coord = (2 * x + 1, y)
                    self.add_qubit(*coord)
                    self.data_coords.append(coord)

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
        lx_coords = [(0, 2 * y) for y in range(dx)]
        self.create_stim_logical(lx_coords, 'X')
        
        # Logical Z: Horizontal line along y=0
        lz_coords = [(2 * x, 0) for x in range(dz)]
        self.create_stim_logical(lz_coords, 'Z')

        self.num_logicals = 1
        
        # -----------------------------------------------------------------------
        # Phase 4: Shift
        # -----------------------------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)
        


    # --- Helpers ---

    def shift_coords(self, dx: float, dy: float):
        """
        Override to update subclass-specific lists.
        Note: self.data_coords and self.syndrome_coords are updated by the base class.
        """
        # 1. Update Master Maps via Base Class
        super().shift_coords(dx, dy)
        
        # 2. Update Subclass Lists (Specific Syndrome Types)
        self.syndrome_coords_x = self._apply_shift_to_list(self.syndrome_coords_x, dx, dy)
        self.syndrome_coords_z = self._apply_shift_to_list(self.syndrome_coords_z, dx, dy)

    def transpose_coords(self):
        """
        Reflects the surface code layout across y=x.
        Logical operators are physically rotated but their code distance remains unchanged.
        """
        # 1. Base class: Update Master Maps, data_coords, syndrome_coords
        super().transpose_coords()

        # 2. Update Subclass Lists
        self.syndrome_coords_x = self._apply_transpose_to_list(self.syndrome_coords_x)
        self.syndrome_coords_z = self._apply_transpose_to_list(self.syndrome_coords_z)
        
        # 3. NO swapping of distances!
        # self.distance_z is still the weight of Logical Z (now vertical).

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

# -----------------------------------------------------------------------------
# Syndrome Extraction Block
# -----------------------------------------------------------------------------
class UnrotatedSurfaceCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Unrotated Surface Code.
    
    This block represents ONE cycle of stabilizer measurements:
    1. Reset syndrome qubits.
    2. Entangling gates (H, CNOTs) following the specific 6-tick scheduling (Li's paper).
    3. Measure syndrome qubits.
    
    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.
    """

    def __init__(self, system):
        """
        Args:
            system: An unrotated surface code patch.
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
        self.circuit.append("TICK", tag="SE_start") # NoiseInjector targets this tag

        # --- Step 2: Preparation (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits to |+> state
        x_syn_indices = [self.system.index_map[c] for c in self.system.syndrome_coords_x]
        self.circuit.append("H", x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 3: Entangling Gates (CNOT Scheduling) ---
        # Unrotated Surface Code scheduling typically involves 4 interactions per stabilizer,
        # but the provided source uses a 6-tick schedule for avoiding conflicts.
        
        # Format: (dx_z, dx_x)
        # dx_z: Offset for Z-stabilizers (Data -> Ancilla)
        # dx_x: Offset for X-stabilizers (Ancilla -> Data)
        # (0, 0) means no operation for that type in that tick.
        tick_deltas = [
            ((-1, 0), (0, 0)),  # Tick 1: Z checks Top
            ((+1, 0), (0, 0)),  # Tick 2: Z checks Bottom
            ((0, +1), (0, +1)), # Tick 3: Z checks Right / X checks Right (or Bottom depending on coord sys)
            ((0, -1), (0, -1)), # Tick 4: Z checks Left / X checks Left
            ((0, 0), (-1, 0)),  # Tick 5: X checks Top
            ((0, 0), (+1, 0))   # Tick 6: X checks Bottom
        ]

        for dx_z, dx_x in tick_deltas:
            cnot_targets = []
            
            # 3.1 Handle X-Stabilizers (Syndrome is Control, Data is Target)
            # Only process if dx_x is not (0,0)
            if dx_x != (0, 0):
                for syn_coord in self.system.syndrome_coords_x:
                    target_data_coord = (syn_coord[0] + dx_x[0], syn_coord[1] + dx_x[1])
                    
                    if target_data_coord in self.system.data_coords:
                        syn_idx = self.system.index_map[syn_coord]
                        data_idx = self.system.index_map[target_data_coord]
                        cnot_targets.extend([syn_idx, data_idx]) # Control -> Target

            # 3.2 Handle Z-Stabilizers (Data is Control, Syndrome is Target)
            # Only process if dx_z is not (0,0)
            if dx_z != (0, 0):
                for syn_coord in self.system.syndrome_coords_z:
                    target_data_coord = (syn_coord[0] + dx_z[0], syn_coord[1] + dx_z[1])
                    
                    if target_data_coord in self.system.data_coords:
                        syn_idx = self.system.index_map[syn_coord]
                        data_idx = self.system.index_map[target_data_coord]
                        cnot_targets.extend([data_idx, syn_idx]) # Control -> Target

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