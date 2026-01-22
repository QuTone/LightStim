from typing import Tuple, Dict, List, Optional, Literal, Set
import stim
from src.ir.qec_patch import QECPatch
from src.ir.coupler import LogicalCoupler
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
        dz = self.distance_z
        dx = self.distance_x

        self.syndrome_indices_x: Set[int] = set()
        self.syndrome_indices_z: Set[int] = set()

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
# Part 2. Syndrome Extraction Block
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
        
        # Format: (dx_x, dx_z)
        # dx_z: Offset for Z-stabilizers (Data -> Ancilla)
        # dx_x: Offset for X-stabilizers (Ancilla -> Data)
        canonical_tick_deltas = [
            ((0, 0), (-1, 0)),  # Tick 1
            ((0, 0), (+1, 0)),  # Tick 2
            ((0, +1), (0, +1)), # Tick 3
            ((0, -1), (0, -1)), # Tick 4
            ((-1, 0), (0, 0)),  # Tick 5
            ((+1, 0),(0, 0))    # Tick 6
        ]

        # current_tick_deltas = []
        # for vec_x, vec_z in canonical_tick_deltas:
        #     global_vec_z = self.system.transform_vector(vec_z)
        #     global_vec_x = self.system.transform_vector(vec_x)
        #     current_tick_deltas.append((global_vec_x, global_vec_z))

        for dx_x, dx_z in canonical_tick_deltas:
            cnot_targets = []
            
            # 3.1 Handle X-Stabilizers (Syndrome is Control, Data is Target)
            # Only process if dx_x is not (0,0)
            if dx_x != (0, 0):
                for syn_coord in self.system.syndrome_coords_x:
                    owner_patch = self.system.patches[self.system.spatial_map[syn_coord]][0]
                    dx_x_global = owner_patch.transform_vector(dx_x)
                    raw_target = (
                        syn_coord[0] + dx_x_global[0], 
                        syn_coord[1] + dx_x_global[1]
                    )
                    target_key = owner_patch.get_grid_key(raw_target)

                    if target_key in owner_patch.grid_map:
                        neighbor_idx = owner_patch.grid_map[target_key]
                        if neighbor_idx in owner_patch.data_qubit_indices:
                            syn_idx = self.system.index_map[syn_coord]
                            data_coord = owner_patch.qubit_coords[neighbor_idx]
                            data_idx = self.system.index_map[data_coord]
                            cnot_targets.extend([syn_idx, data_idx]) # Syndrome -> Data

            # 3.2 Handle Z-Stabilizers (Data is Control, Syndrome is Target)
            # Only process if dx_z is not (0,0)
            if dx_z != (0, 0):
                for syn_coord in self.system.syndrome_coords_z:
                    owner_patch = self.system.patches[self.system.spatial_map[syn_coord]][0]
                    dx_z_global = owner_patch.transform_vector(dx_z)
                    raw_target = (
                        syn_coord[0] + dx_z_global[0], 
                        syn_coord[1] + dx_z_global[1]
                    )
                    target_key = owner_patch.get_grid_key(raw_target)
                    
                    if target_key in owner_patch.grid_map:
                        neighbor_idx = owner_patch.grid_map[target_key]
                        if neighbor_idx in owner_patch.data_qubit_indices:
                            syn_idx = self.system.index_map[syn_coord]
                            data_coord = owner_patch.qubit_coords[neighbor_idx]
                            data_idx = self.system.index_map[data_coord]
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

# -----------------------------------------------------------------------------
# Part 3. Couplers
# -----------------------------------------------------------------------------
class UnrotatedTwoPatchCoupler(LogicalCoupler):
    """
    Implementation of a coupler between two Unrotated Surface Code patches.
    Enforces strict alignment and containment rules for Phase 1 development.
    """
    
    EXPECTED_PATCH_COUNT = 2

    def __init__(self, 
                 patches: List[QECPatch], 
                 interaction_type: Literal['ZZ', 'XX'], 
                 name: str = "coupler", 
                 **kwargs):
        
        self.interaction_type = interaction_type
        # Initialize base, which will call self.build()
        super().__init__(patches, name=name, **kwargs)

    def build(self):
        """
        The main orchestration flow.
        1. Geometry Analysis: Validates constraints and calculates the gap.
        2. Geometry Construction: Fills the gap with qubits.
        3. Topology Definition: Constructs stabilizers based on neighbors.
        """
        patch_a = self.patches[0]
        patch_b = self.patches[1]

        # Step 1: Geometry Analysis
        # Returns:
        # - logical_op_orientation: 'horizontal' or 'vertical'
        # - anchor_patch: The smaller patch used as the starting point for the coupling region construction
        # - gap_bounds: (gap_x_min, gap_x_max) or (gap_y_min, gap_y_max) depending on the logical operator orientation
        logical_op_orientation, anchor_patch, gap_bounds = self._analyze_geometry(patch_a, patch_b)

        # Step 2: Construct Coupling Region (Geometry)
        # This uses the "Anchor" method you plan to implement
        self._construct_coupling_region(logical_op_orientation, gap_bounds, anchor_patch)

        # Step 3: Initialize Stabilizers (Topology)
        # Unified method, agnostic of ZZ/XX
        self._init_stabilizers()

    def _analyze_geometry(self, patch_a: QECPatch, patch_b: QECPatch):
        """
        Performs rigorous checks (Rotation, Containment, Parity) and defines the gap.
        """
        # ====================================================
        # (1) Basic Constraint Checks
        # ====================================================
        
        # (a) Rotation Angle Difference check
        # Assuming QECPatch has rotation_angle (in degrees or radians)
        angle_a = getattr(patch_a, 'rotation_angle', 0)
        angle_b = getattr(patch_b, 'rotation_angle', 0)
        
        # Check alignment (difference must be multiple of pi)
        # Using epsilon for float safety
        diff = abs(angle_a - angle_b)
        if not (math.isclose(diff % np.pi, 0, abs_tol=1e-5)):
            raise ValueError(f"Patches must have aligned orientation. Got angles {angle_a} and {angle_b}.")

        # (b) Absolute Rotation check (0 or 90 only for now)
        if not (math.isclose(angle_a % np.pi/2, 0, abs_tol=1e-5)):
             raise ValueError("Coupler only supports rotation angles of 0 or pi/2.")

        # (c) Transpose check
        trans_a = getattr(patch_a, 'is_transposed', False)
        trans_b = getattr(patch_b, 'is_transposed', False)
        if trans_a != trans_b:
            raise ValueError(f"Patches must have same transposition state. Got {trans_a} vs {trans_b}.")

        # ====================================================
        # (2) Interaction Logic & Orientation
        # ====================================================
        
        # Determine the orientation of the Logical Operator involved in the interaction
        # We need to know if that operator is Horizontal or Vertical.
        logical_op_orientation = self._get_logical_op_orientation(self.interaction_type, angle_a, trans_a)

        # ====================================================
        # (3) Containment & Relative Position Check
        # ====================================================
        
        bounds_a = self._get_bounds(patch_a) # (min_x, max_x, min_y, max_y)
        bounds_b = self._get_bounds(patch_b)

        gap_bounds = None
        anchor_patch = None

        if logical_op_orientation == 'horizontal':
            # ------------------------------------------------
            # Case: Vertical Stacking (Gap in y direction)
            # Requirement: Horizontal Range (x-axis) must contain each other
            # ------------------------------------------------
            
            # 1. Check Containment: A contains B OR B contains A
            a_contains_b = (bounds_a[0] <= bounds_b[0] and bounds_a[1] >= bounds_b[1])
            b_contains_a = (bounds_b[0] <= bounds_a[0] and bounds_b[1] >= bounds_a[1])
            
            if not (a_contains_b or b_contains_a):
                raise ValueError("Vertical interaction requires strictly contained X-ranges.")
            
            anchor_patch = patch_a if b_contains_a else patch_b

            # 2. Determine Gap y-axis Bounds and Anchor Patch
            # Gap bound order matters, always from anchor patch to target patch
            if bounds_a[3] < bounds_b[2]: # A is below B
                if b_contains_a:
                    gap_bounds = (bounds_a[3], bounds_b[2])
                else:
                    gap_bounds = (bounds_b[2], bounds_a[3])
            elif bounds_b[3] < bounds_a[2]: # B is below A
                if b_contains_a:
                    gap_bounds = (bounds_a[2], bounds_b[3])
                else:
                    gap_bounds = (bounds_b[3], bounds_a[2])
            else:
                raise ValueError("Patches overlap or are not vertically separated.")
            
            # 4. Check Distance Parity (Delta Y)
            delta_y = gap_bounds[1] - gap_bounds[0]
            # Unrotated constraint: gap must be multiple of grid step (e.g. 2.0)
            if not self._validate_gap_parity(delta_y):
                raise ValueError(f"Vertical gap size {delta_y} violates parity constraints.")

        else:
            # ------------------------------------------------
            # Case: Horizontal Stacking (Gap in x-axis direction)
            # Requirement: Vertical Range (y-axis) must contain each other
            # ------------------------------------------------
            
            # 1. Check Containment
            a_contains_b = (bounds_a[2] <= bounds_b[2] and bounds_a[3] >= bounds_b[3])
            b_contains_a = (bounds_b[2] <= bounds_a[2] and bounds_b[3] >= bounds_a[3])
            
            if not (a_contains_b or b_contains_a):
                raise ValueError("Horizontal interaction requires strictly contained Y-ranges.")

            # 2. Determine Gap y-Bounds and Select Anchor Patch
            # Gap bound order matters, always from anchor patch to target patch
            anchor_patch = patch_a if b_contains_a else patch_b

            if bounds_a[1] < bounds_b[0]: # A is left of B
                if b_contains_a:
                    gap_bounds = (bounds_a[1], bounds_b[0])
                else:
                    gap_bounds = (bounds_b[0], bounds_a[1])
            elif bounds_b[1] < bounds_a[0]: # B is left of A
                if b_contains_a:
                    gap_bounds = (bounds_a[0], bounds_b[1])
                else:
                    gap_bounds = (bounds_b[1], bounds_a[0])
            else:
                 raise ValueError("Patches overlap or are not horizontally separated.")
            
            # 4. Check Distance Parity (Delta x)
            delta_x = gap_bounds[1] - gap_bounds[0]
            if not self._validate_gap_parity(delta_x):
                raise ValueError(f"Horizontal gap size {delta_x} violates parity constraints.")

        return logical_op_orientation, anchor_patch, gap_bounds

    def _construct_coupling_region(self, logical_op_orientation, gap_bounds, anchor_patch):
        """
        Pure geometry construction of the coupling region. Use "Anchor" method to infer the
        role of each qubit and add them to the coupling region.
        """
        
        target_patch = self.patches[0] if anchor_patch == self.patches[1] else self.patches[1]
        step = np.sign(gap_bounds[1] - gap_bounds[0]) # the sign points from anchor patch to target patch
        
        if logical_op_orientation == 'vertical':
            _, _, y_min_anchor, y_max_anchor = self._get_bounds(anchor_patch)
            _, _, y_min_target, y_max_target = self._get_bounds(target_patch)

            current_x = gap_bounds[0]
            while current_x < gap_bounds[1] + 1e-3: # Robust float comparison
                # before reaching the target patch (the larger one), expand the y range to the target patch
                if current_x.isclose(gap_bounds[1] - 1, abs_tol=1e-3):
                    current_y = y_min_target
                    terminate_y = y_max_target
                else:
                    current_y = y_min_anchor
                    terminate_y = y_max_anchor
                while current_y <= terminate_y + 1e-3:
                    role = self._infer_role_from_anchor(anchor_patch, current_x, current_y)
                    if role:
                        self.add_qubit(current_x, current_y, role=role)
                    current_y += 1
                current_x += step
        elif logical_op_orientation == 'horizontal':
            x_min_anchor, x_max_anchor, _, _ = self._get_bounds(anchor_patch)
            x_min_target, x_max_target, _, _ = self._get_bounds(target_patch)

            current_y = gap_bounds[0]
            while current_y < gap_bounds[1] + 1e-3: # Robust float comparison
                # before reaching the target patch (the larger one), expand the x range to the target patch
                if current_y.isclose(gap_bounds[1] - 1, abs_tol=1e-3):
                    current_x = x_min_target
                    terminate_x = x_max_target
                else:
                    current_x = x_min_anchor
                    terminate_x = x_max_anchor
                while current_x <= terminate_x + 1e-3:
                    role = self._infer_role_from_anchor(anchor_patch, current_x, current_y)
                    if role:
                        self.add_qubit(current_x, current_y, role=role)
                    current_x += 1
                current_y += step
        else:
            raise ValueError(f"Invalid logical operation orientation: {logical_op_orientation}.")

    def _infer_role_from_anchor(self, anchor_patch, x, y):
        """
        Infer the roles of the coupling region qubits according to the anchor qubit coordinates.
        """
        # Reference qubits
        anchor_data_coord = anchor_patch.data_coords[0]
        anchor_syndrome_x_coord = anchor_patch.syndrome_x_coords[0]
        anchor_syndrome_z_coord = anchor_patch.syndrome_z_coords[0]
        
        # Check parity distance
        data_dx = self.snap_coord((x - anchor_data_coord[0]))
        data_dy = self.snap_coord((y - anchor_data_coord[1]))
        syndrome_x_dx = self.snap_coord((x - anchor_syndrome_x_coord[0]))
        syndrome_x_dy = self.snap_coord((y - anchor_syndrome_x_coord[1]))
        syndrome_z_dx = self.snap_coord((x - anchor_syndrome_z_coord[0]))
        syndrome_z_dy = self.snap_coord((y - anchor_syndrome_z_coord[1]))

        # Infer the role based on the parity distance
        if data_dx % 2 == 0 and data_dy % 2 == 0:
            return 'data'
        elif syndrome_x_dx % 2 == 0 and syndrome_x_dy % 2 == 0:
            return 'syndrome_x'
        elif syndrome_z_dx % 2 == 0 and syndrome_z_dy % 2 == 0:
            return 'syndrome_z'
        else:
            raise ValueError(f"Invalid role for qubit {(x, y)}.")

    def _init_stabilizers(self):
        """
        Pure topology construction based on grid adjacency.
        """
        # for data_coord in anchor_patch.data_coords:
        # if self.interaction_type == 'ZZ':
        # for x in range((d_z + 1) // 2): # Z-stabilizers
        #             syn_coord = (4 * x, y)
        #             if x == 0: # Leftmost
        #                 targets = {(4 * x + 1, y - 1): "Z", (4 * x + 1, y + 1): "Z"}
        #             else:
        #                 targets = {(4 * x - 1, y - 1): "Z", (4 * x - 1, y + 1): "Z", 
        #                            (4 * x + 1, y - 1): "Z", (4 * x + 1, y + 1): "Z"}
        #             self.create_stim_stabilizer(targets, syn_coord, 'Z')
        # Logic:
        # Iterate over self.syndrome_qubit_indices (the ones we just added).
        # Check grid_map for neighbors (up, down, left, right).
        # Construct stabilizer based on neighbors found.
        pass

    # =========================================================
    # Helpers
    # =========================================================

    def _get_logical_op_orientation(self, interaction: str, angle: float, is_transposed: bool) -> str:
        """
        Determines if the logical operator is 'horizontal' or 'vertical'.
        Unrotated Orientation by Default: 
        - Z logical is Horizontal (Left-Right)
        - X logical is Vertical (Top-Down)
        """
        is_rotated_pi_2 = math.isclose(angle % np.pi, np.pi/2, abs_tol=1e-3)
        is_flipped = (is_rotated_pi_2 + is_transposed) % 2 == 1
            
        if interaction == 'ZZ':
            # By default, Z is Horizontal.\
            return 'vertical' if is_flipped else 'horizontal'
        elif interaction == 'XX':
            # Standard X is Vertical.
            return 'horizontal' if is_flipped else 'vertical'
        else:
            raise ValueError(f"Invalid interaction type: {interaction}")

    def _get_bounds(self, patch: QECPatch) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y)"""
        xs = [c[0] for c in patch.qubit_coords.values()]
        ys = [c[1] for c in patch.qubit_coords.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def _validate_gap_parity(self, delta: float) -> bool:
        """
        Checks if the gap size matches lattice constraints.
        For Unrotated, typically needs to be integer multiple of 2.0.
        """
        return abs(delta % 2.0) < 1e-5