from typing import Tuple, Dict, List, Optional, Literal, Set
import math
import numpy as np
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.coupler import LogicalCouplerProtocol

# -----------------------------------------------------------------------------
# Part 3. Couplers
# -----------------------------------------------------------------------------
class UnrotatedTwoPatchCoupler(LogicalCouplerProtocol):
    """
    Implementation of a coupler between two Unrotated Surface Code patches.
    Enforces strict alignment and containment rules for Phase 1 development.
    """
    
    EXPECTED_PATCH_COUNT = 2

    def __init__(self):
        super().__init__(name_prefix = "unrotated_coupler")

    def _build_coupler_geometry(self, coupler_patch: QECPatch, patches: List[QECPatch], **params):
        """
        The main orchestration flow.
        1. Geometry Analysis: Validates constraints and calculates the gap.
        2. Geometry Construction: Fills the gap with qubits.
        3. Topology Definition: Constructs stabilizers based on neighbors.
        params:
            interaction_type: 'ZZ' or 'XX'
        """

        interaction_type = params.get('interaction_type')

        # Step 1: Geometry Analysis
        # Returns:
        # - logical_op_orientation: 'horizontal' or 'vertical'
        # - anchor_patch: The smaller patch used as the starting point for the coupling region construction
        # - gap_bounds: (gap_x_min, gap_x_max) or (gap_y_min, gap_y_max) depending on the logical operator orientation
        logical_op_orientation, anchor_patch, gap_bounds = self._analyze_geometry(patches, interaction_type)

        # Step 2: Construct Coupling Region (Geometry)
        # This uses the "Anchor" method you plan to implement
        self._construct_coupling_region(coupler_patch, patches, logical_op_orientation, anchor_patch, gap_bounds)

        # Step 3: Initialize Stabilizers (Topology)
        # Unified method, agnostic of ZZ/XX
        self._init_stabilizers(coupler_patch, patches, logical_op_orientation, anchor_patch, gap_bounds)

    def _analyze_geometry(self, patches: List[QECPatch], interaction_type: Literal['ZZ', 'XX']):
        """
        Performs rigorous checks and defines the gap as a 4-tuple Bounding Box.
        Returns: (orientation, anchor_patch, (gap_x_min, gap_x_max, gap_y_min, gap_y_max))
        """
        patch_a = patches[0]
        patch_b = patches[1]
        # ====================================================
        # (1) Basic Constraint Checks
        # ====================================================
        angle_a = getattr(patch_a, 'rotation_angle', 0)
        angle_b = getattr(patch_b, 'rotation_angle', 0)
        
        # Check alignment
        diff = abs(angle_a - angle_b)
        if not (math.isclose(diff % np.pi, 0, abs_tol=1e-5)):
            raise ValueError(f"Patches must have aligned orientation. Got {angle_a} and {angle_b}.")
        
        # Check absolute rotation (Only support 0 and pi/2)
        if not (math.isclose(angle_a % (np.pi/2), 0, abs_tol=1e-5)):
             raise ValueError("Coupler currently only supports rotation angles of 0 or pi/2.")

        # Check transpose
        trans_a = getattr(patch_a, 'is_transposed', False)
        trans_b = getattr(patch_b, 'is_transposed', False)
        if trans_a != trans_b:
            raise ValueError(f"Patches must have same transposition state.")

        # ====================================================
        # (2) Interaction Logic & Orientation
        # ====================================================
        logical_op_orientation = self._get_logical_op_orientation(interaction_type, angle_a, trans_a)

        # ====================================================
        # (3) Containment & Gap Definition
        # ====================================================
        bounds_a = patch_a._get_bounds() # (min_x, max_x, min_y, max_y)
        bounds_b = patch_b._get_bounds()
        
        # Initialize
        gap_bounds = None # Will be (min_x, max_x, min_y, max_y)
        anchor_patch = None

        if logical_op_orientation == 'horizontal':
            # ------------------------------------------------
            # Case: Vertical Stacking (Patches are Top/Bottom) -> Gap is Horizontal Strip
            # ------------------------------------------------
            # 1. Check Horizontal Containment (X-axis)
            a_contains_b = (bounds_a[0] <= bounds_b[0] and bounds_a[1] >= bounds_b[1])
            b_contains_a = (bounds_b[0] <= bounds_a[0] and bounds_b[1] >= bounds_a[1])
            
            if not (a_contains_b or b_contains_a):
                raise ValueError("Vertical interaction requires strictly contained X-ranges.")
            
            anchor_patch = patch_a if b_contains_a else patch_b

            # 2. Determine Y-gap
            # Sort bounds to ensure min < max
            if bounds_a[3] < bounds_b[2]:   # A is below B
                gap_y_min, gap_y_max = bounds_a[3], bounds_b[2]
            elif bounds_b[3] < bounds_a[2]: # B is below A
                gap_y_min, gap_y_max = bounds_b[3], bounds_a[2]
            else:
                raise ValueError("Patches overlap or are not vertically separated.")
            
            # 3. Determine X-gap (The widest X range)
            gap_x_max = max(bounds_a[0], bounds_b[0])
            gap_x_min = min(bounds_a[1], bounds_b[1])
            
            # 4. Parity Check
            delta_y = gap_y_max - gap_y_min
            delta_x = bounds_a[0] - bounds_b[0]
            is_valid_x = (math.isclose(delta_x % 2.0, 0, abs_tol=1e-3))
            is_valid_y = (math.isclose(delta_y % 2.0, 0, abs_tol=1e-3))
            if not (is_valid_x and is_valid_y):
                 raise ValueError(f"X-gap size {delta_x} or Y-gap size {delta_y} is not valid.")

            gap_bounds = (gap_x_min, gap_x_max, gap_y_min, gap_y_max)

        else: # logical_op_orientation == 'vertical'
            # ------------------------------------------------
            # Case: Horizontal Stacking (Patches are Left/Right) -> Gap is Vertical Strip
            # ------------------------------------------------
            # 1. Check Vertical Containment (Y-axis)
            a_contains_b = (bounds_a[2] <= bounds_b[2] and bounds_a[3] >= bounds_b[3])
            b_contains_a = (bounds_b[2] <= bounds_a[2] and bounds_b[3] >= bounds_a[3])
            
            if not (a_contains_b or b_contains_a):
                raise ValueError("Horizontal interaction requires strictly contained Y-ranges.")

            anchor_patch = patch_a if b_contains_a else patch_b

            # 2. Determine X-gap
            if bounds_a[1] < bounds_b[0]:   # A is left of B
                gap_x_min, gap_x_max = bounds_a[1], bounds_b[0]
            elif bounds_b[1] < bounds_a[0]: # B is left of A
                gap_x_min, gap_x_max = bounds_b[1], bounds_a[0]
            else:
                 raise ValueError("Patches overlap or are not horizontally separated.")
            
            # 3. Determine Y-gap (the widest Y range)
            gap_y_min = min(bounds_a[2], bounds_b[2])
            gap_y_max = max(bounds_a[3], bounds_b[3])
            
            # 4. Parity Check
            delta_x = gap_x_max - gap_x_min
            delta_y = bounds_a[2] - bounds_b[2]
            is_valid_x = (math.isclose(delta_x % 2.0, 0, abs_tol=1e-3))
            is_valid_y = (math.isclose(delta_y % 2.0, 0, abs_tol=1e-3))
            if not (is_valid_x and is_valid_y):
                 raise ValueError(f"X-gap size {delta_x} or Y-gap size {delta_y} is not valid.")

            gap_bounds = (gap_x_min, gap_x_max, gap_y_min, gap_y_max)

        return logical_op_orientation, anchor_patch, gap_bounds


    def _construct_coupling_region(self, coupler_patch: QECPatch, patches: List[QECPatch], logical_op_orientation, anchor_patch, gap_bounds):
        """
        Pure geometry construction.
        Uses 4-tuple gap_bounds (min_x, max_x, min_y, max_y).
        Dynamically determines sweep direction based on anchor position.
        """
        gx_min, gx_max, gy_min, gy_max = gap_bounds
        
        # Determine Target Patch
        target_patch = patches[0] if anchor_patch == patches[1] else patches[1]
        
        # Estimate Grid Step (needed for loop)
        grid_step = 1.0
        
        if logical_op_orientation == 'vertical':
            # ---------------------------------------------------------
            # Horizontal Stacking (Left/Right) -> Sweep along X
            # ---------------------------------------------------------
            # Determine Sweep Direction: Are we going Left->Right or Right->Left?
            anchor_bounds = anchor_patch._get_bounds()
            
            # If Anchor is on the Left (max_x <= gap_min), we sweep +step
            if anchor_bounds[1] <= gx_min + 1e-3:
                step = grid_step
                start_x = gx_min + step
                end_x = gx_max # Exclusive in loop logic, not reached
            else: # Anchor is on the Right, sweep -step
                step = -grid_step
                start_x = gx_max + step # Start from inside gap
                end_x = gx_min 
            
            # Y-range targets
            _, _, y_min_anchor, y_max_anchor = anchor_patch._get_bounds()
            _, _, y_min_target, y_max_target = target_patch._get_bounds()

            current_x = start_x
            
            # Loop condition: Check distance to end (Robust float check)
            # We continue as long as we haven't passed the end point (considering direction)
            def keep_going(curr, end, s):
                return (curr < end - 1e-3) if s > 0 else (curr > end + 1e-3)

            while keep_going(current_x, end_x, step):
                
                # Check if this is the "Last Column" (adjacent to target)
                is_last_column = math.isclose(current_x, end_x - step, abs_tol=1e-3)

                if is_last_column:
                    current_y_min, current_y_max = y_min_target, y_max_target
                else:
                    current_y_min, current_y_max = y_min_anchor, y_max_anchor
                
                # Fill Column
                current_y = current_y_min
                while current_y <= current_y_max + 1e-3:
                    role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor_patch, current_x, current_y)
                    if role:
                        coupler_patch.add_qubit(current_x, current_y, role=role)
                    current_y += grid_step # Y step is always positive for column fill
                
                current_x += step

        elif logical_op_orientation == 'horizontal':
            # ---------------------------------------------------------
            # Vertical Stacking (Top/Bottom) -> Sweep along Y
            # ---------------------------------------------------------
            anchor_bounds = anchor_patch._get_bounds()
            
            # If Anchor is Bottom (max_y <= gap_min), sweep +step (Up)
            if anchor_bounds[3] <= gy_min + 1e-3:
                step = grid_step
                start_y = gy_min + step
                end_y = gy_max
            else: # Anchor is Top, sweep -step (Down)
                step = -grid_step
                start_y = gy_max + step
                end_y = gy_min
                
            x_min_anchor, x_max_anchor, _, _ = anchor_patch._get_bounds()
            x_min_target, x_max_target, _, _ = target_patch._get_bounds()

            current_y = start_y
            
            def keep_going(curr, end, s):
                return (curr < end - 1e-3) if s > 0 else (curr > end + 1e-3)

            while keep_going(current_y, end_y, step):
                
                is_last_row = math.isclose(current_y, end_y - step, abs_tol=1e-3)
                
                if is_last_row:
                    current_x_min, current_x_max = x_min_target, x_max_target
                else:
                    current_x_min, current_x_max = x_min_anchor, x_max_anchor
                
                current_x = current_x_min
                while current_x <= current_x_max + 1e-3:
                    role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor_patch, current_x, current_y)
                    if role:
                        coupler_patch.add_qubit(current_x, current_y, role=role)
                    current_x += grid_step
                
                current_y += step

        else:
            raise ValueError(f"Invalid logical op orientation: {logical_op_orientation}")


    @staticmethod
    def _infer_role_from_anchor(anchor_patch: QECPatch, x, y):
        """
        Infer the roles of the coupling region qubits according to the anchor qubit coordinates.
        """
        # Reference qubits
        anchor_data_coord = anchor_patch.data_coords[0]
        anchor_syndrome_coord_x = anchor_patch.syndrome_coords_x[0]
        anchor_syndrome_coord_z = anchor_patch.syndrome_coords_z[0]
        
        # Check parity distance
        data_delta = (x - anchor_data_coord[0], y - anchor_data_coord[1])
        syndrome_x_delta = (x - anchor_syndrome_coord_x[0], y - anchor_syndrome_coord_x[1])
        syndrome_z_delta = (x - anchor_syndrome_coord_z[0], y - anchor_syndrome_coord_z[1])

        # Infer the role based on the parity distance
        is_data = (data_delta[0] % 2 == 0 and data_delta[1] % 2 == 0) or (data_delta[0] % 2 == 1 and data_delta[1] % 2 == 1)
        is_syndrome_x = (syndrome_x_delta[0] % 2 == 0 and syndrome_x_delta[1] % 2 == 0)
        is_syndrome_z = (syndrome_z_delta[0] % 2 == 0 and syndrome_z_delta[1] % 2 == 0)

        if is_data:
            return 'data'
        elif is_syndrome_x:
            return 'syndrome_x'
        elif is_syndrome_z:
            return 'syndrome_z'
        else:
            raise ValueError(f"Invalid role for qubit {(x, y)}.")

    def _init_stabilizers(self, coupler_patch: QECPatch, patches: List[QECPatch], logical_op_orientation, anchor_patch, gap_bounds):
        """
        Topology Definition.
        Phase 1: Define stabilizers for NEW gap syndrome qubits.
        Phase 2: Redefine stabilizers for EXISTING boundary syndrome qubits.
        """

        coupler_patch.conflicting_stabilizer_coords = set() # Ensure initialization

        # ====================================================
        # Phase 1: Gap Internal Qubits (New)
        # ====================================================
        for uid in coupler_patch.syndrome_indices:
            syn_coord = coupler_patch.qubit_coords[uid]
            
            # Determine Type for New Qubits
            if uid in coupler_patch.syndrome_indices_x:
                stype = 'X'
            elif uid in coupler_patch.syndrome_indices_z:
                stype = 'Z'
            else:
                raise ValueError(f"Invalid syndrome qubit: {uid}, role is undefined.")
            
            self._probe_and_create_stabilizer(coupler_patch, patches, syn_coord, stype)

        # ====================================================
        # Phase 2: Boundary Qubits (Existing)
        # ====================================================
        # "Spatial Mask" Strategy: Find boundary syndrome qubits.
        boundary_candidates = UnrotatedTwoPatchCoupler._find_boundary_syndrome_candidates(patches, logical_op_orientation, gap_bounds)

        for syn_coord in boundary_candidates:
            # 1. Determine the type of the existing syndrome qubit
            stype = UnrotatedTwoPatchCoupler._resolve_existing_syndrome_type(patches, syn_coord)
            if not stype: continue

            # 2. Probe and create stabilizer
            success = self._probe_and_create_stabilizer(coupler_patch, patches, syn_coord, stype)

            if success:
                # 3. Add the existing syndrome qubit to the conflicting stabilizer coords
                coupler_patch.conflicting_stabilizer_coords.add(syn_coord)

    # ====================================================
    # Helper Functions (The Clean-up Crew)
    # ====================================================

    def _probe_and_create_stabilizer(self, coupler_patch, patches: List[QECPatch], syn_coord, stype) -> bool:
        """
        Core Logic: Probes 4 directions, finds data neighbors, creates GeometricStabilizer.
        Returns True if a stabilizer was created.
        """
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            target_x = syn_coord[0] + dx
            target_y = syn_coord[1] + dy
            
            # Check existence using the robust cross-boundary check
            if UnrotatedTwoPatchCoupler._is_data_qubit_at(coupler_patch, patches, target_x, target_y):
                neighbors.append((target_x, target_y))

        if neighbors:
            stabilizer_record = {
                'pauli': {coord: stype for coord in neighbors},
                'type': stype,
                'syn_coord': syn_coord,
            }
            # Note: When initializing the coupler patch, the stabilizer records is of this incomplete format.
            # The final format will be completed when the coupler patch added to the QECSystem using the QECSystem.add_patch method.
            coupler_patch.stabilizers.append(stabilizer_record)
            return True
            
        return False

    @staticmethod
    def _find_boundary_syndrome_candidates(patches: List[QECPatch], logical_op_orientation, gap_bounds) -> List[Tuple[float, float]]:
        """
        Spatial Mask: Finds existing syndrome qubits close to the gap.
        """
        candidates = []
        
        gx_min, gx_max, gy_min, gy_max = gap_bounds

        if logical_op_orientation == 'vertical':
            for patch in patches:
                for coord in patch.syndrome_coords: 
                    x, y = coord
                    if math.isclose(x, gx_min, abs_tol=1e-3) or math.isclose(x, gx_max, abs_tol=1e-3):
                        candidates.append(coord)
        else: # logical_op_orientation == 'horizontal':
            for patch in patches:
                for coord in patch.syndrome_coords: 
                    x, y = coord
                    if math.isclose(y, gy_min, abs_tol=1e-3) or math.isclose(y, gy_max, abs_tol=1e-3):
                        candidates.append(coord)
        return candidates

    @staticmethod
    def _resolve_existing_syndrome_type(patches: List[QECPatch], coord) -> Optional[str]:
        """
        Asks the interacting patches: "What type is the syndrome at this coordinate?"
        """
        for patch in patches:
            if hasattr(patch, 'syndrome_coords_x') and coord in patch.syndrome_coords_x:
                return 'X'
            if hasattr(patch, 'syndrome_coords_z') and coord in patch.syndrome_coords_z:
                return 'Z'
        return None

    @staticmethod
    def _is_data_qubit_at(coupler_patch: QECPatch, patches: List[QECPatch], x: float, y: float) -> bool:
        """
        Helper to check if a data qubit exists at (x, y).
        Checks:
        1. Self (Coupler)
        2. Interacting Patches (Patch A, Patch B)
        """
        # Check Self, the coupler
        if (x, y) in coupler_patch.index_map:
            uid = coupler_patch.index_map[(x, y)]
            if uid in coupler_patch.data_indices:
                return True
        
        # Check Interacting Patches (Cross-Boundary Check)
        for patch in patches:
            # Use the index_map of the interacting patch
            if (x, y) in patch.index_map:
                uid = patch.index_map[(x, y)]
                if uid in patch.data_indices:
                    return True
                    
        return False

    # =========================================================
    # Helpers
    # =========================================================

    def _get_logical_op_orientation(self, interaction_type: str, angle: float, is_transposed: bool) -> str:
        """
        Determines if the logical operator is 'horizontal' or 'vertical'.
        Unrotated Orientation by Default: 
        - Z logical is Horizontal (Left-Right)
        - X logical is Vertical (Top-Down)
        """
        is_rotated_pi_2 = math.isclose(angle % np.pi, np.pi/2, abs_tol=1e-3)
        is_flipped = (is_rotated_pi_2 + is_transposed) % 2 == 1
            
        if interaction_type == 'ZZ':
            # By default, Z is Horizontal.\
            return 'vertical' if is_flipped else 'horizontal'
        elif interaction_type == 'XX':
            # Standard X is Vertical.
            return 'horizontal' if is_flipped else 'vertical'
        else:
            raise ValueError(f"Invalid interaction type: {interaction_type}")