import stim
from typing import Dict, List, Tuple, Set, Any, TYPE_CHECKING
from dataclasses import dataclass
from src.ir.qec_patch import QECPatch

if TYPE_CHECKING:
    from src.ir.coupler import BaseCoupler

class QECSystem:
    """
    The Global Canvas for Multi-Patch QEC Experiments. Consist of QEC patches and couplers.
    
    Responsibilities:
    1. Manages the global aggregation of coordinate system, qubit indexing, stabilizers, 
    and logical operators from multiple patches.
    2. Define couplers to enable logical operations between patches.
    3. Provides a unified view for the CircuitBuilder.
    """
    
    def __init__(self):
        # 1. Components Registry
        # name -> (patch_object, offset_tuple)
        self.patches: Dict[str, Tuple[QECPatch, Tuple[int, int]]] = {} 

        # 2. Global State & Indexing
        self.index_map: Dict[Tuple[int, int], int] = {} # (x, y) coordinate -> global_index
        self.qubit_coords: Dict[int, Tuple[int, int]] = {} # global_index -> (x, y) coordinate
        self.next_index = 0
        
        # Spatial Map: (x, y) -> patch_name 
        # Used for collision detection and debugging
        self.spatial_map: Dict[Tuple[int, int], str] = {}

        # 3. Aggregated System Properties (Global View)
        # These lists store GLOBAL coordinates
        self.data_coords: List[Tuple[int, int]] = []
        self.syndrome_coords_x: List[Tuple[int, int]] = []
        self.syndrome_coords_z: List[Tuple[int, int]] = []
        self.syndrome_coords: List[Tuple[int, int]] = [] # All syndromes
        
        # 4. Global Stabilizers
        # Stores stabilizers with GLOBAL INDICES, ready for CircuitBuilder
        # Same format as in qec_patch.py
        self.stabilizers: List[Dict[str, Any]] = []
        self.logical_ops: List[Dict[str, Any]] = []
        self.num_logicals: int = 0

        # 5. Couplers. Special for QEC Systems, enabling logical operations between patches.
        self.couplers: Dict[str, 'BaseCoupler'] = {} 
        self.active_couplers = set[str]()

    @property
    def num_qubits(self) -> int:
        return self.next_index
    
    # ======================================================================
    # Part 1. Patch Management, Information Aggregation
    # ======================================================================
    def add_patch(self, name: str, patch: QECPatch, offset: Tuple[int, int] = (0, 0)):
        """
        Registers a QECPatch onto the global canvas at a specific offset.
        
        Args:
            name: Unique identifier for the patch (e.g., "logical_1").
            patch: The QECPatch object (contains local coords and stabilizers).
            offset: (x_shift, y_shift) to place the patch on the canvas.
        """
        if name in self.patches:
            raise ValueError(f"Patch '{name}' already exists in the system.")
        
        # 1. Store reference
        self.patches[name] = (patch, offset)
        off_x, off_y = offset

        # ======================================================================
        # Step 1: Global Identity Registration
        # ======================================================================
        for local_coord in patch.index_map.keys():
            global_coord = (local_coord[0] + off_x, local_coord[1] + off_y)
            
            # Collision Check
            if global_coord in self.index_map:
                existing_owner = self.spatial_map[global_coord]
                raise ValueError(
                    f"Coordinate collision at {global_coord}. "
                    f"Trying to add patch '{name}', but occupied by '{existing_owner}'."
                )
            
            # Assign Unique Global Index
            idx = self.next_index
            self.index_map[global_coord] = idx
            self.qubit_coords[idx] = global_coord
            self.spatial_map[global_coord] = name
            self.next_index += 1

        # ======================================================================
        # Step 2: Classification / Categorization
        # ======================================================================
        # Helper to shift coords and extend global lists
        def shift_and_extend(source_list_name: str, target_global_list: List[Tuple[int, int]]):
            # Still using getattr here strictly for OPTIONAL sub-class attributes 
            # (like syndrome_coords_x/z which might not exist in all patches)
            local_list = getattr(patch, source_list_name, [])
            for local in local_list:
                target_global_list.append((local[0] + off_x, local[1] + off_y))

        shift_and_extend('data_coords', self.data_coords)
        shift_and_extend('syndrome_coords', self.syndrome_coords)
        shift_and_extend('syndrome_coords_x', self.syndrome_coords_x)
        shift_and_extend('syndrome_coords_z', self.syndrome_coords_z)

        # ======================================================================
        # Step 3: Stabilizer and Logical Operator Translation
        # ======================================================================
        for i, stab in enumerate(patch.stabilizers):
            global_stab = self._translate_stabilizer(stab, patch, offset)
            global_stab['patch_name'] = name
            global_stab['local_index'] = i
            self.stabilizers.append(global_stab)

        for i, op in enumerate(patch.logical_ops):
            global_op = self._translate_logical_op(op, patch, offset)
            global_op['patch_name'] = name
            global_op['local_index'] = i
            self.logical_ops.append(global_op)
        
        # 4. Add number of logical qubits from the patch
        self.num_logicals += patch.num_logicals

    def _translate_stabilizer(self, local_stab: Dict, patch: QECPatch, offset: Tuple[int, int]) -> Dict:
        off_x, off_y = offset
        
        # 1. Translate syndrome qubit
        loc_syn_coord = local_stab.get('syn_coord')
        if loc_syn_coord is None:
             raise ValueError("Stabilizer record missing 'syn_coord'.")
        
        glob_syn_coord = (loc_syn_coord[0] + off_x, loc_syn_coord[1] + off_y)
        if glob_syn_coord not in self.index_map:
            raise KeyError(f"Stabilizer ancilla {glob_syn_coord} not found in global map.")
        glob_syn_idx = self.index_map[glob_syn_coord]
        
        # 2. Translate Data & Pauli Basis
        glob_data_indices = []
        glob_paulis = {} # Replaces stim.PauliString
        
        local_ps = local_stab.get('pauli') # stim.PauliString object from patch
        local_data_indices = local_stab.get('data_indices', [])
        
        for loc_idx in local_data_indices:
            # A. Convert Index: Local -> Coord -> Global
            if loc_idx not in patch.qubit_coords:
                 raise KeyError(f"Local qubit index {loc_idx} not found in patch.")
            loc_coord = patch.qubit_coords[loc_idx]
            glob_coord = (loc_coord[0] + off_x, loc_coord[1] + off_y)
            
            if glob_coord not in self.index_map:
                raise KeyError(f"Stabilizer data qubit {glob_coord} not found in global map.")
            
            glob_idx = self.index_map[glob_coord]
            glob_data_indices.append(glob_idx)
            
            # B. Extract Pauli Basis (Crucial for mixed stabilizer like XZZX)
            if local_ps is not None:
                # 0=I, 1=X, 2=Y, 3=Z. We usually map to char for readability.
                basis_code = local_ps[loc_idx] 
                basis_char = {1: 'X', 2: 'Y', 3: 'Z', 0: 'I'}.get(basis_code, 'I')
                glob_paulis[glob_idx] = basis_char
            
        return {
            'paulis': glob_paulis,          # <--- Dictionary instead of Stim Object
            'type': local_stab.get('type'),
            'data_indices': glob_data_indices,
            'syn_coord': glob_syn_coord,
            'syn_idx': glob_syn_idx
        }
    
    def _translate_logical_op(self, local_op: Dict, patch: QECPatch, offset: Tuple[int, int]) -> Dict:
        off_x, off_y = offset
        
        local_ps = local_op.get('pauli')
        local_indices = local_op.get('data_indices', [])
        
        glob_indices = []
        glob_paulis = {}
        
        for loc_idx in local_indices:
            # Index Translation
            if loc_idx not in patch.qubit_coords:
                 raise KeyError(f"Local logical qubit index {loc_idx} not found.")
            loc_coord = patch.qubit_coords[loc_idx]
            glob_coord = (loc_coord[0] + off_x, loc_coord[1] + off_y)
            
            if glob_coord not in self.index_map:
                raise KeyError(f"Logical qubit {glob_coord} not found in global map.")
            
            glob_idx = self.index_map[glob_coord]
            glob_indices.append(glob_idx)
            
            # Basis Extraction
            if local_ps is not None:
                basis_code = local_ps[loc_idx]
                basis_char = {1: 'X', 2: 'Y', 3: 'Z', 0: 'I'}.get(basis_code, 'I')
                glob_paulis[glob_idx] = basis_char
            
        return {
            'paulis': glob_paulis, # Important if logical operator is mixed
            'type': local_op.get('type'),
            'data_indices': glob_indices,
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Returns a summary of the entire QEC system.
        Combines global stats with individual patch info.
        """
        patch_infos = {}
        for name, (patch, offset) in self.patches.items():
            # Get patch raw info and append offset info
            p_info = patch.get_info()
            p_info['system_offset'] = offset
            patch_infos[name] = p_info

        return {
            'num_patches': len(self.patches),
            'num_qubits': self.num_qubits,
            'num_logicals': self.num_logicals,
            'num_stabilizers': len(self.stabilizers),
            'patches': patch_infos,
            'global_layout': {
                'num_data': len(self.data_coords),
                'num_syndrome': len(self.syndrome_coords),
                'bbox': self._get_bounding_box()
            },
        }

    def _get_bounding_box(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Helper to find the (min_x, min_y) and (max_x, max_y) of the system."""
        if not self.index_map:
            return ((0,0), (0,0))
        all_coords = list(self.index_map.keys())
        min_x = min(c[0] for c in all_coords)
        max_x = max(c[0] for c in all_coords)
        min_y = min(c[1] for c in all_coords)
        max_y = max(c[1] for c in all_coords)
        return ((min_x, min_y), (max_x, max_y))


    # ======================================================================
    # Part 2. Coupler Management
    # ======================================================================
    def add_coupler(self, coupler: 'BaseCoupler'):
        """
        Registers a Coupler.
        1. Reuse add_patch to handle registration of the coupler's qubits.
        2. Store reference in self.couplers for conflict handling.
        """
        # 1. Treat Coupler as a Patch (Physical Registration)
        # Usually Couplers are defined with absolute coords or relative to patches,
        # so we pass offset=(0,0).
        self.add_patch(coupler.name, coupler, offset=(0,0))
        
        # 2. Register Logic
        self.couplers[coupler.name] = coupler

    def activate_coupler(self, name: str):
        if name not in self.couplers:
            raise ValueError(f"Unknown coupler: {name}")
        self.active_couplers.add(name)

    def deactivate_coupler(self, name: str):
        if name not in self.active_couplers:
            return # Already inactive or unknown
        self.active_couplers.remove(name)

    # ==========================================================================
    # Dynamic Stabilizer Filtering: Current stabilizers of the QEC system
    # ==========================================================================
    
    def get_current_stabilizers(self) -> List[Dict]:
        """
        Returns the effective list of stabilizers of the QEC system based on the active couplers.
        Logic: All Patch Stabs - Conflicts + Active Coupler Stabs
        """
        
        # Calculate the 'Kill set' (stabilizers to be disabled in the QEC system)
        kill_set = set()
        for name in self.active_couplers:
            # We assume the object in self.couplers has the 'conflicting_stabilizer_coords' attribute
            # We can use getattr to be safe if strictly typing as QECPatch
            coupler = self.couplers[name]
            conflicts = getattr(coupler, 'conflicting_stabilizer_coords', set())
            kill_set.update(conflicts)

        active_list = []
        
        for stab in self.stabilizers:
            owner = stab['patch_name']
            syn_coord = stab['syn_coord']
            
            # Case A: Stabilizer belongs to a Coupler
            if owner in self.couplers:
                # Only show if ACTIVE
                if owner in self.active_couplers:
                    active_list.append(stab)
            
            # Case B: Stabilizer belongs to a normal QEC Patch (not a coupler, encoding logical qubits)
            else:
                # Show UNLESS it's in the Kill set
                if syn_coord not in kill_set:
                    active_list.append(stab)
                    
        return active_list