import stim
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass
from lightstim.ir.qec_patch import QECPatch
import copy
from lightstim.ir.coupler import LogicalCouplerProtocol

class QECSystem:
    """
    The Global Canvas for Multi-Patch QEC Experiments. Consist of QEC patches and couplers.
    
    Responsibilities:
    1. Manages the global aggregation of coordinate system, qubit indexing, stabilizers, 
    and logical operators from multiple patches.
    2. Define couplers to enable logical operations between patches.
    3. Tracking Stabilizer Activities (masked/unmasked) for the CircuitBuilder.
    """
    
    def __init__(self):
        # 1. Components Registry
        # name -> (patch_object, offset_tuple)
        self.patches: Dict[str, Tuple[QECPatch, Tuple[float, float]]] = {} # including code patches and coupler patches
        self.coupler_protocols: Dict[str, 'LogicalCouplerProtocol'] = {}
        self.coupler_patches: Dict[str, QECPatch] = {}
        # self._coupler_cache: Dict[str, str] = {} # For next phase of development


        # 2. Global State & Indexing
        self.index_map: Dict[Tuple[float, float], int] = {} # (x, y) coordinate -> global_index
        self.qubit_coords: Dict[int, Tuple[float, float]] = {} # global_index -> (x, y) coordinate
        self.grid_map: Dict[Tuple[int, int], int] = {} # (x, y) coordinate -> global_index
        self.next_index = 0

        # 3. Qubit Categorization (Global Sets of Indices)
        self.data_indices: Set[int] = set()
        self.syndrome_indices: Set[int] = set()
        self.syndrome_indices_x: Set[int] = set()
        self.syndrome_indices_z: Set[int] = set()

        # 4. Global Stabilizers (The Master List)
        # The list index IS the stabilizer unique ID.
        self.stabilizers: List[Dict[str, Any]] = [] 
        self._stabilizer_signatures: Dict[str, Any] = {}
        self.logical_ops: List[Dict[str, Any]] = [] 
        self.num_logicals: int = 0

        # 5. Dynamic Stabilizer Activities
        # Stores indices of self.stabilizers that are currently ON.
        self.active_stabilizer_indices: Set[int] = set()
        # coupler_name -> stabilizer_indices
        # The stabilizers indices that are paused because of the coupler's activation.
        self.paused_stabilizer_indices: Dict[str, Set[int]] = {}
        
        # 6. Active Qubit Tracking (Logical Lifetime)
        # Qubits are "active" from initialization to measurement.
        # Dormant qubits (measured, not yet re-initialized) can be reused by new couplers.
        self.active_qubit_indices: Set[int] = set()

        # Owner Map: (x, y) -> patch_name, Determine the owner of the qubit.
        # Used for collision detection and debugging
        self.coord_to_owner_map: Dict[Tuple[float, float], str] = {}
        self.index_to_owner_map: Dict[int, str] = {}
        # Local to Global Map: patch_name -> local_index -> global_index
        self.local_to_global_map: Dict[str, Dict[int, int]] = {}

        # Define-by-run: optional tracker and builder to auto-sync when add_patch adds new qubits
        self._tracker: Optional[Any] = None
        self._builder: Optional[Any] = None

    def register_tracker(self, tracker: Any):
        """Register a SyndromeTracker for define-by-run. When add_patch adds qubits, tracker.expand() is called automatically."""
        self._tracker = tracker

    def register_builder(self, builder: Any):
        """Register a CircuitBuilder for define-by-run. When add_patch adds qubits, QUBIT_COORDS for new qubits are appended automatically."""
        self._builder = builder

    @property
    def num_qubits(self) -> int:
        return self.next_index
    
    @property
    def data_coords(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.data_indices]
    
    @property
    def syndrome_coords(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.syndrome_indices]
    
    @property
    def syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.syndrome_indices_x]
    
    @property
    def syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.syndrome_indices_z]
    
    @property
    def active_stabilizers(self) -> List[Dict[str, Any]]:
        return [self.stabilizers[idx] for idx in sorted(self.active_stabilizer_indices)]
    
    @ property
    def active_stabilizers_x(self) -> List[Dict[str, Any]]:
        return [self.stabilizers[idx] for idx in sorted(self.active_stabilizer_indices) if self.stabilizers[idx].get('type') == 'X']
    
    @ property
    def active_stabilizers_z(self) -> List[Dict[str, Any]]:
        return [self.stabilizers[idx] for idx in sorted(self.active_stabilizer_indices) if self.stabilizers[idx].get('type') == 'Z']

    @property
    def active_syndrome_indices(self) -> List[int]:
        """
        Returns the set of unique global indices for syndrome qubits 
        that belong to CURRENTLY ACTIVE stabilizers.
        Used to generate 'R' and 'M' instructions only for relevant qubits.
        """
        return [
            self.stabilizers[uid]['syn_idx'] 
            for uid in self.active_stabilizer_indices
        ]

    @property
    def active_syndrome_indices_x(self) -> List[int]:
        """
        Returns indices of syndrome qubits measuring active X-stabilizers.
        Used to determine where to apply Hadamard gates (or basis change).
        """
        return [
            self.stabilizers[uid]['syn_idx'] 
            for uid in self.active_stabilizer_indices 
            if self.stabilizers[uid].get('type') == 'X'
        ]

    @property
    def active_syndrome_indices_z(self) -> List[int]:
        """
        Returns indices of syndrome qubits measuring active Z-stabilizers.
        """
        return [
            self.stabilizers[uid]['syn_idx'] 
            for uid in self.active_stabilizer_indices 
            if self.stabilizers[uid].get('type') == 'Z'
        ]

    @property
    def active_syndrome_coords(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.active_syndrome_indices]
    
    @property
    def active_syndrome_coords_x(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.active_syndrome_indices_x]
    
    @property
    def active_syndrome_coords_z(self) -> List[Tuple[float, float]]:
        return [self.qubit_coords[idx] for idx in self.active_syndrome_indices_z]


    # ======================================================================
    # Part 1. Patch Management, Information Aggregation
    # ======================================================================
    def add_patch(self, patch: QECPatch, offset: Tuple[float, float] = (0, 0), name: str = None, is_active: bool = True):
        """
        Registers a QECPatch onto the global canvas at a specific offset.
        
        Args:
            name: Unique identifier for the patch (e.g., "logical_1").
            patch: The QECPatch object (contains local coords and stabilizers).
            offset: (x_shift, y_shift) to place the patch on the canvas.
            is_active: If True, immediately unmasks the patch's stabilizers.
        """
        if name in self.patches:
            raise ValueError(f"Patch '{name}' already exists in the system.")
        
        if name is None:
            name = f"patch_{len(self.patches)+ len(self.coupler_patches)}"

        n_old = self.next_index  # for define-by-run: new qubits will be n_old, n_old+1, ...
        
        # 1. Store reference
        patch = copy.deepcopy(patch)
        patch.shift_coords(offset[0], offset[1])
        self.patches[name] = (patch, offset)
        # Note: the patch is already shifted, we just record the offset for the patch.

        # 2. Global Identity Registration
        local_to_global_map = {}
        self.local_to_global_map[name] = {}

        for global_coord, local_index in patch.index_map.items(): # the patch is already shifted to the global coordinate system

            if global_coord in self.index_map:
                existing_idx = self.index_map[global_coord]
                if existing_idx in self.active_qubit_indices:
                    existing_owner = self.coord_to_owner_map.get(global_coord, '?')
                    raise ValueError(
                        f"Coordinate collision with ACTIVE qubit at {global_coord}. "
                        f"Trying to add patch '{name}', but occupied by active '{existing_owner}'."
                    )
                # Reuse dormant qubit index (measured, ready for reuse)
                idx = existing_idx
            else:
                # New qubit — assign fresh index
                idx = self.next_index
                self.next_index += 1

            self.index_map[global_coord] = idx
            self.qubit_coords[idx] = global_coord
            self.coord_to_owner_map[global_coord] = name
            grid_key = patch.get_grid_key(global_coord)
            self.grid_map[grid_key] = idx
            local_to_global_map[local_index] = idx
            self.local_to_global_map[name][local_index] = idx
            self.index_to_owner_map[idx] = name

            # Categorize
            if local_index in patch.data_indices:
                self.data_indices.add(idx)
            if local_index in patch.syndrome_indices:
                self.syndrome_indices.add(idx)
            if hasattr(patch, 'syndrome_indices_x'):
                if local_index in patch.syndrome_indices_x:
                    self.syndrome_indices_x.add(idx)
            if hasattr(patch, 'syndrome_indices_z'):
                if local_index in patch.syndrome_indices_z:
                    self.syndrome_indices_z.add(idx)

        # 3. Stabilizer and Logical Operator Translation
        stabilizer_indices = []

        for stab in patch.stabilizers:
            global_stab = self._translate_record(stab, local_to_global_map)
            global_stab['patch_name'] = name
            # Generate the stabilizer signature
            pauli_indices = sorted(global_stab['data_indices'])
            signature = (global_stab['type'], tuple(pauli_indices))

            if signature in self._stabilizer_signatures:
                existing_uid = self._stabilizer_signatures[signature]
                stabilizer_indices.append(existing_uid)
                continue
            
            new_uid = len(self.stabilizers)
            self.stabilizers.append(global_stab)
            self._stabilizer_signatures[signature] = new_uid
            stabilizer_indices.append(new_uid)

        for op in patch.logical_ops:
            global_op = self._translate_record(op, local_to_global_map)
            global_op['patch_name'] = name
            self.logical_ops.append(global_op)
            # We don't need to track the logical operator indices for now.
        
        # 4. Add number of logical qubits from the patch
        self.num_logicals += patch.num_logicals

        # 5. Store registered stabilizer UIDs on the patch (for coupler activate/deactivate)
        patch._registered_stabilizer_uids = set(stabilizer_indices)

        # 6. Set Initial Active Stabilizers
        if is_active:
            self.active_stabilizer_indices.update(stabilizer_indices)

        # 6. Create and return global patch view (with global indices)
        # This is a deep copy of the patch with all indices converted to global
        global_patch = copy.deepcopy(patch)
        
        # Convert data_indices from local to global
        global_patch.data_indices = {local_to_global_map[local_idx] for local_idx in patch.data_indices}
        
        # Convert syndrome_indices from local to global
        global_patch.syndrome_indices = {local_to_global_map[local_idx] for local_idx in patch.syndrome_indices}
        
        # Convert syndrome_indices_x and syndrome_indices_z if they exist
        if hasattr(patch, 'syndrome_indices_x'):
            global_patch.syndrome_indices_x = {local_to_global_map[local_idx] for local_idx in patch.syndrome_indices_x}
        if hasattr(patch, 'syndrome_indices_z'):
            global_patch.syndrome_indices_z = {local_to_global_map[local_idx] for local_idx in patch.syndrome_indices_z}
        
        # Convert logical_ops indices from local to global
        for logical_op in global_patch.logical_ops:
            # Update pauli string indices
            new_pauli = {}
            for local_idx, pauli_type in logical_op['pauli'].items():
                if local_idx in local_to_global_map:
                    new_pauli[local_to_global_map[local_idx]] = pauli_type
            logical_op['pauli'] = new_pauli
            
            # Update data_indices
            logical_op['data_indices'] = [local_to_global_map[local_idx] for local_idx in logical_op['data_indices'] if local_idx in local_to_global_map]
        
        # Convert stabilizers indices from local to global
        for stabilizer in global_patch.stabilizers:
            raw_pauli = stabilizer.get('pauli', {})
            new_pauli = {}
            for key, pauli_type in raw_pauli.items():
                if isinstance(key, int):  # Code Patch (local index)
                    if key in local_to_global_map:
                        new_pauli[local_to_global_map[key]] = pauli_type
                elif isinstance(key, tuple):  # Coupler Patch (coord)
                    if key in self.index_map:
                        new_pauli[self.index_map[key]] = pauli_type
            stabilizer['pauli'] = new_pauli

            # Update data_indices (Code Patch has it; Coupler derives from pauli)
            if 'data_indices' in stabilizer:
                stabilizer['data_indices'] = [local_to_global_map[i] for i in stabilizer['data_indices'] if i in local_to_global_map]
            else:
                stabilizer['data_indices'] = sorted(list(new_pauli.keys()))

            # Update syn_idx if it exists
            if stabilizer.get('syn_idx') is not None and stabilizer['syn_idx'] in local_to_global_map:
                stabilizer['syn_idx'] = local_to_global_map[stabilizer['syn_idx']]
            elif stabilizer.get('syn_coord') is not None:
                coord = stabilizer['syn_coord']
                if coord in self.index_map:
                    stabilizer['syn_idx'] = self.index_map[coord]

        # Define-by-run: auto-expand tracker, sync expected_num_logicals, append QUBIT_COORDS for new qubits
        if self._tracker is not None:
            n_new = self.num_qubits
            if n_new > self._tracker.num_qubits:
                self._tracker.expand(n_new - self._tracker.num_qubits)
            self._tracker.expected_num_logicals = self.num_logicals
        if self._builder is not None and n_old < self.num_qubits:
            self._builder.append_coordinates_for_new_qubits(n_old)

        return global_patch

    # ======================================================================
    # Helper Methods
    # ======================================================================

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
            'coupler_patches': self.coupler_patches,
            'coupler_protocols': self.coupler_protocols,
            'global_layout': {
                'num_data': len(self.data_coords),
                'num_syndrome': len(self.syndrome_coords),
            },
        }
    
    def _translate_record(self, record: Dict, local_to_global_map: Dict[int, int]) -> Dict:
        """
        Translates a stabilizer/logical record from local/coordinate context to global index context.
        
        Handles two cases:
        1. Code Patch (Standard): {"pauli": {id: pauli_type}, "type": ..., "data_indices": ..., 
        "syn_coord": ..., "syn_idx": ...}
        2. Coupler Patch (Hybrid): {'pauli': {coord: pauli_type}, 'type': ..., 'syn_coord': ...}
        
        Args:
            record: The raw stabilizer record from the patch.
            local_to_global_map: Mapping from local_uid -> global_uid for the CURRENT patch.
            
        Returns:
            A unified record with global indices: 
            {'paulis': {global_idx: type}, 'syn_idx': global_idx, ...}
        """
        new_record = record.copy()
    
        # --- 1. Process Pauli Support (Data Qubits) ---
        global_pauli = {}
        raw_pauli = record.get('pauli', {}) 
        
        for key, pauli_type in raw_pauli.items():
            if isinstance(key, int): # Code Patch (Local Index)
                if key in local_to_global_map:
                    global_pauli[local_to_global_map[key]] = pauli_type
                else:
                    raise ValueError(f"Local index {key} not found in current patch map.")
            elif isinstance(key, tuple): # Coupler Patch (Global Coordinate)
                if key in self.index_map:
                    global_pauli[self.index_map[key]] = pauli_type
                else:
                    raise ValueError(f"Coordinate {key} not found in System.")
        
        new_record['pauli'] = global_pauli

        # --- 2. Process Data Indices ---
        if 'data_indices' in record:
            # Case A: Standard Code Patch
            raw_data_indices = record['data_indices']
            global_data_indices = [local_to_global_map[local_idx] for local_idx in raw_data_indices]
            new_record['data_indices'] = global_data_indices
        else:
            # Case B: Coupler Patch (Auto-generate from pauli keys)
            new_record['data_indices'] = sorted(list(global_pauli.keys()))
        
        # --- 3. Process Syndrome Index ---
        if 'syn_idx' in record:
            # Case A: Standard Code Patch (Local Index)
            # Allow syn_idx=None for stabilizers without syndrome ancilla (e.g. PQRM X stabs)
            if record['syn_idx'] is not None:
                new_record['syn_idx'] = local_to_global_map[record['syn_idx']]
            else:
                new_record['syn_idx'] = None
        elif 'syn_coord' in record:
            # Case B: Coupler Patch (Coord -> Global Index)
            # Since Coupler format lacks syn_idx, we must look it up via coord
            coord = record['syn_coord']
            if coord in self.index_map:
                new_record['syn_idx'] = self.index_map[coord]
            else:
                raise ValueError(f"Syndrome coordinate {coord} not found in System.")

        return new_record

    # ======================================================================
    # Part 2. Coupler Management
    # ======================================================================
    def add_coupler_protocol(self, protocol: 'LogicalCouplerProtocol', name: str):
        """
        Registers a Coupler Protocol.
        """
        self.coupler_protocols[name] = protocol
    

    def register_coupler(self, 
                         protocol: 'LogicalCouplerProtocol', 
                         patch_names: List[str], 
                         name: str = None, 
                         **kwargs):
        """
        Uses a Protocol to generate a coupler patch between existing patches,
        and registers it to the system (default: Inactive).
        """
        # 1. Retrieve Physical Patches & Offsets
        patches = []
        for p_name in patch_names:
            if p_name not in self.patches:
                raise ValueError(f"Patch '{p_name}' not found in system.")
            p_obj, p_off = self.patches[p_name]
            patches.append(p_obj)
            
        # 2. Use the Factory (Protocol) to generate the 'Connector' Patch
        # This does the heavy lifting: geometry analysis, stabilizer generation
        coupler_patch = protocol.create_coupler_patch(patches, name=name, **kwargs)
        
        # 3. Register it into the System
        # [Crucial]: Add it with is_active=False
        # Physically it exists (qubits allocated), but Logically it's OFF.
        # This will internally call self.add_patch(...)
        self.add_patch(coupler_patch, offset=(0,0), name=coupler_patch.name, is_active=False)
        
        # 4. Store the registered patch (the deep-copied version from add_patch,
        # which has _registered_stabilizer_uids set)
        self.coupler_patches[coupler_patch.name] = self.patches[coupler_patch.name][0]

        return coupler_patch
    
    def activate_coupler(self, coupler_name: str):
        """
        Activates a registered coupler.
        1. Deactivates conflicting stabilizers on the code patches.
        2. Activates the coupler's stabilizers.
        3. Saves the 'killed' stabilizers to history for later restoration.
        """
        if coupler_name not in self.coupler_patches:
            raise ValueError(f"Coupler '{coupler_name}' not found.")

        # 0. Idempotency Check, avoid re-activation
        if coupler_name in self.paused_stabilizer_indices:
            print(f"Coupler '{coupler_name}' is already active.")
            return

        coupler_patch = self.coupler_patches[coupler_name]

        # 1. Identify Coupler Stabilizers from registered UIDs
        # (stored during add_patch, includes deduplicated stabilizers)
        coupler_stabilizer_uids = getattr(coupler_patch, '_registered_stabilizer_uids', set())

        # 2. Identify Conflicting Stabilizers (code patch stabs at boundary positions)
        conflict_coords = getattr(coupler_patch, 'conflicting_stabilizer_coords', set())
        
        conflict_uids = set()
        if conflict_coords:
            # We need to map Coordinates -> Stabilizer UIDs
            for uid in self.active_stabilizer_indices:
                stab = self.stabilizers[uid]
                if stab['syn_coord'] in conflict_coords:
                    conflict_uids.add(uid)

        # 3. Execute the stabilizer masking/unmasking
        
        # A. Save history (Critical for Deactivation)
        self.paused_stabilizer_indices[coupler_name] = conflict_uids
        
        # B. Deactivate Conflicts and update the stabilizer tableau
        self.active_stabilizer_indices.difference_update(conflict_uids)

        # C. Activate Coupler (exclude conflict UIDs — they're the originals being replaced)
        self.active_stabilizer_indices.update(coupler_stabilizer_uids - conflict_uids)


    def deactivate_coupler(self, coupler_name: str):
        """
        Deactivates a coupler.
        1. Deactivates the coupler's stabilizers.
        2. Restores (Re-activates) the stabilizers that were paused by this coupler.
        """
        if coupler_name not in self.paused_stabilizer_indices:
            print(f"Coupler '{coupler_name}' is not currently active (or wasn't activated via this method).")
            return

        # 1. Identify Coupler Stabilizers from registered UIDs
        coupler_patch = self.coupler_patches[coupler_name]
        coupler_stabilizer_uids = getattr(coupler_patch, '_registered_stabilizer_uids', set())

        # 2. Retrieve History
        restored_uids = self.paused_stabilizer_indices.pop(coupler_name)

        # 3. Execute the stabilizer masking/unmasking
        
        # A. Deactivate Coupler
        self.active_stabilizer_indices.difference_update(coupler_stabilizer_uids)
        
        # B. Restore Original Stabilizers
        self.active_stabilizer_indices.update(restored_uids)

    def remove_coupler(self, coupler_name: str):
        """
        Remove a coupler patch from the system so the same corridor space
        can be reused by a new coupler registration.

        Must be called AFTER deactivate_coupler. The coupler's qubit indices
        become orphans (never measured again) but stay allocated to avoid
        re-indexing. The coordinate maps are cleared so new couplers can
        register at the same positions.
        """
        if coupler_name not in self.coupler_patches:
            raise ValueError(f"Coupler '{coupler_name}' not found.")
        if coupler_name in self.paused_stabilizer_indices:
            raise ValueError(f"Coupler '{coupler_name}' is still active. Deactivate first.")

        coupler_patch = self.coupler_patches[coupler_name]

        # 1. Collect global indices owned by this coupler
        coupler_global_indices = set()
        if coupler_name in self.local_to_global_map:
            coupler_global_indices = set(self.local_to_global_map[coupler_name].values())

        # 2. Clear coordinate maps (allows re-registration at same coords)
        coords_to_remove = []
        for coord, owner in list(self.coord_to_owner_map.items()):
            if owner == coupler_name:
                coords_to_remove.append(coord)

        for coord in coords_to_remove:
            del self.coord_to_owner_map[coord]
            if coord in self.index_map:
                del self.index_map[coord]
            grid_key = coupler_patch.get_grid_key(coord)
            if grid_key in self.grid_map:
                del self.grid_map[grid_key]

        # 3. Remove from qubit category sets (data, syndrome)
        self.data_indices.difference_update(coupler_global_indices)
        self.syndrome_indices.difference_update(coupler_global_indices)
        self.syndrome_indices_x.difference_update(coupler_global_indices)
        self.syndrome_indices_z.difference_update(coupler_global_indices)

        # 4. Remove coupler stabilizers from active set
        coupler_stab_uids = {
            i for i, s in enumerate(self.stabilizers)
            if s.get('patch_name') == coupler_name
        }
        self.active_stabilizer_indices.difference_update(coupler_stab_uids)

        # 5. Mark stabilizers as removed (set patch_name to None so they're skipped)
        for uid in coupler_stab_uids:
            self.stabilizers[uid]['patch_name'] = None

        # 6. Clean up signature cache for removed stabilizers
        sigs_to_remove = [sig for sig, uid in self._stabilizer_signatures.items()
                          if uid in coupler_stab_uids]
        for sig in sigs_to_remove:
            del self._stabilizer_signatures[sig]

        # 7. Remove from index_to_owner_map
        for idx in coupler_global_indices:
            if idx in self.index_to_owner_map:
                del self.index_to_owner_map[idx]

        # 8. Remove qubit_coords for coupler qubits (prevents ghost qubits in diagrams)
        for idx in coupler_global_indices:
            if idx in self.qubit_coords:
                del self.qubit_coords[idx]

        # 9. Remove from patch/coupler registries
        del self.local_to_global_map[coupler_name]
        del self.coupler_patches[coupler_name]
        del self.patches[coupler_name]