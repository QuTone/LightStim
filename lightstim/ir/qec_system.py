import stim
import numpy as np
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

        # grow_patch bookkeeping for repair_grow_logicals: patch_name -> orphan info
        self._grow_repairs: Dict[str, Dict[str, Any]] = {}

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
        return sorted({
            self.stabilizers[uid]['syn_idx']
            for uid in self.active_stabilizer_indices
            if self.stabilizers[uid].get('syn_idx') is not None
        })

    @property
    def active_syndrome_indices_x(self) -> List[int]:
        """
        Returns indices of syndrome qubits measuring active X-stabilizers.
        Used to determine where to apply Hadamard gates (or basis change).
        """
        return sorted({
            self.stabilizers[uid]['syn_idx']
            for uid in self.active_stabilizer_indices
            if (
                self.stabilizers[uid].get('type') == 'X'
                and self.stabilizers[uid].get('syn_idx') is not None
            )
        })

    @property
    def active_syndrome_indices_z(self) -> List[int]:
        """
        Returns indices of syndrome qubits measuring active Z-stabilizers.
        """
        return sorted({
            self.stabilizers[uid]['syn_idx']
            for uid in self.active_stabilizer_indices
            if (
                self.stabilizers[uid].get('type') == 'Z'
                and self.stabilizers[uid].get('syn_idx') is not None
            )
        })

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
        n_log_old = self.num_logicals  # define-by-run: budget grows by THIS patch's logicals

        # 1. Store reference
        patch = copy.deepcopy(patch)
        patch.shift_coords(offset[0], offset[1])
        self.patches[name] = (patch, offset)
        # Note: the patch is already shifted, we just record the offset for the patch.

        # 2. Global Identity Registration
        local_to_global_map = {}
        self.local_to_global_map[name] = {}
        reused_indices = []

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
                reused_indices.append(idx)
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

        # A reused dormant index must stop counting as retired: the retire
        # bookkeeping (Fix C absorbed resolution, retired-qubit masking)
        # would otherwise treat the LIVE qubit as measured-out.  The tracker
        # rows were eliminated from these columns at retirement, so any
        # remaining support means the retirement was incomplete — fail loud
        # rather than build on a stale tableau.
        tracker = getattr(self, "_tracker", None)
        if tracker is not None and reused_indices:
            stale = sorted(q for q in reused_indices
                           if q in tracker.retired_qubits)
            if stale:
                n_t = tracker.num_qubits
                cols = [q for q in stale] + [n_t + q for q in stale]
                for tab_label, tab in (("stabilizers", tracker.stabilizers),
                                       ("logicals", tracker.logicals),
                                       ("absorbed_ops", tracker.absorbed_ops)):
                    if tab.count and tab.matrix[:, cols].any():
                        raise RuntimeError(
                            f"add_patch('{name}'): reusing retired qubits "
                            f"{stale}, but tracker {tab_label} rows still "
                            f"reference them — stale retirement state.")
                tracker.retired_qubits.difference_update(stale)

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

        # 7. Create and return global patch view (with global indices)
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

        # Define-by-run: auto-expand tracker, GROW the logical budget by exactly this
        # patch's logicals, append QUBIT_COORDS for new qubits.
        if self._tracker is not None:
            n_new = self.num_qubits
            if n_new > self._tracker.num_qubits:
                self._tracker.expand(n_new - self._tracker.num_qubits)
            # Increment, do NOT reset to self.num_logicals: a mid-sequence registration
            # (after an earlier PPM already consumed logical DOF) must not clobber the
            # reduced budget back up to the stale static total. For registration before
            # any consumption this equals the old `expected := num_logicals` (they match
            # then); a coupler adds 0 logicals, so it leaves the budget untouched.
            self._tracker.expected_num_logicals += self.num_logicals - n_log_old
        if self._builder is not None and n_old < self.num_qubits:
            self._builder.append_coordinates_for_new_qubits(n_old)

        return global_patch

    def grow_patch(self, name: str, new_patch: QECPatch, offset: Tuple[float, float] = (0, 0)):
        """Grow a LIVE patch onto a LARGER footprint (Litinski's *moving boundaries*,
        `A Game of Surface Codes` Fig 40c) — the first atom of a patch rotation.

        ``new_patch`` is the enlarged reference geometry (e.g.
        ``RotatedSurfaceCode(distance_x=d, distance_z=2*d+1)``) placed at ``offset`` so
        that it COVERS the patch's current qubits at their current coordinates.  Those
        shared qubits keep their existing global indices, so every index-keyed record and
        the index-based ``SyndromeTracker`` stay valid, and — because a stabilizer's
        signature is ``(type, global data indices)`` — every check that the enlarged code
        shares with the old one **keeps its uid** and therefore its detector history.  That
        is what anchors the growth: extending a ``d×d`` patch by one tile keeps 7 of its 8
        checks verbatim (only the weight-2 on the boundary being extended through changes),
        so there is no blind round.

        The freshly added data qubits are returned; the caller initialises them in the
        basis OF THE LOGICAL BEING EXTENDED — growth through a Z boundary extends Z̄,
        so the new region is |0⟩ (NOT |+⟩: that leaves a one-round detector-free
        handoff window, measured graphlike distance 1) — and runs ``d`` SE rounds with
        the DIAGONAL schedule (``DiagonalSurfaceCodeExtractionBlock``; every N/Z
        zigzag caps the deformed patch's distance).  Under that init every new
        same-type check is first-round deterministic, so the retiring growth-edge
        lobes' enlarged successors continue their banked values (bridging detectors)
        and the handoff window is fully detector-spanned — see
        docs/specs/litinski-patch-rotation.md (bridging detectors).

        Returns ``(new_data_indices, new_syndrome_indices)``.
        """
        if name not in self.patches:
            raise ValueError(f"patch {name!r} is not in the system")
        old_idxs = {i for i, o in self.index_to_owner_map.items() if o == name}
        if not old_idxs:
            raise ValueError(f"patch {name!r} has no qubits in the system")

        grown = copy.deepcopy(new_patch)
        grown.shift_coords(offset[0], offset[1])

        # every current DATA qubit of the patch must still be covered by the grown
        # geometry, else the deformation would silently drop live information.
        # Syndrome ancillas MAY be dropped (move_corners retires boundary lobes whose
        # ancilla site the new boundary no longer uses) — a dropped ancilla simply
        # stops appearing in active_syndrome_indices once its check deactivates; its
        # cell stays owned by the patch until a shrink/retire releases it.
        grown_coords = set(grown.index_map)
        # retired data qubits (an earlier shrink's bar, an absorbed corridor)
        # carry no live information — their records are folded into the
        # tracker — so they are exempt from the coverage requirement; the
        # geometry engine below re-initialises any the new geometry re-occupies
        retired = (self._tracker.retired_qubits if self._tracker is not None
                   else frozenset())
        missing = {self.qubit_coords[i] for i in old_idxs
                   if i in self.data_indices and i not in retired} - grown_coords
        if missing:
            raise ValueError(
                f"grow_patch({name!r}): the new geometry does not cover {len(missing)} "
                f"of the patch's current data qubits (e.g. {sorted(missing)[:3]}); it "
                f"must cover them all, placed so shared qubits keep their coordinates")
        return self._apply_patch_geometry(name, grown, offset)

    def _apply_patch_geometry(self, name: str, grown: QECPatch,
                              offset: Tuple[float, float]):
        """Shared deformation engine behind grow_patch / move_corners / shrink_patch.

        ``grown`` is the ALREADY-SHIFTED reference geometry.  Maps its local indices
        onto global ones (shared coords keep their index), re-registers the
        stabilizers (identical signatures keep their uid — detector continuity),
        records retiring-check/orphan bookkeeping in ``_grow_repairs[name]``, swaps in
        the geometry's logical operator records, and syncs tracker/builder.
        Returns ``(new_data_indices, new_syndrome_indices)``."""
        n_old_qubits = self.next_index
        old_idxs = {i for i, o in self.index_to_owner_map.items() if o == name}

        # ---- 1. map the grown geometry's local indices onto global ones -------------
        l2g = {}
        new_data, new_syn = [], []
        for coord, local_index in grown.index_map.items():
            existing = self.index_map.get(coord)
            if existing is not None:
                owner = self.coord_to_owner_map.get(coord)
                if owner not in (None, name) and existing in self.active_qubit_indices:
                    raise ValueError(
                        f"grow_patch({name!r}): coordinate {coord} is occupied by ACTIVE "
                        f"patch {owner!r}; the growth region must be free")
                idx = existing
                # a DORMANT qubit of this patch (measured out by an earlier shrink,
                # flagged retired in the tracker) that the new geometry re-occupies is
                # FRESH again: it needs re-initialisation, so it must be returned in
                # new_data — the rotation's move-back (Fig 45 step 7→8) regrows the
                # bar over the cells the first shrink retired.
                was_retired = (self._tracker is not None
                               and idx in self._tracker.retired_qubits)
                fresh = idx not in old_idxs or was_retired
                if was_retired:
                    self._tracker.retired_qubits.discard(idx)
            else:
                idx = self.next_index
                self.next_index += 1
                fresh = True
            self.index_map[coord] = idx
            self.qubit_coords[idx] = coord
            self.coord_to_owner_map[coord] = name
            self.grid_map[grown.get_grid_key(coord)] = idx
            self.index_to_owner_map[idx] = name
            l2g[local_index] = idx
            if local_index in grown.data_indices:
                self.data_indices.add(idx)
                if fresh:
                    new_data.append(idx)
            if local_index in grown.syndrome_indices:
                self.syndrome_indices.add(idx)
                if fresh:
                    new_syn.append(idx)
            if local_index in getattr(grown, 'syndrome_indices_x', ()):
                self.syndrome_indices_x.add(idx)
            if local_index in getattr(grown, 'syndrome_indices_z', ()):
                self.syndrome_indices_z.add(idx)
            # NOTE: deliberately NOT touching active_qubit_indices — add_patch does not
            # populate it either (it is the dormant-reuse ledger that retire_* maintains),
            # and diverging here would make a grown patch behave differently from a
            # normally-added one when its cells are later reused.
        self.local_to_global_map[name] = dict(l2g)

        # ---- 2. re-register the stabilizers; shared ones keep their uid -------------
        old_uids = {uid for uid, s in enumerate(self.stabilizers)
                    if s.get('patch_name') == name}
        self.active_stabilizer_indices.difference_update(old_uids)
        grown_uids = []
        for stab in grown.stabilizers:
            g = self._translate_record(stab, l2g)
            g['patch_name'] = name
            signature = (g['type'], tuple(sorted(g['data_indices'])))
            uid = self._stabilizer_signatures.get(signature)
            if uid is None:
                uid = len(self.stabilizers)
                self.stabilizers.append(g)
                self._stabilizer_signatures[signature] = uid
            else:
                # the signature can also match a DEAD region's record (an
                # absorbed corridor this geometry regrows over) — reclaim
                # ownership, or every patch_name-keyed scan (shrink coverage
                # criterion, retire, SE selection) misses the revived check
                self.stabilizers[uid]['patch_name'] = name
            grown_uids.append(uid)
        self.active_stabilizer_indices.update(grown_uids)
        grown._registered_stabilizer_uids = set(grown_uids)

        # ---- 2b. orphan bookkeeping for repair_grow_logicals ------------------------
        # A retiring growth-edge check's data qubit is ORPHANED when every same-type
        # check covering it in the grown code is NEW (first outcome random under the
        # fresh-qubit init) — the one-round handoff there is not detector-spanned.
        retiring = old_uids - set(grown_uids)
        kept = old_uids & set(grown_uids)
        orphans = set()
        for uid in retiring:
            s = self.stabilizers[uid]
            for q in s['data_indices']:
                covered = any(
                    q in self.stabilizers[u]['data_indices']
                    and self.stabilizers[u]['type'] == s['type']
                    for u in kept)
                if not covered:
                    orphans.add(q)
        self._grow_repairs[name] = {'orphans': orphans, 'retired_uids': retiring}

        # ---- 3. swap in the enlarged logical operators (the patch still holds 1) ----
        self.logical_ops = [l for l in self.logical_ops if l.get('patch_name') != name]
        for op in grown.logical_ops:
            g = self._translate_record(op, l2g)
            g['patch_name'] = name
            self.logical_ops.append(g)

        self.patches[name] = (grown, offset)

        # ---- 4. define-by-run bookkeeping (qubit count only; k is unchanged) --------
        if self._tracker is not None and self.num_qubits > self._tracker.num_qubits:
            self._tracker.expand(self.num_qubits - self._tracker.num_qubits)
        if self._builder is not None and n_old_qubits < self.num_qubits:
            self._builder.append_coordinates_for_new_qubits(n_old_qubits)
        return new_data, new_syn

    def grow_init_bases(self, name: str, new_data: List[int], boundary_type: str = 'Z'):
        """Per-qubit init bases for the qubits added by the last :meth:`grow_patch`.

        Growing through a ``boundary_type`` boundary initialises the new region in the
        CONJUGATE basis (Z boundary → |+⟩, X boundary → |0⟩) — Fig 40c/41's ancilla
        prescription, which makes the conjugate-type new checks trivial-outcome
        (deterministic) on their first round.

        EXCEPT: each RETIRING growth-edge check dissolves into an enlarged successor,
        and the successor's first outcome must deterministically continue the retired
        check's banked value, else the handoff window is spanned by no detector and a
        single data error on the retired check's outer qubit flips the logical frame
        silently (measured: graphlike distance 1).  The qubits the successor ADDS are
        therefore initialised in the retiring check's OWN basis (Z lobe → |0⟩), which
        pins the added factor to +1 and turns the successor's first round into the
        bridging detector.  This mixed product state is exactly the fused equivalent of
        the decomposed surgery's "prepare the |+̄⟩ ancilla patch and let its boundary
        settle, then merge": an X̄ = +1 ancilla state whose merge-facing boundary lobe
        is already sharp.
        """
        info = self._grow_repairs.get(name)
        conj = {'Z': 'X', 'X': 'Z'}
        bases = {q: conj[boundary_type] for q in new_data}
        if info:
            grown_uids = {uid for uid, s in enumerate(self.stabilizers)
                          if s.get('patch_name') == name
                          and uid in self.active_stabilizer_indices}
            for uid in info['retired_uids']:
                r = self.stabilizers[uid]
                r_sup = set(r['data_indices'])
                for u in grown_uids:
                    s = self.stabilizers[u]
                    if s['type'] == r['type'] and r_sup < set(s['data_indices']):
                        for q in set(s['data_indices']) - r_sup:
                            if q in bases:
                                bases[q] = r['type']
        return bases

    def move_corners(self, name: str, new_patch: QECPatch, offset: Tuple[float, float] = (0, 0)):
        """Move the corners of a LIVE patch (Litinski Fig 40b; Fig 45 step 5→6) —
        the second atom of a patch rotation.

        ``new_patch`` is a reference geometry with the SAME data-qubit footprint but a
        different boundary stabilizer set (e.g. one long edge flipped X↔Z, which slides
        the same-type corner wrap from one end of the bar to the other).  The bulk and
        every unchanged boundary check keep their uid — signature-based dedup — so
        their detector history continues; the retiring boundary checks deactivate and
        the replacement checks are measured fresh (their first outcomes gauge-fix
        against the retired anticommuting lobes, so they get no first-round detector).
        No data qubits are added or removed and nothing is initialised; the caller
        just runs **d** SE rounds afterwards (diagonal schedule — the deformed
        logical representatives bend).

        Returns ``(new_data, new_syn)`` from the underlying re-registration;
        ``new_data`` is asserted empty (``new_syn`` may contain the fresh boundary
        ancillas).  Retiring-check bookkeeping lands in ``_grow_repairs[name]``,
        overwriting the previous deformation's entry.
        """
        retired = (self._tracker.retired_qubits if self._tracker is not None
                   else frozenset())
        old_data_coords = {self.qubit_coords[i]
                           for i, o in self.index_to_owner_map.items()
                           if o == name and i in self.data_indices
                           and i not in retired}
        shifted = copy.deepcopy(new_patch)
        shifted.shift_coords(offset[0], offset[1])
        new_data_coords = {shifted.qubit_coords[i] for i in shifted.data_indices}
        if new_data_coords != old_data_coords:
            raise ValueError(
                f"move_corners({name!r}): data footprint must be unchanged "
                f"(+{len(new_data_coords - old_data_coords)}/"
                f"-{len(old_data_coords - new_data_coords)} qubits); "
                f"use grow_patch/shrink_patch for footprint changes")
        new_data, new_syn = self.grow_patch(name, new_patch, offset)
        assert not new_data
        return new_data, new_syn

    def shrink_patch(self, name: str, new_patch: QECPatch,
                     offset: Tuple[float, float] = (0, 0)):
        """Shrink a LIVE patch onto a SMALLER footprint (Fig 40c's shortening;
        Fig 45 step 6→7) — the third atom of a patch rotation.

        Call AFTER destructively measuring the removed region through the builder::

            removed = [data qubits leaving the patch]
            builder.apply_data_readout(final_measurements={q: 'X' for q in removed})
            system.shrink_patch(name, shrunk_geometry, offset)

        The measurement basis is X when the patch was grown/extended through a Z
        boundary (Fig 40c: "measuring the left two thirds of physical qubits in the
        X basis"; K&F reference implementation: ``measurement="x"``).

        Mechanics: checks fully inside the surviving footprint keep their uid; checks
        touching the removed region deactivate; a cut-line check reappears TRUNCATED
        (same type, support = old ∩ surviving, new uid) and its value continuity is
        automatic — ``tracker.retire_qubits`` folds the readout records into the
        truncated row, so its first post-shrink measurement is deterministic (a
        bridging detector).  The detection-coverage criterion (handoff §3.2) is
        asserted: every surviving data qubit keeps both-type coverage via kept-uid
        checks or truncated successors; a violation means a silent handoff window and
        raises instead of building a distance-broken circuit.

        The removed data qubits become dormant (reusable cells); their indices stay
        allocated (masked) so global indexing is stable.  Returns them sorted.
        """
        if name not in self.patches:
            raise ValueError(f"patch {name!r} is not in the system")
        old_idxs = {i for i, o in self.index_to_owner_map.items() if o == name}
        # retired data qubits (dormant cells from an earlier shrink/absorb) are
        # not part of the live footprint: they must not enter removed/surviving
        # or the coverage criterion below would demand checks on dead qubits
        retired = (self._tracker.retired_qubits if self._tracker is not None
                   else frozenset())
        old_data = {i for i in old_idxs
                    if i in self.data_indices and i not in retired}
        if not old_data:
            raise ValueError(f"patch {name!r} has no data qubits in the system")
        shrunk = copy.deepcopy(new_patch)
        shrunk.shift_coords(offset[0], offset[1])
        new_coords = {shrunk.qubit_coords[i] for i in shrunk.data_indices}
        old_coords = {self.qubit_coords[i] for i in old_data}
        extra = new_coords - old_coords
        if extra:
            raise ValueError(
                f"shrink_patch({name!r}): the new geometry adds {len(extra)} data "
                f"qubits (e.g. {sorted(extra)[:3]}); use grow_patch to enlarge")
        removed = {i for i in old_data if self.qubit_coords[i] not in new_coords}
        if not removed:
            raise ValueError(
                f"shrink_patch({name!r}): nothing to remove — use move_corners for "
                f"same-footprint boundary changes")

        # snapshot pre-shrink active checks for the coverage criterion below
        old_active = {u for u in self.active_stabilizer_indices
                      if self.stabilizers[u].get('patch_name') == name}
        old_checks = [(self.stabilizers[u]['type'],
                       frozenset(self.stabilizers[u]['data_indices']))
                      for u in old_active]

        new_data, new_syn = self._apply_patch_geometry(name, shrunk, offset)
        assert not new_data, "shrink_patch must not add data qubits"

        # ---- detection-coverage handoff criterion (handoff §3.2) --------------------
        surviving = old_data - removed
        safe_checks = []
        for u in self.active_stabilizer_indices:
            s = self.stabilizers[u]
            if s.get('patch_name') != name:
                continue
            sup = frozenset(s['data_indices'])
            kept = u in old_active
            truncated = any(t == s['type'] and sup == (osup & surviving)
                            for t, osup in old_checks)
            if kept or truncated:
                safe_checks.append((s['type'], sup))
        holes = []
        for q in sorted(surviving):
            for etype, catcher in (('X', 'Z'), ('Z', 'X')):
                if not any(t == catcher and q in sup for t, sup in safe_checks):
                    holes.append((q, etype))
        if holes:
            raise ValueError(
                f"shrink_patch({name!r}): detection-coverage handoff broken at "
                f"{holes[:4]}{'…' if len(holes) > 4 else ''} — a window error there "
                f"would be silently absorbed (handoff §3.2 criterion)")

        # ---- tracker: retire the measured-out qubits (folds readout records into
        # truncated stabilizer rows and logical rows), free the cells ---------------
        if self._tracker is not None:
            self._tracker.retire_qubits(removed)
            self.num_logicals = self._tracker.expected_num_logicals
        self.active_qubit_indices.difference_update(removed)
        return sorted(removed)

    def repair_grow_logicals(self, name: str, tracker: Optional[Any] = None):
        """Re-anchor the tracker's logical rows after :meth:`grow_patch`.  Call ONCE,
        after the first post-grow syndrome-extraction call and before any readout.

        Growing retires the growth-edge boundary checks (their plaquettes gain the new
        data qubits and become bulk).  A data qubit whose same-type coverage was ONLY
        those retired checks is ORPHANED across the handoff: the enlarged check that
        takes over starts with a RANDOM first outcome (fresh |+>/|0> neighbours), so no
        detector spans the last-old-round → first-new-round window there.  Any logical
        row whose data support rests on an orphaned qubit is flippable by a SINGLE
        error in that window (measured: graphlike distance 1 for Z memory across a
        Z-boundary grow, from the retired lobe's outer corner qubit).

        The repair multiplies each affected logical row by grown-code checks (GF(2)
        solve) so its support avoids every orphaned qubit, and installs the result via
        ``tracker.logical_canonicalization``, which folds the correct records — the
        straddling new check's own banked record replaces the orphaned data legs, and
        that record IS detector-protected from its second round on.
        """
        from lightstim.utils.linear_algebra import solve_linear_decomposition

        tracker = tracker if tracker is not None else self._tracker
        if tracker is None:
            raise ValueError(
                "repair_grow_logicals needs a tracker (register_tracker or pass one)")
        info = self._grow_repairs.get(name)
        if not info or not info['orphans']:
            return
        orphans = sorted(info['orphans'])
        n = tracker.num_qubits

        def vec_of(stab):
            v = np.zeros(2 * n, dtype=np.uint8)
            for q, p in stab['pauli'].items():
                v[q if p == 'X' else n + q] = 1
            return v

        check_rows = [vec_of(self.stabilizers[uid])
                      for uid in sorted(self.active_stabilizer_indices)
                      if self.stabilizers[uid].get('patch_name') == name]
        if not check_rows:
            return
        C = np.array(check_rows, dtype=np.uint8)
        ocols = [q for q in orphans] + [n + q for q in orphans]

        for r in range(tracker.logicals.count):
            row = tracker.logicals.matrix[r]
            if not row[ocols].any():
                continue
            coeffs, dep, _ = solve_linear_decomposition(
                basis=C[:, ocols],
                targets=row[ocols].reshape(1, -1),
                reduce_weight=False,
            )
            if not dep[0]:
                raise ValueError(
                    f"repair_grow_logicals({name!r}): logical row {r} cannot be moved "
                    f"off orphaned qubits {orphans} with the grown checks — no "
                    f"fault-tolerant representative exists in the grown code")
            target = ((row + coeffs[0].astype(np.uint8) @ C) % 2).astype(np.uint8)
            tracker.logical_canonicalization({r: target})

    def rotate_patch(self, name: str, clockwise: bool = True):
        """Rotate a LIVE patch 90° in place, about its own bounding-box centre.

        Only COORDINATE metadata moves: the patch's qubit coords in the global
        maps (``qubit_coords`` / ``index_map`` / ``grid_map`` / owner map) are
        rotated and snapped back onto the integer lattice, and each of the
        patch's stabilizer ``syn_coord`` is refreshed to its qubit's new coord.
        The index-keyed stabilizer/logical records are left untouched — they
        follow the qubits — so the index-based ``SyndromeTracker`` is unaffected.

        A d×d rotated surface-code patch's full qubit footprint is invariant
        under a 90° turn about its centre (data + syndrome coords map onto the
        same set), so no coord collides with another patch; only the
        coord↔index assignment inside this patch permutes.  The resulting
        checkerboard is byte-identical to the conjugate-convention construction
        the routed coupler hosts natively (rule-0 lock) — see
        ``tests/test_patch_rotation.py`` — so a rotated patch merges at full
        distance when its next coupler is registered with the flipped
        orientation."""
        idxs = [i for i, o in self.index_to_owner_map.items() if o == name]
        if not idxs:
            raise ValueError(f"patch {name!r} has no qubits in the system")
        xs = [self.qubit_coords[i][0] for i in idxs]
        ys = [self.qubit_coords[i][1] for i in idxs]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        theta = -np.pi / 2 if clockwise else np.pi / 2
        old = {i: self.qubit_coords[i] for i in idxs}
        # drop the pre-rotation coord keys for this patch (footprint is invariant,
        # so the rotated coords re-fill exactly this set — no leftover, no clash)
        for i, c in old.items():
            if self.index_map.get(c) == i:
                del self.index_map[c]
            gk = QECPatch.get_grid_key(c)
            if self.grid_map.get(gk) == i:
                del self.grid_map[gk]
            if self.coord_to_owner_map.get(c) == name:
                del self.coord_to_owner_map[c]
        for i, c in old.items():
            nc = QECPatch.snap_coord(QECPatch._rotate_transform(c, theta, (cx, cy)))
            self.qubit_coords[i] = nc
            self.index_map[nc] = i
            self.grid_map[QECPatch.get_grid_key(nc)] = i
            self.coord_to_owner_map[nc] = name
        for stab in self.stabilizers:
            if stab.get('patch_name') == name:
                si = stab.get('syn_idx')
                if si is not None and si in self.qubit_coords:
                    stab['syn_coord'] = self.qubit_coords[si]

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

    def retire_measured_patch(self, name):
        """After a patch's data qubits have been destructively measured out, free
        it: deactivate its stabilizers and retire its qubits from the tracker
        (clean shrink), then re-sync the logical budget to the standing logical
        count. Qubit indices stay allocated (masked) so global indexing is stable."""
        # NOTE: unlike remove_coupler, we intentionally do NOT prune data_indices /
        # index_to_owner_map / qubit_coords here — the retired patch's coords linger
        # harmlessly (SE only measures ACTIVE stabilizers) and the end-readout + noise
        # injection still resolve them. Masking (not deleting) keeps global indices stable.
        dq = {q for q in self.data_indices if self.index_to_owner_map.get(q) == name}
        if not dq:
            return
        dead = {uid for uid in list(self.active_stabilizer_indices)
                if self.stabilizers[uid].get('patch_name') == name}
        self.active_stabilizer_indices.difference_update(dead)
        if self._tracker is not None:
            self._tracker.retire_qubits(dq)
            # NOTE: the budget (expected_num_logicals) is NOT touched here. The patch's
            # destructive readout — apply_data_readout, run just before this call —
            # already decremented `expected` by exactly the number of live logical DOFs
            # it resolved (free ones consumed and/or absorbed relations read out). We
            # deliberately do NOT re-sync expected := standing (+ absorbed): that would
            # slave the independent budget to the tableau matrices and silently erase any
            # prior discrepancy the guardrail is meant to catch.
            # Keep the system's STATIC logical count mirroring the tracker's dynamic
            # budget so a LATER define-by-run registration (add_patch) grows from the
            # right base instead of a stale global count.
            self.num_logicals = self._tracker.expected_num_logicals
        # Make the retired data qubits DORMANT so their coarse cells can be reused as a
        # later coupler's corridor: add_patch reuses an existing coord's index only when
        # that index is NOT in active_qubit_indices (else it raises an ACTIVE collision).
        self.active_qubit_indices.difference_update(dq)

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
