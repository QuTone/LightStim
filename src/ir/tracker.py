import stim
import warnings
import numpy as np
from ..utils.linear_algebra import check_commutativity, solve_linear_decomposition
from ..utils.tableau_utils import stabilizers_to_symplectic
from .tableau import PauliTableau
from typing import List, Dict, Tuple, Optional, Set, Any

# Tag for post-selection: detectors with this tag are used for post-selection filtering
POST_SELECT_TAG = "post-select"

# Sentinel for unmeasured stabilizer rows: treated as has_record (stays stabilizer), excluded from detector construction
UNMEASURED_STAB_RECORD = -1


def _append_detector(
    circuit: stim.Circuit,
    args: list,
    coords: list,
    post_select: bool = False,
) -> None:
    """Append a DETECTOR instruction, optionally with post-select tag."""
    if post_select:
        try:
            circuit.append("DETECTOR", args, coords, tag=POST_SELECT_TAG)
        except TypeError:
            circuit.append("DETECTOR", args, coords)
    else:
        circuit.append("DETECTOR", args, coords)


class SyndromeTracker:
    def __init__(
        self,
        num_qubits: int,
        expected_num_logicals: int = 0,
        post_select_detector_coords: Optional[Set[Tuple[float, ...]]] = None,
    ):
        # Number of physical qubits (include data, syndrome, and potentially other ancilla qubits)
        self.num_qubits = num_qubits
        # Number of logical qubits (should be the total number of logical qubits in the QEC system)
        self.expected_num_logicals = expected_num_logicals
        # Total number of measurements
        self.total_measurements = 0
        self.meas_rec_to_idx_map = {}
        
        # Track the current stabilizers and logicals of the system 
        # Note 1: Technically, logicals are also stabilizers of the system, define the logical states
        # Note 2: Stabilizer tableau allows linear dependencies between rows (e.g. toric code, BB code), but logicals do not in general..
        self.stabilizers = PauliTableau(num_qubits)
        self.logicals = PauliTableau(num_qubits)
        self.stabilizer_with_logical_components = set()  # Row indices of stabilizers that contain logical components
        self._gauge_logical_vectors = []  # GF(2) vectors over logical indices for rank computation
        self.post_select_detector_coords = post_select_detector_coords or set()
        self.post_select_row_indices = set()  # Stabilizer row indices to post-select in process_final_measurement

    def set_expected_logicals(self, k: int):
        """
        Call this after the logical count is adjusted  
        (e.g. some logicals are fixed into stabilizers)
        """
        self.expected_num_logicals = k

    def expand(self, delta: int):
        """
        Expand the tracker to include delta new qubits (define-by-run).
        New qubits act as identity on existing stabilizers/logicals.
        """
        if delta <= 0:
            return
        self.stabilizers.expand(delta)
        self.logicals.expand(delta)
        self.num_qubits += delta

    def reset_records_for_qubits(self, qubit_indices):
        """
        Clean up tracker state for qubits being re-initialized mid-circuit.

        Three cases for each stabilizer row:
        - Support ONLY on target qubits → remove (re-init adds fresh rows)
        - Support on BOTH target and non-target → keep but clear records
          (prevents detectors from comparing across state-change boundary)
        - No target support → keep as-is

        Same logic applied to logical rows.
        """
        n = self.num_qubits
        qubit_set = set(qubit_indices)

        def _clean_rows(tableau):
            new_indices = []
            new_records = []
            for i in range(tableau.count):
                row = tableau.matrix[i]
                has_target = any(row[q] != 0 or row[q + n] != 0 for q in qubit_set)
                if not has_target:
                    # No target support → keep as-is
                    new_indices.append(i)
                    new_records.append(tableau.records[i])
                elif has_target:
                    # Check if support extends beyond target qubits
                    has_non_target = False
                    for q in range(n):
                        if q not in qubit_set and (row[q] != 0 or row[q + n] != 0):
                            has_non_target = True
                            break
                    if has_non_target:
                        # Mixed support → keep but:
                        # 1. Mark as unmeasured (prevents stale detector comparison)
                        # 2. Zero out target qubit columns (partial trace over
                        #    orphaned qubits — they're never measured again)
                        new_indices.append(i)
                        new_records.append([UNMEASURED_STAB_RECORD])
                        for q in qubit_set:
                            tableau.matrix[i, q] = 0
                            tableau.matrix[i, q + n] = 0
                    # else: support ONLY on target → remove (skip)

            if new_indices:
                tableau.matrix = tableau.matrix[new_indices]
            else:
                tableau.matrix = np.zeros((0, 2 * n), dtype=np.uint8)
            tableau.records = new_records

        _clean_rows(self.stabilizers)
        _clean_rows(self.logicals)

    def stabilizer_canonicalization(
        self,
        system: Any,
        stabilizer_uids: Optional[Set[int]] = None,
    ) -> None:
        """
        Re-organize stabilizer tableau into stabilizers vs logicals BEFORE any SE measurement.
        Basis = canonical stabilizer set (active_stabilizers or stabilizer_uids).
        Uses new_basis_indices (Logical Basis) to extract minimal logical dimension, not is_dependent.
        Aligned with process_mid_measurement Step 3.

        Call after encoding, before SE. Raises if logical count does not match expected.
        """
        n = self.num_qubits
        if stabilizer_uids is not None:
            stab_dicts = [system.stabilizers[i] for i in range(len(system.stabilizers)) if i in stabilizer_uids]
        else:
            stab_dicts = [system.stabilizers[i] for i in sorted(system.active_stabilizer_indices)]
        canonical_basis = stabilizers_to_symplectic(system, stab_dicts, n)

        if canonical_basis.shape[0] == 0:
            return

        num_stabs = self.stabilizers.count
        if num_stabs == 0:
            return

        # Full tableau = stabilizers + logicals (same structure as process_mid_measurement)
        existing_log_matrix = self.logicals.matrix
        existing_log_records = self.logicals.records
        if existing_log_matrix.shape[0] > 0:
            full_matrix = np.vstack([self.stabilizers.matrix, existing_log_matrix])
            full_records = self.stabilizers.records + existing_log_records
        else:
            full_matrix = self.stabilizers.matrix
            full_records = self.stabilizers.records

        _, _, new_basis_indices = solve_linear_decomposition(
            basis=canonical_basis,
            targets=full_matrix,
            reduce_weight=True,
        )

        # new_basis_indices = Logical Basis (pivot columns). Rows not in it -> stabilizers.
        old_stab_indices = [i for i in range(num_stabs) if i not in new_basis_indices]
        new_log_basis_indices = list(new_basis_indices)

        new_stab_matrix = full_matrix[old_stab_indices]
        new_stab_records = [
            full_records[i] if full_records[i] else [UNMEASURED_STAB_RECORD]
            for i in old_stab_indices
        ]
        new_log_matrix = full_matrix[new_log_basis_indices]
        new_log_records = [full_records[i] for i in new_log_basis_indices]

        self.stabilizers.matrix = new_stab_matrix
        self.stabilizers.records = new_stab_records
        self.logicals.matrix = new_log_matrix
        self.logicals.records = new_log_records

        if self.logicals.count != self.expected_num_logicals:
            raise RuntimeError(
                f"After stabilizer_canonicalization: logical count {self.logicals.count} "
                f"!= expected {self.expected_num_logicals}. "
                "Unitary encoding circuit may be incorrect."
            )
    
    def process_initialization(self, init_tableau: np.ndarray):
        """
        Registers new stabilizers from initialization into the tracker.
        
        For t=0 (System Start): Populates the empty tableau.
        For New Patch: Appends new independent stabilizers to the existing set.
        
        Args:
            init_tableau: Shape (k, 2n). 
        """
        self.stabilizers.add_stabilizers(init_tableau)


    def process_unitary_block(self, circuit_chunk: stim.Circuit):
        """
        Evolves the internal stabilizer/logical tableau by applying a unitary circuit chunk.
        Update Rule: S_new = S_old @ Symplectic_Matrix
        
        Args:
            circuit_chunk: A Stim circuit containing only unitary operations (no measurements/resets).
        """
        # 1. Convert Circuit to Tableau (Forward Evolution)
        # We ignore noise/measurement to treat it as a pure Clifford unitary
        # Note: Unlike back-propagation, we do NOT invert the tableau here.
        # We want the forward evolution U.
        u_tableau = stim.Tableau.from_circuit(
            circuit_chunk, 
            ignore_noise=True, 
            ignore_measurement=True, 
            ignore_reset=True
        )
        
        # 2. Get Symplectic Components
        # Output shapes are (n_chunk, n_chunk)
        # x2x: X component of X generators' image
        # x2z: Z component of X generators' image
        # z2x: X component of Z generators' image
        # z2z: Z component of Z generators' image
        x2x, x2z, z2x, z2z, _, _ = u_tableau.to_numpy()
        
        # Convert to standard integer numpy arrays (uint8 is sufficient for mod 2)
        x2x = x2x.astype(np.uint8)
        x2z = x2z.astype(np.uint8)
        z2x = z2x.astype(np.uint8)
        z2z = z2z.astype(np.uint8)
        
        # 3. Padding Logic (Alignment to System Size)
        n_chunk = len(u_tableau)
        n_sys = self.num_qubits
        
        if n_chunk > n_sys:
            raise ValueError(f"Circuit chunk involves qubit {n_chunk-1}, exceeding system size {n_sys}.")
        
        # We need to construct the full 2N x 2N symplectic matrix M.
        # The structure of M for right-multiplication (Row Vector @ M) is:
        # M = [ x2x  x2z ]
        #     [ z2x  z2z ]
        
        if n_chunk == n_sys:
            # No padding needed, construct directly
            top = np.hstack([x2x, x2z])      # (N, 2N)
            bottom = np.hstack([z2x, z2z])   # (N, 2N)
            symplectic_matrix = np.vstack([top, bottom]) # (2N, 2N)
            
        else:
            # Need padding. The operation is Identity on qubits [n_chunk, n_sys).
            # Identity Symplectic Matrix structure:
            # [ I  0 ]
            # [ 0  I ]
            
            # 3.1 Initialize full matrix as Identity
            full_M = np.eye(2 * n_sys, dtype=np.uint8)
            
            # 3.2 Fill the active region
            # Top-Left (X->X)
            full_M[:n_chunk, :n_chunk] = x2x
            # Top-Right (X->Z)
            full_M[:n_chunk, n_sys:n_sys+n_chunk] = x2z
            # Bottom-Left (Z->X)
            full_M[n_sys:n_sys+n_chunk, :n_chunk] = z2x
            # Bottom-Right (Z->Z)
            full_M[n_sys:n_sys+n_chunk, n_sys:n_sys+n_chunk] = z2z
            
            symplectic_matrix = full_M

        # 4. Perform Symplectic Update (Conjugation) for Stabilizers
        # self.stabilizers.matrix shape: (K_stabilizers, 2*n_sys)
        # symplectic_matrix shape: (2*n_sys, 2*n_sys)
        # Result: Each row P becomes P' = P @ M
        if self.stabilizers.count > 0:
            # Using matmul (@) and modulo 2
            self.stabilizers.matrix = (self.stabilizers.matrix @ symplectic_matrix) % 2
            self.stabilizers.matrix = self.stabilizers.matrix.astype(np.uint8)

        # 5. Perform Symplectic Update for Logicals
        if self.logicals.count > 0:
            # Using matmul (@) and modulo 2
            self.logicals.matrix = (self.logicals.matrix @ symplectic_matrix) % 2
            self.logicals.matrix = self.logicals.matrix.astype(np.uint8)

    def process_mid_measurement(self, 
                                circuit: stim.Circuit, 
                                back_propagated_paulis: np.ndarray, 
                                syn_coords: list):
        """
        Handles Mid-circuit measurements (assumed to be on syndrome qubits).
        """
        num_meas = back_propagated_paulis.shape[0]
        current_base_idx = self.total_measurements
        self.total_measurements += num_meas

        # Reset per-round tracking (these are only meaningful within a single PMM call)
        self.stabilizer_with_logical_components = set()
        self._gauge_logical_vectors = []
        
        # ======================================================================
        # Step 1: Combine Stabilizers and Logicals into Full Tableau
        # ======================================================================
        num_stabs = self.stabilizers.count 
        num_logs = self.logicals.count
        
        # If logicals is empty, full_tableau is just stabilizers
        if num_logs > 0:
            full_matrix = np.vstack([self.stabilizers.matrix, self.logicals.matrix])
            # Flatten records list
            full_records = self.stabilizers.records + self.logicals.records
        else:
            full_matrix = self.stabilizers.matrix.copy() # Copy to avoid reference issues during loop
            full_records = list(self.stabilizers.records) # Deep copy of list structure
        
        # ======================================================================
        # Step 2: Process Back-propagated Pauli measurements (Update / Detector)
        # ======================================================================
        # Use a temporary tableau view so we can call update_row / replace_row.
        temp_full = PauliTableau(self.num_qubits)
        temp_full.matrix = full_matrix
        temp_full.records = full_records

        for i in range(num_meas):
            meas_pauli = back_propagated_paulis[i]
            meas_row = meas_pauli.reshape(1, -1)
            meas_abs_idx = current_base_idx + i

            # Check commutativity against existing stabilizers and logicals
            comm_check = check_commutativity(meas_row, full_matrix)
            anti_comm_indices = np.where(comm_check[0])[0]

            if len(anti_comm_indices) > 0:
                # --- Case A: Anti-commutes (State Update) ---
                pivot = anti_comm_indices[0]
                # Update other anti-commuting rows
                for other in anti_comm_indices[1:]:
                    temp_full.update_row(other, pivot) # (target, source)

                # Replace the pivot with the back_propagated_paulis
                temp_full.replace_row(pivot, meas_pauli, [meas_abs_idx])

                if pivot >= num_stabs:
                    # If the pivot is a logical operator and is replaced by a measurement, decreases one degree of freedom
                    self.expected_num_logicals -= 1

            else:
                # --- Case B: Commutes (Detector) ---
                # Detector is formed by decomposing Back-propagated Pauli Measurements into existing STABILIZERS only (rows in the stabilizer tableau).
                # (Logicals do not contribute to the decomposition)

                if num_stabs > 0:
                    # First check if meas_row is exactly one row in curr_stab_matrix
                    # Directly compare meas_row against current stabilizer rows
                    curr_stab_matrix = full_matrix[:num_stabs]
                    matching_rows = np.where(np.all(curr_stab_matrix == meas_row, axis=1))[0]
                    if len(matching_rows) > 0:
                        # Raise warning if there are multiple identical stabilizer rows matching this measurement
                        if len(matching_rows) > 1:
                            warnings.warn(
                                f"Found {len(matching_rows)} identical stabilizer rows matching this measurement. "
                                "Check that the stabilizer tableau has no duplicate rows.",
                                UserWarning,
                                stacklevel=2,
                            )
                        # Directly construct the detector
                        row_idx = matching_rows[0]  # Take the first matching row
                        args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                        for r in full_records[row_idx]:
                            if r >= 0:
                                args.append(stim.target_rec(r - self.total_measurements))
                        coords = list(syn_coords[i]) + [0]
                        _append_detector(
                            circuit, args, coords,
                            post_select=tuple(coords) in self.post_select_detector_coords,
                        )
                    else: # meas_row is not exactly one row in curr_stab_matrix, but a linear combination of rows in the full matrix
                        # decompose meas_row into existing stabilizers
                        coeffs, is_dependent, _ = solve_linear_decomposition(
                            basis=full_matrix,
                            targets=meas_row
                        )
                        # Note: Here we use the full matrix as the basis, not just the stabilizer tableau.
                        # The back-propagated Pauli may contain present logical operator components. e.g., Logical ZZ over two |0> states, then
                        # the last Z gauge operator consisting ZZ measurements can be written as the linear combination of previous Z gauges and two logical Z operators of two patches.
                        # If we don't use the full matrix as the basis, these measurements will be identified as independent basis and treated as logicals, which is incorrect.
                        # However, these measurements, although they can be decomposed, cannot be detectors, because their logical operator components cannot be measured in the middle of the circuit
                        # and cannot give syndrome information. This will be identified when we construct detectors.

                        # Construct a detector
                        if is_dependent[0]:
                            args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                            comp_indices = np.where(coeffs[0])[0]
                            if max(comp_indices) >= num_stabs:
                                # The measurement contains a logical component, and cannot be a detector
                                # Flag this row for further logical observable construction
                                self.stabilizer_with_logical_components.add(i)
                                # Track which logical indices this measurement involves (for rank computation)
                                log_vec = np.zeros(num_logs, dtype=np.uint8)
                                for c in comp_indices:
                                    if c >= num_stabs:
                                        log_vec[c - num_stabs] = 1
                                self._gauge_logical_vectors.append(log_vec)
                                continue
                            # Otherwise, purely depends on stabilizers, construct a detector
                            # Use set-based XOR for O(1) toggle instead of O(n) list scan
                            args_set = set(args)
                            for c_idx in comp_indices:
                                # Map back to full records (skip UNMEASURED_STAB_RECORD)
                                for r in full_records[c_idx]:
                                    if r < 0:
                                        continue
                                    rec_to_append = stim.target_rec(r - self.total_measurements)
                                    if rec_to_append in args_set:
                                        args_set.remove(rec_to_append)
                                    else:
                                        args_set.add(rec_to_append)  # addition modulo 2
                            args = list(args_set)

                            coords = list(syn_coords[i]) + [0]
                            _append_detector(
                                circuit, args, coords,
                                post_select=tuple(coords) in self.post_select_detector_coords,
                            )
                        else:
                            # Measurement row commute but is independent of the current full tableau,
                            # but this should never happen in a well-defined full tableau, unless there are degrees of freedom missing.
                            raise RuntimeError(
                                f"Measurement {i} commutes with all rows in the current full tableau (stabilizers + logicals) but is linearly independent.\n"
                                f"This implies the Full (Stabilizer + Logicals) Tableau is incomplete (Rank < num_qubits).\n"
                                f"Please ensure all qubits are initialized and added to the tracker before measurement."
                            )

        # ======================================================================
        # Step 3: Write Back with "Clean" Basis
        # ======================================================================
        # After detector construction, we always decompose the system into the "Clean Basis" of the measurements we just performed.
        # - Dependent rows in Full Tableau -> Replaced by Clean Measurements (Stabilizers).
        # - Independent rows in Full Tableau -> Identified as Logicals.
        # This automatically determines the right logicals, e.g. after first round of syndrome extraction

        # Basis: The clean measurements (Back-propagated Paulis)
        # Targets: The updated Full Tableau (System State)
        # new_basis_indices: Indices in full_matrix that form the Logical Basis.
        #
        # IMPORTANT: Reorder targets so logical rows come BEFORE stabilizer rows.
        # This gives logicals priority for RREF pivots, preventing stabilizer rows
        # from "stealing" a logical's pivot when they share the same independent direction.
        # Without this, logicals can be lost (e.g., PQRM logical Z in CrossLS Z-state).
        if num_logs > 0:
            reordered_targets = np.vstack([full_matrix[num_stabs:], full_matrix[:num_stabs]])
            reordered_records = full_records[num_stabs:] + full_records[:num_stabs]
        else:
            reordered_targets = full_matrix
            reordered_records = full_records

        _, is_dependent, new_basis_indices = solve_linear_decomposition(
            basis=back_propagated_paulis,
            targets=reordered_targets
        )

        # Map reordered indices back to original full_matrix semantics:
        # reordered[0..num_logs-1]       = original logicals  (full_matrix[num_stabs..end])
        # reordered[num_logs..num_logs+num_stabs-1] = original stabs (full_matrix[0..num_stabs-1])
        def _remap(idx):
            if idx < num_logs:
                return num_stabs + idx          # logical row
            else:
                return idx - num_logs            # stabilizer row

        # 1. Update Logicals
        # Independent rows fall into:
        # (a) Logical rows: stay as logicals
        # (b) Stab rows with records: stay as old stabilizers (not measured this round)
        # (c) Stab rows without records: become logicals
        if len(new_basis_indices) > 0:
            new_log_basis_indices = []
            old_stab_basis_indices = []

            for ri in new_basis_indices:
                orig_idx = _remap(ri)
                orig_record = reordered_records[ri]
                if orig_idx >= num_stabs:
                    # Logical row → stays as logical
                    new_log_basis_indices.append(orig_idx)
                elif orig_record == []:
                    # Stab row without records → becomes logical
                    new_log_basis_indices.append(orig_idx)
                else:
                    # Stab row with records → stays as old stabilizer
                    old_stab_basis_indices.append(orig_idx)

            self.logicals.matrix = full_matrix[new_log_basis_indices]
            self.logicals.records = [full_records[i] for i in new_log_basis_indices]
        else:
            new_log_basis_indices = []
            old_stab_basis_indices = []
            self.logicals = PauliTableau(self.num_qubits)  # empty logicals

        # 2. Update Stabilizers
        # Reset stabilizer tableau to the canonical measurement basis.
        new_stab_records = [[self.total_measurements - num_meas + i] for i in range(num_meas)]

        # Build old_stab part: rows with records that stayed as stabilizers
        old_stab_matrix = full_matrix[old_stab_basis_indices]
        old_stab_records = [full_records[i] for i in old_stab_basis_indices]

        self.stabilizers.matrix = np.vstack([back_propagated_paulis, old_stab_matrix])
        self.stabilizers.records = new_stab_records + old_stab_records

        # Update post_select_row_indices: map old full_matrix indices to new stabilizer indices.
        # - Old stab rows in old_stab_basis_indices → new indices num_meas, num_meas+1, ...
        # - Dependent rows (absorbed into measurement basis) → find which measurement rows captured them
        if self.post_select_row_indices:
            new_ps = set()
            # Map old stab rows that survived
            for j, old_idx in enumerate(old_stab_basis_indices):
                if old_idx in self.post_select_row_indices:
                    new_ps.add(num_meas + j)
            # Map dependent rows: decompose against measurement basis to find which rows captured them
            # Map dependent rows: decompose against measurement basis to find which rows captured them
            # Note: is_dependent uses reordered indices; map post_select old_idx to reordered idx
            coeffs_ps, _, _ = solve_linear_decomposition(
                basis=back_propagated_paulis,
                targets=full_matrix,
                reduce_weight=False,
            )
            for old_idx in self.post_select_row_indices:
                if old_idx < full_matrix.shape[0]:
                    # Decompose against original full_matrix (not reordered)
                    row = full_matrix[old_idx:old_idx+1]
                    c, dep, _ = solve_linear_decomposition(basis=back_propagated_paulis, targets=row, reduce_weight=False)
                    if dep[0]:
                        meas_indices = np.where(c[0])[0]
                        new_ps.update(int(m) for m in meas_indices)
            self.post_select_row_indices = new_ps

        # 3. Final Sanity Check (The Guardrail)
        # Count the number of independent logical degrees of freedom consumed by gauge measurements.
        # A logical is "consumed" only if it appears in gauge vectors AND did NOT survive as a logical.
        # Logicals that survive are still alive — gauge measurements referencing them are just
        # measurements that CONTAIN a live logical component, not ones that absorbed it.
        if self._gauge_logical_vectors:
            # Determine which logical indices survived in the logicals tableau
            surviving_log_indices = set(
                i - num_stabs for i in new_log_basis_indices
                if i >= num_stabs
            ) if 'new_log_basis_indices' in dir() else set()
            # Zero out gauge vector entries for surviving logicals
            gauge_matrix = np.array(self._gauge_logical_vectors, dtype=np.uint8).copy()
            for j in surviving_log_indices:
                if j < gauge_matrix.shape[1]:
                    gauge_matrix[:, j] = 0
            num_absorbed = int(np.linalg.matrix_rank(gauge_matrix.astype(float)))
        else:
            num_absorbed = 0

        if (num_absorbed + self.logicals.count) != self.expected_num_logicals:
            raise RuntimeError(
                 f"[Error] Logical Count Mismatch!\n"
                 f"Expected: {self.expected_num_logicals}, \n"
                 f"Found: (1) Logicals {self.logicals.count}, \n"
                 f"(2) Absorbed Logical DOF (rank) {num_absorbed}\n"
                 f"(3) Measurements with Logical Components {len(self.stabilizer_with_logical_components)}\n"
             )


    def process_final_measurement(self, 
                                  circuit: stim.Circuit, 
                                  final_paulis: np.ndarray,
                                  idx_to_coord_map: Dict[int, Tuple[float, float]]):
        """
        Handles Final Data Qubit Measurements using Gaussian Elimination.
        
        Args:
            circuit: Stim circuit to append to.
            final_paulis: (M, 2N) numpy array. The measurement basis.
                          Does NOT need to be single-qubit Paulis (can be general).
            idx_to_coord_map: Mapping from qubit index to coordinate. Determine the coordinate of the detector in the decoding graph.
        """

        num_new_meas = final_paulis.shape[0]
        base_meas_idx = self.total_measurements
        self.total_measurements += num_new_meas
        
        num_stabs = self.stabilizers.count
        num_logs = self.logicals.count

        # ======================================================================
        # Step 1: Combine Full Tableau
        # ======================================================================
        if num_logs > 0:
            full_matrix = np.vstack([self.stabilizers.matrix, self.logicals.matrix])
            full_records = self.stabilizers.records + self.logicals.records
        else:
            full_matrix = self.stabilizers.matrix.copy()
            full_records = list(self.stabilizers.records)

        # ======================================================================
        # Step 2: Update Tableau (Resolve Anti-commutation)
        # ======================================================================
        # Use a temporary tableau view so we can call update_row / replace_row.
        temp_full = PauliTableau(self.num_qubits)
        temp_full.matrix = full_matrix
        temp_full.records = full_records

        # We MUST track which rows represent stabilizers that are destroyed by the measurement.
        # A destroyed stabilizer cannot form a deterministic detector.
        destroyed_rows = set()

        for i in range(num_new_meas):
            meas_pauli = final_paulis[i]
            meas_row = meas_pauli.reshape(1, -1)
            meas_abs_idx = base_meas_idx + i

            comm_check = check_commutativity(meas_row, full_matrix)
            anti_comm_indices = [j for j in np.where(comm_check[0])[0] if j not in destroyed_rows]

            if len(anti_comm_indices) > 0:
                # Prefer rows that are: (1) not logicals, (2) not swlc rows
                # This preserves logicals and gauge-measurement rows for observable construction.
                safe = [j for j in anti_comm_indices
                        if j < num_stabs and j not in self.stabilizer_with_logical_components]
                if safe:
                    pivot = safe[0]
                else:
                    stab_candidates = [j for j in anti_comm_indices if j < num_stabs]
                    pivot = stab_candidates[0] if stab_candidates else anti_comm_indices[0]
                destroyed_rows.add(pivot) # Mark pivot as destroyed

                for other in anti_comm_indices[1:]:
                    temp_full.update_row(other, pivot)

                # Replace pivot to maintain valid tableau for subsequent loop steps
                temp_full.replace_row(pivot, meas_pauli, [meas_abs_idx])

        # ======================================================================
        # Step 3: Decomposition, Detectors/Logical Observables Construction
        # ======================================================================
        # Basis: The Final Measurements we just performed.
        # Targets: The Updated System State (Stabilizers + Logicals).
        # reduce_weight=False: detector/observable construction only needs correct
        #   linear combination, not minimal-weight; _greedy_reduce_weight is O(k^2)
        #   and can dominate runtime for large codes (e.g. BB [[144,12,12]]).
        coeffs, is_dependent, _ = solve_linear_decomposition(
            basis=final_paulis,
            targets=full_matrix,
            reduce_weight=False,
        )

        num_rows = full_matrix.shape[0]
        logical_observable_idx = 0
        for k in range(num_rows):
            # Condition 1: Must NOT be destroyed (anti-commuted).
            if k in destroyed_rows:
                continue

            # Condition 2: Must be fully determined by the measurements (Linear Dependent).
            if not is_dependent[k]:
                continue

            # --- Construct Detector / Observable ---
            args = []

            # 1. Measurement Components (The decomposition result)
            basis_indices = np.where(coeffs[k])[0]
            for b_idx in basis_indices:
                stim_rec_target = b_idx - num_new_meas
                args.append(stim.target_rec(stim_rec_target))

            # 2. Historical Record Components — use set-based XOR for O(1) toggle
            det_coord = None
            args_set = set(args)
            for r in full_records[k]:
                if r < 0:
                    continue
                rec_to_append = stim.target_rec(r - self.total_measurements)
                if rec_to_append in args_set:
                    args_set.remove(rec_to_append)
                else:
                    args_set.add(rec_to_append)
                qubit_idx = self.meas_rec_to_idx_map.get(r)
                if qubit_idx is not None and qubit_idx in idx_to_coord_map:
                    # Only use this coord if the qubit is in this row's support
                    # (avoids inheriting coords from unrelated syndrome qubits via row updates)
                    row_k = full_matrix[k]
                    n_q = self.num_qubits
                    if row_k[qubit_idx] or row_k[n_q + qubit_idx]:
                        det_coord = idx_to_coord_map[qubit_idx]
            args = list(args_set)

            # Fallback when no syndrome records (e.g. X stabilizers with Z-only SE, final X measure)
            if det_coord is None:
                row = full_matrix[k]
                n = self.num_qubits
                first_support = next(
                    (i for i in range(n) if row[i] or row[n + i]),
                    None,
                )
                det_coord = (
                    idx_to_coord_map[first_support]
                    if first_support is not None and first_support in idx_to_coord_map
                    else next(iter(idx_to_coord_map.values()), (0, 0))
                )

            # 3. Output:
            if k < num_stabs and k not in self.stabilizer_with_logical_components:
                coords = list(det_coord) + [1]
                _append_detector(
                    circuit, args, coords,
                    post_select=(tuple(coords) in self.post_select_detector_coords
                                 or k in self.post_select_row_indices),
                )
            else:
                circuit.append("OBSERVABLE_INCLUDE", args, [logical_observable_idx])
                logical_observable_idx += 1

        # ======================================================================
        # Step 4: Partial reduction of remaining rows (incremental support)
        # ======================================================================
        # Rows that commute with measurements but share support on measured
        # qubits (e.g. Z_L = Z1*Z2*Z3 when Z2,Z3 are measured in Z basis)
        # must be reduced by XOR-ing out the measurement Paulis. This ensures
        # remaining rows only depend on unmeasured qubits, enabling correct
        # decomposition in subsequent process_final_measurement calls.
        emitted_rows = {k for k in range(num_rows)
                        if k not in destroyed_rows and is_dependent[k]}
        rows_to_remove = destroyed_rows | emitted_rows

        # Build pivot map: for each single-qubit measurement, identify its
        # pivot column (the nonzero column in the 2n symplectic vector).
        # MZ(q) -> pivot at n+q; MX(q) -> pivot at q; MY(q) -> pivot at q.
        n = self.num_qubits
        meas_pivot_map = {}  # pivot_col -> (measurement index i, meas Pauli vector)
        for i in range(num_new_meas):
            meas = final_paulis[i]
            nonzero_cols = np.where(meas)[0]
            if len(nonzero_cols) == 0:
                continue
            # Use the first nonzero column as pivot
            pivot_col = int(nonzero_cols[0])
            meas_pivot_map[pivot_col] = (i, meas)

        for k in range(num_rows):
            if k in rows_to_remove:
                continue
            row = full_matrix[k]
            for pivot_col, (i, meas) in meas_pivot_map.items():
                if row[pivot_col]:
                    # XOR measurement Pauli out of this row
                    full_matrix[k] = row = row ^ meas
                    # Update records: symmetric difference with measurement index
                    meas_abs_idx = base_meas_idx + i
                    rec_set = set(full_records[k])
                    rec_set.symmetric_difference_update({meas_abs_idx})
                    full_records[k] = list(rec_set)

        # ======================================================================
        # Step 5: Persist updated tableau state
        # ======================================================================
        remaining_stab_rows = [k for k in range(num_stabs) if k not in rows_to_remove]
        remaining_log_rows = [k for k in range(num_stabs, num_rows) if k not in rows_to_remove]

        if remaining_stab_rows:
            self.stabilizers.matrix = full_matrix[remaining_stab_rows]
            self.stabilizers.records = [full_records[k] for k in remaining_stab_rows]
        else:
            self.stabilizers.matrix = np.zeros((0, 2 * self.num_qubits), dtype=np.uint8)
            self.stabilizers.records = []

        if remaining_log_rows:
            self.logicals.matrix = full_matrix[remaining_log_rows]
            self.logicals.records = [full_records[k] for k in remaining_log_rows]
        else:
            self.logicals.matrix = np.zeros((0, 2 * self.num_qubits), dtype=np.uint8)
            self.logicals.records = []

        # Reset per-call tracking that uses row indices (now invalidated)
        self.stabilizer_with_logical_components = set()