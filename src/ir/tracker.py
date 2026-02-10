import stim
import numpy as np
from ..utils.linear_algebra import check_commutativity, solve_linear_decomposition
from .tableau import PauliTableau
from typing import List, Dict, Tuple

class SyndromeTracker:
    def __init__(self, num_qubits: int, expected_num_logicals: int = 0):
        # Number of physical qubits (include data, syndrome, and potentially other ancilla qubits)
        self.num_qubits = num_qubits
        # Number of logical qubits (should be the total number of logical qubits in the QEC system)
        self.expected_num_logicals = expected_num_logicals
        # Total number of measurements
        self.total_measurements = 0
        self.meas_rec_to_idx_map = {}
        
        # Track the current stabilizers and logicals of the system 
        # Note: Technically, logicals are also stabilizers of the system, define the logical states
        self.stabilizers = PauliTableau(num_qubits)
        self.logicals = PauliTableau(num_qubits)
        self.stabilizer_with_logical_components = set() # Row indices of stabilizers that contain logical components

    def set_expected_logicals(self, k: int):
        """
        Call this after the logical count is adjusted  
        (e.g. some logicals are fixed into stabilizers)
        """
        self.expected_num_logicals = k
    
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
        # M = [ X->X  X->Z ]
        #     [ Z->X  Z->Z ]
        #
        # So:
        # Top-Left: x2x
        # Top-Right: x2z
        # Bottom-Left: z2x
        # Bottom-Right: z2z
        
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
            self.stabilizers._rebuild_map()
        
        # 5. Perform Symplectic Update for Logicals
        if self.logicals.count > 0:
            # Using matmul (@) and modulo 2
            self.logicals.matrix = (self.logicals.matrix @ symplectic_matrix) % 2
            self.logicals.matrix = self.logicals.matrix.astype(np.uint8)
            self.logicals._rebuild_map()


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
        
        for i in range(num_meas):
            meas_pauli = back_propagated_paulis[i]
            meas_row_view = meas_pauli.reshape(1, -1)
            meas_abs_idx = current_base_idx + i
            
            # Check commutativity against existing stabilizers and logicals
            comm_check = check_commutativity(meas_row_view, full_matrix)
            anti_comm_indices = np.where(comm_check[0])[0]
            
            if len(anti_comm_indices) > 0:
                # --- Case A: Anti-commutes (State Update) ---
                pivot = anti_comm_indices[0]
                
                # Update other anti-commuting rows
                for other in anti_comm_indices[1:]:
                    # Row[other] ^= Row[pivot]
                    full_matrix[other] ^= full_matrix[pivot]
                    full_records[other].extend(full_records[pivot])
                
                # Replace the pivot with the back_propagated_paulis
                # They may replace an original stabilizer or logical (the pivot), which will be determined in Step 3. 
                full_matrix[pivot] = meas_pauli
                full_records[pivot] = [meas_abs_idx]

                if pivot >= num_stabs:
                    # The pivot is a logical operator, decreases one degree of freedom
                    self.expected_num_logicals -= 1
                
            else:
                # --- Case B: Commutes (Detector) ---
                # Detector is formed by decomposing Back-propagated Pauli Measurements into existing STABILIZERS only.
                # (Logicals do not contribute to the decomposition)
                
                # Extract current stabilizers from full_matrix
                if num_stabs > 0:
                    # First check if meas_row_view is exactly one row in curr_stab_matrix
                    # Directly compare meas_row_view against current stabilizer rows
                    curr_stab_matrix = full_matrix[:num_stabs]
                    matching_rows = np.where(np.all(curr_stab_matrix == meas_row_view, axis=1))[0]
                    if len(matching_rows) > 0:
                        # Directly construct the detector
                        row_idx = matching_rows[0]  # Take the first matching row
                        args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                        for r in full_records[row_idx]:
                            args.append(stim.target_rec(r - self.total_measurements))
                        circuit.append("DETECTOR", args, list(syn_coords[i]) + [0])
                    else:
                        # meas_row_view is decomposed into existing stabilizers
                        coeffs, is_dependent, _ = solve_linear_decomposition(
                            basis=full_matrix, # full matrix! the back-propagated Pauli may contain present logical operator components
                            targets=meas_row_view
                        )
                        
                        if is_dependent[0]: # Construct a detector
                            args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                            comp_indices = np.where(coeffs[0])[0]
                            if max(comp_indices) >= num_stabs:
                                # The measurement contains a logical component, and cannot be a detector
                                # Flag this row for further logical observable construction
                                self.stabilizer_with_logical_components.add(i)
                                continue
                            # Otherwise, purely depends on stabilizers, construct a detector
                            for c_idx in comp_indices:
                                # Map back to full records
                                for r in full_records[c_idx]:
                                    # Clean the format: if there are the same target, remove the duplicates
                                    rec_to_append = stim.target_rec(r - self.total_measurements)
                                    if rec_to_append in args:
                                        args.remove(rec_to_append)
                                    else:
                                        args.append(rec_to_append) # this logic is essentially the addition modulo 2
            
                            circuit.append("DETECTOR", args, list(syn_coords[i]) + [0])
                        else:
                            # New stabilizer that's missing in the current stabilizer tableau
                            # This should never happen in a well-defined stabilizer tableau
                            # raise RuntimeError(
                            #     f"Measurement {i} commutes with all current stabilizers but is linearly independent.\n"
                            #     f"This implies the Full Stabilizer + Logicals Tableau is incomplete (Rank < num_qubits).\n"
                            #     f"Please ensure all qubits are initialized and added to the tracker before measurement."
                            # )
                            pass

        # ======================================================================
        # Step 3: Write Back with "Clean" Basis
        # ======================================================================
        # User's Insight: Regardless of the state, we always decompose the system 
        # into the "Clean Basis" of the measurements we just performed.
        # - Dependent rows in Full Tableau -> Replaced by Clean Measurements (Stabilizers).
        # - Independent rows in Full Tableau -> Identified as Logicals.
        # This automatically determines the right logicals, e.g. after first round of syndrome extraction

        # Basis: The clean measurements (Back-propagated Paulis)
        # Targets: The updated Full Tableau (System State)
        # new_basis_indices: Indices in full_matrix that form the Logical Basis.
        _, _, new_basis_indices = solve_linear_decomposition(
            basis=back_propagated_paulis, 
            targets=full_matrix
        )
        
        # 1. Update Logicals
        # We extract the pivot independent rows identified by RREF. There are two possibilities:
        # Case 1: The logical operators with no measurement records.
        # Case 2: The stabilizers with measurement records that stay in the system, not being measured this round.
        if len(new_basis_indices) > 0:
            # self.logicals.matrix = full_matrix[new_basis_indices]
            # self.logicals.records = [full_records[i] for i in new_basis_indices]
            # self.logicals._rebuild_map()
            new_log_basis_indices = []
            old_stab_basis_indices = []
            for new_idx in new_basis_indices:
                if new_idx < num_stabs: 
                    if full_records[new_idx] == []: # new logical operators, usually appear in the first round SE after initialization
                        new_log_basis_indices.append(new_idx)
                    else: # old stabilizers with measurement records that stay in the system, not being measured this round
                        old_stab_basis_indices.append(new_idx)
                else:
                    new_log_basis_indices.append(new_idx) # Original logical operator rows stay in the logical tableau

            self.logicals.matrix = full_matrix[new_log_basis_indices]
            self.logicals.records = [full_records[i] for i in new_log_basis_indices]
            self.logicals._rebuild_map()
        else:
            self.logicals = PauliTableau(self.num_qubits) # empty logicals

        # 2. Update Stabilizers
        # Reset stabilizer tableau to the canonical measurement basis.
        # This keeps the tableau sparse and prevents "messy" linear combinations.
        new_stab_records = [[self.total_measurements - num_meas + i] for i in range(num_meas)]

        # self.stabilizers.matrix = back_propagated_paulis
        # self.stabilizers.records = new_stab_records
        # self.stabilizers._rebuild_map()
        old_stab_matrix = full_matrix[old_stab_basis_indices]
        old_stab_records = [full_records[i] for i in old_stab_basis_indices]
        self.stabilizers.matrix = np.vstack([back_propagated_paulis, old_stab_matrix])
        self.stabilizers.records = new_stab_records + old_stab_records
        self.stabilizers._rebuild_map()
        
        # 3. Final Sanity Check (The Guardrail)
        if (len(self.stabilizer_with_logical_components) + self.logicals.count) != self.expected_num_logicals:
            """ Probably need to replace this condition with rank check"""
            raise RuntimeError(
                 f"[Error] Logical Count Mismatch!\n"
                 f"Expected: {self.expected_num_logicals}, \n"
                 f"Found: (1) Logicals {self.logicals.count}, \n"
                 f"(2) Stabilizers with Logical Components {len(self.stabilizer_with_logical_components)}\n"
                 f"This implies the measurements defined a subspace with incorrect dimensions.\n"
                #  f"- If Found > Expected: System Underspecified (Missing measurements?).\n"
                #  f"- If Found < Expected: System Overspecified (Measured a Logical?)."
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
            syndrome_coords: List of coordinates corresponding to each row in self.stabilizers.
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
        
        # We MUST track which rows represent stabilizers that are destroyed by the measurement.
        # A destroyed stabilizer cannot form a deterministic detector.
        destroyed_rows = set()
        
        for i in range(num_new_meas):
            meas_pauli = final_paulis[i]
            meas_row_view = meas_pauli.reshape(1, -1)
            meas_abs_idx = base_meas_idx + i
            
            comm_check = check_commutativity(meas_row_view, full_matrix)
            anti_comm_indices = np.where(comm_check[0])[0]
            
            if len(anti_comm_indices) > 0:
                pivot = anti_comm_indices[0]
                destroyed_rows.add(pivot) # Mark pivot as destroyed

                for other in anti_comm_indices[1:]:
                    full_matrix[other] ^= full_matrix[pivot]
                    full_records[other].extend(full_records[pivot])
                
                # Replace pivot to maintain valid tableau for subsequent loop steps
                full_matrix[pivot] = meas_pauli
                full_records[pivot] = [meas_abs_idx]

        # ======================================================================
        # Step 3: Decomposition, Detectors/Logical Observables Construction
        # ======================================================================
        # Basis: The Final Measurements we just performed.
        # Targets: The Updated System State (Stabilizers + Logicals).
        
        coeffs, is_dependent, _ = solve_linear_decomposition(
            basis=final_paulis, 
            targets=full_matrix
        )
        
        num_rows = full_matrix.shape[0]
        
        log_idx = 0
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
            # coeffs[k] tells us which Final Measurements (Basis) sum up to this Operator.
            # basis_indices contains indices 0..M-1
            basis_indices = np.where(coeffs[k])[0]
            
            for b_idx in basis_indices:
                # Map relative basis index to absolute record index
                # The i-th final measurement has record target: (base + i) - total
                # = i - num_new_meas
                stim_rec_target = b_idx - num_new_meas
                args.append(stim.target_rec(stim_rec_target)) # data qubit component
            
            # 2. Historical Record Components
            for r in full_records[k]: # syndrome qubit record components
                rec_to_append = stim.target_rec(r - self.total_measurements)
                if rec_to_append in args:
                    args.remove(rec_to_append)
                else:
                    args.append(rec_to_append) # this logic is essentially the addition modulo 2
                det_coord = idx_to_coord_map[self.meas_rec_to_idx_map[r]] # Set the detector coordinate to be the last syndrome qubit coordinate involved
            
            # 3. Output:
            if k < num_stabs and k not in self.stabilizer_with_logical_components:
                # Stabilizer -> DETECTOR
                circuit.append("DETECTOR", args, list(det_coord) + [1])
            else:
                # Logical -> OBSERVABLE
                circuit.append("OBSERVABLE_INCLUDE", args, [log_idx])
                log_idx += 1