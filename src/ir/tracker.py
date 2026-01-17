import stim
import numpy as np
from .utils import check_commutativity, solve_linear_decomposition
from .tableau import StabilizerTableau
from typing import List

class SyndromeTracker:
    def __init__(self, num_qubits: int, expected_num_logicals: int = 0):
        # Number of physical qubits (include data, syndrome, and potentially other ancilla qubits)
        self.num_qubits = num_qubits
        # Number of logical qubits (should be the total number of logical qubits in the QEC system)
        self.expected_num_logicals = expected_num_logicals
        # Total number of measurements
        self.total_measurements = 0
        
        # Track the current stabilizers and logicals of the system 
        # Note: Technically, logicals are also stabilizers of the system, define the logical states
        self.stabilizers = StabilizerTableau(num_qubits)
        self.logicals = StabilizerTableau(num_qubits)

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
        # Step 2: Process Back-propagated measurements (Update / Detector)
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
                
                # Replace the pivot with the measurement
                # Note: We keep the pivot in the Full Tableau. 
                # Whether it ends up in Stabilizers or Logicals depends on Step 3.
                full_matrix[pivot] = meas_pauli
                full_records[pivot] = [meas_abs_idx]
            
            else:
                # --- Case B: Commutes (Detector) ---
                # Detector is formed by decomposing Meas into STABILIZERS only.
                # (Logicals do not contribute to deterministic checks in SE)
                
                # Extract current stabilizers from full_matrix
                if num_stabs > 0:
                    curr_stab_matrix = full_matrix[:num_stabs]
                    
                    coeffs, is_dependent, _ = solve_linear_decomposition(
                        basis=curr_stab_matrix, 
                        targets=meas_row_view
                    )
                    
                    if is_dependent[0]: # Construct a detector
                        args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                        comp_indices = np.where(coeffs[0])[0]
                        for c_idx in comp_indices:
                            # Map back to full records
                            # c_idx is index in curr_stab_matrix, which matches full_records[:num_stabs]
                            for r in full_records[c_idx]:
                                args.append(stim.target_rec(r - self.total_measurements))
                        
                        circuit.append("DETECTOR", args, list(syn_coords[i]) + [0])
                    else:
                        # New stabilizer that's missing in the current stabilizer tableau
                        # This should never happen in a well-defined stabilizer tableau
                        raise RuntimeError(
                            f"Measurement {i} commutes with all current stabilizers but is linearly independent.\n"
                            f"This implies the Full Stabilizer + Logicals Tableau is incomplete (Rank < num_qubits).\n"
                            f"Please ensure all qubits are initialized and added to the tracker before measurement."
                        )

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
        # We extract the pivot independent rows identified by RREF, which form the Logical Basis.
        if len(new_basis_indices) > 0:
            new_log_matrix = full_matrix[new_basis_indices]
            new_log_records = [full_records[i] for i in new_basis_indices]
            
            self.logicals.matrix = new_log_matrix
            self.logicals.records = new_log_records
        else:
            self.logicals = StabilizerTableau(self.num_qubits)

        # 2. Update Stabilizers
        # Reset stabilizers to the canonical measurement basis.
        # This keeps the tableau sparse and prevents "messy" linear combinations.
        new_stab_records = [[self.total_measurements - num_meas + i] for i in range(num_meas)]
        
        self.stabilizers.matrix = back_propagated_paulis
        self.stabilizers.records = new_stab_records

        # 3. Final Sanity Check (The Guardrail)
        if self.logicals.count != self.expected_num_logicals:
             raise RuntimeError(
                 f"[Error] Logical Count Mismatch!\n"
                 f"Expected: {self.expected_num_logicals}, Found: {self.logicals.count}\n"
                 f"This implies the measurements defined a subspace with incorrect dimensions.\n"
                 f"- If Found > Expected: System Underspecified (Missing measurements?).\n"
                 f"- If Found < Expected: System Overspecified (Measured a Logical?)."
             )

             
    def process_final_measurement(self, 
                                  circuit: stim.Circuit, 
                                  final_paulis: np.ndarray,
                                  syndrome_coords: List[List[float]]):
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
        
        for k in range(num_rows):
            # Condition 1: Must NOT be destroyed (anti-commuted).
            if k in destroyed_rows:
                continue
            
            # Condition 2: Must be fully determined by the measurements (Linear Dependent).
            # e.g., If we measure Z, but the row is X, it is independent -> Skip.
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
                args.append(stim.target_rec(stim_rec_target))
            
            # 2. Historical Record Components
            for r in full_records[k]:
                args.append(stim.target_rec(r - self.total_measurements))
            
            # 3. Output
            if k < num_stabs:
                # Stabilizer -> DETECTOR
                det_coord = syndrome_coords[k]
                circuit.append("DETECTOR", args, det_coord)
            else:
                # Logical -> OBSERVABLE
                log_idx = k - num_stabs
                circuit.append("OBSERVABLE_INCLUDE", args, [log_idx])