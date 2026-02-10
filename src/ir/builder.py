import stim
import numpy as np
from typing import List, Dict, Any, Optional, Union, Literal, Set
from dataclasses import dataclass

from ..ir.tracker import SyndromeTracker
from ..noise.config import NoiseConfig
from ..noise.injector import NoiseInjector

class CircuitBuilder:
    """
    Constructs the Stim circuit for QEC experiments. 
    SyndromeTracker automatically generates detectors and logical observables.
    NoiseInjector automatically injects noise to appropriate places according to the given noise model.
    """

    def __init__(self, 
                 tracker: SyndromeTracker, 
                 system_config: Any,
                 if_detector: bool = True):
        """
        Args:
            tracker: Initialized SyndromeTracker instance.
            system_config: Object containing system specs:
                           - qubit_coords: Dict[int, List[float]] OR List[List[float]]
                           - data_indices: List[int]
                           - syndrome_indices: List[int]
                           - syndrome_coords: List[List[float]]
        """
        self.tracker = tracker
        self.system = system_config
        self.circuit = stim.Circuit()
        self.if_detector = if_detector

    # --------------------------------------------------------------------------
    # A. Setup & Initialization
    # --------------------------------------------------------------------------
    def write_coordinates(self):
        """
        Generates QUBIT_COORDS instructions based on the system's layout.
        Essential for visualization.
        """
        # Handle both dict and list formats for coords
        coords_iterable = None
        if isinstance(self.system.qubit_coords, dict):
            coords_iterable = self.system.qubit_coords.items()
        elif isinstance(self.system.qubit_coords, list):
            coords_iterable = enumerate(self.system.qubit_coords)
        
        if coords_iterable:
            for q_index, coords in coords_iterable:
                self.circuit.append("QUBIT_COORDS", [q_index], list(coords))

    def initialize(self, init_dict: Dict[int, str], n: int):
        """
        Resets specific qubits in a given basis.
        """
        qubit_indices_x = [q for q, b in init_dict.items() if b == 'X']
        qubit_indices_z = [q for q, b in init_dict.items() if b == 'Z']
        qubit_indices_y = [q for q, b in init_dict.items() if b == 'Y']

        # Apply Reset Gate
        if qubit_indices_x:
            self.circuit.append("RX", qubit_indices_x)
        if qubit_indices_z:
            self.circuit.append("R", qubit_indices_z)
        if qubit_indices_y:
            self.circuit.append("RY", qubit_indices_y)
        
        init_tableau = self._get_initialization_tableau(qubit_indices_x, qubit_indices_z, qubit_indices_y, n)

        self.tracker.process_initialization(init_tableau)

    # --------------------------------------------------------------------------
    # B. Syndrome Extraction
    # --------------------------------------------------------------------------
    def apply_syndrome_extraction(self, 
                                  circuit_chunk: stim.Circuit, 
                                  rounds: int = 1):
        """
        Applies syndrome extraction with automated Tracker integration.
        
        Args:
            circuit_chunk: A Stim circuit representing ONE round of stabilizer measurement.
                           Only includes circuit operations. The last instruction has to be syndrome qubit measurement.
            rounds: Number of times to repeat.
        """
        if rounds < 1: return

        # ======================================================================
        # Phase 1: First Round (Tracker-Driven)
        # ======================================================================
        print("Applying first round of syndrome extraction...")
        # Analyze Ideal Basis for the Tracker
        back_propagated_paulis, syn_qubit_indices = self._get_back_propagated_pauli(circuit_chunk, self.tracker.num_qubits)
        syn_coords = [self.system.qubit_coords[i] for i in syn_qubit_indices] # extract from circuit_chunk, more robust
        
        # Append clean chunk to actual circuit
        self.circuit += circuit_chunk
        
        total_measurements = self.tracker.total_measurements
        meas_rec_to_idx_map_update = {total_measurements + i: syn_idx for i, syn_idx in enumerate(syn_qubit_indices)}
        self.tracker.meas_rec_to_idx_map.update(meas_rec_to_idx_map_update)
        # Ask Tracker to process it (Update Tableau + Generate Detectors)
        if self.if_detector:
            self.tracker.process_mid_measurement(
                circuit=self.circuit,
                back_propagated_paulis=back_propagated_paulis,
                syn_coords=syn_coords
            )
        
        
        # ======================================================================
        # Phase 2: Repeat Rounds (Stim Loop)
        # ======================================================================
        if rounds > 1:
            print("Applying rest rounds of syndrome extraction...")
            loop_body = stim.Circuit()
            num_syn = len(syn_coords)
            
            # Circuit operations for the repeated block
            loop_body.append("TICK")
            loop_body += circuit_chunk
            
            if self.if_detector:
                # Time Shift for visualization
                loop_body.append("SHIFT_COORDS", [], [0, 0, 1])
                
                # Construct Repeated Detectors: rec[-k] ^ rec[-k-num_syn]
                for i in range(num_syn):
                    # Stim record indices are relative to the current moment in the loop
                    rec_current = -num_syn + i
                    rec_prev = -num_syn + i - num_syn
                    
                    coord = syn_coords[i]
                    loop_body.append("DETECTOR", [stim.target_rec(rec_current), stim.target_rec(rec_prev)], list(coord) + [0]) 
            
            self.circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop_body))
            
            # Update the meas_rec_to_idx_map for the repeated rounds
            total_measurements = self.tracker.total_measurements
            for r in range(rounds - 1):
                self.tracker.meas_rec_to_idx_map.update({total_measurements + num_syn * r + i: syn_idx for i, syn_idx in enumerate(syn_qubit_indices)})
            
            # Update Tracker Counter, but the tableau does not need to be updated again
            meas_record_offset = num_syn * (rounds - 1)
            self.tracker.total_measurements += meas_record_offset
            for i in range(len(syn_coords)):
                records = self.tracker.stabilizers.records[i]
                shift_records = [rec + meas_record_offset for rec in records]
                self.tracker.stabilizers.records[i] = shift_records
            
    # --------------------------------------------------------------------------
    # C. Logical Gate & Unitary Operations
    # --------------------------------------------------------------------------
    def apply_unitary_block(self, unitary_block: stim.Circuit):
        """
        Applies a unitary circuit block and updates the tracker's tableau.
        
        This method is used for logical operations (e.g., transversal CNOT) that
        need to update the stabilizer tableau to reflect the unitary transformation.
        
        Args:
            unitary_block: A Stim circuit containing only unitary operations (no measurements/resets).
        """
        # Append the unitary block to the circuit
        self.circuit += unitary_block
        
        # Update the tracker's tableau to reflect the unitary transformation
        if self.if_detector:
            self.tracker.process_unitary_block(unitary_block)

    # --------------------------------------------------------------------------
    # D. Logical Coupler Activity, Stabilizer Masking/Unmasking
    # --------------------------------------------------------------------------
    def activate_coupler(self, name: str):
        """
        Turn on the logical coupler. A wrapper for QECSystem.activate_coupler.
        This changes the active stabilizer set for the NEXT round of extraction.
        """
        # Call the system's state manager
        self.system.activate_coupler(name)

    def deactivate_coupler(self, name: str):
        """
        Turn off the logical coupler and restore original patch boundaries.
        A wrapper for QECSystem.deactivate_coupler.
        """
        self.system.deactivate_coupler(name)

    def mask_stabilizers(self, ids: Set[int]):
        """
        Mask (Deactivate) the stabilizers with the given ids.
        To be implemented.
        """
        pass

    def unmask_stabilizers(self, ids: Set[int]):
        """
        Unmask (Activate) the stabilizers with the given ids.
        To be implemented.
        """
        pass


    # --------------------------------------------------------------------------
    # E. Data Qubit Measurement
    # --------------------------------------------------------------------------
    def apply_data_readout(self, final_measurements: Dict[int, str] = None):
        """
        Applies destructive measurement on data qubits and calls Tracker to 
        resolve remaining stabilizers into Detectors/Observables.
        """
        if final_measurements is None:
            final_measurements = {q: 'Z' for q in self.system.data_indices}
            
        zs = [q for q, b in final_measurements.items() if b == 'Z']
        xs = [q for q, b in final_measurements.items() if b == 'X']
        
        # Append gates (No manual noise here)
        if xs: self.circuit.append("MX", xs)
        if zs: self.circuit.append("M", zs)
        
        # Prepare Basis for Tracker
        sorted_indices = xs + zs
        n = self.tracker.num_qubits
        final_paulis = np.zeros((len(sorted_indices), 2 * n), dtype=np.uint8)
        
        for i, q in enumerate(sorted_indices):
            basis = final_measurements[q]
            if basis == 'X': final_paulis[i, q] = 1
            else: final_paulis[i, n + q] = 1

        # Call Tracker
        if self.if_detector:
            self.tracker.process_final_measurement(
                circuit=self.circuit,
                final_paulis=final_paulis,
                idx_to_coord_map=self.system.qubit_coords 
            )

    # --------------------------------------------------------------------------
    # F. Final Build & Noise Injection
    # --------------------------------------------------------------------------
    def build_noisy_circuit(
        self, 
        noise_params: NoiseConfig,
        noise_model: str = 'circuit_level'
    ) -> stim.Circuit:
        """
        Consumes the clean circuit and applies noise using the specified model strategy.
        
        Args:
            noise_params: Noise parameters (NoiseConfig).
            noise_model: The name of the factory method in NoiseInjector to use.
                         e.g., 'circuit_level' -> calls NoiseInjector.from_circuit_level(...)
                         e.g., 'custom_test'   -> calls NoiseInjector.from_custom_test(...)
        """
        # 1. Construct the expected factory method name
        method_name = f"from_{noise_model}"
        
        # 2. Dynamically retrieve the method from the NoiseInjector class
        if not hasattr(NoiseInjector, method_name):
            # Fallback or nice error message showing available options
            valid_methods = [m.replace("from_", "") for m in dir(NoiseInjector) if m.startswith("from_")]
            raise ValueError(f"Unknown noise model '{noise_model}'. "
                             f"Expected one of: {valid_methods}")
        
        factory_method = getattr(NoiseInjector, method_name)
        
        # 3. Inject noise
        # Assumptions: All factory methods must accept (config, data_qubits)
        data_indices = [self.system.index_map[coord] for coord in self.system.data_coords]
        injector = factory_method(noise_params, data_indices)
        noisy_circuit = injector.inject_noise(self.circuit)

        return noisy_circuit

    # --------------------------------------------------------------------------
    # G. Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _get_initialization_tableau(qubit_indices_x: List[int], qubit_indices_z: List[int], qubit_indices_y: List[int], n: int):
        """
        Generates the tableau for the given qubit indices in X, Z, Y basis.
        Args:
            qubit_indices_x: List[int]
            qubit_indices_z: List[int]
            qubit_indices_y: List[int]
            n: int
        Returns:
            initialized_tableau: np.ndarray
        """
        # 1. X Basis: (X=1, Z=0)
        # Shape is (len, 2n). If len=0, it's safe.
        t_x = np.zeros((len(qubit_indices_x), 2 * n), dtype=int)
        if qubit_indices_x:
            t_x[np.arange(len(qubit_indices_x)), qubit_indices_x] = 1

        # 2. Z Basis: (X=0, Z=1)
        t_z = np.zeros((len(qubit_indices_z), 2 * n), dtype=int)
        if qubit_indices_z:
            # Use list comprehension or numpy add to shift index by n
            cols = [i + n for i in qubit_indices_z]
            t_z[np.arange(len(qubit_indices_z)), cols] = 1

        # 3. Y Basis: (X=1, Z=1) -> Critical Fix here
        t_y = np.zeros((len(qubit_indices_y), 2 * n), dtype=int)
        if qubit_indices_y:
            # Set X part
            t_y[np.arange(len(qubit_indices_y)), qubit_indices_y] = 1
            # Set Z part (same row!)
            cols_z = [i + n for i in qubit_indices_y]
            t_y[np.arange(len(qubit_indices_y)), cols_z] = 1

        # 4. Stack
        # Since all sub-matrices have 2*n columns, vstack works even if some have 0 rows.
        return np.vstack([t_x, t_z, t_y])

    @staticmethod
    def _get_back_propagated_pauli(circuit_chunk: stim.Circuit, num_qubits: int) -> np.ndarray:
        """
        Analyzes the circuit chunk to find the measured Pauli basis using Tableau inversion.
        """
        # Sanity check: The last instruction has to be syndrome qubit measurement
        # Note: We check the last instruction of the provided chunk.
        last_instr = circuit_chunk[-1]
        
        meas_basis = ""
        if last_instr.name in ["M", "MZ", "MR", "MRZ"]:
            meas_basis = "Z"
        elif last_instr.name in ["MX", "MRX"]:
            meas_basis = "X"
        else: 
            raise ValueError(f"The last instruction has to be X/Z measurement on syndrome qubits. Now {last_instr.name}.")

        # Step 1. Get the back-propagated Pauli string
        # Convert to tableau and get numpy representation (binary array in symplectic representation)
        # We ignore measurement/reset to treat the circuit as a unitary operation for analysis.
        se_tableau = stim.Tableau.from_circuit(circuit_chunk, ignore_noise=True, ignore_measurement=True, ignore_reset=True)
        se_tableau_inverse = se_tableau.inverse()
        
        # 6 outputs: x2x, x2z, z2x, z2z, x_signs, z_signs
        x2x, x2z, z2x, z2z, _, _ = se_tableau_inverse.to_numpy()
        
        # Convert to int
        x2x_int = x2x.astype(int)
        x2z_int = x2z.astype(int)
        z2x_int = z2x.astype(int)
        z2z_int = z2z.astype(int)

        syn_meas_targets = last_instr.targets_copy()
        syn_qubit_indices = [item.value for item in syn_meas_targets if item.is_qubit_target]

        back_pauli_x = None
        back_pauli_z = None

        if meas_basis == "Z":
            # Make the syndrome qubit location zero, focus on the data qubit support
            # (without this step the back-propagated Pauli string will involve both syndrome and data qubits)
            # We zero out the columns corresponding to the syndrome qubits themselves
            z2x_int[syn_qubit_indices, syn_qubit_indices] = 0
            z2z_int[syn_qubit_indices, syn_qubit_indices] = 0
            
            # Get the back-propagated Pauli string corresponding to the measured qubits
            # We take the rows corresponding to the syndrome indices
            back_pauli_x = z2x_int[syn_qubit_indices, :]
            back_pauli_z = z2z_int[syn_qubit_indices, :]
            
        elif meas_basis == "X":
            # Make the syndrome qubit location zero, focus on the data qubit support
            x2x_int[syn_qubit_indices, syn_qubit_indices] = 0
            x2z_int[syn_qubit_indices, syn_qubit_indices] = 0
            
            # Get the back-propagated Pauli string
            back_pauli_x = x2x_int[syn_qubit_indices, :]
            back_pauli_z = x2z_int[syn_qubit_indices, :]
        else:
            raise ValueError(f"Measurement basis {meas_basis} not supported.")

        # Padding the back-propagated Pauli string to the full size of the system
        current_size = back_pauli_x.shape[1]
        
        if current_size < num_qubits:
            pad_width = num_qubits - current_size
            # Pad indices: ((top, bottom), (left, right))
            # We only pad columns on the right with 0s (Identity)
            back_pauli_x = np.pad(back_pauli_x, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            back_pauli_z = np.pad(back_pauli_z, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif current_size > num_qubits:
            raise ValueError(f"Circuit chunk has qubit index {current_size-1} which exceeds system size {num_qubits}.")

        # stack x_part and z_part to get the full 2n-bitstring
        back_pauli = np.hstack([back_pauli_x, back_pauli_z])
        
        return back_pauli, syn_qubit_indices

    def to_stim_circuit(self) -> stim.Circuit:
        return self.circuit