from typing import Tuple, Dict, List
import stim
from src.ir.qec_patch import QECPatch

# -----------------------------------------------------------------------------
# Code Patch Class Definition
# -----------------------------------------------------------------------------
class RepetitionCode(QECPatch):
    """
    Implementation of the Repetition Code (Bit-Flip / Z-Stabilizer version).
    
    Layout: A linear chain of qubits along the x-axis.
    - Data qubits are at even x-coordinates: (0,0), (2,0), ...
    - Syndrome qubits are at odd x-coordinates: (1,0), (3,0), ...
    
    Stabilizers:
    - Z-checks between adjacent data qubits: Z_i * Z_{i+1}
    
    Logicals:
    - Logical Z: Z on the first data qubit (or any single data qubit).
    - Logical X: X on all data qubits.

    Parameters:
    -----------
    distance : int
        The number of data qubits. Must be >= 2.
    shift : Tuple[float, float], optional
        Global coordinate offset.

    Examples:
    ---------
    1. Create a standard distance-5 repetition code:
    >>> code = RepetitionCode(distance=5)

    2. Create a code shifted to a specific location (e.g., for layout alignment):
    >>> code = RepetitionCode(distance=5, shift=(10, 0))

    3. Construct from a configuration dictionary:
    >>> config = {'distance': 7, 'shift': (0, 2)}
    >>> code = RepetitionCode.from_config(config)
    """

    def _process_params(self):
        self.distance = self.params.get("distance")
        self.shift = self.params.get("shift", (0, 0))

        if self.distance is None:
            raise ValueError("Parameter 'distance' must be provided.")
        if self.distance < 2:
            raise ValueError("Distance must be at least 2.")

    def build(self):
        d = self.distance
        
        # Initialize lists
        self.data_coords = []
        self.syndrome_coords = []

        # -----------------------------------------------------------------------
        # Phase 1: Geometry Registration (Linear Chain at y=0)
        # -----------------------------------------------------------------------
        # Total length = 2*d - 1
        for x in range(2 * d - 1):
            coord = (x, 0)
            self.add_qubit(*coord)
            
            if x % 2 == 0:
                self.data_coords.append(coord)
            else:
                self.syndrome_coords.append(coord)

        # -----------------------------------------------------------------------
        # Phase 2: Physics Construction (Stabilizers)
        # -----------------------------------------------------------------------
        # Iterate over syndrome qubits (which are at odd indices 1, 3, ...)
        for k, syn_coord in enumerate(self.syndrome_coords):
            # Syndrome at x = 2k + 1
            # Left data neighbor: 2k
            # Right data neighbor: 2k + 2
            
            left_neighbor = (syn_coord[0] - 1, 0)
            right_neighbor = (syn_coord[0] + 1, 0)
            
            targets = {
                left_neighbor: 'Z',
                right_neighbor: 'Z'
            }
            self.create_stim_stabilizer(targets, syn_coord, 'Z')

        # -----------------------------------------------------------------------
        # Phase 3: Logical Operators
        # -----------------------------------------------------------------------
        # Logical Z: Z on the first data qubit (0,0)
        # Note: In repetition code, Z_L can be Z on any single data qubit.
        self.create_stim_logical([(0, 0)], 'Z')

        # Logical X: X on ALL data qubits (transversal)
        self.create_stim_logical(self.data_coords, 'X')

        self.num_logicals = 1

        # -----------------------------------------------------------------------
        # Phase 4: Shift
        # -----------------------------------------------------------------------
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)

    # --- Helpers ---

    # No need to override shift_coords, because the syndrome qubits are already shifted in the base class.

    def get_info(self):
        info = super().get_info()
        info.update({
            'distance': self.distance,
            'num_data_qubits': len(self.data_coords),
            'num_syndrome_qubits': len(self.syndrome_coords),
            'data_coords': self.data_coords,
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
class RepetitionCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Repetition Code.
    
    This block represents ONE cycle of stabilizer measurements (Z-check):
    1. Reset syndrome qubits.
    2. CNOT Layer 1: Data(Left) -> Syndrome.
    3. CNOT Layer 2: Data(Right) -> Syndrome.
    4. Measure syndrome qubits.
    
    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.
    """

    def __init__(self, system):
        """
        Args:
            system: The System object containing layout and index maps.
                    Expected attributes:
                    - index_map: Dict[(row, col), int]
                    - data_coords: List[(row, col)]
                    - syndrome_coords: List[(row, col)]
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
        
        # Critical Tag for NoiseInjector
        # Injectors (like CodeCapacity) look for this to inject errors on data qubits
        self.circuit.append("TICK", tag="SE_start") 

        # --- Step 2: CNOT Layer 1 (Left Neighbor) ---
        # According to source: "First layer of CNOTs from data to ancilla"
        # Based on layout, we assume this is Left Data -> Syndrome
        cnot_targets_1 = []
        
        for syn_coord in self.system.syndrome_coords:
            # syn_coord is (x, 0)
            # Left neighbor data is at (x-1, 0)
            left_data_coord = (syn_coord[0] - 1, 0)
            
            # Check if neighbor exists and is a data qubit
            if left_data_coord in self.system.data_coords:
                syn_idx = self.system.index_map[syn_coord]
                data_idx = self.system.index_map[left_data_coord]
                
                # Z-Stabilizer: Data is Control, Syndrome is Target
                cnot_targets_1.extend([data_idx, syn_idx])
        
        if cnot_targets_1:
            self.circuit.append("CNOT", cnot_targets_1)
        
        self.circuit.append("TICK")

        # --- Step 3: CNOT Layer 2 (Right Neighbor) ---
        # According to source: "Second layer... reverse order" -> Right Data -> Syndrome
        cnot_targets_2 = []
        
        for syn_coord in self.system.syndrome_coords:
            # Right neighbor data is at (x+1, 0)
            right_data_coord = (syn_coord[0] + 1, 0)
            
            if right_data_coord in self.system.data_coords:
                syn_idx = self.system.index_map[syn_coord]
                data_idx = self.system.index_map[right_data_coord]
                
                # Z-Stabilizer: Data is Control, Syndrome is Target
                cnot_targets_2.extend([data_idx, syn_idx])
        
        if cnot_targets_2:
            self.circuit.append("CNOT", cnot_targets_2)
            
        self.circuit.append("TICK")

        # --- Step 4: Measurement ---
        # Measure all syndrome qubits
        # Note: Source used 'MR', but we split into 'R' at start and 'M' at end
        # to fit the standard loop structure better.
        self.circuit.append("M", syn_indices)