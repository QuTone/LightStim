from typing import Set, Tuple, List, Dict
from src.ir.qec_patch import QECPatch

class BaseCoupler(QECPatch):
    """
    A Coupler IS A QECPatch, but (1) has no logical operators, and
    (2) has extra logic for enabling interactions between other patches encoding logical qubits.
    
    Inherited Attributes:
    - qubit_coords, index_map: Manages the qubits used by the coupler.
    - stabilizers: These are the "New Stabilizers" of the coupler, and measuring them enables logical measurements between the patches.
    - data_coords, syndrome_coords: Categorized qubit coordinates used by the coupler.
    
    New Attributes:
    - conflicting_stabilizer_coords: A set of syndrome coordinates whose associated stabilizers from other patches
      must be DISABLED when this coupler is active. Instead, the stabilizers of the coupler sharing the same syndrome coordinates are ACTIVATED.
    """
    def __init__(self, name: str, **kwargs):
        # Initialize as a standard patch
        super().__init__(**kwargs) 
        self.name = name
        self.conflicting_stabilizer_coords: Set[Tuple[int, int]] = set()
        # Note: No logical operators are needed for a coupler.

    def build(self):
        """
        Subclasses implement this to:
        1. Populate self.qubit_coords, self.data_coords... 
        2. Populate self.stabilizers 
        3. Populate self.conflicting_stabilizers
        """
        raise NotImplementedError