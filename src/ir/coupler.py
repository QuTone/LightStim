from typing import List, Set, Tuple, Optional
from src.ir.qec_patch import QECPatch

class LogicalCoupler(QECPatch):
    """
    Base class for all couplers.
    Manages connections between N patches.
    """
    
    # Subclasses must override this attribute to define the expected number of patches.
    # If None, it indicates support for any number of patches (variable length).
    EXPECTED_PATCH_COUNT: Optional[int] = None 

    def __init__(self, patches: List[QECPatch], name: str = "coupler", **kwargs):
        # 1. Init Base (QECPatch)
        # This initializes empty containers for qubit_coords, stabilizers, etc.
        super().__init__(**kwargs)
        
        self.name = name
        self.patches = patches
        
        # 2. Validation Logic
        self._validate_patch_count()
        
        # 3. Conflict Management Container (populated by subclass build or detect_conflicts)
        self.conflicting_stabilizer_coords: Set[Tuple[int, int]] = set()

        # 4. Trigger Build Process (The Builder Pattern)
        self.build()

    def _validate_patch_count(self):
        """
        Enforces the patch count constraint defined by the subclass.
        """
        expected = self.EXPECTED_PATCH_COUNT
        actual = len(self.patches)
        
        if expected is not None and actual != expected:
            raise ValueError(
                f"{self.__class__.__name__} expects {expected} patches, "
                f"but got {actual}."
            )

    def build(self):
        """
        [Abstract Interface]
        Subclasses MUST implement this to generate the physical geometry.
        1. Add qubits (self.add_qubit) to fill the gap.
        2. Define coupler stabilizers (self.stabilizers).
        """
        raise NotImplementedError("Subclasses must implement 'build' to define geometry.")

    def detect_conflicts(self):
        """
        [Automation Logic]
        Generic logic to find commutativity conflicts.
        (Shared across all coupler types)
        """
        # ... your conflict detection code ...
        pass