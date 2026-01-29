from typing import List, Set, Tuple, Optional
from src.ir.qec_patch import QECPatch
from dataclasses import dataclass
from abc import ABC, abstractmethod

class LogicalCouplerProtocol(ABC):
    """
    Base class for all coupler protocols.

    This class does NOT hold physical qubits itself. Instead, it defines the 
    'Rules of Engagement' (Protocol). It acts as a factory that takes a list 
    of interacting QECPatches and generates a NEW QECPatch instance (the coupler) 
    containing the necessary boundary stabilizers and ancilla qubits (data + syndrome).
    """
    
    # Subclasses must override this attribute to define the expected number of patches.
    # If None, it indicates support for any number of patches (variable length).
    EXPECTED_PATCH_COUNT: Optional[int] = None 

    def __init__(self, name_prefix: str = "coupler"):
        """
        Args:
            **kwargs: Protocol-specific configuration (e.g., interaction_type='XX').
        """
        # Store protocol configuration (State independent of specific patches)
        self.name_prefix = name_prefix
    
    def create_coupler_patch(self, 
                             patches: List[QECPatch], 
                             name: Optional[str] = None,
                             **params) -> QECPatch:
        """
        The Public Interface (Factory Method).
        
        Analyzes the geometry of the provided patches and generates a new QECPatch 
        representing the active coupling region.
        
        Args:
            patches: List of QECPatch objects to couple.
            name: Optional name for the new coupler patch.
            
        Returns:
            A new QECPatch instance containing coupler stabilizers and ancillae.
        """
        # 1. Validation
        self._validate_patch_count(patches)
        
        # 2. Name Resolution
        if name is None:
            # e.g. "coupler_patchA_patchB"
            patch_names = "_".join([p.name for p in patches])
            name = f"{self.name_prefix}_{patch_names}"
            
        # 3. Create the Container (The Coupler Instance)
        # This patch will own the new syndrome qubits (if any).
        coupler_patch = LogicalCouplerPatch(name=name)
        # 4. Delegate to Subclass to fill the container
        self._build_coupler_geometry(coupler_patch, patches, **params)
        
        return coupler_patch

    @abstractmethod
    def _build_coupler_geometry(self, coupler_patch: QECPatch, patches: List[QECPatch], **params):
        """
        Implementation of the abstract method from QECPatch.
        """
        raise NotImplementedError("Subclasses must implement '_build_coupler_patch' to define geometry.")
    

    def _validate_patch_count(self, patches: List[QECPatch]):
        """
        Enforces the patch count constraint defined by the subclass.
        """
        if self.EXPECTED_PATCH_COUNT is not None and len(patches) != self.EXPECTED_PATCH_COUNT:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.EXPECTED_PATCH_COUNT} patches, "
                f"but got {len(patches)}."
            )

class LogicalCouplerPatch(QECPatch):
    """
    Helper class: A concrete QECPatch that acts solely as a container.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conflicting_stabilizer_coords = set()
        self.syndrome_indices_x: Set[int] = set()
        self.syndrome_indices_z: Set[int] = set()

        self._process_params()

    def _process_params(self):
        self.name = self.params.get('name')
    
    def build(self):
        pass
    