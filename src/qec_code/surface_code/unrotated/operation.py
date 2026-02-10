from src.ir.operation import LogicalOpSet
from src.ir.qec_patch import QECPatch
import stim

class UnrotatedSurfaceCodeLogicalOpSet(LogicalOpSet):
    """
    Inherits CNOT/Init from CSSLogicalOpSet.
    Adds Surface-codespecific operations.
    """
    def __init__(self):
        super().__init__("UnrotatedSurfaceCode")

    def fold_transversal_Hadamard(self, patch: QECPatch) -> stim.Circuit:
        """
        Applies a fold-transversal Hadamard gate using H-SWAP gates.
        """
        pass

    def LS_Hadamard(self, patch: QECPatch) -> stim.Circuit:
        """
        Applies a Hadamard gate using transversal H with patch rotation.
        """
        pass

    def fold_transversal_s(self, patch: QECPatch) -> stim.Circuit:
        """
        Applies a fold-transversal S gate using S/S^dagger-CZ gates.
        """
        pass