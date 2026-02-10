from __future__ import annotations

from typing import Dict, List, Type

from src.ir.builder import CircuitBuilder
from src.ir.operation import LogicalOpSet
from src.ir.qec_patch import QECPatch

class LogicalExecutor:
    def __init__(self, builder: CircuitBuilder):
        self.builder = builder
        # Map patch type -> op-set implementation
        self.op_sets: Dict[Type[QECPatch], LogicalOpSet] = {}
    
    def register_op_set(self, patch_type: Type[QECPatch], op_set: LogicalOpSet):
        """
        Register an operation set for a given patch type.
        """
        self.op_sets[patch_type] = op_set

    def apply_logical_operation(self, op_name: str, patches: List['QECPatch'], **kwargs):
        """
        The main API called by Experiment.
        """
        if not patches: return

        # 1. Routing: find OpSet based on Patch type
        primary_patch = patches[0]
        p_type = type(primary_patch)
        
        if p_type not in self.op_sets:
            raise ValueError(f"No LogicalOpSet registered for {p_type.__name__}")
        op_set = self.op_sets[p_type]

        # 2. Find method (Reflection)
        if not hasattr(op_set, op_name):
            raise ValueError(f"Operation '{op_name}' not supported by {p_type.__name__}")
        
        method = getattr(op_set, op_name)

        # 3. Execute and inject (Injection)
        # Pass self.builder into the method!
        method(self.builder, *patches, **kwargs)