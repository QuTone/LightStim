"""
Logical S-gate teleportation protocols.

This module provides a two-patch S-state teleportation experiment with two
implementation choices:

* ``method="ZZ"`` performs the parity measurement with two-patch ZZ lattice
  surgery.
* ``method="cnot_trans"`` performs the same teleportation primitive with a
  transversal CNOT layer.

The resource |Y>_L state can be prepared either by corner injection or by
fault-tolerant logical S on a |+>_L patch.  The latter is currently available
for the unrotated surface code only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Type

import stim

from lightstim.ir.experiment import QECExperiment
from lightstim.ir.qec_patch import QECPatch
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
    RotatedTwoPatchCoupler,
)
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
    UnrotatedTwoPatchCoupler,
)


CodeName = Literal["unrotated_sc", "rotated_sc"]
Method = Literal["ZZ", "cnot_trans"]
StatePrep = Literal["inject", "logical_gate"]


@dataclass(frozen=True)
class _CodeSpec:
    patch_class: Type[QECPatch]
    extraction_block_class: Type
    op_set_class: Type
    coupler_class: Optional[Type]
    default_resource_offset: Tuple[float, float]
    supports_logical_s: bool
    supports_transversal_cnot: bool


def _code_spec(code: CodeName, distance: int, method: Method) -> _CodeSpec:
    if code == "unrotated_sc":
        step = (2 * distance - 1) + 1
        return _CodeSpec(
            patch_class=UnrotatedSurfaceCode,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            op_set_class=UnrotatedSurfaceCodeLogicalOpSet,
            coupler_class=UnrotatedTwoPatchCoupler,
            default_resource_offset=(0.0, float(step)) if method == "ZZ" else (float(2 * step), 0.0),
            supports_logical_s=True,
            supports_transversal_cnot=True,
        )
    if code == "rotated_sc":
        step = 2 * distance + 2
        return _CodeSpec(
            patch_class=RotatedSurfaceCode,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            op_set_class=RotatedSurfaceCodeLogicalOpSet,
            coupler_class=RotatedTwoPatchCoupler,
            default_resource_offset=(0.0, float(step)) if method == "ZZ" else (float(step), 0.0),
            supports_logical_s=False,
            supports_transversal_cnot=True,
        )
    raise ValueError(f"Unknown code {code!r}. Choose 'unrotated_sc' or 'rotated_sc'.")


def _patch_data(system, owner: str) -> Dict[int, str]:
    return {
        q: owner
        for q in system.data_indices
        if system.index_to_owner_map.get(q) == owner
    }


def _coupler_data(system, coupler_name: str) -> list[int]:
    return [
        system.local_to_global_map[coupler_name][q]
        for q in system.coupler_patches[coupler_name].data_indices
    ]


class SGateTeleportExperiment(QECExperiment):
    """
    Two-patch logical S-gate teleportation benchmark circuit.

    The input patch starts in |+>_L.  The resource patch is prepared as |Y>_L
    either by state injection or by applying logical S to |+>_L.  The two patches
    are then coupled by the selected teleportation primitive and read out in a
    basis that verifies the teleported S action.
    """

    def __init__(
        self,
        distance: int = 3,
        code: CodeName = "unrotated_sc",
        method: Method = "ZZ",
        state_prep: StatePrep = "logical_gate",
        rounds_prep: Optional[int] = None,
        rounds_gate: Optional[int] = None,
        resource_offset: Optional[Tuple[float, float]] = None,
        injection_protocol: Literal["corner", "middle"] = "corner",
        noise_params: Optional[NoiseConfig] = None,
        noise_model: Optional[str] = "circuit_level",
        if_detector: bool = True,
    ):
        method = "ZZ" if method.upper() == "ZZ" else method
        if method not in ("ZZ", "cnot_trans"):
            raise ValueError(f"Unknown method {method!r}. Choose 'ZZ' or 'cnot_trans'.")
        if state_prep not in ("inject", "logical_gate"):
            raise ValueError(
                f"Unknown state_prep {state_prep!r}. Choose 'inject' or 'logical_gate'."
            )

        self.distance = distance
        self.code = code
        self.method = method
        self.state_prep = state_prep
        self.rounds_prep = rounds_prep if rounds_prep is not None else distance
        self.rounds_gate = rounds_gate if rounds_gate is not None else distance
        self.injection_protocol = injection_protocol
        self.spec = _code_spec(code, distance, method)
        self.resource_offset = (
            tuple(resource_offset)
            if resource_offset is not None
            else self.spec.default_resource_offset
        )

        if state_prep == "logical_gate" and not self.spec.supports_logical_s:
            raise ValueError(
                "state_prep='logical_gate' requires a code with logical S support. "
                "Use code='unrotated_sc' or state_prep='inject'."
            )
        if method == "cnot_trans" and not self.spec.supports_transversal_cnot:
            raise ValueError(
                "method='cnot_trans' is currently supported for unrotated_sc only. "
                "Use method='ZZ' for rotated_sc."
            )

        super().__init__(
            extraction_block_class=self.spec.extraction_block_class,
            rounds=self.rounds_gate,
            noise_params=noise_params,
            noise_model=noise_model,
            if_detector=if_detector,
        )

        self.input_name = "input"
        self.resource_name = "resource"
        self.coupler_name = "zz_coupler"
        self.circuit_info: Dict[str, Any] = {}

    def _do_se(self, rounds: int):
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(
            circuit_chunk=se_block.circuit,
            rounds=rounds,
        )

    def _register_ops(self):
        op_set = self.spec.op_set_class(self.extraction_block_class)
        self.logical_executor.register_op_set(self.spec.patch_class, op_set)
        return op_set

    def _prepare_input_and_resource(self, resource_global_patch, resource_registered_patch):
        input_data = _patch_data(self.system, self.input_name)
        resource_data = _patch_data(self.system, self.resource_name)

        if self.state_prep == "logical_gate":
            init_dict = {q: "X" for q in input_data}
            init_dict.update({q: "X" for q in resource_data})
            self.builder.initialize(init_dict=init_dict, n=self.system.num_qubits)
            self._do_se(self.rounds_prep)
            self.logical_executor.apply_logical_operation(
                "fold_transversal_s",
                patches=[resource_registered_patch],
            )
            self._do_se(self.rounds_gate)
            return

        self.builder.initialize(
            init_dict={q: "X" for q in input_data},
            n=self.system.num_qubits,
        )
        self.logical_executor.apply_logical_operation(
            "state_injection",
            patches=[resource_global_patch],
            inject_state="Y",
            protocol=self.injection_protocol,
            rounds=0,
            post_select_coords=set(),
        )
        self._do_se(self.rounds_prep)

    def _apply_zz_teleport(self):
        self.builder.activate_coupler(self.coupler_name)
        coupler_data = _coupler_data(self.system, self.coupler_name)
        self.builder.initialize(
            init_dict={q: "X" for q in coupler_data},
            n=self.system.num_qubits,
        )
        self._do_se(self.rounds_gate)
        return coupler_data

    def _apply_cnot_teleport(self, input_global_patch, resource_global_patch):
        self.logical_executor.apply_logical_operation(
            "transversal_cnot",
            patches=[input_global_patch, resource_global_patch],
        )
        self._do_se(self.rounds_gate)
        return []

    def _verify_and_readout(self, input_registered_patch, coupler_data):
        if self.spec.supports_logical_s:
            self.logical_executor.apply_logical_operation(
                "fold_transversal_s_dag",
                patches=[input_registered_patch],
                noiseless=True,
            )
            input_basis = "X"
        else:
            input_basis = "Y"

        final_meas: Dict[int, str] = {}
        final_meas.update({q: input_basis for q in _patch_data(self.system, self.input_name)})
        resource_basis = "X" if self.method == "ZZ" else "Z"
        final_meas.update({q: resource_basis for q in _patch_data(self.system, self.resource_name)})
        final_meas.update({q: "X" for q in coupler_data})
        self.builder.apply_data_readout(final_measurements=final_meas)

    def build(self) -> stim.Circuit:
        from lightstim.ir.qec_system import QECSystem

        self.system = QECSystem()
        input_global_patch = self.system.add_patch(
            self.spec.patch_class(distance=self.distance),
            name=self.input_name,
            offset=(0.0, 0.0),
        )
        resource_global_patch = self.system.add_patch(
            self.spec.patch_class(distance=self.distance),
            name=self.resource_name,
            offset=self.resource_offset,
        )
        input_registered_patch = self.system.patches[self.input_name][0]
        resource_registered_patch = self.system.patches[self.resource_name][0]

        if self.method == "ZZ":
            self.system.register_coupler(
                self.spec.coupler_class(),
                patch_names=[self.input_name, self.resource_name],
                name=self.coupler_name,
                interaction_type="ZZ",
            )

        self._setup_experiment()
        self._register_ops()
        self.builder.write_coordinates()

        self._prepare_input_and_resource(resource_global_patch, resource_registered_patch)

        if self.method == "ZZ":
            coupler_data = self._apply_zz_teleport()
        else:
            coupler_data = self._apply_cnot_teleport(input_global_patch, resource_global_patch)

        self._verify_and_readout(input_registered_patch, coupler_data)

        circuit = self._inject_noise(self.builder.circuit)
        self.circuit_info = {
            "code": self.code,
            "method": self.method,
            "state_prep": self.state_prep,
            "distance": self.distance,
            "rounds_prep": self.rounds_prep,
            "rounds_gate": self.rounds_gate,
            "num_qubits": circuit.num_qubits,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
        }
        return circuit
