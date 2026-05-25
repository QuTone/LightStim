"""
Bell State Teleportation experiments (three variants).

All three circuits teleport |ψ⟩_L from patch1 to patch3 via an entangled patch2.

Variants
--------
BellTeleportTG    — Transversal-CNOT-based (no lattice surgery)
BellTeleportZZLS  — Two sequential ZZ lattice-surgery measurements (vertical layout)
BellTeleportXXLS  — Two sequential XX lattice-surgery measurements (horizontal layout)
"""
import stim
from typing import Literal, Optional

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler,
)


def _coupler_globals(system: QECSystem, name: str):
    local = system.coupler_patches[name].data_indices
    return [system.local_to_global_map[name][q] for q in local]


class BellTeleportTG:
    """
    Transversal-gate Bell teleportation on 3 unrotated surface code patches.

    Protocol
    --------
    patch1 |ψ⟩  ──────SE──●──SE──── Meas X
                          │
    patch2 |+⟩  ───●──SE──⊕──SE──── Meas Z
                   │
    patch3 |0⟩  ───⊕──SE──── SE──── Meas <teleport_state>  →  |ψ⟩_L
    """

    def __init__(
        self,
        distance: int = 3,
        rounds_pre: Optional[int] = None,
        rounds_mid: int = 1,
        rounds_post: int = 1,
        teleport_state: Literal["X", "Z"] = "Z",
        if_detector: bool = True,
        noise_params: Optional[NoiseConfig] = None,
        noise_model: str = "circuit_level",
    ):
        self.distance = distance
        self.rounds_pre = rounds_pre if rounds_pre is not None else distance
        self.rounds_mid = rounds_mid
        self.rounds_post = rounds_post
        self.teleport_state = teleport_state.upper()
        self.if_detector = if_detector
        self.noise_params = noise_params
        self.noise_model = noise_model

    def build(self) -> stim.Circuit:
        d = self.distance
        dx = 2 * (2 * d - 1) - 2  # patch spacing (no overlap, gap=2)

        system = QECSystem()
        p1 = system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch1")
        p2 = system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch2", offset=(dx, 0))
        p3 = system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch3", offset=(2 * dx, 0))

        tracker = SyndromeTracker(
            num_qubits=system.num_qubits,
            expected_num_logicals=system.num_logicals,
        )
        builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=self.if_detector)
        builder.write_coordinates()

        # Initialise: patch1=|ψ⟩, patch2=|+⟩(X), patch3=|0⟩(Z)
        INIT = {"patch1": self.teleport_state, "patch2": "X", "patch3": "Z"}
        init_dict = {q: INIT[system.index_to_owner_map[q]] for q in system.data_indices}
        builder.initialize(init_dict=init_dict, n=system.num_qubits)

        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_pre)

        executor = LogicalExecutor(builder=builder)
        executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())

        # CNOT(patch2 → patch3): prepare Bell pair
        executor.apply_logical_operation("transversal_cnot", patches=[p2, p3])
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_mid)

        # CNOT(patch1 → patch2): entangle patch1 with the Bell pair
        executor.apply_logical_operation("transversal_cnot", patches=[p1, p2])
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_post)

        # Final readout: patch1=X, patch2=Z, patch3=teleport_state
        MEAS = {"patch1": "X", "patch2": "Z", "patch3": self.teleport_state}
        meas_dict = {q: MEAS[system.index_to_owner_map[q]] for q in system.data_indices}
        builder.apply_data_readout(final_measurements=meas_dict)

        if self.noise_params is not None:
            return builder.build_noisy_circuit(
                noise_params=self.noise_params, noise_model=self.noise_model
            )
        return builder.circuit


class BellTeleportZZLS:
    """
    ZZ-lattice-surgery Bell teleportation on 3 unrotated surface code patches.

    Protocol (vertical column layout)
    ----------------------------------
    patch1 |ψ⟩  ─────────┤ZZ├───── Meas X
                         │12│
    patch2 |+⟩  ──┤ZZ├───┤ZZ├───── Meas X
                  │23│
    patch3 |+⟩  ──┤ZZ├──────────── Meas <teleport_state>  →  |ψ⟩_L

    Coupler_23 data qubits are measured *immediately* when coupler_23 is
    deactivated, before coupler_12 is activated.  This prevents idle-error
    accumulation in the detector chain.
    """

    def __init__(
        self,
        distance: int = 3,
        rounds_pre: Optional[int] = None,
        rounds_ls: Optional[int] = None,
        teleport_state: Literal["X", "Z"] = "Z",
        if_detector: bool = True,
        noise_params: Optional[NoiseConfig] = None,
        noise_model: str = "circuit_level",
    ):
        self.distance = distance
        self.rounds_pre = rounds_pre if rounds_pre is not None else distance
        self.rounds_ls = rounds_ls if rounds_ls is not None else distance
        self.teleport_state = teleport_state.upper()
        self.if_detector = if_detector
        self.noise_params = noise_params
        self.noise_model = noise_model

    def build(self) -> stim.Circuit:
        d = self.distance
        step = (2 * d - 1) + 1  # d_size + gap=1 → gap between patches is even ✓

        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch1")
        system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch2", offset=(0, step))
        system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch3", offset=(0, 2 * step))

        coupler_proto = UnrotatedTwoPatchCoupler()
        system.register_coupler(coupler_proto, patch_names=["patch2", "patch3"],
                                name="coupler_23", interaction_type="ZZ")
        system.register_coupler(coupler_proto, patch_names=["patch1", "patch2"],
                                name="coupler_12", interaction_type="ZZ")

        tracker = SyndromeTracker(
            num_qubits=system.num_qubits,
            expected_num_logicals=system.num_logicals,
        )
        builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=self.if_detector)
        builder.write_coordinates()

        # Initialise code patches: patch1=|ψ⟩, patch2=|+⟩(X), patch3=|+⟩(X)
        INIT = {"patch1": self.teleport_state, "patch2": "X", "patch3": "X"}
        init_dict = {
            q: INIT[system.index_to_owner_map[q]]
            for q in system.data_indices
            if system.index_to_owner_map[q] in INIT
        }
        builder.initialize(init_dict=init_dict, n=system.num_qubits)

        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_pre)

        # ── First LS: ZZ(patch2, patch3) ────────────────────────────────
        builder.activate_coupler("coupler_23")
        cp23 = _coupler_globals(system, "coupler_23")
        builder.initialize(init_dict={q: "X" for q in cp23}, n=system.num_qubits)
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_ls)
        builder.deactivate_coupler("coupler_23")

        # ── Measure coupler_23 data qubits IMMEDIATELY at deactivation ──
        # If deferred to final readout, these qubits sit idle through the
        # entire coupler_12 phase — corrupting the detector chain.
        builder.apply_data_readout(final_measurements={q: "X" for q in cp23})

        # ── Second LS: ZZ(patch1, patch2) ───────────────────────────────
        builder.activate_coupler("coupler_12")
        cp12 = _coupler_globals(system, "coupler_12")
        builder.initialize(init_dict={q: "X" for q in cp12}, n=system.num_qubits)
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_ls)

        # ── Final readout: code patches + coupler_12 ────────────────────
        MEAS = {"patch1": "X", "patch2": "X", "patch3": self.teleport_state}
        meas_dict = {
            q: MEAS[system.index_to_owner_map[q]]
            for q in system.data_indices
            if system.index_to_owner_map[q] in MEAS
        }
        meas_dict.update({q: "X" for q in cp12})
        builder.apply_data_readout(final_measurements=meas_dict)

        if self.noise_params is not None:
            return builder.build_noisy_circuit(
                noise_params=self.noise_params, noise_model=self.noise_model
            )
        return builder.circuit


class BellTeleportXXLS:
    """
    XX-lattice-surgery Bell teleportation on 3 unrotated surface code patches.

    Protocol (horizontal row layout)
    ----------------------------------
    patch1 |ψ⟩  ─────────┤XX├───── Meas Z
                         │12│
    patch2 |0⟩  ──┤XX├───┤XX├───── Meas Z
                  │23│
    patch3 |0⟩  ──┤XX├──────────── Meas <teleport_state>  →  |ψ⟩_L

    Coupler_23 data qubits are measured *immediately* when coupler_23 is
    deactivated, before coupler_12 is activated.
    """

    def __init__(
        self,
        distance: int = 3,
        rounds_pre: Optional[int] = None,
        rounds_ls: Optional[int] = None,
        teleport_state: Literal["X", "Z"] = "X",
        if_detector: bool = True,
        noise_params: Optional[NoiseConfig] = None,
        noise_model: str = "circuit_level",
    ):
        self.distance = distance
        self.rounds_pre = rounds_pre if rounds_pre is not None else distance
        self.rounds_ls = rounds_ls if rounds_ls is not None else distance
        self.teleport_state = teleport_state.upper()
        self.if_detector = if_detector
        self.noise_params = noise_params
        self.noise_model = noise_model

    def build(self) -> stim.Circuit:
        d = self.distance
        step = (2 * d - 1) + 1  # d_size + gap=1 → gap between patches is even ✓

        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch1")
        system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch2", offset=(step, 0))
        system.add_patch(UnrotatedSurfaceCode(distance=d), name="patch3", offset=(2 * step, 0))

        coupler_proto = UnrotatedTwoPatchCoupler()
        system.register_coupler(coupler_proto, patch_names=["patch2", "patch3"],
                                name="coupler_23", interaction_type="XX")
        system.register_coupler(coupler_proto, patch_names=["patch1", "patch2"],
                                name="coupler_12", interaction_type="XX")

        tracker = SyndromeTracker(
            num_qubits=system.num_qubits,
            expected_num_logicals=system.num_logicals,
        )
        builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=self.if_detector)
        builder.write_coordinates()

        # Initialise code patches: patch1=|ψ⟩, patch2=|0⟩(Z), patch3=|0⟩(Z)
        INIT = {"patch1": self.teleport_state, "patch2": "Z", "patch3": "Z"}
        init_dict = {
            q: INIT[system.index_to_owner_map[q]]
            for q in system.data_indices
            if system.index_to_owner_map[q] in INIT
        }
        builder.initialize(init_dict=init_dict, n=system.num_qubits)

        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_pre)

        # ── First LS: XX(patch2, patch3) ────────────────────────────────
        builder.activate_coupler("coupler_23")
        cp23 = _coupler_globals(system, "coupler_23")
        builder.initialize(init_dict={q: "Z" for q in cp23}, n=system.num_qubits)
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_ls)
        builder.deactivate_coupler("coupler_23")

        # ── Measure coupler_23 data qubits IMMEDIATELY at deactivation ──
        builder.apply_data_readout(final_measurements={q: "Z" for q in cp23})

        # ── Second LS: XX(patch1, patch2) ───────────────────────────────
        builder.activate_coupler("coupler_12")
        cp12 = _coupler_globals(system, "coupler_12")
        builder.initialize(init_dict={q: "Z" for q in cp12}, n=system.num_qubits)
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=self.rounds_ls)

        # ── Final readout: code patches + coupler_12 ────────────────────
        MEAS = {"patch1": "Z", "patch2": "Z", "patch3": self.teleport_state}
        meas_dict = {
            q: MEAS[system.index_to_owner_map[q]]
            for q in system.data_indices
            if system.index_to_owner_map[q] in MEAS
        }
        meas_dict.update({q: "Z" for q in cp12})
        builder.apply_data_readout(final_measurements=meas_dict)

        if self.noise_params is not None:
            return builder.build_noisy_circuit(
                noise_params=self.noise_params, noise_model=self.noise_model
            )
        return builder.circuit
