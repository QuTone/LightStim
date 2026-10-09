"""
CrossLS Experiment: Surface-PQRM lattice surgery.

Flow: Init Surface |+>, one round Surface SE; PQRM hypercube encode; init coupler;
SE rounds; final measure PQRM+ancilla in X, Surface in Z/X per PQRM_state.
"""

from typing import Dict, List, Optional

import numpy as np
import stim

from lightstim.ir.experiment import QECExperiment
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedSurfaceCodeLogicalOpSet
from lightstim.noise.config import NoiseConfig
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic

from lightstim.qec_code.PQRM.pqrm_patch import PQRMPatch, LOG_PQRM_LEN_DICT
from lightstim.qec_code.PQRM.pqrm_operation import PQRMLogicalOpSet
from .surface_pqrm_coupler import SurfacePQRMCoupler
from .surface_pqrm_se_block import SurfacePQRMSEBlock


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_pqrm_and_ancilla_data_indices(system: QECSystem) -> List[int]:
    """Global indices for PQRM data + coupler ancilla data (for X measurement)."""
    indices = []
    if "pqrm" in system.local_to_global_map:
        l2g = system.local_to_global_map["pqrm"]
        patch = system.patches["pqrm"][0]
        for lid in patch.data_indices:
            if lid in l2g:
                indices.append(l2g[lid])
    for cname, coupler in system.coupler_patches.items():
        if cname in system.local_to_global_map:
            l2g = system.local_to_global_map[cname]
            for lid in coupler.data_indices:
                if lid in l2g:
                    indices.append(l2g[lid])
    return sorted(set(indices))


def _get_surface_data_indices(system: QECSystem) -> List[int]:
    """Global indices for Surface data qubits."""
    name = "surface"
    if name not in system.local_to_global_map:
        return []
    l2g = system.local_to_global_map[name]
    patch = system.patches[name][0]
    return [l2g[lid] for lid in patch.data_indices if lid in l2g]


# -----------------------------------------------------------------------------
# CrossLSExperiment
# -----------------------------------------------------------------------------

class CrossLSExperiment(QECExperiment):
    """
    CrossLS: Lattice surgery between Surface code and PQRM.
    """

    def __init__(
        self,
        PQRM_para: List[int],
        d_surf: int,
        rounds: int = 2,
        PQRM_state: str = "Z",
        surf_state: str = "X",
        noise_params: Optional[NoiseConfig] = None,
        noise_model: str = "circuit_level",
        if_detector: bool = True,
        post_select_surface_se: bool = False,
        post_select_hybrid: bool = False,
        canonical_pqrm_logical: bool = False,
    ):
        rx, rz, m = PQRM_para

        super().__init__(
            extraction_block_class=SurfacePQRMSEBlock,
            rounds=rounds,
            noise_params=noise_params,
            noise_model=noise_model,
            if_detector=if_detector,
        )
        self.PQRM_para = PQRM_para
        self.post_select_surface_se = post_select_surface_se
        self.post_select_hybrid = post_select_hybrid
        self.canonical_pqrm_logical = canonical_pqrm_logical
        self.d_surf = d_surf
        self.PQRM_state = PQRM_state
        self.surf_state = surf_state
        self.rx, self.rz, self.m = rx, rz, m

    def _mark_terminal_pqrm_x_checks_for_postselection(self) -> None:
        """Select the PQRM X-check closures in the final tracker basis.

        Combined SE dresses some PQRM X checks with bridge-data X factors.
        They have no syndrome ancillas and are checked by the terminal MX.
        Identify their current representatives by their PQRM projection and
        physical support, not by a record sentinel or an earlier row index.
        The optional surface/hybrid coordinate selection remains separate.
        """
        checks = [
            stab for stab in self.system.stabilizers
            if stab.get("patch_name") == "pqrm" and stab.get("type") == "X"
        ]
        x_space = stabilizers_to_symplectic(
            self.system, checks, self.system.num_qubits)
        rows = self.tracker.stabilizers.matrix
        n = self.system.num_qubits
        pqrm = self.system.patches["pqrm"][0]
        l2g = self.system.local_to_global_map["pqrm"]
        pqrm_data = {l2g[q] for q in pqrm.data_indices}
        mx_data = set(_get_pqrm_and_ancilla_data_indices(self.system))
        outside_pqrm = sorted(set(range(n)) - pqrm_data)
        outside_mx = sorted(set(range(n)) - mx_data)
        projected = rows.copy()
        projected[:, outside_pqrm] = 0
        projected[:, n:] = 0
        _, in_x_space, _ = solve_linear_decomposition(
            basis=x_space, targets=projected, reduce_weight=False)
        selected = [
            int(k) for k in np.flatnonzero(in_x_space)
            if projected[k].any()
            and not rows[k, n:].any()
            and not rows[k, outside_mx].any()
            and k not in self.tracker.stabilizer_with_logical_components
        ]
        _, captured, _ = solve_linear_decomposition(
            basis=projected[selected], targets=x_space, reduce_weight=False)
        if not captured.all():
            raise RuntimeError(
                "CrossLS terminal MX is missing a representative of the "
                "PQRM X-check subspace required for postselection.")
        self.tracker.post_select_row_indices.update(selected)

    def build(self) -> stim.Circuit:
        """
        Define-by-run flow:
        1. Add Surface only -> Logical State prep (tracker/sys num_logicals=1)
        2. Add PQRM, init + encode -> num_logicals auto-syncs to 2
        3. Register coupler, init coupler ancilla
        4. Activate coupler
        5. SE rounds (first round identifies PQRM logical; no set_expected_logicals)
        """
        d = self.d_surf
        rx, rz, m = self.PQRM_para
        surf_state = self.surf_state

        # --- 1. Build system: Surface only (define-by-run) ---
        self.system = QECSystem()

        surface = UnrotatedSurfaceCode(distance=d, shift=(0, 0))
        surface.rotate_coords(np.pi / 2)
        surface.reset_rotation_angle()
        surface.shift_coords(-2 * d, 2)

        self.system.add_patch(surface, offset=(0, 0), name="surface", is_active=True)

        self._setup_experiment()
        # tracker.expected_num_logicals = system.num_logicals = 1 at this point
        self.logical_executor.register_op_set(PQRMPatch, PQRMLogicalOpSet())

        builder = self.builder
        n = self.system.num_qubits

        # --- 2. Write coords ---
        builder.write_coordinates()

        # --- 3. Logical State prep: Init Surface |+>, one round Surface SE ---
        sf_data = _get_surface_data_indices(self.system)
        init_dict = {q: surf_state for q in sf_data}
        builder.initialize(init_dict=init_dict, n=n)
        builder.circuit.append("TICK")

        _surf_ps_coords = set()
        if self.post_select_hybrid:
            for i in self.system.active_stabilizer_indices:
                s = self.system.stabilizers[i]
                if s.get("patch_name") == "surface":
                    syn_coord = s.get("syn_coord")
                    if syn_coord:
                        x, y = syn_coord[0], syn_coord[1]
                        if x in (-2, -3) or y in (2, 3):
                            coord_tuple = tuple(syn_coord) + (0,)
                            _surf_ps_coords.add(coord_tuple)
        if _surf_ps_coords:
            self.tracker.post_select_detector_coords.update(_surf_ps_coords)

        surface_se = UnrotatedSurfaceCodeExtractionBlock(self.system)
        builder.apply_syndrome_extraction(surface_se.circuit, rounds=1)
        builder.circuit.append("TICK")

        if _surf_ps_coords:
            self.tracker.post_select_detector_coords -= _surf_ps_coords

        self._noiseless_prefix_len = len(builder.circuit)

        # --- 4. Add PQRM (define-by-run): tracker expands, expected_num_logicals -> 2 ---
        pqrm = PQRMPatch(rx=rx, rz=rz, m=m, punctured=True, shift=(0, 0))
        self.system.add_patch(pqrm, offset=(0, 0), name="pqrm", is_active=True)
        n = self.system.num_qubits

        # --- 5. PQRM encoding (includes init via hypercube diagonal) ---
        self.logical_executor.apply_logical_operation(
            "hypercube_encode",
            patches=[pqrm],
            target_state=self.PQRM_state,
            patch_name="pqrm",
        )

        # --- 5b. Stabilizer canonicalization ---
        builder.stabilizer_canonicalization()

        # --- 5c. Logical canonicalization (optional) ---
        if self.canonical_pqrm_logical:
            canonical = self._build_canonical_pqrm_logical(pqrm)
            if canonical:
                builder.logical_canonicalization(canonical)

        # --- 6. Register coupler ---
        protocol = SurfacePQRMCoupler()
        self.system.register_coupler(protocol, ["surface", "pqrm"], name="surface_pqrm_coupler")
        n = self.system.num_qubits

        # --- 7. Init coupler data qubits only ---
        coupler_name = "surface_pqrm_coupler"
        coupler = self.system.coupler_patches[coupler_name]
        l2g = self.system.local_to_global_map.get(coupler_name, {})
        coupler_data_init = {l2g[i]: "X" for i in coupler.data_indices if i in l2g}
        builder.initialize(init_dict=coupler_data_init, n=n)
        builder.circuit.append("TICK")

        # --- 8. Activate coupler + PQRM ---
        self.system.activate_coupler(coupler_name)
        pqrm_stab_uids = {
            i for i, s in enumerate(self.system.stabilizers)
            if s.get("patch_name") == "pqrm"
        }
        self.system.active_stabilizer_indices.update(pqrm_stab_uids)

        # --- 9. SE rounds ---
        se_block = SurfacePQRMSEBlock(self.system)
        builder.apply_syndrome_extraction(se_block.circuit, rounds=self.rounds)
        builder.circuit.append("TICK")

        # --- 10. Final measure ---
        pqrm_ancilla_indices = _get_pqrm_and_ancilla_data_indices(self.system)
        surface_indices = _get_surface_data_indices(self.system)

        if self.PQRM_state == "Y":
            surface_patch = self.system.patches["surface"][0]
            surf_op_set = UnrotatedSurfaceCodeLogicalOpSet()
            surf_op_set.fold_transversal_s_dag(builder, surface_patch, noiseless=True)
            builder.circuit.append("TICK")

        final_measurements = {q: "X" for q in pqrm_ancilla_indices}
        for q in surface_indices:
            if self.PQRM_state == "Z":
                final_measurements[q] = "Z"
            elif self.PQRM_state == "X":
                final_measurements[q] = "X"
            elif self.PQRM_state == "Y":
                final_measurements[q] = "X"
            else:
                raise ValueError(f"Invalid PQRM state: {self.PQRM_state}")

        if self.post_select_surface_se:
            for i in self.system.active_stabilizer_indices:
                s = self.system.stabilizers[i]
                if s.get("type") == "X" and s.get("patch_name") == "surface":
                    syn_coord = s.get("syn_coord")
                    if syn_coord:
                        self.tracker.post_select_detector_coords.add(
                            (float(syn_coord[0]), float(syn_coord[1]), 1)
                        )
        elif self.post_select_hybrid:
            for coord in self.system.qubit_coords.values():
                x, y = float(coord[0]), float(coord[1])
                if x in (-2, -3) or y in (2, 3):
                    self.tracker.post_select_detector_coords.add((x, y, 1))

        if self.if_detector:
            self._mark_terminal_pqrm_x_checks_for_postselection()
        builder.apply_data_readout(final_measurements=final_measurements)
        builder.to_stim_circuit()

        # --- 11. Noise (optional) ---
        if self.noise_params is not None:
            prefix_len = getattr(self, '_noiseless_prefix_len', 0)
            if prefix_len > 0:
                from lightstim.noise.injector import NoiseInjector
                clean_prefix = builder.circuit[:prefix_len]
                clean_suffix = builder.circuit[prefix_len:]
                data_indices = [self.system.index_map[c] for c in self.system.data_coords]
                method_name = f"from_{self.noise_model}"
                factory_method = getattr(NoiseInjector, method_name)
                injector = factory_method(self.noise_params, data_indices)
                noisy_suffix = injector.inject_noise(clean_suffix)
                return clean_prefix + noisy_suffix
            return self._inject_noise(builder.circuit)
        return builder.to_stim_circuit()

    def _build_canonical_pqrm_logical(self, pqrm_patch: PQRMPatch) -> Optional[Dict[int, np.ndarray]]:
        """
        Build canonical PQRM logical operator for logical_canonicalization.
        """
        n = self.system.num_qubits
        log_len = LOG_PQRM_LEN_DICT[(self.rx, self.rz, self.m)]
        l2g = self.system.local_to_global_map.get("pqrm", {})

        canonical_pauli = np.zeros(2 * n, dtype=np.uint8)

        if self.PQRM_state == "Z":
            for y in range(log_len):
                local_coord = (0, 2 * (y + 1))
                for local_idx in pqrm_patch.data_indices:
                    if pqrm_patch.qubit_coords[local_idx] == local_coord:
                        global_idx = l2g[local_idx]
                        canonical_pauli[n + global_idx] = 1  # Z part
                        break
        elif self.PQRM_state in ("X", "Y"):
            if self.m == 4:
                target_y = {0, 2}
            else:
                target_y = {0, 2, 4, 6}

            for local_idx in pqrm_patch.data_indices:
                coord = pqrm_patch.qubit_coords[local_idx]
                if coord[1] in target_y and coord != (0, 0):
                    global_idx = l2g[local_idx]
                    canonical_pauli[global_idx] = 1  # X part
                    if self.PQRM_state == "Y":
                        canonical_pauli[n + global_idx] = 1  # Z part too → Y = XZ

        if np.sum(canonical_pauli) == 0:
            return None

        return {1: canonical_pauli}
