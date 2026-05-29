"""
Protocol Smoke Tests — every LightStim protocol builds a valid circuit.

Each test verifies (noiseless, d=3):
  1. num_detectors > 0, num_observables > 0
  2. Zero detection events on 200 noiseless shots
  3. DEM construction succeeds

Purpose: catch tracker / builder / coupler regressions that would silently
break a protocol without touching its own module.

Run:  pytest tests/test_protocols.py -m smoke -q
"""
import io
import contextlib
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import assert_valid_circuit, assert_noiseless, assert_dem_valid, build_quiet


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestMemory:

    def _run(self, code, block_cls, basis="Z", rounds=3):
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.memory import MemoryExperiment
        system = QECSystem()
        system.add_patch(code, name="patch")
        exp = MemoryExperiment(qec_system=system, extraction_block_class=block_cls,
                               rounds=rounds, noise_params=None, noise_model="circuit_level",
                               basis=basis)
        return build_quiet(exp.build)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_rotated_surface_code(self, basis):
        from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
        c = self._run(RotatedSurfaceCode(distance=3), RotatedSurfaceCodeExtractionBlock, basis)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_unrotated_surface_code(self, basis):
        from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
        c = self._run(UnrotatedSurfaceCode(distance=3), UnrotatedSurfaceCodeExtractionBlock, basis)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_toric_code(self):
        from lightstim.qec_code.surface_code.toric import ToricCode, ToricCodeExtractionBlock
        c = self._run(ToricCode(distance=3), ToricCodeExtractionBlock)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_color_code(self):
        from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
        c = self._run(ColorCode(distance=3), ColorCodeExtractionBlock)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_repetition_code(self):
        from lightstim.qec_code.repetition import RepetitionCode, RepetitionCodeExtractionBlock
        c = self._run(RepetitionCode(distance=5), RepetitionCodeExtractionBlock)
        assert c.num_detectors > 0; assert_noiseless(c)

    def test_bb_code(self):
        from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.memory import MemoryExperiment
        code = BBCode(l=6, m=6, A=[[3,0],[0,1],[0,2]], B=[[0,3],[1,0],[2,0]])
        system = QECSystem(); system.add_patch(code, name="bb")
        exp = MemoryExperiment(qec_system=system, extraction_block_class=BBCodeExtractionBlock,
                               rounds=6, noise_params=None, noise_model="circuit_level", basis="Z")
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_four_d_geo_code(self):
        from lightstim.qec_code.four_d_geo_code import FourDGeoCode, FourDGeoCodeExtractionBlock
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.memory import MemoryExperiment
        code = FourDGeoCode(L=[[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,3]], d=3)
        system = QECSystem(); system.add_patch(code, name="4d")
        exp = MemoryExperiment(qec_system=system, extraction_block_class=FourDGeoCodeExtractionBlock,
                               rounds=3, noise_params=None, noise_model="circuit_level", basis="Z")
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGICAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestLogicalOps:

    def test_two_patch_ls_zz(self):
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3}, patch2_config={"distance": 3},
            offset=(0, 8), interaction_type="ZZ",
            initial_state_patch1="X", initial_state_patch2="Z",
            measure_state_patch1="X", measure_state_patch2="Z",
            rounds=2, noise_params=None,
        )
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_two_patch_ls_xx(self):
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3}, patch2_config={"distance": 3},
            offset=(8, 0), interaction_type="XX",
            initial_state_patch1="Z", initial_state_patch2="X",
            measure_state_patch1="Z", measure_state_patch2="X",
            rounds=2, noise_params=None,
        )
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_multi_patch_ls(self):
        """3-patch ZZZ product measurement."""
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedMultiPatchCoupler)
        from lightstim.ir.qec_system import QECSystem
        from lightstim.ir.tracker import SyndromeTracker
        from lightstim.ir.builder import CircuitBuilder
        d = 3; step = float(d * 4)
        system = QECSystem()
        names = []
        for i, off in enumerate([(0.,0.), (step,0.), (0.,step)]):
            nm = f"p{i+1}"; system.add_patch(UnrotatedSurfaceCode(distance=d), name=nm, offset=off); names.append(nm)
        with contextlib.redirect_stdout(io.StringIO()):  # type: ignore[attr-defined]
            system.register_coupler(UnrotatedMultiPatchCoupler(), names, "c",
                                    path_axis="vertical", center_axis=step/2)
        tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker, system)
        builder.write_coordinates()
        nc = {q: "X" for q in system.data_indices if system.index_to_owner_map.get(q) != "c"}
        builder.initialize(nc, n=system.num_qubits)
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(se.circuit, rounds=2)
        builder.activate_coupler("c")
        cd = {system.local_to_global_map["c"][q]: "X" for q in system.coupler_patches["c"].data_indices}
        builder.initialize(cd, n=system.num_qubits)
        se2 = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(se2.circuit, rounds=2)
        builder.apply_data_readout({**nc, **cd})
        c = builder.circuit
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_transversal_cnot(self):
        from lightstim.protocols.cnot_trans import CNOTTransExperiment
        from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
        exp = CNOTTransExperiment(
            code_patch_class=UnrotatedSurfaceCode,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            code_params_control={"distance": 3}, code_params_target={"distance": 3},
            offset_target=(12., 0.), rounds_before=2, rounds_after=2, noise_params=None,
        )
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_cnot_ls(self):
        from lightstim.protocols.cnot_ls import CNOTLSExperiment
        exp = CNOTLSExperiment(
            patch_configs={"a": {"distance": 3}, "c": {"distance": 3}, "t": {"distance": 3}},
            offset_ta=(6., 0.), offset_ca=(0., 6.), rounds=2, noise_params=None,
        )
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_logical_h(self):
        from lightstim.protocols.fold_transversal import build_gate_verification_circuit
        c = build_quiet(lambda: build_gate_verification_circuit(
            distance=3, gates=["fold_transversal_hadamard"],
            init_basis="Z", measure_basis="X", rounds=2, unencode=False, noise_params=None,
        ))
        assert c.num_detectors > 0; assert_noiseless(c); assert_dem_valid(c)

    def test_logical_s_roundtrip(self):
        from lightstim.protocols.fold_transversal import build_s_roundtrip_circuit
        c = build_quiet(lambda: build_s_roundtrip_circuit(distance=3, rounds=2, noise_params=None))
        assert c.num_detectors > 0; assert_noiseless(c); assert_dem_valid(c)

    def test_ghz(self):
        from lightstim.protocols.ghz import GHZExperiment
        exp = GHZExperiment(distance=3, rounds_before=2, rounds_after=2, noise_params=None)
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    @pytest.mark.parametrize("state", ["Z", "X", "Y"])
    def test_state_injection(self, state):
        from lightstim.protocols.state_injection import StateInjectionExperiment
        exp = StateInjectionExperiment(distance=3, rounds=2, inject_state=state, noise_params=None)
        c = build_quiet(exp.build)
        assert c.num_detectors > 0; assert_noiseless(c); assert_dem_valid(c)

    def test_cross_ls(self):
        from lightstim.protocols.cross_ls import CrossLSExperiment
        exp = CrossLSExperiment(PQRM_para=[1, 2, 4], d_surf=3, rounds=2, noise_params=None)
        c = build_quiet(exp.build)
        assert c.num_detectors > 0; assert_noiseless(c)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGICAL CIRCUITS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestLogicalCircuits:

    @pytest.mark.parametrize("variant,state", [("tg", "Z"), ("zz_ls", "Z"), ("xx_ls", "X")])
    def test_bell_teleport(self, variant, state):
        from lightstim.protocols.bell_teleportation import BellTeleportTG, BellTeleportZZLS, BellTeleportXXLS
        cls = {"tg": BellTeleportTG, "zz_ls": BellTeleportZZLS, "xx_ls": BellTeleportXXLS}[variant]
        if variant == "tg":
            exp = cls(distance=3, rounds_pre=2, rounds_mid=1, rounds_post=1,
                      teleport_state=state, noise_params=None)
        else:
            exp = cls(distance=3, rounds_pre=2, rounds_ls=2,
                      teleport_state=state, noise_params=None)
        c = build_quiet(exp.build)
        assert c.num_detectors > 0; assert_noiseless(c); assert_dem_valid(c)

    def test_tg_distillation_build(self):
        """Noiseless circuit structure only — noise injection tested separately."""
        from lightstim.protocols.tg_distillation import build_distillation_circuit
        circuit, info, _ = build_quiet(lambda: build_distillation_circuit(d=3, rounds_init=3, rounds_gate=1))
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
        assert info["num_detectors"] == circuit.num_detectors

    def test_ls_distillation_build(self):
        from lightstim.protocols.ls_distillation import build_distillation_circuit
        circuit, info, _ = build_quiet(lambda: build_distillation_circuit(d=3, rounds=3))
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
