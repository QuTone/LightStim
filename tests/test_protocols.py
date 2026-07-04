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

    # ── XZZX surface code: per-qubit checkerboard basis via data_basis_map ──

    def _build_xzzx(self, d=3, basis="Z", noise_params=None, map_edit=None, **exp_kwargs):
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.memory import MemoryExperiment
        from lightstim.qec_code.surface_code.xzzx import (
            XZZXSurfaceCode, XZZXSurfaceCodeExtractionBlock, xzzx_memory_basis)
        system = QECSystem()
        system.add_patch(XZZXSurfaceCode(distance=d), name="xzzx_sc")
        basis_map = xzzx_memory_basis(system, basis)
        if map_edit is not None:
            basis_map = map_edit(basis_map)
        exp = MemoryExperiment(qec_system=system,
                               extraction_block_class=XZZXSurfaceCodeExtractionBlock,
                               rounds=d, noise_params=noise_params, noise_model="circuit_level",
                               basis=basis, data_basis_map=basis_map, **exp_kwargs)
        return build_quiet(exp.build)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_xzzx_surface_code(self, basis):
        c = self._build_xzzx(basis=basis)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_xzzx_checkerboard_recovers_distance_3(self, basis):
        """Uniform basis on XZZX is degenerate (distance 1); the checkerboard map must restore d=3.
        Covers both memory bases: a flip-dict bug in the X map would pass the noiseless
        smoke test (a degenerate circuit also has zero noiseless detection events)."""
        from lightstim.noise.config import NoiseConfig
        p = 1e-3
        noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
        c = self._build_xzzx(basis=basis, noise_params=noise)
        assert c.num_detectors == 24, f"expected rotated-d3-like detector structure, got {c.num_detectors}"
        assert len(c.shortest_graphlike_error()) == 3

    def test_xzzx_basis_map_z_only_rejected(self):
        """z_only readout path assumes uniform Z measurement — must refuse a mixed basis map."""
        with pytest.raises(ValueError, match="z_only"):
            self._build_xzzx(z_only=True)

    def test_xzzx_basis_map_wrong_keys_rejected(self):
        """A map not covering exactly the data qubits (e.g. local patch indices) must fail fast."""
        def drop_one(m):
            m.pop(next(iter(m)))
            return m
        with pytest.raises(ValueError, match="data_basis_map"):
            self._build_xzzx(map_edit=drop_one)

    def test_xzzx_basis_map_invalid_values_rejected(self):
        """Values outside X/Y/Z must be rejected at construction — builder.initialize()
        would otherwise silently drop the qubit from every reset group."""
        def corrupt(m):
            return {q: "Q" for q in m}
        with pytest.raises(ValueError, match="values must be"):
            self._build_xzzx(map_edit=corrupt)

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

    def test_two_patch_system_has_two_observables(self):
        """Two independent patches without coupler must have num_observables == 2."""
        from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.memory import MemoryExperiment
        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p1")
        system.add_patch(UnrotatedSurfaceCode(distance=3), offset=(8, 0), name="p2")
        exp = MemoryExperiment(qec_system=system,
                               extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                               rounds=3, noise_params=None, noise_model="circuit_level", basis="Z")
        c = build_quiet(exp.build)
        assert c.num_observables == 2, f"two independent patches must have 2 logical qubits, got {c.num_observables}"
        assert_noiseless(c); assert_dem_valid(c)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGICAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestLogicalOps:

    @staticmethod
    def _expected_terminal_block(patch, side, width):
        x0, x1, y0, y1 = map(lambda v: int(round(v)), patch._get_bounds())
        if side == "left":
            xs = range(x0 - width, x0)
            ys = range(y0, y1 + 1)
        elif side == "right":
            xs = range(x1 + 1, x1 + 1 + width)
            ys = range(y0, y1 + 1)
        elif side == "top":
            xs = range(x0, x1 + 1)
            ys = range(y0 - width, y0)
        elif side == "bottom":
            xs = range(x0, x1 + 1)
            ys = range(y1 + 1, y1 + 1 + width)
        else:
            raise ValueError(side)
        return {(float(x), float(y)) for x in xs for y in ys}

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

    def test_routed_multi_patch_coupler(self):
        """Bent routed coupler can connect explicitly selected patch sides."""
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedRoutedMultiPatchCoupler)
        from lightstim.ir.qec_system import QECSystem

        system = QECSystem()
        layout = {
            "p1": (0, 0),
            "p2": (12, 0),
            "p3": (0, 12),
            "p4": (12, 12),
        }
        for name, off in layout.items():
            system.add_patch(UnrotatedSurfaceCode(distance=3), name=name, offset=off)

        system.register_coupler(
            UnrotatedRoutedMultiPatchCoupler(),
            ["p1", "p2", "p3", "p4"],
            "route",
            sides=["right", "left", "right", "left"],
            route_padding=8,
        )
        cp = system.coupler_patches["route"]
        assert cp.route_width == 5
        assert len(cp.data_indices) > 0
        assert len(cp.syndrome_indices) > 0
        assert len(cp.conflicting_stabilizer_coords) > 0
        for name, side in zip(["p1", "p2", "p3", "p4"], ["right", "left", "right", "left"]):
            expected = self._expected_terminal_block(system.patches[name][0], side, cp.route_width)
            assert expected.issubset(set(cp.index_map.keys()))

    def test_routed_full_width_requires_coarse_patch_grid(self):
        """Full ancillary-patch routing rejects data patches between patch-block cells."""
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedRoutedMultiPatchCoupler)
        from lightstim.ir.qec_system import QECSystem

        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p1", offset=(0, 0))
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p2", offset=(14, 8))

        with pytest.raises(ValueError, match="coarse grid"):
            system.register_coupler(
                UnrotatedRoutedMultiPatchCoupler(),
                ["p1", "p2"],
                "bad_route",
                sides=["right", "top"],
                route_width=5,
            )

    def test_routed_zzzx_pauli_product(self):
        """ZZZX is implemented as H on the X patch, routed ZZZZ, then H back."""
        from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.routed_multi_patch_ls import (
            build_routed_pauli_product_readout_circuit)

        system = QECSystem()
        layout = {
            "p1": (0, 0),
            "p2": (12, 0),
            "p3": (0, 12),
            "p4": (12, 12),
        }
        for name, off in layout.items():
            system.add_patch(UnrotatedSurfaceCode(distance=3), name=name, offset=off)

        circuit, info, _ = build_quiet(lambda: build_routed_pauli_product_readout_circuit(
            system=system,
            patch_names=["p1", "p2", "p3", "p4"],
            paulis="ZZZX",
            sides=["right", "left", "right", "left"],
            interface_paulis=["Z", "Z", "Z", "Z"],
            rounds=2,
            coupler_name="route",
            route_padding=8,
        ))
        assert info["h_basis_change_indices"] == [3]
        assert_valid_circuit(circuit); assert_noiseless(circuit); assert_dem_valid(circuit)

    def test_routed_basis_aware_h_decision(self):
        """H decisions compare requested Paulis against the selected interface basis."""
        from lightstim.protocols.routed_multi_patch_ls import (
            basis_change_indices_for_interfaces)

        assert basis_change_indices_for_interfaces("XZ", ["X", "Z"]) == []
        assert basis_change_indices_for_interfaces("XZ", ["Z", "Z"]) == [0]
        assert basis_change_indices_for_interfaces("ZZZX", ["Z", "Z", "Z", "Z"]) == [3]
        assert basis_change_indices_for_interfaces("ZZZX", ["Z", "Z", "Z", "X"]) == []

    def test_experimental_mixed_routed_templates(self):
        """Mixed routed coupler creates local X/Z seam templates without assuming weight 4."""
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedRoutedMultiPatchCoupler,
            UnrotatedSurfaceCodeExtractionBlock)
        from lightstim.ir.qec_system import QECSystem

        def commute(a, b):
            parity = 0
            for q, p in a["pauli"].items():
                other = b["pauli"].get(q)
                if other is not None and other != p:
                    parity ^= 1
            return parity == 0

        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p1", offset=(0, 0))
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p2", offset=(12, 12))
        system.register_coupler(
            UnrotatedRoutedMultiPatchCoupler(),
            ["p1", "p2"],
            "xz",
            sides=["right", "top"],
            interface_paulis=["X", "Z"],
            target_paulis=["X", "Z"],
            mixed_stabilizers=True,
            route_padding=8,
        )
        cp = system.coupler_patches["xz"]
        assert cp.route_width == 5
        assert any(stab["type"] == "MIXED" for stab in cp.stabilizers)
        assert any(len(stab["pauli"]) == 3 for stab in cp.stabilizers)
        for name, side in zip(["p1", "p2"], ["right", "top"]):
            expected = self._expected_terminal_block(system.patches[name][0], side, cp.route_width)
            assert expected.issubset(set(cp.index_map.keys()))

        system.activate_coupler("xz")
        se_ticks = sum(
            1 for inst in UnrotatedSurfaceCodeExtractionBlock(system).circuit
            if inst.name == "TICK"
        )
        assert se_ticks <= 20
        stabs = system.active_stabilizers
        for i, a in enumerate(stabs):
            for b in stabs[i + 1:]:
                assert commute(a, b)

    def test_mixed_xz_syndrome_product_extractor(self):
        """X1Z2 is recovered by solving routed syndrome/readout algebra."""
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedRoutedMultiPatchCoupler)
        from lightstim.ir.qec_system import QECSystem
        from lightstim.protocols.routed_multi_patch_ls import (
            solve_routed_pauli_product_syndromes,
        )

        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p1", offset=(0, 0))
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p2", offset=(12, 12))
        system.register_coupler(
            UnrotatedRoutedMultiPatchCoupler(),
            ["p1", "p2"],
            "xz",
            sides=["right", "top"],
            interface_paulis=["X", "Z"],
            target_paulis=["X", "Z"],
            mixed_stabilizers=True,
            route_padding=8,
        )

        with pytest.raises(ValueError):
            solve_routed_pauli_product_syndromes(
                system=system,
                patch_names=["p1", "p2"],
                paulis="XZ",
                coupler_name="xz",
                include_patch_stabilizers=False,
                include_ancilla_readout_terms=False,
            )
        direct_decomp = solve_routed_pauli_product_syndromes(
            system=system,
            patch_names=["p1", "p2"],
            paulis="XZ",
            coupler_name="xz",
            include_patch_stabilizers=True,
            include_ancilla_readout_terms=False,
        )
        assert direct_decomp.verified
        assert not direct_decomp.selected_ancilla_terms

        assert direct_decomp.target_paulis == ["X", "Z"]
        assert direct_decomp.selected_coupler_terms
        assert direct_decomp.selected_patch_terms
        assert any(term.stype == "MIXED" for term in direct_decomp.selected_coupler_terms)
        assert all(term.rec_offset < 0 for term in direct_decomp.selected_terms)
        assert "xz" not in system.paused_stabilizer_indices

    def test_native_mixed_routed_boundary_templates(self):
        """Mixed routed templates must not recolor mismatched data-patch boundary checks."""
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedRoutedMultiPatchCoupler,
            UnrotatedSurfaceCodeExtractionBlock)
        from lightstim.ir.qec_system import QECSystem
        from lightstim.ir.tracker import SyndromeTracker
        from lightstim.ir.builder import CircuitBuilder
        from lightstim.protocols.routed_multi_patch_ls import (
            infer_interface_paulis,
            routed_coupler_data_basis,
            solve_routed_pauli_product_long_ancilla,
            solve_routed_pauli_product_merge_checks,
            solve_routed_pauli_product_syndromes,
        )

        d = 3
        patch_names = ["p1", "p2", "p3", "p4"]
        sides = ["bottom", "top", "top", "left"]
        system = QECSystem()
        for name, off in {
            "p1": (0, 0),
            "p2": (24, 0),
            "p3": (12, 24),
            "p4": (36, 24),
        }.items():
            system.add_patch(UnrotatedSurfaceCode(distance=d), name=name, offset=off)

        native_interfaces = infer_interface_paulis(system, patch_names, sides)
        system.register_coupler(
            UnrotatedRoutedMultiPatchCoupler(),
            patch_names,
            "mixed_geom",
            sides=sides,
            interface_paulis=native_interfaces,
            target_paulis=list("ZZZX"),
            mixed_stabilizers=True,
            route_padding=8,
            route_width=2 * d - 1,
        )

        prep_basis = routed_coupler_data_basis(system, "mixed_geom", mode="opposite")

        p4 = system.patches["p4"][0]
        p4_left_x = p4._get_bounds()[0]
        p4_left_boundary_syndromes = {
            coord for coord in p4.syndrome_coords
            if coord[0] == p4_left_x
        }
        assert p4_left_boundary_syndromes == {(36, 25), (36, 27)}
        p4_boundary_templates = [
            stab for stab in system.coupler_patches["mixed_geom"].stabilizers
            if stab["syn_coord"] in p4_left_boundary_syndromes
        ]
        assert len(p4_boundary_templates) == 2
        assert all(stab["type"] == "X" for stab in p4_boundary_templates)
        assert all(len(stab["pauli"]) == 4 for stab in p4_boundary_templates)

        with pytest.raises(ValueError):
            solve_routed_pauli_product_syndromes(
                system=system,
                patch_names=patch_names,
                paulis="ZZZX",
                coupler_name="mixed_geom",
                include_ancilla_readout_terms=False,
            )
        merge_decomp = solve_routed_pauli_product_merge_checks(
            system=system,
            patch_names=patch_names,
            paulis="ZZZX",
            coupler_name="mixed_geom",
        )
        assert not merge_decomp.verified
        assert merge_decomp.selected_merge_terms
        assert merge_decomp.selection_rule == "basis_matched_geometry_no_trim"
        assert not merge_decomp.trimmed_boundary_terms
        assert merge_decomp.residual_terms
        selected_records = [
            system.stabilizers[term.stabilizer_uid]
            for term in merge_decomp.selected_merge_terms
            if term.patch_name == "mixed_geom"
        ]
        route_basis = system.coupler_patches["mixed_geom"].route_coord_basis
        for stab in selected_records:
            if stab["type"] in ("X", "Z"):
                assert route_basis[stab["syn_coord"]] == stab["type"]
        assert len(merge_decomp.patch_correction_terms) == 6
        assert all(term.weight == term.base_weight for term in merge_decomp.selected_merge_terms)

        paper_system = QECSystem()
        for name, off in {
            "p1": (0, 0),
            "p2": (24, 0),
            "p3": (12, 24),
            "p4": (36, 24),
        }.items():
            paper_system.add_patch(UnrotatedSurfaceCode(distance=d), name=name, offset=off)
        paper_interfaces = infer_interface_paulis(paper_system, patch_names, sides)
        paper_system.register_coupler(
            UnrotatedRoutedMultiPatchCoupler(),
            patch_names,
            "paper_long_ancilla",
            sides=sides,
            interface_paulis=paper_interfaces,
            target_paulis=list("ZZZX"),
            mixed_stabilizers=True,
            route_padding=8,
            route_width=2 * d - 1,
        )
        long_ancilla_decomp = solve_routed_pauli_product_long_ancilla(
            system=paper_system,
            patch_names=patch_names,
            paulis="ZZZX",
            coupler_name="paper_long_ancilla",
        )
        assert long_ancilla_decomp.verified
        assert len(long_ancilla_decomp.selected_coupler_terms) > 50
        assert len(long_ancilla_decomp.selected_patch_terms) == 6
        assert any(term.stype == "MIXED" for term in long_ancilla_decomp.selected_coupler_terms)
        assert len(long_ancilla_decomp.selected_ancilla_logical_terms) == 1
        logical_factor = long_ancilla_decomp.selected_ancilla_logical_terms[0]
        assert logical_factor.pauli == "MIXED"
        assert len(logical_factor.support_terms) == 69
        assert {
            term.pauli for term in logical_factor.support_terms
        } == {"X", "Z"}
        assert all(
            paper_system.index_to_owner_map[term.global_qubit_index] == "paper_long_ancilla"
            for term in logical_factor.support_terms
        )

        tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=False)
        builder.write_coordinates()
        data_prep = {
            q: "Z"
            for q in system.data_indices
            if system.index_to_owner_map.get(q) != "mixed_geom"
        }
        builder.initialize(data_prep, n=system.num_qubits)
        builder.apply_syndrome_extraction(
            UnrotatedSurfaceCodeExtractionBlock(system).circuit,
            rounds=1,
        )
        builder.activate_coupler("mixed_geom")
        builder.initialize(prep_basis, n=system.num_qubits)
        builder.apply_syndrome_extraction(
            UnrotatedSurfaceCodeExtractionBlock(system).circuit,
            rounds=1,
        )
        builder.apply_data_readout(data_prep)

        circuit = builder.circuit
        assert circuit.num_qubits > 0
        assert circuit.num_measurements > 0
        assert_dem_valid(circuit)

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

    @pytest.mark.parametrize("init_meas", [("Z", "X"), ("X", "Z")])
    def test_logical_h_rotated(self, init_meas):
        """Rotated logical H via transversal-H + 90-degree rotation (X_L <-> Z_L)."""
        from lightstim.protocols.fold_transversal import build_gate_verification_circuit
        from lightstim.qec_code.surface_code.rotated import (
            RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
        )
        ib, mb = init_meas
        c = build_quiet(lambda: build_gate_verification_circuit(
            distance=3, gates=["fold_transversal_hadamard"],
            init_basis=ib, measure_basis=mb, rounds=2, unencode=False, noise_params=None,
            code_patch_class=RotatedSurfaceCode,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        ))
        assert c.num_detectors > 0; assert_noiseless(c); assert_dem_valid(c)

    def test_logical_s_roundtrip(self):
        from lightstim.protocols.fold_transversal import build_s_roundtrip_circuit
        c = build_quiet(lambda: build_s_roundtrip_circuit(distance=3, rounds=2, noise_params=None))
        assert c.num_detectors > 0; assert_noiseless(c); assert_dem_valid(c)

    def test_logical_s_to_the_fourth_is_identity(self):
        """S⁴ = I: applying S four times must return to initial state (mathematical invariant)."""
        from lightstim.protocols.fold_transversal import build_gate_verification_circuit
        c = build_quiet(lambda: build_gate_verification_circuit(
            distance=3,
            gates=["fold_transversal_s"] * 4,  # S^4 = I
            init_basis="X", measure_basis="X",
            rounds=2, unencode=False, noise_params=None,
        ))
        assert c.num_detectors > 0
        assert_noiseless(c)  # S^4|+> = |+>, so X-basis measurement must always agree

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

    @pytest.mark.parametrize("method,state_prep", [
        ("ZZ", "logical_gate"),
        ("ZZ", "inject"),
        ("cnot_trans", "logical_gate"),
        ("cnot_trans", "inject"),
    ])
    def test_s_gate_teleport_unrotated(self, method, state_prep):
        from lightstim.protocols.gate_teleport import SGateTeleportExperiment
        exp = SGateTeleportExperiment(
            distance=3,
            code="unrotated_sc",
            method=method,
            state_prep=state_prep,
            rounds_prep=2,
            rounds_gate=2,
            noise_params=None,
        )
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    @pytest.mark.parametrize("method", ["ZZ", "cnot_trans"])
    def test_s_gate_teleport_rotated_injection(self, method):
        from lightstim.protocols.gate_teleport import SGateTeleportExperiment
        exp = SGateTeleportExperiment(
            distance=3,
            code="rotated_sc",
            method=method,
            state_prep="inject",
            rounds_prep=2,
            rounds_gate=2,
            noise_params=None,
        )
        c = build_quiet(exp.build)
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    def test_s_gate_teleport_rotated_logical_gate_rejected(self):
        from lightstim.protocols.gate_teleport import SGateTeleportExperiment
        with pytest.raises(ValueError, match="logical S support"):
            SGateTeleportExperiment(
                distance=3,
                code="rotated_sc",
                method="ZZ",
                state_prep="logical_gate",
            )

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
