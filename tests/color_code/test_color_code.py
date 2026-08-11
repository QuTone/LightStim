"""Tests for Color Code geometry, extraction blocks, and memory circuits."""

import math
import pytest
import numpy as np
import stim
from pathlib import Path

from lightstim.qec_code.color_code import (
    ColorCode,
    ColorCodeTile,
    ColorCodeBellFlaggingBlock,
    ColorCodeBellMultiplexingBlock,
    ColorCodeExtractionBlock,
    ColorCodeMiddleOutBlock,
    ColorCodeSpaceMultiplexingBlock,
    ColorCodeTimeMultiplexingBlock,
    RectangleMiddleOutPlan,
)
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.utils.linear_algebra import row_echelon, solve_linear_decomposition


# ---- Geometry Tests ----

class TestColorCodeGeometry:
    """Verify qubit placement, face counts, and coordinate properties."""

    @pytest.mark.parametrize("d, expected_data, expected_faces", [
        (3, 7, 3),
        (5, 19, 9),
        (7, 37, 18),
    ])
    def test_qubit_counts(self, d, expected_data, expected_faces):
        code = ColorCode(distance=d)
        assert len(code.data_indices) == expected_data
        assert len(code.faces) == expected_faces
        assert len(code.syndrome_indices_x) == expected_faces
        assert len(code.syndrome_indices_z) == expected_faces
        # Total qubits = data + 2 * faces (X and Z ancillas)
        assert code.num_qubits == expected_data + 2 * expected_faces

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_operator_weight(self, d):
        code = ColorCode(distance=d)
        for lo in code.logical_ops:
            assert len(lo['data_indices']) == d

    def test_invalid_distance(self):
        with pytest.raises(ValueError):
            ColorCode(distance=2)  # even
        with pytest.raises(ValueError):
            ColorCode(distance=1)  # too small

    def test_invalid_layout(self):
        with pytest.raises(ValueError, match="layout"):
            ColorCode(distance=3, layout="unknown")

    @pytest.mark.parametrize("layout, n_qubits, n_syndrome, n_syndrome_x, n_syndrome_z", [
        ("superdense", 37, 18, 9, 9),
        ("raw", 28, 9, 9, 9),
        ("triangular", 28, 9, 9, 9),
        ("rectangle", 23, 0, 0, 0),
    ])
    def test_layout_qubit_roles(self, layout, n_qubits, n_syndrome, n_syndrome_x, n_syndrome_z):
        code = ColorCode(distance=5, layout=layout)
        assert len(code.data_indices) == n_qubits - n_syndrome
        assert len(code.syndrome_indices) == n_syndrome
        assert len(code.syndrome_indices_x) == n_syndrome_x
        assert len(code.syndrome_indices_z) == n_syndrome_z
        assert code.num_qubits == n_qubits

    def test_raw_layout_uses_row_position_coordinates(self):
        code = ColorCode(distance=5, layout="raw")
        expected = {
            (position, row)
            for row in range(7)
            for position in range(7 - row)
        }
        assert set(code.qubit_coords.values()) == expected

    def test_triangular_layout_uses_hex_lattice_coordinates(self):
        code = ColorCode(distance=5, layout="triangular")
        expected = {
            (round(position + row / 2, 6), round(row * math.sqrt(3) / 2, 6))
            for row in range(7)
            for position in range(7 - row)
        }
        assert set(code.qubit_coords.values()) == expected

    @pytest.mark.parametrize("layout", ["raw", "triangular"])
    def test_lattice_matches_patch_coords(self, layout):
        code = ColorCode(distance=5, layout=layout)
        lattice = code.lattice
        assert lattice is not None
        assert set(lattice.data_coords) == {
            code.qubit_coords[i] for i in code.data_indices
        }
        assert {face.center for face in lattice.faces} == {
            code.qubit_coords[i] for i in code.syndrome_indices
        }

    @pytest.mark.parametrize("layout", ["superdense", "raw", "triangular"])
    def test_layout_neighbor_weights(self, layout):
        code = ColorCode(distance=5, layout=layout)
        weights = {sum(n is not None for n in face['data_neighbors']) for face in code.faces}
        assert weights == {4, 6}

    @pytest.mark.parametrize("layout, expected_tiles", [
        ("superdense", 18),
        ("raw", 18),
        ("triangular", 18),
        ("rectangle", 11),
    ])
    def test_layout_tiles_are_exposed(self, layout, expected_tiles):
        code = ColorCode(distance=5, layout=layout)
        assert all(isinstance(tile, ColorCodeTile) for tile in code.tiles)
        assert len(code.tiles) == expected_tiles
        assert {tile.color for tile in code.tiles} == {"r", "g", "b"}

    @pytest.mark.parametrize("layout", ["superdense", "raw", "triangular"])
    def test_face_tiles_pair_x_and_z_checks(self, layout):
        code = ColorCode(distance=5, layout=layout)
        assert len(code.faces) == 9
        for face in code.faces:
            x_tile = face["x_tile"]
            z_tile = face["z_tile"]
            assert x_tile.basis == "X"
            assert z_tile.basis == "Z"
            assert x_tile.ordered_data_coords == z_tile.ordered_data_coords
            assert {x_tile.weight, z_tile.weight} <= {4, 6}
            if layout in {"raw", "triangular"}:
                assert x_tile.measurement_coord == z_tile.measurement_coord
                assert face["x_ancilla_idx"] == face["z_ancilla_idx"]
            else:
                assert x_tile.measurement_coord != z_tile.measurement_coord
                assert face["x_ancilla_idx"] != face["z_ancilla_idx"]

    def test_rectangle_tiles_are_single_rgb_layer(self):
        code = ColorCode(distance=5, layout="rectangle")
        assert len(code.faces) == 11
        assert {tile.basis for tile in code.tiles} == {None}
        assert {tile.color for tile in code.tiles} == {"r", "g", "b"}
        assert {tile.weight for tile in code.tiles} == {2, 4, 6}
        assert len(code.stabilizers) == 22
        assert {stabilizer["type"] for stabilizer in code.stabilizers} == {"X", "Z"}
        assert all(stabilizer["syn_idx"] is None for stabilizer in code.stabilizers)
        assert code.syndrome_indices == set()
        assert code.num_logicals == 1

    @pytest.mark.parametrize("distance, sample_name", [
        (5, "midout_color_code_d5_r10_p1000.stim"),
        (9, "midout_color_code_d9_r36_p1000.stim"),
    ])
    def test_rectangle_layout_matches_midout_samples(self, distance, sample_name):
        code = ColorCode(distance=distance, layout="rectangle")
        sample_coords = _shift_coords(_sample_qubit_coords(sample_name), dy=-1)
        assert code.qubit_coords == sample_coords

    @pytest.mark.parametrize("distance", [3, 5, 7, 9])
    def test_rectangle_layout_builds_odd_distances(self, distance):
        code = ColorCode(distance=distance, layout="rectangle")
        plan = RectangleMiddleOutPlan.from_patch(code)
        assert code.num_qubits > 0
        assert len(code.syndrome_indices) == 0
        assert len(plan.cnot_layers) == 6
        assert not hasattr(code, "midout_plan")
        assert code.num_logicals == 1
        assert len(code.stabilizers) == 2 * len(code.tiles)
        assert len(code.logical_ops) == 2

    @pytest.mark.parametrize("distance, first_half_x_coords, first_half_z_coords", [
        (
            3,
            [(1, 1), (2, 3)],
            [(1, 2), (1, 4)],
        ),
        (
            7,
            [(1, 1), (1, 5), (3, 1), (3, 3), (3, 5), (3, 7), (4, 9),
             (5, 1), (5, 3), (5, 5), (6, 3)],
            [(1, 2), (1, 4), (1, 6), (3, 2), (3, 4), (3, 6), (3, 8),
             (3, 10), (5, 2), (5, 4)],
        ),
        (
            11,
            [(1, 1), (1, 5), (3, 1), (3, 3), (3, 5), (3, 7), (3, 11),
             (5, 1), (5, 3), (5, 5), (5, 7), (5, 9), (5, 11), (5, 13),
             (6, 15), (7, 1), (7, 3), (7, 5), (7, 7), (7, 9), (7, 11),
             (8, 9), (9, 1), (9, 3), (9, 5), (10, 3)],
            [(1, 2), (1, 4), (1, 6), (3, 2), (3, 4), (3, 6), (3, 8),
             (3, 10), (3, 12), (5, 2), (5, 4), (5, 6), (5, 8), (5, 10),
             (5, 12), (5, 14), (5, 16), (7, 2), (7, 4), (7, 6), (7, 8),
             (7, 10), (9, 2), (9, 4)],
        ),
    ])
    def test_rectangle_middle_out_first_half_measure_coords(
        self,
        distance,
        first_half_x_coords,
        first_half_z_coords,
    ):
        code = ColorCode(distance=distance, layout="rectangle")
        plan = RectangleMiddleOutPlan.from_patch(code)
        assert [code.qubit_coords[i] for i in plan.x_measure_indices] == [
            (x, y - 1) for x, y in first_half_x_coords
        ]
        assert [code.qubit_coords[i] for i in plan.z_measure_indices] == [
            (x, y - 1) for x, y in first_half_z_coords
        ]

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_three_face_colors(self, d):
        code = ColorCode(distance=d)
        colors = {face['color'] for face in code.faces}
        # All three colors should be present for d >= 5; d=3 may have fewer
        if d >= 5:
            assert colors == {'r', 'g', 'b'}


# ---- Algebraic Tests ----

class TestColorCodeAlgebra:
    """Verify CSS properties: commutativity, anti-commutativity, independence."""

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_stabilizer_commutativity(self, d):
        """All stabilizers must commute pairwise."""
        code = ColorCode(distance=d)
        n = code.num_qubits
        vecs = self._get_symplectic_vectors(code.stabilizers, n)

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sip = self._symplectic_inner_product(vecs[i], vecs[j])
                assert sip == 0, f"Stabilizers {i} and {j} anti-commute"

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_anti_commutativity(self, d):
        """X_L and Z_L must anti-commute."""
        code = ColorCode(distance=d)
        n = code.num_qubits
        log_vecs = self._get_symplectic_vectors(code.logical_ops, n)
        assert len(log_vecs) == 2

        sip = self._symplectic_inner_product(log_vecs[0], log_vecs[1])
        assert sip == 1, "Logical X and Z must anti-commute"

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logicals_commute_with_stabilizers(self, d):
        """Each logical must commute with all stabilizers."""
        code = ColorCode(distance=d)
        n = code.num_qubits
        stab_vecs = self._get_symplectic_vectors(code.stabilizers, n)
        log_vecs = self._get_symplectic_vectors(code.logical_ops, n)

        for li, lv in enumerate(log_vecs):
            for si, sv in enumerate(stab_vecs):
                sip = self._symplectic_inner_product(lv, sv)
                assert sip == 0, f"Logical {li} anti-commutes with stabilizer {si}"

    @pytest.mark.parametrize("d", [3, 5, 7, 9])
    def test_rectangle_midpoint_code_has_one_logical_qubit(self, d):
        code = ColorCode(distance=d, layout="rectangle")
        hx, hz = code.get_parity_check_matrix()
        n = code.num_qubits

        assert hx.shape == hz.shape == (len(code.tiles), n)
        assert np.array_equal(hx, hz)
        assert row_echelon(hx)[1] == len(code.tiles)
        assert n - row_echelon(hx)[1] - row_echelon(hz)[1] == 1

        stabilizers = self._get_symplectic_vectors(code.stabilizers, n)
        logicals = self._get_symplectic_vectors(code.logical_ops, n)
        assert len(logicals) == 2
        assert self._symplectic_inner_product(logicals[0], logicals[1]) == 1
        assert all(
            self._symplectic_inner_product(logical, stabilizer) == 0
            for logical in logicals
            for stabilizer in stabilizers
        )

    # Helpers
    @staticmethod
    def _get_symplectic_vectors(ops, n):
        vecs = []
        for op in ops:
            x_vec = np.zeros(n, dtype=int)
            z_vec = np.zeros(n, dtype=int)
            for idx, pauli_type in op['pauli'].items():
                if pauli_type in ('X', 'Y'):
                    x_vec[idx] = 1
                if pauli_type in ('Z', 'Y'):
                    z_vec[idx] = 1
            vecs.append((x_vec, z_vec))
        return vecs

    @staticmethod
    def _symplectic_inner_product(v1, v2):
        x1, z1 = v1
        x2, z2 = v2
        return (x1 @ z2 + z1 @ x2) % 2


# ---- SE Block Tests ----

class TestColorCodeSEBlock:
    """Verify syndrome extraction circuit properties."""

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_no_cnot_collisions(self, d):
        """No qubit should appear twice in the same CNOT timeslice."""
        code = ColorCode(distance=d)

        for tick_idx, (z_pos, x_pos) in enumerate(ColorCodeExtractionBlock.SCHEDULE):
            used = set()

            if z_pos is not None:
                for face in code.faces:
                    neighbor = face['data_neighbors'][z_pos]
                    if neighbor is not None:
                        data_coord, data_idx = neighbor
                        if data_idx in code.data_indices:
                            assert data_idx not in used, f"Collision at tick {tick_idx}"
                            assert face['z_ancilla_idx'] not in used, f"Collision at tick {tick_idx}"
                            used.add(data_idx)
                            used.add(face['z_ancilla_idx'])

            if x_pos is not None:
                for face in code.faces:
                    neighbor = face['data_neighbors'][x_pos]
                    if neighbor is not None:
                        data_coord, data_idx = neighbor
                        if data_idx in code.data_indices:
                            assert data_idx not in used, f"Collision at tick {tick_idx}"
                            assert face['x_ancilla_idx'] not in used, f"Collision at tick {tick_idx}"
                            used.add(data_idx)
                            used.add(face['x_ancilla_idx'])

    def test_time_multiplexing_separates_reset_and_first_cnot(self):
        block = _build_block("triangular", ColorCodeTimeMultiplexingBlock)
        names = [inst.name for inst in block.circuit[:4]]
        assert names == ["RX", "TICK", "CX", "TICK"]

    @pytest.mark.parametrize("distance", [3, 5, 7, 9])
    def test_time_multiplexing_builds_without_cnot_collisions(self, distance):
        block = _build_block("triangular", ColorCodeTimeMultiplexingBlock, distance=distance)
        assert _cnot_collision_layers(block.circuit) == []

    def test_time_multiplexing_blocks_measure_x_and_z_faces(self):
        code = ColorCode(distance=5, layout="triangular")
        system = QECSystem()
        system.add_patch(code, name="color")
        block = ColorCodeTimeMultiplexingBlock(system)
        n = code.num_qubits

        assert len(block.measurement_blocks) == 2
        assert block.measurement_blocks[0] + block.measurement_blocks[1] == block.circuit
        assert [part.num_measurements for part in block.measurement_blocks] == [9, 9]

        for measurement_block, basis in zip(block.measurement_blocks, ("X", "Z")):
            back_paulis, measurement_indices, measurement_bases = (
                CircuitBuilder._get_back_propagated_pauli(
                    measurement_block,
                    n,
                    include_measurement_bases=True,
                )
            )
            basis_slice = slice(0, n) if basis == "X" else slice(n, 2 * n)
            other_slice = slice(n, 2 * n) if basis == "X" else slice(0, n)
            supports = {
                frozenset(np.flatnonzero(row[basis_slice]))
                for row in back_paulis
            }
            expected = {
                frozenset(stabilizer["pauli"])
                for stabilizer in code.stabilizers
                if stabilizer["type"] == basis
            }

            assert supports == expected
            assert not np.any(back_paulis[:, other_slice])
            assert measurement_indices == sorted(
                system.active_syndrome_indices
            )
            assert measurement_bases == [basis] * len(code.faces)

    def test_bell_multiplexing_cnot_tick_count(self):
        block = _build_block("superdense", ColorCodeBellMultiplexingBlock)
        assert len(_cnot_layers(block.circuit)) == 8
        assert _cnot_collision_layers(block.circuit) == []

    def test_bell_multiplexing_backprop_removes_explicit_reset_qubits(self):
        code = ColorCode(distance=5, layout="superdense")
        system = QECSystem()
        system.add_patch(code, name="color")
        block = ColorCodeBellMultiplexingBlock(system)

        back_paulis, _, _ = CircuitBuilder._get_back_propagated_pauli(
            block.circuit,
            code.num_qubits,
            include_measurement_bases=True,
        )

        syndrome_indices = sorted(system.active_syndrome_indices)
        n = code.num_qubits
        assert not np.any(back_paulis[:, syndrome_indices])
        assert not np.any(back_paulis[:, n + np.asarray(syndrome_indices)])

        num_faces = len(code.faces)
        expected_x = {
            frozenset(stabilizer["pauli"])
            for stabilizer in code.stabilizers
            if stabilizer["type"] == "X"
        }
        expected_z = {
            frozenset(stabilizer["pauli"])
            for stabilizer in code.stabilizers
            if stabilizer["type"] == "Z"
        }
        actual_x = {
            frozenset(np.flatnonzero(row[:n]))
            for row in back_paulis[:num_faces]
        }
        actual_z = {
            frozenset(np.flatnonzero(row[n:]))
            for row in back_paulis[num_faces:]
        }
        assert actual_x == expected_x
        assert actual_z == expected_z

    def test_bell_flagging_cnot_tick_count(self):
        block = _build_block("superdense", ColorCodeBellFlaggingBlock)
        assert len(block.measurement_blocks) == 2
        assert block.measurement_blocks[0] + block.measurement_blocks[1] == block.circuit
        assert [part.num_measurements for part in block.measurement_blocks] == [18, 18]
        assert len(_cnot_layers(block.circuit)) == 10
        assert _cnot_collision_layers(block.circuit) == []
        assert block.circuit.num_ticks == 13
        assert all(inst.name != "H" for inst in block.circuit)
        assert [
            inst.name for inst in block.circuit if inst.name in ("MX", "M")
        ] == ["MX", "M", "MX", "M"]

    def test_bell_flagging_prepares_x_check_ancillas_with_rx(self):
        block = _build_block("superdense", ColorCodeBellFlaggingBlock)
        names = [inst.name for inst in block.circuit[:4]]
        assert names == ["R", "RX", "TICK", "CX"]

    def test_bell_flagging_blocks_measure_faces_and_trivial_flags(self):
        code = ColorCode(distance=5, layout="superdense")
        system = QECSystem()
        system.add_patch(code, name="color")
        block = ColorCodeBellFlaggingBlock(system)
        n = code.num_qubits

        for measurement_block, basis in zip(block.measurement_blocks, ("X", "Z")):
            back_paulis, _, _ = CircuitBuilder._get_back_propagated_pauli(
                measurement_block,
                n,
                include_measurement_bases=True,
            )
            basis_slice = slice(0, n) if basis == "X" else slice(n, 2 * n)
            other_slice = slice(n, 2 * n) if basis == "X" else slice(0, n)
            supports = [
                frozenset(np.flatnonzero(row[basis_slice]))
                for row in back_paulis
                if np.any(row)
            ]
            expected = {
                frozenset(stabilizer["pauli"])
                for stabilizer in code.stabilizers
                if stabilizer["type"] == basis
            }

            assert len(supports) == len(code.faces)
            assert set(supports) == expected
            assert sum(not np.any(row) for row in back_paulis) == len(code.faces)
            assert not np.any(back_paulis[:, other_slice])

    @pytest.mark.parametrize("distance, sample_name", [
        (5, "midout_color_code_d5_r10_p1000.stim"),
        (9, "midout_color_code_d9_r36_p1000.stim"),
    ])
    def test_middle_out_cnot_layers_match_samples(self, distance, sample_name):
        block = _build_block("rectangle", ColorCodeMiddleOutBlock, distance=distance)
        assert _cnot_layers(block.circuit)[:12] == _sample_cnot_layers(sample_name, 12)

    def test_middle_out_exposes_two_reset_unitary_measurement_blocks(self):
        block = _build_block("rectangle", ColorCodeMiddleOutBlock, distance=5)
        plan = RectangleMiddleOutPlan.from_patch(block._patch)
        x_measure = sorted(block._g(q) for q in plan.x_measure_indices)
        z_measure = sorted(block._g(q) for q in plan.z_measure_indices)

        assert len(block.measurement_blocks) == 2
        assert block.measurement_blocks[0] + block.measurement_blocks[1] == block.circuit
        assert [part.num_measurements for part in block.measurement_blocks] == [11, 11]
        for part in block.measurement_blocks:
            instructions = list(part)
            first_measurement = next(
                k
                for k, instruction in enumerate(instructions)
                if instruction.name in ("M", "MX")
            )
            assert all(
                instruction.name in ("M", "MX")
                for instruction in instructions[
                    first_measurement:
                ]
            )

        layer_a, layer_b = block.measurement_blocks
        assert [instruction.name for instruction in layer_a[:3]] == [
            "RX", "R", "TICK",
        ]
        assert [target.value for target in layer_a[0].targets_copy()] == z_measure
        assert [target.value for target in layer_a[1].targets_copy()] == x_measure
        assert [instruction.name for instruction in layer_b[:4]] == [
            "TICK", "RX", "R", "TICK",
        ]
        assert [target.value for target in layer_b[1].targets_copy()] == x_measure
        assert [target.value for target in layer_b[2].targets_copy()] == z_measure

    def test_middle_out_layer_a_resets_representatives_without_syndrome_roles(self):
        code = ColorCode(distance=5, layout="rectangle")
        system = QECSystem()
        system.add_patch(code, name="color")
        layer_a = ColorCodeMiddleOutBlock(system).measurement_blocks[0]
        tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker, system, if_detector=True)

        analysis = builder._analyze_measurement_block(
            layer_a,
            z_only=False,
        )

        assert system.active_syndrome_indices == []
        assert analysis.reset_paulis is not None
        assert analysis.reset_paulis.shape == analysis.back_propagated_paulis.shape
        assert analysis.discarded_measurement_qubit_indices == set()

    @pytest.mark.parametrize("distance", [3, 5, 7, 9])
    def test_middle_out_builds_without_cnot_collisions(self, distance):
        block = _build_block("rectangle", ColorCodeMiddleOutBlock, distance=distance)
        assert _cnot_collision_layers(block.circuit) == []


# ---- Integration Tests ----

class TestColorCodeMemory:
    """End-to-end memory experiment tests."""

    def test_space_multiplexing_tracks_both_terminal_measurement_bases(self):
        code = ColorCode(distance=3, layout="superdense")
        system = QECSystem()
        system.add_patch(code, name="color")
        block = ColorCodeSpaceMultiplexingBlock(system)

        back_paulis, syndrome_indices, _ = CircuitBuilder._get_back_propagated_pauli(
            block.circuit,
            code.num_qubits,
            include_measurement_bases=True,
        )

        num_faces = len(code.faces)
        n = code.num_qubits
        assert syndrome_indices == (
            sorted(system.active_syndrome_indices_z)
            + sorted(system.active_syndrome_indices_x)
        )
        assert back_paulis.shape == (2 * num_faces, 2 * n)

        expected_z = {
            frozenset(stabilizer["pauli"])
            for stabilizer in code.stabilizers
            if stabilizer["type"] == "Z"
        }
        expected_x = {
            frozenset(stabilizer["pauli"])
            for stabilizer in code.stabilizers
            if stabilizer["type"] == "X"
        }
        actual_z = {
            frozenset(np.flatnonzero(row[n:]))
            for row in back_paulis[:num_faces]
        }
        actual_x = {
            frozenset(np.flatnonzero(row[:n]))
            for row in back_paulis[num_faces:]
        }
        assert not np.any(back_paulis[:num_faces, :n])
        assert not np.any(back_paulis[num_faces:, n:])
        assert actual_z == expected_z
        assert actual_x == expected_x

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_space_multiplexing_native_readout_with_detectors(self, basis):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=3, layout="superdense"),
            extraction_block_class=ColorCodeSpaceMultiplexingBlock,
            rounds=3,
            basis=basis,
            if_detector=True,
        ).build()

        samples = circuit.compile_detector_sampler().sample(
            100,
            append_observables=True,
        )
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1
        assert samples.sum() == 0

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_bell_multiplexing_first_round_boundaries(self, basis):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="superdense"),
            extraction_block_class=ColorCodeBellMultiplexingBlock,
            rounds=1,
            basis=basis,
            if_detector=True,
        ).build()

        samples = circuit.compile_detector_sampler().sample(
            100,
            append_observables=True,
        )
        assert circuit.num_detectors == 18
        assert circuit.num_observables == 1
        assert samples.sum() == 0

    def test_bell_multiplexing_forward_writeback_tracks_neighbor_record(self):
        code = ColorCode(distance=5, layout="superdense")
        system = QECSystem()
        system.add_patch(code, name="color")
        tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker, system, if_detector=True)
        builder.initialize(
            {q: "Z" for q in system.data_indices},
            n=code.num_qubits,
        )
        block = ColorCodeBellMultiplexingBlock(system)
        back_paulis, _, _ = CircuitBuilder._get_back_propagated_pauli(
            block.circuit,
            code.num_qubits,
            include_measurement_bases=True,
        )

        builder.apply_syndrome_extraction(block.circuit, rounds=1)

        p9_rows = [
            i
            for i, row in enumerate(tracker.stabilizers.matrix)
            if np.array_equal(row, back_paulis[9])
        ]
        assert len(p9_rows) == 1
        assert tracker.stabilizers.records[p9_rows[0]] == [9, 10]

    def test_bell_multiplexing_second_round_detector_uses_output_parity(self):
        code = ColorCode(distance=5, layout="superdense")
        system = QECSystem()
        system.add_patch(code, name="color")
        tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker, system, if_detector=True)
        builder.initialize(
            {q: "X" for q in system.data_indices},
            n=code.num_qubits,
        )
        builder.apply_syndrome_extraction(
            ColorCodeBellMultiplexingBlock(system).circuit,
            rounds=2,
        )

        detectors = [
            inst
            for inst in builder.circuit.flattened()
            if inst.name == "DETECTOR"
        ]
        d18_records = {
            target.value for target in detectors[18].targets_copy()
        }
        assert d18_records == {-9, -27, -26}

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_bell_multiplexing_repeat_reuses_explicit_second_round(self, basis):
        def build(*, combined_rounds):
            code = ColorCode(distance=5, layout="superdense")
            system = QECSystem()
            system.add_patch(code, name="color")
            tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
            builder = CircuitBuilder(tracker, system, if_detector=True)
            builder.initialize(
                {q: basis for q in system.data_indices},
                n=code.num_qubits,
            )
            block = ColorCodeBellMultiplexingBlock(system)
            if combined_rounds:
                builder.apply_syndrome_extraction(block.circuit, rounds=3)
            else:
                for _ in range(3):
                    builder.apply_syndrome_extraction(block.circuit, rounds=1)
            return builder.circuit

        repeated = build(combined_rounds=True)
        explicit = build(combined_rounds=False)

        def detector_record_targets(circuit):
            return [
                tuple(sorted(target.value for target in inst.targets_copy()))
                for inst in circuit.flattened()
                if inst.name == "DETECTOR"
            ]

        assert detector_record_targets(repeated) == detector_record_targets(explicit)
        assert repeated.compile_detector_sampler().sample(100).sum() == 0
        assert explicit.compile_detector_sampler().sample(100).sum() == 0

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_bell_multiplexing_repeat_compresses_logical_frame(self, basis):
        def build(rounds):
            return MemoryExperiment(
                qec_patch=ColorCode(distance=5, layout="superdense"),
                extraction_block_class=ColorCodeBellMultiplexingBlock,
                rounds=rounds,
                basis=basis,
                if_detector=True,
            ).build()

        short = build(3)
        long = build(1000)
        repeat_blocks = [
            inst for inst in long
            if isinstance(inst, stim.CircuitRepeatBlock)
        ]
        assert len(repeat_blocks) == 1
        assert repeat_blocks[0].repeat_count == 998

        body_observables = [
            inst for inst in repeat_blocks[0].body_copy()
            if not isinstance(inst, stim.CircuitRepeatBlock)
            and inst.name == "OBSERVABLE_INCLUDE"
        ]
        if basis == "X":
            assert body_observables == []
        else:
            assert len(body_observables) == 1
            assert tuple(
                target.value
                for target in body_observables[0].targets_copy()
            ) == (-8, -5)

        def top_level_observable_target_count(circuit):
            return sum(
                len(inst.targets_copy())
                for inst in circuit
                if not isinstance(inst, stim.CircuitRepeatBlock)
                and inst.name == "OBSERVABLE_INCLUDE"
            )

        assert (
            top_level_observable_target_count(long)
            == top_level_observable_target_count(short)
        )
        dets, observables = long.compile_detector_sampler().sample(
            100,
            separate_observables=True,
        )
        assert dets.sum() == 0
        assert observables.sum() == 0

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_bell_multiplexing_compressed_observable_is_detector_equivalent(self, basis):
        def build(*, combined_rounds):
            code = ColorCode(distance=5, layout="superdense")
            system = QECSystem()
            system.add_patch(code, name="color")
            tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
            builder = CircuitBuilder(tracker, system, if_detector=True)
            builder.initialize(
                {q: basis for q in system.data_indices},
                n=code.num_qubits,
            )
            block = ColorCodeBellMultiplexingBlock(system)
            if combined_rounds:
                builder.apply_syndrome_extraction(block.circuit, rounds=5)
            else:
                for _ in range(5):
                    builder.apply_syndrome_extraction(block.circuit, rounds=1)
            builder.apply_data_readout(
                {q: basis for q in system.data_indices}
            )
            return builder.circuit

        compressed_detectors, compressed_observable = _annotation_parities(
            build(combined_rounds=True)
        )
        explicit_detectors, explicit_observable = _annotation_parities(
            build(combined_rounds=False)
        )

        assert np.array_equal(compressed_detectors, explicit_detectors)
        observable_difference = (
            compressed_observable ^ explicit_observable
        ).reshape(1, -1)
        _, is_detector_product, _ = solve_linear_decomposition(
            basis=compressed_detectors,
            targets=observable_difference,
            reduce_weight=False,
        )
        assert is_detector_product[0]

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_bell_multiplexing_repeated_memory_boundaries(self, basis):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="superdense"),
            extraction_block_class=ColorCodeBellMultiplexingBlock,
            rounds=3,
            basis=basis,
            if_detector=True,
        ).build()

        samples = circuit.compile_detector_sampler().sample(
            100,
            append_observables=True,
        )
        assert circuit.num_detectors == 54
        assert circuit.num_observables == 1
        assert samples.sum() == 0

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    @pytest.mark.parametrize("rounds", [1, 3, 1000])
    def test_bell_flagging_repeated_memory_boundaries(self, basis, rounds):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="superdense"),
            extraction_block_class=ColorCodeBellFlaggingBlock,
            rounds=rounds,
            basis=basis,
            if_detector=True,
        ).build()

        dets, observables = circuit.compile_detector_sampler().sample(
            100,
            separate_observables=True,
        )
        single_record_detectors = sum(
            len(inst.targets_copy()) == 1
            for inst in circuit.flattened()
            if inst.name == "DETECTOR"
        )

        assert circuit.num_detectors == 36 * rounds
        assert circuit.num_observables == 1
        assert single_record_detectors >= 18 * rounds
        assert dets.sum() == 0
        assert observables.sum() == 0

        if rounds > 2:
            repeat_blocks = [
                inst
                for inst in circuit
                if isinstance(inst, stim.CircuitRepeatBlock)
            ]
            assert len(repeat_blocks) == 1
            assert repeat_blocks[0].repeat_count == rounds - 2
            assert repeat_blocks[0].body_copy().num_measurements == 36
            assert repeat_blocks[0].body_copy().num_detectors == 36

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    @pytest.mark.parametrize("rounds", [1, 3, 1000])
    def test_time_multiplexing_repeated_memory_boundaries(self, basis, rounds):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="triangular"),
            extraction_block_class=ColorCodeTimeMultiplexingBlock,
            rounds=rounds,
            basis=basis,
            if_detector=True,
        ).build()

        dets, observables = circuit.compile_detector_sampler().sample(
            100,
            separate_observables=True,
        )
        circuit.detector_error_model()

        assert circuit.num_detectors == 18 * rounds
        assert circuit.num_observables == 1
        assert dets.sum() == 0
        assert observables.sum() == 0

        if rounds > 2:
            repeat_blocks = [
                inst
                for inst in circuit
                if isinstance(inst, stim.CircuitRepeatBlock)
            ]
            assert len(repeat_blocks) == 1
            assert repeat_blocks[0].repeat_count == rounds - 2
            assert repeat_blocks[0].body_copy().num_measurements == 18
            assert repeat_blocks[0].body_copy().num_detectors == 18

    @pytest.mark.parametrize("d", [3, 5])
    @pytest.mark.parametrize("basis", ['X', 'Z'])
    def test_noiseless_memory(self, d, basis):
        """Noiseless circuit should have 0 detector events and 0 logical errors."""
        code = ColorCode(distance=d)
        system = QECSystem()
        system.add_patch(code, name=f'color_d{d}')

        experiment = MemoryExperiment(
            qec_system=system,
            extraction_block_class=ColorCodeExtractionBlock,
            rounds=d,
            basis=basis,
        )
        circuit = experiment.build()

        assert circuit.num_qubits == code.num_qubits
        assert circuit.num_observables == 1
        assert circuit.num_detectors > 0

        det_sampler = circuit.compile_detector_sampler()
        det_samples = det_sampler.sample(1000, append_observables=True)
        assert det_samples.sum() == 0, "Noiseless circuit should have no events"

    @pytest.mark.parametrize("d", [3, 5])
    def test_dem_validity(self, d):
        """DEM should be extractable from noisy circuit."""
        code = ColorCode(distance=d)
        system = QECSystem()
        system.add_patch(code, name=f'color_d{d}')

        noise = NoiseConfig(p_1q=0.001, p_2q=0.001, p_meas=0.001, p_reset=0.001, p_idle=0.001)
        experiment = MemoryExperiment(
            qec_system=system,
            extraction_block_class=ColorCodeExtractionBlock,
            rounds=d,
            noise_params=noise,
            noise_model='circuit_level',
            basis='Z',
        )
        circuit = experiment.build()
        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        assert dem.num_errors > 0
        assert dem.num_observables == 1
        assert dem.num_detectors == circuit.num_detectors

    @pytest.mark.parametrize("layout, block_cls", [
        ("superdense", ColorCodeExtractionBlock),
        ("superdense", ColorCodeBellMultiplexingBlock),
        ("superdense", ColorCodeBellFlaggingBlock),
        ("triangular", ColorCodeTimeMultiplexingBlock),
        ("rectangle", ColorCodeMiddleOutBlock),
    ])
    def test_paper_blocks_build_detector_off_memory(self, layout, block_cls):
        code = ColorCode(distance=5, layout=layout)
        circuit = MemoryExperiment(
            qec_patch=code,
            extraction_block_class=block_cls,
            rounds=1,
            basis="Z",
            if_detector=False,
        ).build()

        assert circuit.num_qubits == code.num_qubits
        assert circuit.num_detectors == 0
        assert circuit.num_measurements > 0

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    @pytest.mark.parametrize("layout, block_cls", [
        ("superdense", ColorCodeSpaceMultiplexingBlock),
        ("superdense", ColorCodeBellMultiplexingBlock),
        ("superdense", ColorCodeBellFlaggingBlock),
        ("triangular", ColorCodeTimeMultiplexingBlock),
    ])
    def test_memory_interface_selects_basis_layout_and_block(self, basis, layout, block_cls):
        experiment = MemoryExperiment(
            qec_patch=ColorCode(distance=3, layout=layout),
            extraction_block_class=block_cls,
            rounds=1,
            basis=basis,
            if_detector=False,
        )
        circuit = experiment.build()

        assert experiment.basis == basis
        assert experiment.patch.layout == layout
        assert experiment.block_class is block_cls
        assert circuit.num_detectors == 0

    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_middle_out_memory_interface_selects_basis(self, basis):
        experiment = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=1,
            basis=basis,
            if_detector=False,
        )
        circuit = experiment.build()

        assert experiment.basis == basis
        assert circuit.num_detectors == 0

    @pytest.mark.parametrize("distance", [3, 5, 7])
    @pytest.mark.parametrize("basis", ["X", "Z"])
    @pytest.mark.parametrize("rounds", [1, 3])
    def test_middle_out_detector_memory_is_deterministic(
        self,
        distance,
        basis,
        rounds,
    ):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=distance, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=rounds,
            basis=basis,
            if_detector=True,
        ).build()

        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1
        samples = circuit.compile_detector_sampler().sample(
            256,
            append_observables=True,
        )
        assert not np.any(samples)

    def test_stateful_reset_transfers_old_parity_before_clearing_pivot(self):
        tracker = SyndromeTracker(num_qubits=2)
        tracker.stabilizers.matrix = np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        tracker.stabilizers.records = [[0], [1]]
        reset_z0 = np.array([[0, 0, 1, 0]], dtype=np.uint8)

        tracker.process_resets(reset_z0)

        rows_and_records = {
            tuple(row): frozenset(records)
            for row, records in zip(
                tracker.stabilizers.matrix,
                tracker.stabilizers.records,
            )
        }
        assert rows_and_records[(0, 0, 1, 0)] == frozenset()
        assert rows_and_records[(0, 0, 0, 1)] == frozenset({0, 1})

    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_middle_out_detector_error_model_is_extractable(self, basis):
        noise = NoiseConfig(
            p_1q=0.001,
            p_2q=0.001,
            p_meas=0.001,
            p_reset=0.001,
            p_idle=0.001,
        )
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=3, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=3,
            basis=basis,
            if_detector=True,
            noise_params=noise,
            noise_model="circuit_level",
        ).build()

        dem = circuit.detector_error_model()
        assert dem.num_errors > 0
        assert dem.num_detectors == circuit.num_detectors
        assert dem.num_observables == 1

    @pytest.mark.parametrize("distance, sample_name", [
        (5, "midout_color_code_d5_r10_p1000.stim"),
        (9, "midout_color_code_d9_r36_p1000.stim"),
    ])
    def test_middle_out_first_x_measurements_match_trusted_detectors(
        self,
        distance,
        sample_name,
    ):
        generated = MemoryExperiment(
            qec_patch=ColorCode(distance=distance, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=1,
            basis="X",
            if_detector=True,
        ).build()
        trusted = stim.Circuit(_sample_path(sample_name).read_text())

        generated_first = _measurement_layers(generated)[0]
        trusted_first = _measurement_layers(trusted)[0]
        generated_x_measurements = generated_first["measurements"][0]
        trusted_x_measurements = trusted_first["measurements"][0]
        expected_single_record_detectors = [
            (record,)
            for record in generated_x_measurements["records"]
        ]

        assert generated_x_measurements["gate"] == "MX"
        assert generated_x_measurements == trusted_x_measurements
        assert generated_first["detectors"] == expected_single_record_detectors
        assert trusted_first["detectors"] == expected_single_record_detectors

    def test_middle_out_se_detectors_match_trusted_d5_prefix(self):
        generated = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=5,
            basis="X",
            if_detector=True,
        ).build()
        trusted = stim.Circuit(
            _sample_path("midout_color_code_d5_r10_p1000.stim").read_text()
        )
        generated_layers = _measurement_layers(generated)
        trusted_layers = _measurement_layers(trusted)
        num_se_layers = len(trusted_layers) - 1

        generated_counts = [
            (layer["measurement_count"], len(layer["detectors"]))
            for layer in generated_layers[:num_se_layers]
        ]
        trusted_counts = [
            (layer["measurement_count"], len(layer["detectors"]))
            for layer in trusted_layers[:num_se_layers]
        ]
        generated_generators = [
            set(layer["detectors"])
            for layer in generated_layers[:num_se_layers]
        ]
        trusted_generators = [
            set(layer["detectors"])
            for layer in trusted_layers[:num_se_layers]
        ]

        assert generated_counts == trusted_counts
        assert generated_generators == trusted_generators

    @pytest.mark.parametrize(("basis", "first_layer_singles"), [
        ("X", 6),
        ("Z", 5),
    ])
    def test_middle_out_steady_single_record_detectors_are_reset_boundaries(
        self,
        basis,
        first_layer_singles,
    ):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=5,
            basis=basis,
            if_detector=True,
        ).build()
        se_layers = _measurement_layers(circuit)[:-1]
        single_counts = [
            sum(
                len(records) == 1
                for records in layer["detectors"]
            )
            for layer in se_layers
        ]

        assert single_counts == [
            first_layer_singles,
            *([4] * (len(se_layers) - 1)),
        ]

    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_middle_out_repeated_rounds_are_compressed(self, basis):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=1000,
            basis=basis,
            if_detector=True,
        ).build()

        repeat_blocks = [
            instruction
            for instruction in circuit
            if isinstance(instruction, stim.CircuitRepeatBlock)
        ]
        assert len(repeat_blocks) == 1
        assert repeat_blocks[0].repeat_count == 998
        assert repeat_blocks[0].body_copy().num_measurements == 22
        assert repeat_blocks[0].body_copy().num_detectors == 22

        detectors, observables = (
            circuit.compile_detector_sampler().sample(
                100,
                separate_observables=True,
            )
        )
        assert detectors.sum() == 0
        assert observables.sum() == 0

    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_middle_out_compressed_detectors_span_explicit_detectors(self, basis):
        compressed = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=5,
            basis=basis,
            if_detector=True,
        ).build()
        explicit = _build_explicit_middle_out_memory(
            basis=basis,
            rounds=5,
        )

        compressed_detectors, compressed_observable = _annotation_parities(
            compressed
        )
        explicit_detectors, explicit_observable = _annotation_parities(
            explicit
        )
        _, explicit_in_compressed_span, _ = solve_linear_decomposition(
            basis=compressed_detectors,
            targets=explicit_detectors,
            reduce_weight=False,
        )
        _, compressed_in_explicit_span, _ = solve_linear_decomposition(
            basis=explicit_detectors,
            targets=compressed_detectors,
            reduce_weight=False,
        )

        assert explicit_in_compressed_span.all()
        assert compressed_in_explicit_span.all()
        assert np.array_equal(compressed_observable, explicit_observable)

    def test_middle_out_memory_interface_rejects_unverified_y_basis(self):
        with pytest.raises(ValueError, match="Middle-out memory basis"):
            MemoryExperiment(
                qec_patch=ColorCode(distance=5, layout="rectangle"),
                extraction_block_class=ColorCodeMiddleOutBlock,
                rounds=1,
                basis="Y",
                if_detector=False,
            )

    @pytest.mark.parametrize("distance, sample_name", [
        (5, "midout_color_code_d5_r10_p1000.stim"),
        (9, "midout_color_code_d9_r36_p1000.stim"),
    ])
    def test_middle_out_memory_initialization_matches_samples(self, distance, sample_name):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=distance, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            basis="X",
            rounds=1,
            if_detector=False,
        ).build()
        assert set(_reset_targets_before_first_tick(circuit, "RX")) == set(
            _sample_first_reset_targets(sample_name, "RX")
        )
        assert set(_reset_targets_before_first_tick(circuit, "R")) == set(
            _sample_first_reset_targets(sample_name, "R")
        )

    def test_middle_out_memory_basis_changes_only_passive_data_qubits(self):
        code = ColorCode(distance=5, layout="rectangle")
        plan = RectangleMiddleOutPlan.from_patch(code)
        x_basis_map = plan.initial_basis_map(code.data_indices, memory_basis="X")
        z_basis_map = plan.initial_basis_map(code.data_indices, memory_basis="Z")

        representatives = set(plan.x_measure_indices) | set(plan.z_measure_indices)
        passive_data = code.data_indices - representatives
        assert all(x_basis_map[q] == z_basis_map[q] for q in representatives)
        assert {x_basis_map[q] for q in passive_data} == {"X"}
        assert {z_basis_map[q] for q in passive_data} == {"Z"}

    def test_middle_out_memory_readout_separated_from_extraction_measurements(self):
        circuit = MemoryExperiment(
            qec_patch=ColorCode(distance=5, layout="rectangle"),
            extraction_block_class=ColorCodeMiddleOutBlock,
            rounds=1,
            basis="Z",
            if_detector=False,
        ).build()
        assert _measurement_collision_layers(circuit) == []

    @pytest.mark.parametrize("layout, block_cls", [
        ("triangular", ColorCodeExtractionBlock),
        ("rectangle", ColorCodeTimeMultiplexingBlock),
        ("superdense", ColorCodeMiddleOutBlock),
    ])
    def test_paper_blocks_reject_incompatible_layouts(self, layout, block_cls):
        code = ColorCode(distance=5, layout=layout)
        system = QECSystem()
        system.add_patch(code, name=f'color_{layout}')

        experiment = MemoryExperiment(
            qec_system=system,
            extraction_block_class=block_cls,
            rounds=1,
            basis='Z',
            if_detector=False,
        )
        with pytest.raises(ValueError, match="layout"):
            experiment.build()


def _build_block(layout, block_cls, distance=5):
    code = ColorCode(distance=distance, layout=layout)
    system = QECSystem()
    system.add_patch(code, name=f"color_{layout}")
    return block_cls(system)


def _cnot_layers(circuit):
    layers = []
    for inst in circuit:
        if inst.name in ("CX", "CNOT"):
            targets = [target.value for target in inst.targets_copy()]
            layers.append([
                (targets[i], targets[i + 1])
                for i in range(0, len(targets), 2)
            ])
    return layers


def _cnot_collision_layers(circuit):
    collisions = []
    for tick, inst in enumerate(circuit):
        if inst.name not in ("CX", "CNOT"):
            continue
        targets = [target.value for target in inst.targets_copy()]
        if len(targets) != len(set(targets)):
            collisions.append(tick)
    return collisions


def _measurement_collision_layers(circuit):
    collisions = []
    measured_in_moment = set()
    moment = 0
    measurement_gates = {"M", "MX", "MY", "MR", "MRX", "MRY", "MRZ"}

    for inst in circuit:
        if inst.name == "TICK":
            measured_in_moment.clear()
            moment += 1
            continue
        if inst.name not in measurement_gates:
            continue
        targets = [target.value for target in inst.targets_copy()]
        repeated = [q for q in targets if q in measured_in_moment]
        if repeated:
            collisions.append((moment, inst.name, sorted(set(repeated))))
        measured_in_moment.update(targets)

    return collisions


def _sample_path(sample_name):
    return Path(__file__).resolve().parent / "data" / sample_name


def _sample_qubit_coords(sample_name):
    coords = {}
    for line in _sample_path(sample_name).read_text().splitlines():
        if not line.startswith("QUBIT_COORDS"):
            continue
        args = line.split("(", 1)[1].split(")", 1)[0]
        x, y = (int(v.strip()) for v in args.split(","))
        idx = int(line.split()[-1])
        coords[idx] = (x, y)
    return coords


def _shift_coords(coords, *, dx=0, dy=0):
    return {
        idx: (x + dx, y + dy)
        for idx, (x, y) in coords.items()
    }


def _sample_cnot_layers(sample_name, count):
    layers = []
    for line in _sample_path(sample_name).read_text().splitlines():
        if not line.startswith("CX "):
            continue
        targets = [int(v) for v in line.split()[1:]]
        layers.append([
            (targets[i], targets[i + 1])
            for i in range(0, len(targets), 2)
        ])
        if len(layers) == count:
            break
    return layers


def _reset_targets_before_first_tick(circuit, gate_name):
    result = []
    for instruction in circuit:
        if instruction.name == "TICK":
            break
        if instruction.name == gate_name:
            result.extend(
                target.value
                for target in instruction.targets_copy()
            )
    return result


def _sample_first_reset_targets(sample_name, gate_name):
    for line in _sample_path(sample_name).read_text().splitlines():
        if line.startswith(f"{gate_name} "):
            return [int(v) for v in line.split()[1:]]
    return []


def _measurement_layers(circuit):
    measurement_gates = {
        "M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY",
    }
    layers = []
    current_layer = None
    in_measurement_layer = False
    measurement_count = 0

    for instruction in circuit.flattened():
        if instruction.name in measurement_gates:
            if not in_measurement_layer:
                current_layer = {
                    "measurement_count": 0,
                    "measurements": [],
                    "detectors": [],
                }
                layers.append(current_layer)
                in_measurement_layer = True

            qubits = [
                target.value
                for target in instruction.targets_copy()
                if target.is_qubit_target
            ]
            records = list(range(
                measurement_count,
                measurement_count + len(qubits),
            ))
            current_layer["measurements"].append({
                "gate": instruction.name,
                "qubits": qubits,
                "records": records,
            })
            current_layer["measurement_count"] += len(qubits)
            measurement_count += len(qubits)
            continue

        if instruction.name == "DETECTOR" and current_layer is not None:
            current_layer["detectors"].append(tuple(sorted(
                measurement_count + target.value
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            )))
        elif instruction.name != "OBSERVABLE_INCLUDE":
            in_measurement_layer = False

    return layers


def _annotation_parities(circuit):
    circuit = circuit.flattened()
    num_measurements = circuit.num_measurements
    measurement_count = 0
    detector_rows = []
    observable_row = np.zeros(num_measurements, dtype=np.uint8)

    for instruction in circuit:
        if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            row = np.zeros(num_measurements, dtype=np.uint8)
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    row[measurement_count + target.value] ^= 1
            if instruction.name == "DETECTOR":
                detector_rows.append(row)
            else:
                assert instruction.gate_args_copy() == [0.0]
                observable_row ^= row
            continue

        single_instruction = stim.Circuit()
        single_instruction.append(
            instruction.name,
            instruction.targets_copy(),
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
        measurement_count += single_instruction.num_measurements

    return np.asarray(detector_rows), observable_row


def _build_explicit_middle_out_memory(*, basis, rounds):
    code = ColorCode(distance=5, layout="rectangle")
    system = QECSystem()
    system.add_patch(code, name="color")
    tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=True)

    local_to_global = system.local_to_global_map["color"]
    basis_map = {
        local_to_global[qubit]: qubit_basis
        for qubit, qubit_basis in (
            ColorCodeMiddleOutBlock.memory_data_basis_map(code, basis).items()
        )
    }
    block = ColorCodeMiddleOutBlock(system)
    init_map = {
        qubit: qubit_basis
        for qubit, qubit_basis in basis_map.items()
        if qubit not in block.data_qubits_initialized_by_block
    }
    builder.initialize(init_map, code.num_qubits)
    system.active_qubit_indices.update(system.data_indices)
    for _ in range(rounds):
        builder.apply_syndrome_extraction(
            block.circuit,
            rounds=1,
            measurement_blocks=block.measurement_blocks,
        )
    builder.apply_data_readout(basis_map)
    return builder.circuit
