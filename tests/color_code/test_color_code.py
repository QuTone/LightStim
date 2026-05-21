"""Tests for the Triangular Color Code implementation."""

import pytest
import numpy as np
import stim

from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig


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


# ---- Integration Tests ----

class TestColorCodeMemory:
    """End-to-end memory experiment tests."""

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
