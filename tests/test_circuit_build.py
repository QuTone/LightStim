"""
Circuit Build Regression Tests — covers all notebook experiments.

Each test builds a noiseless circuit and verifies:
  1. circuit.num_detectors > 0 and circuit.num_observables > 0
  2. No detection events and no observable errors on 200 noiseless shots
  3. detector_error_model(decompose_errors=True) succeeds

Covers: memory_experiment, test_trans_CNOT, test_LS_CNOT, test_injection,
        test_ghz, test_LS_two_patch, fold_transversal, test_qecSys notebooks.
"""

import io
import contextlib
import numpy as np
import pytest
import stim

# ── Code imports ──────────────────────────────────────────────────────────────
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
)
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
)
from src.qec_code.surface_code.toric import ToricCode, ToricCodeExtractionBlock
from src.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
from src.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from src.ir.qec_system import QECSystem
from src.noise.config import NoiseConfig

# ── Experiment imports ────────────────────────────────────────────────────────
from experiments.memory import MemoryExperiment
from experiments.CNOT_trans import CNOTTransExperiment
from experiments.CNOT_LS import CNOTLSExperiment
from experiments.state_injection import StateInjectionExperiment
from experiments.ghz import GHZExperiment
from experiments.two_patch_LS_unrotated import TwoPatchLSExperiment
from experiments.fold_transversal import build_gate_verification_circuit

SHOTS = 200  # noiseless shots — fast but catches any stray detection events


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_quiet(fn):
    """Run fn() suppressing stdout (circuit build progress prints)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def assert_valid_circuit(circuit: stim.Circuit):
    assert circuit.num_qubits > 0
    assert circuit.num_detectors > 0
    assert circuit.num_observables > 0


def assert_noiseless(circuit: stim.Circuit, shots: int = SHOTS):
    """No detection events and no observable errors in a noiseless circuit."""
    dets, obs = circuit.compile_detector_sampler(seed=0).sample(
        shots=shots, separate_observables=True
    )
    assert not np.any(dets), f"Unexpected detection events in noiseless circuit"
    assert not np.any(obs),  f"Unexpected observable errors in noiseless circuit"


def assert_dem_valid(circuit: stim.Circuit):
    """DEM construction should not raise."""
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == circuit.num_observables


def _memory_circuit(code, block_cls, basis, rounds=3):
    system = QECSystem()
    system.add_patch(code, name="patch")
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=block_cls,
        rounds=rounds,
        noise_params=None,
        noise_model="circuit_level",
        basis=basis,
    )
    return _build_quiet(exp.build)


# ── memory_experiment.ipynb ───────────────────────────────────────────────────

class TestMemoryExperiment:

    @pytest.mark.parametrize("distance,basis", [(3, "Z"), (3, "X"), (5, "Z")])
    def test_rotated_surface_code(self, distance, basis):
        code = RotatedSurfaceCode(distance=distance)
        code.rotate_coords(np.pi / 4)
        circuit = _memory_circuit(code, RotatedSurfaceCodeExtractionBlock, basis)
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)

    @pytest.mark.parametrize("distance,basis", [(3, "Z"), (3, "X"), (5, "Z")])
    def test_unrotated_surface_code(self, distance, basis):
        circuit = _memory_circuit(
            UnrotatedSurfaceCode(distance=distance),
            UnrotatedSurfaceCodeExtractionBlock, basis,
        )
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_toric_code(self, basis):
        circuit = _memory_circuit(
            ToricCode(distance=3), ToricCodeExtractionBlock, basis,
        )
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
        assert circuit.num_observables > 0
        assert_noiseless(circuit)
        assert_dem_valid(circuit)

    @pytest.mark.parametrize("distance,basis", [(3, "Z"), (3, "X"), (5, "Z")])
    def test_color_code(self, distance, basis):
        circuit = _memory_circuit(
            ColorCode(distance=distance), ColorCodeExtractionBlock, basis,
        )
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)

    def test_bb_code_72_12_6(self):
        code = BBCode(l=6, m=6, A=[[3,0],[0,1],[0,2]], B=[[0,3],[1,0],[2,0]])
        system = QECSystem()
        system.add_patch(code, name="bb")
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=BBCodeExtractionBlock,
            rounds=6,  # rounds = d
            noise_params=None,
            noise_model="circuit_level",
            basis="Z",
        )
        circuit = _build_quiet(exp.build)
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)

    def test_multi_patch_memory(self):
        """test_qecSys.ipynb: two patches in one system."""
        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=3), name="p1")
        system.add_patch(UnrotatedSurfaceCode(distance=3), offset=(8, 0), name="p2")
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            rounds=3,
            noise_params=None,
            noise_model="circuit_level",
            basis="Z",
        )
        circuit = _build_quiet(exp.build)
        assert circuit.num_observables == 2   # two logical qubits
        assert_noiseless(circuit)
        assert_dem_valid(circuit)


# ── test_trans_CNOT.ipynb ─────────────────────────────────────────────────────

class TestTransversalCNOT:

    @pytest.mark.parametrize("code_cls,block_cls", [
        (UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock),
    ])
    def test_cnot_trans(self, code_cls, block_cls):
        exp = CNOTTransExperiment(
            code_patch_class=code_cls,
            extraction_block_class=block_cls,
            code_params_control={"distance": 3},
            rounds_before=2,
            rounds_after=2,
            noise_params=None,
        )
        circuit = _build_quiet(exp.build)
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)


# ── test_LS_CNOT.ipynb ────────────────────────────────────────────────────────

class TestLatticeSurgeryCNOT:

    def test_cnot_ls_unrotated(self):
        exp = CNOTLSExperiment(
            patch_configs={
                "c": {"distance": 3},
                "t": {"distance": 3},
                "a": {"distance": 3},
            },
            offset_ta=(6, 0),
            offset_ca=(0, 6),
            rounds=2,
            noise_params=None,
        )
        circuit = _build_quiet(exp.build)
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)


# ── test_injection.ipynb ──────────────────────────────────────────────────────

class TestStateInjection:

    @pytest.mark.parametrize("state", ["Z", "X", "Y"])
    def test_rotated_sc_injection(self, state):
        exp = StateInjectionExperiment(
            code_patch_class=RotatedSurfaceCode,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            distance=3,
            rounds=2,
            inject_state=state,
            noise_params=None,
        )
        circuit = _build_quiet(exp.build)
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
        assert_noiseless(circuit)
        assert_dem_valid(circuit)


# ── test_ghz.ipynb ────────────────────────────────────────────────────────────

class TestGHZ:

    def test_ghz_three_patch(self):
        exp = GHZExperiment(
            distance=3,
            rounds_before=2,
            rounds_after=2,
            noise_params=None,
        )
        circuit = _build_quiet(exp.build)
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)


# ── test_LS_two_patch.ipynb ───────────────────────────────────────────────────

class TestTwoPatchLS:

    @pytest.mark.parametrize("interaction", ["XX", "ZZ"])
    def test_two_patch_ls(self, interaction):
        init1, init2 = ("X", "Z") if interaction == "XX" else ("Z", "X")
        meas1, meas2 = ("Z", "X") if interaction == "XX" else ("X", "Z")
        # XX: patches side-by-side (horizontal); ZZ: patches stacked (vertical)
        offset = (8, 0) if interaction == "XX" else (0, 8)
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3},
            patch2_config={"distance": 3},
            offset=offset,
            interaction_type=interaction,
            initial_state_patch1=init1,
            initial_state_patch2=init2,
            measure_state_patch1=meas1,
            measure_state_patch2=meas2,
            rounds=2,
            noise_params=None,
        )
        circuit = _build_quiet(exp.build)
        assert_valid_circuit(circuit)
        assert_noiseless(circuit)
        assert_dem_valid(circuit)


# ── fold_transversal.ipynb ────────────────────────────────────────────────────

class TestFoldTransversal:

    @pytest.mark.parametrize("gates,init_basis,measure_basis", [
        (["fold_transversal_hadamard"], "Z", "X"),
        (["fold_transversal_s"],        "X", "Y"),
        (["fold_transversal_s", "fold_transversal_s",
          "fold_transversal_s", "fold_transversal_s"], "X", "X"),  # S^4 = I roundtrip
    ])
    def test_gate_verification(self, gates, init_basis, measure_basis):
        circuit = _build_quiet(lambda: build_gate_verification_circuit(
            distance=3,
            gates=gates,
            init_basis=init_basis,
            measure_basis=measure_basis,
            rounds=2,
            noise_params=None,
        ))
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
        assert_noiseless(circuit)
        assert_dem_valid(circuit)
