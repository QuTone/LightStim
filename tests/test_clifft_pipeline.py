"""Exercise Clifft through the existing Clifford decoding pipeline."""

from contextlib import redirect_stdout
from io import StringIO

import pytest
import stim

from lightstim.protocols.cnot_trans import CNOTTransExperiment
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.simulation.decoder_backend import SimulationPipeline

from .test_clifft_backend import clifft_python


pytestmark = pytest.mark.smoke


def _pipeline(clifft_python, num_workers=1):
    return SimulationPipeline(
        sampler_backend="clifft",
        clifft_python=clifft_python,
        max_shots=32,
        max_errors=33,
        batch_size=8,
        num_workers=num_workers,
        print_progress=False,
        progress_poll_interval_sec=0.01,
    )


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("fault_probability", [0, 1])
def test_postselection_matches_stim_reference(clifft_python, num_workers, fault_probability):
    circuit = stim.Circuit(f"""
        R 0 1
        X 0 1
        E({fault_probability}) X0 X1
        M 0 1
        DETECTOR[post-select] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)

    stats = _pipeline(clifft_python, num_workers).run(circuit)

    assert stats.shots == 32
    assert stats.post_selected_shots == (32 if fault_probability == 0 else 0)
    assert stats.errors == 0


def test_deterministic_logical_fault_counts_as_error(clifft_python):
    circuit = stim.Circuit("""
        R 0 1
        X 0
        X_ERROR(1) 0
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)

    stats = _pipeline(clifft_python).run(circuit)

    assert stats.shots == stats.post_selected_shots == stats.errors == 32


def _memory_experiment():
    return MemoryExperiment(
        qec_patch=RotatedSurfaceCode(distance=3),
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=3,
        basis="X",
    )


def _transversal_cnot_experiment():
    return CNOTTransExperiment(
        code_patch_class=UnrotatedSurfaceCode,
        extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
        code_params_control={"distance": 3},
        offset_target=(12, 0),
        rounds_before=2,
        rounds_after=2,
    )


@pytest.mark.parametrize("experiment", [_memory_experiment, _transversal_cnot_experiment])
def test_noiseless_protocols_use_clifft(clifft_python, experiment):
    with redirect_stdout(StringIO()):
        circuit = experiment().build()

    stats = _pipeline(clifft_python).run(circuit)

    assert stats.shots == stats.post_selected_shots == 32
    assert stats.errors == 0


def test_unavailable_clifft_does_not_fall_back_to_stim(tmp_path):
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")

    with pytest.raises(RuntimeError, match="Clifft"):
        _pipeline(str(tmp_path / "missing-clifft-python")).run(circuit)


def test_repeated_pauli_target_observable_is_rejected():
    circuit = stim.Circuit("""
        REPEAT 2 {
            R 0 1
            X_ERROR(1) 0
            OBSERVABLE_INCLUDE(0) Z0
            M 0 1
            DETECTOR rec[-1]
        }
    """)

    with pytest.raises(ValueError, match="Pauli targets in OBSERVABLE_INCLUDE"):
        _pipeline(None).run(circuit)
