"""
Pipeline Smoke Tests — noisy circuit → SimulationPipeline → LER > 0.

Tests the full stack: noise injection → DEM → PyMatching decode → stats.
Uses d=3, p=0.05 (high noise so errors appear in ~300 shots), max_errors=3.

Purpose: catch regressions in SimulationPipeline, NoiseInjector, or decoder
wiring that would produce LER=0 or crash without touching protocol code.

Run:  pytest tests/test_pipeline.py -m smoke -q
"""
import io
import contextlib
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

NOISE = NoiseConfig(p_1q=0.05, p_2q=0.05, p_meas=0.05, p_reset=0.05)
PIPELINE = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_shots=500,
    max_errors=3,
    num_workers=1,  # single-worker for reproducibility in CI
)


def _run_pipeline(circuit):
    stats = PIPELINE.run(circuit)
    return stats


@pytest.mark.smoke
def test_pipeline_rotated_memory():
    """Rotated SC Z-memory with circuit-level noise → LER > 0."""
    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
    from lightstim.ir.qec_system import QECSystem
    from lightstim.ir.tracker import SyndromeTracker
    from lightstim.ir.builder import CircuitBuilder

    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="main")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    se_cls = RotatedSurfaceCodeExtractionBlock

    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=3)
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    noisy = builder.build_noisy_circuit(NOISE, noise_model="circuit_level")

    stats = _run_pipeline(noisy)
    assert stats.shots > 0,  "pipeline returned 0 shots"
    assert stats.errors > 0, f"LER=0 at p=0.05 — noise injection or decoder may be broken"
    assert 0 < stats.logical_error_rate < 1, f"LER out of range: {stats.logical_error_rate}"


@pytest.mark.smoke
def test_pipeline_two_patch_ls():
    """Two-patch ZZ LS with circuit-level noise → pipeline runs, LER > 0."""
    from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment

    with contextlib.redirect_stdout(io.StringIO()):
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3}, patch2_config={"distance": 3},
            offset=(0, 8), interaction_type="ZZ",
            initial_state_patch1="X", initial_state_patch2="Z",
            measure_state_patch1="X", measure_state_patch2="Z",
            rounds=2,
            noise_params=NOISE, noise_model="circuit_level",
        )
        noisy = exp.build()

    stats = _run_pipeline(noisy)
    assert stats.shots > 0
    assert stats.errors > 0, "LER=0 at p=0.05 — noise injection or decoder may be broken"
    assert 0 < stats.logical_error_rate < 1


@pytest.mark.smoke
def test_pipeline_noise_models():
    """Code-capacity and phenomenological models produce valid LER."""
    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
    from lightstim.ir.qec_system import QECSystem
    from lightstim.ir.tracker import SyndromeTracker
    from lightstim.ir.builder import CircuitBuilder

    for nm, noise in [
        ("code_capacity",    NoiseConfig(p_idle=0.05)),
        ("phenomenological", NoiseConfig(p_idle=0.05, p_meas=0.05)),
    ]:
        system = QECSystem()
        system.add_patch(RotatedSurfaceCode(distance=3), name="main")
        tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker, system)
        builder.write_coordinates()
        builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
        builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=3)
        builder.apply_data_readout({q: "Z" for q in system.data_indices})
        noisy = builder.build_noisy_circuit(noise, noise_model=nm)

        stats = _run_pipeline(noisy)
        assert stats.errors > 0, f"{nm}: LER=0 — p_idle may not be applied"
