"""
Export Function Tests — lightstim.frontend.export_all() schema validation.

Verifies that export_all() returns JSON matching the front-end schemas for
both raw and decomposed DEM, across two different circuit types.

Run:  pytest tests/test_export.py -m smoke -q
"""
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.frontend import export_all

NOISE = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3)


def _rotated_memory_circuit(d=3, rounds=3):
    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
    from lightstim.ir.qec_system import QECSystem
    from lightstim.ir.tracker import SyndromeTracker
    from lightstim.ir.builder import CircuitBuilder
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=d), name="main")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(RotatedSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    return builder.build_noisy_circuit(NOISE, noise_model="circuit_level")


@pytest.fixture(scope="module")
def rotated_memory_payload():
    circuit = _rotated_memory_circuit()
    return export_all(circuit, source="rotated_memory_z", distance=3, rounds=3,
                      noise_model="circuit_level", physical_error_rate=1e-3)


@pytest.mark.smoke
def test_export_top_level_keys(rotated_memory_payload):
    assert set(rotated_memory_payload.keys()) == {"dem", "timeline", "detslice"}


@pytest.mark.smoke
class TestDEMSchema:

    def test_required_keys(self, rotated_memory_payload):
        dem = rotated_memory_payload["dem"]
        assert all(k in dem for k in ("metadata", "detectors", "observables", "error_mechanisms"))

    def test_detectors_non_empty(self, rotated_memory_payload):
        dets = rotated_memory_payload["dem"]["detectors"]
        assert len(dets) == 24  # d=3 rotated SC, 3 rounds → 24 detectors

    def test_detector_has_xyz(self, rotated_memory_payload):
        for d in rotated_memory_payload["dem"]["detectors"]:
            assert "id" in d
            assert {"x", "y", "t"} <= set(d["coords"])

    def test_error_mechanisms_non_empty(self, rotated_memory_payload):
        assert len(rotated_memory_payload["dem"]["error_mechanisms"]) == 219

    def test_error_mechanism_keys(self, rotated_memory_payload):
        for e in rotated_memory_payload["dem"]["error_mechanisms"]:
            assert "probability" in e
            assert "detector_ids" in e
            assert "observable_ids" in e
            assert 0 < e["probability"] < 1

    def test_observables(self, rotated_memory_payload):
        obs = rotated_memory_payload["dem"]["observables"]
        assert len(obs) == 1  # 1 logical qubit
        assert obs[0]["id"] == 0


@pytest.mark.smoke
class TestTimelineSchema:

    def test_required_keys(self, rotated_memory_payload):
        tl = rotated_memory_payload["timeline"]
        assert all(k in tl for k in ("metadata", "qubits", "ticks", "detectors"))

    def test_metadata_fields(self, rotated_memory_payload):
        meta = rotated_memory_payload["timeline"]["metadata"]
        assert meta["num_qubits"] > 0
        assert meta["num_ticks"] > 0
        assert meta["num_detectors"] == 24

    def test_ticks_have_operations(self, rotated_memory_payload):
        ticks = rotated_memory_payload["timeline"]["ticks"]
        assert len(ticks) > 0
        for tick in ticks:
            assert "operations" in tick
            assert "noise" in tick


@pytest.mark.smoke
class TestDetSliceSchema:

    def test_required_keys(self, rotated_memory_payload):
        ds = rotated_memory_payload["detslice"]
        assert all(k in ds for k in ("metadata", "qubits", "detector_coordinates", "slices"))

    def test_qubits_have_coords(self, rotated_memory_payload):
        for q in rotated_memory_payload["detslice"]["qubits"]:
            assert "id" in q
            assert "coords" in q
            assert "x" in q["coords"] and "y" in q["coords"]

    def test_slices_have_detectors(self, rotated_memory_payload):
        for sl in rotated_memory_payload["detslice"]["slices"]:
            assert "tick" in sl
            assert "detectors" in sl
            for det in sl["detectors"]:
                assert "detector_id" in det
                assert "pauli_support" in det
                for ps in det["pauli_support"]:
                    assert ps["pauli"] in ("X", "Y", "Z")


@pytest.mark.smoke
def test_export_decomposed():
    """Decomposed DEM should have only degree-1 and degree-2 edges."""
    circuit = _rotated_memory_circuit()
    payload = export_all(circuit, source="test", distance=3, rounds=3,
                         noise_model="circuit_level", decompose_errors=True)
    for e in payload["dem"]["error_mechanisms"]:
        assert len(e["detector_ids"]) <= 2, f"decomposed DEM has hyperedge: {e}"


@pytest.mark.smoke
def test_export_two_patch_ls():
    """Two-patch LS export: two patches → y-span should cover both."""
    import contextlib, io
    from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
    with contextlib.redirect_stdout(io.StringIO()):
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3}, patch2_config={"distance": 3},
            offset=(0, 10), interaction_type="ZZ",
            initial_state_patch1="X", initial_state_patch2="Z",
            measure_state_patch1="X", measure_state_patch2="Z",
            rounds=3, noise_params=NOISE, noise_model="circuit_level",
        )
        circuit = exp.build()
    payload = export_all(circuit, source="two_patch_ls_zz", distance=3, rounds=3,
                         noise_model="circuit_level")
    dets = payload["dem"]["detectors"]
    ys = [d["coords"]["y"] for d in dets]
    assert max(ys) > min(ys) + 5, "two-patch LS should span at least 5 units in y"
