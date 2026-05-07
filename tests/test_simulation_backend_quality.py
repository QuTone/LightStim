import pytest
import stim

from src.simulation.decoder_backend import DecoderConfig, SimulationPipeline, list_decoders
from src.simulation.simulator import ExperimentTask, QECSimulator


def _simple_observable_circuit(error_probability: float = 0.0) -> stim.Circuit:
    circuit = stim.Circuit()
    if error_probability:
        circuit.append("X_ERROR", [0], error_probability)
    circuit.append("M", [0])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


def test_multiprocess_unknown_decoder_raises_in_parent():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("definitely_missing"),
        max_shots=10,
        max_errors=1,
        batch_size=5,
        num_workers=2,
        print_progress=False,
    )

    with pytest.raises(ValueError, match="Unknown decoder 'definitely_missing'"):
        pipeline.run(_simple_observable_circuit())


def test_legacy_nvidia_gpu_backend_is_disabled():
    simulator = QECSimulator(backend="nvidia_gpu", num_workers=1)

    with pytest.raises(NotImplementedError, match="placeholder decoder"):
        simulator.run_batch([ExperimentTask(_simple_observable_circuit())])


def test_list_decoders_deduplicates_canonical_names():
    names = list_decoders()

    assert names == sorted(set(names))


def test_post_selection_can_reject_all_shots_without_decoding():
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=5,
        max_errors=1,
        batch_size=5,
        num_workers=1,
        post_select_detector_indices=[0],
        print_progress=False,
    )

    stats = pipeline.run(_simple_observable_circuit(error_probability=1.0))

    assert stats.shots == 5
    assert stats.post_selected_shots == 0
    assert stats.errors == 0
    assert stats.post_selection_rate == 0.0
