"""Packaged shrink experiment uses the native executor and noise pipeline."""

import numpy as np
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.protocols import RotatedSurfaceShrinkExperiment


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize(
    ("distance", "target_distance", "rounds"),
    [(5, 3, 0), (7, 3, 1), (7, 5, 2)],
)
def test_experiment_is_repeatable_and_closes_the_retained_patch(
    basis, distance, target_distance, rounds,
):
    experiment = RotatedSurfaceShrinkExperiment(
        distance=distance, target_distance=target_distance, basis=basis,
        source_rounds=2, rounds=rounds,
    )
    circuit = experiment.build()
    assert experiment.source_patch.distance_x == distance
    assert experiment.output_patch.distance_x == target_distance
    assert experiment.system.patches["patch"][0].distance_x == target_distance
    assert experiment.output_patch.data_indices < experiment.source_patch.data_indices
    assert circuit.num_qubits == 2 * distance**2 - 1
    assert circuit.num_measurements == (
        (distance**2 - 1) * 2 + (target_distance**2 - 1) * rounds + distance**2
    )
    assert sum(i.name == "CX" for i in circuit.flattened()) == 4 * (2 + rounds)
    assert circuit.num_observables == 1
    circuit.detector_error_model(decompose_errors=True)
    dets, obs = circuit.compile_detector_sampler(seed=52).sample(
        128, separate_observables=True,
    )
    assert not np.any(dets)
    assert not np.any(obs)
    assert not any(i.tag == "post-select" for i in circuit.flattened())
    assert experiment.build() == circuit


def test_default_shrink_has_no_target_extraction_round():
    experiment = RotatedSurfaceShrinkExperiment()
    circuit = experiment.build()
    assert experiment.rounds == 0
    assert experiment.source_patch.distance_x == 5
    assert experiment.output_patch.distance_x == 3
    assert circuit.num_measurements == 24 + 16 + 9
    assert sum(i.name == "CX" for i in circuit.flattened()) == 4


@pytest.mark.parametrize("rounds", [0, 1])
def test_experiment_injects_noise_into_preparation_shrink_and_readout(rounds):
    experiment = RotatedSurfaceShrinkExperiment(
        rounds=rounds,
        noise_params=NoiseConfig(p_1q=.001, p_2q=.001, p_reset=.001, p_meas=.001),
    )
    noisy = experiment.build()
    clean = experiment.builder.to_stim_circuit()
    assert "DEPOLARIZE2" in str(noisy)
    assert "DEPOLARIZE" not in str(clean)
    assert noisy.without_noise() == clean
    assert not any(i.tag == "noiseless" for i in clean.flattened())
    assert noisy.detector_error_model(decompose_errors=True).num_errors > 0


@pytest.mark.parametrize("kwargs", [
    {"distance": 1}, {"distance": True}, {"target_distance": 5},
    {"target_distance": 7}, {"target_distance": 2}, {"target_distance": 3.0},
    {"rounds": -1}, {"rounds": True}, {"source_rounds": 0},
    {"source_rounds": True}, {"basis": "Y"}, {"direction": "+x,+y"},
])
def test_experiment_rejects_unsupported_configuration(kwargs):
    with pytest.raises(ValueError):
        RotatedSurfaceShrinkExperiment(**kwargs)
