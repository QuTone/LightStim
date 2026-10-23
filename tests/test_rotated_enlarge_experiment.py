"""Packaged growth experiment uses the native executor and noise pipeline."""

import numpy as np
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.protocols import RotatedSurfaceEnlargeExperiment


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_experiment_is_repeatable_and_closes_the_enlarged_patch(basis):
    experiment = RotatedSurfaceEnlargeExperiment(
        basis=basis, target_distance=5, source_rounds=2, rounds=3,
    )
    circuit = experiment.build()
    assert experiment.source_patch.distance_x == 3
    assert experiment.output_patch.distance_x == 5
    assert experiment.system.patches["patch"][0].distance_x == 5
    assert circuit.num_qubits == 49
    assert circuit.num_measurements == 8 * 2 + 24 * 3 + 25
    assert sum(i.name == "CX" for i in circuit.flattened()) == 4 * (2 + 3)
    assert circuit.num_observables == 1
    circuit.detector_error_model(decompose_errors=True)
    dets, obs = circuit.compile_detector_sampler(seed=52).sample(
        128, separate_observables=True,
    )
    assert not np.any(dets)
    assert not np.any(obs)
    assert not any(i.tag == "post-select" for i in circuit.flattened())
    assert experiment.build() == circuit


def test_experiment_injects_noise_into_preparation_growth_and_readout():
    experiment = RotatedSurfaceEnlargeExperiment(
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
    {"distance": 1}, {"target_distance": 3}, {"target_distance": 6},
    {"rounds": 0}, {"rounds": True}, {"source_rounds": 0},
    {"basis": "Y"}, {"direction": "-x,-y"},
])
def test_experiment_rejects_unsupported_configuration(kwargs):
    with pytest.raises(ValueError):
        RotatedSurfaceEnlargeExperiment(**kwargs)
