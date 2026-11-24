from copy import deepcopy
from importlib.resources import files
import json

import pytest

from lightstim.config import ExperimentConfig, SCHEMA_VERSION, expand_sweep, load_config
from lightstim.noise.config import NOISE_MODELS, NoiseConfig
from lightstim.simulation.decoder_backend.config import DecoderConfig, normalize_decoder


@pytest.fixture
def spec():
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "example-memory",
        "executable": False,
        "implementation": {"entrypoint": "example.Memory"},
        "parameters": {
            "distance": 3, "rounds": {"$param": "distance"}, "p": 0.01,
            "basis": "Z", "readout": "Z",
        },
        "system": {"patches": [{
            "id": "q", "code_family": "surface", "code": "rotated_sc",
            "params": {"distance": {"$param": "distance"}},
            "offset": [0, {"$param": "rounds", "times": 2, "plus": -1}],
        }]},
        "operations": [{
            "op": "initialize", "targets": ["q"], "params": {"basis": {"q": {"$param": "basis"}}},
        }],
        "noise": {"model": "circuit_level", "p": {"$param": "p"}},
        "decoder": {"name": "pymatching"},
        "simulation": {"mode": "decode", "max_shots": 1000, "max_errors": 10},
    }


def test_cartesian_product_with_zip_and_affine_aliases(spec):
    spec["sweep"] = {
        "axes": {"distance": [3, 5], "p": [0.1, 0.2]},
        "zip": [{"basis": ["X", "Z"], "readout": ["Z", "X"]}],
    }
    config = ExperimentConfig.from_dict(spec)
    points = list(expand_sweep(config))
    assert len(points) == 8
    assert points[-1].coordinates == {"distance": 5, "p": 0.2, "basis": "Z", "readout": "X"}
    assert points[-1]["parameters"]["rounds"] == 5
    assert points[-1]["system"]["patches"][0]["offset"] == [0, 9]
    assert points[-1]["operations"][0]["params"]["basis"] == {"q": "Z"}
    assert "sweep" not in points[-1]
    points[-1]["simulation"]["mode"] = "hybrid"
    assert points[0]["simulation"]["mode"] == config["simulation"]["mode"] == "decode"
    assert config["parameters"]["rounds"] == {"$param": "distance"}


def test_draft_file_and_default_point(spec, tmp_path):
    spec["schema_version"] = "lightstim-config/v1-draft"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    config = load_config(path)
    point, = expand_sweep(config)
    assert point.coordinates == {}
    assert point["schema_version"] == SCHEMA_VERSION
    assert point["noise"]["p"] == 0.01
    assert point["parameters"]["basis"] == "Z"
    assert config.resolve({"distance": 7})["parameters"]["rounds"] == 7


@pytest.mark.parametrize(("sweep", "message"), [
    ({"axes": {"typo": [1]}}, "undeclared"),
    ({"axes": {"p": [0]}, "zip": [{"basis": ["X"], "p": [0]}]}, "more than once"),
    ({"zip": [{"basis": ["X"], "p": [0, 0.1]}]}, "equal lengths"),
])
def test_invalid_sweep(spec, sweep, message):
    spec["sweep"] = sweep
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_dict(spec)


def test_contextual_reference_errors(spec):
    spec["parameters"]["distance"] = {"$param": "rounds"}
    with pytest.raises(ValueError, match="alias cycle"):
        ExperimentConfig.from_dict(spec)
    spec["parameters"]["distance"] = {"$param": "typo"}
    with pytest.raises(ValueError, match="Unknown parameter"):
        ExperimentConfig.from_dict(spec)
    spec["parameters"]["distance"] = "X"
    with pytest.raises(ValueError, match="numeric value"):
        ExperimentConfig.from_dict(spec)
    spec["parameters"]["distance"] = 3
    config = ExperimentConfig.from_dict(spec)
    with pytest.raises(ValueError, match="noise"):
        config.resolve({"p": 0.8})
    with pytest.raises(ValueError, match="undeclared"):
        config.resolve({"dist": 5})


@pytest.mark.parametrize("value", [float("inf"), float("nan"), float("-inf")])
def test_nonfinite_numbers_in_unrestricted_fields(spec, value):
    spec["decoder"]["params"] = {"values": [value]}
    with pytest.raises(ValueError, match="non-finite JSON number"):
        ExperimentConfig.from_dict(spec)


def test_nonfinite_json_literal_and_affine_overflow(spec, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec).replace("0.01", "1e9999"))
    with pytest.raises(ValueError, match="non-finite"):
        load_config(path)
    spec["parameters"]["distance"] = 1e308
    with pytest.raises(ValueError, match="non-finite"):
        ExperimentConfig.from_dict(spec)


def test_reject_invalid_references_and_resolved_offsets(spec):
    spec["operations"][0]["params"]["basis"]["q"] = {"$param": "basis", "multiply": 3}
    with pytest.raises(ValueError, match="operations"):
        ExperimentConfig.from_dict(spec)
    spec["operations"][0]["params"]["basis"]["q"] = {"$param": "basis"}
    spec["system"]["patches"][0]["offset"][0] = {"$param": "basis"}
    with pytest.raises(ValueError, match="offset"):
        ExperimentConfig.from_dict(spec)


def test_noise_defaults_overrides_and_null():
    spec = {
        "model": "XZ_biased", "p": 0.01,
        "overrides": {"p_idle": 0, "p_1q": None}, "custom_params": {"p_1q_x": 0.001},
    }
    noise = NoiseConfig.from_spec(spec)
    assert (noise.p_idle, noise.p_1q, noise.p_2q, noise.p_meas, noise.p_reset) == (0, 0.01, 0.01, 0.01, 0.01)
    assert noise.get("p_1q_x") == 0.001
    with pytest.raises(ValueError, match="Resolve"):
        NoiseConfig.from_spec({**spec, "p": {"$param": "p"}})
    schema = json.loads(files("lightstim.config").joinpath("schema.json").read_text())
    assert set(schema["$defs"]["noise"]["properties"]["model"]["enum"]) == set(NOISE_MODELS)


def test_decoder_chain_and_aliases():
    spec = {"chain": [
        {"name": "bposd", "backend": "gpu", "params": {"osd_order": 4}},
        {"name": "mle-ilp", "on_decode_failure": "discard", "params": {"time_limit": 2}},
    ]}
    original = deepcopy(spec)
    chain = DecoderConfig.from_spec(spec)
    assert chain.name == "chain" and chain.on_decode_failure == "discard"
    bp, mle = chain.params["stages"]
    assert (bp.name, bp.backend, bp.params) == ("bposd", "gpu", {"osd_order": 4})
    assert mle.name == "mle-ilp"
    bp.params["osd_order"] = 0
    assert spec == original
    with pytest.warns(FutureWarning, match="deprecated"):
        assert DecoderConfig.from_spec({"name": "cpu_bposd"}) == DecoderConfig("bposd")
    with pytest.raises(ValueError, match="conflicts"):
        normalize_decoder("gpu_bposd", "cpu")


def test_seed_fits_sampler_range(spec):
    spec["simulation"]["seed"] = 2**64
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig.from_dict(spec)


def test_simulation_json_integers_can_be_passed_to_sampling_apis(spec):
    spec["simulation"].update(seed=2.0, batch_size=3.0, mode="hybrid",
                              target_observable_indices=[0.0], postselection={"detector_indices": [1.0]})
    sim = ExperimentConfig.from_dict(spec).resolve()["simulation"]
    assert all(type(value) is int for value in (
        sim["seed"], sim["batch_size"], sim["target_observable_indices"][0],
        sim["postselection"]["detector_indices"][0],
    ))
