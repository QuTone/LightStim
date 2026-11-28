"""Fixed-shot Relay benchmark accounting, spawn execution, and resume safety."""
from dataclasses import dataclass
import json
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sparse
import stim

from benchmarks.logical_ops import bb_ppm_runner as runner
from benchmarks.logical_ops.run_bb_ppm import parse_args
from tests.conftest import importorskip_safe


@pytest.fixture
def toy_inputs(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    sparse.save_npz(source / "check_matrix.npz", sparse.eye(2, dtype=np.uint8, format="csc"))
    sparse.save_npz(source / "observable_matrix.npz", sparse.csc_matrix([[1, 0]], dtype=np.uint8))
    np.save(source / "error_priors.npy", np.array([.1, .2]))
    det = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    # Two intentionally mismatched truths check the actual Observable criterion.
    obs = np.array([[0], [0], [1], [1]], dtype=np.uint8)
    np.save(source / "detectors.npy", np.packbits(det, axis=1, bitorder="little"))
    np.save(source / "observables.npy", np.packbits(obs, axis=1, bitorder="little"))
    runner._atomic_json(source / "manifest.json", dict(status="prepared", shots=4,
        artifact_hashes={name: runner._sha(source / name) for name in (*runner.MATRIX_FILES, *runner.SAMPLE_FILES)}))
    return source


def test_spawn_decode_resume_rechecks_hc_oc_without_repeating_tasks(toy_inputs, tmp_path, monkeypatch):
    importorskip_safe("relay_bp")
    destination = tmp_path / "decoded"
    rows, complete, stopped = runner.decode_saved_inputs(toy_inputs, destination, workers=2)
    assert complete and not stopped
    assert len(rows) == 4 and all(r["converged"] for r in rows)
    assert sum(r["logical_failure"] for r in rows) == 2
    markers = sorted(destination.glob("per_shot/*/complete.json"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in markers}

    def no_pool(*args, **kwargs):
        pytest.fail("A fully completed resume must not start any decoder worker")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", no_pool)
    assert runner.decode_saved_inputs(toy_inputs, destination, workers=1)[:2] == (rows, True)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in markers}
    with pytest.raises(ValueError, match="Decode configuration changed"):
        runner.decode_saved_inputs(toy_inputs, destination, workers=1, decoder_params={"gamma0": .25})
    marker = markers[0]
    saved = runner._read(marker)
    saved["row"]["converged"] = False
    runner._atomic_json(marker, saved)
    with pytest.raises(ValueError, match="Saved Hc/Oc"):
        runner.decode_saved_inputs(toy_inputs, destination, workers=1)


def test_input_hash_and_parameter_changes_refused_before_workers(toy_inputs, tmp_path, monkeypatch):
    destination = tmp_path / "decode"
    with patch.object(runner, "ProcessPoolExecutor", side_effect=AssertionError("unexpected worker")):
        with (toy_inputs / "detectors.npy").open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(ValueError, match="Saved input changed"):
            runner.decode_saved_inputs(toy_inputs, destination, workers=1)
    with pytest.raises(ValueError, match="Unknown Relay"):
        runner.normalize_decoder_params({"use_lsd": True})
    with pytest.raises(ValueError, match="decoder seed"):
        runner.normalize_decoder_params({"seed": 1})
    assert runner.normalize_decoder_params({"gamma_dist_interval": [-.1, .5]})["gamma_dist_interval"] == (-.1, .5)


def test_four_categories_and_zero_failure_exact_interval():
    rows = [dict(shot_index=i, converged=valid, logical_failure=wrong, decode_cpu_seconds=1.)
            for i, (valid, wrong) in enumerate([(True, False), (True, True), (False, False), (False, True)])]
    stats = runner.summarize(rows, 4)
    assert stats["ler"] == stats["nonconvergence_rate"] == .5
    assert all(stats[key] == 1 for key in ("converged_correct", "converged_logical_failure",
                                         "unconverged_correct_observables", "unconverged_logical_failure"))
    with pytest.raises(ValueError, match="every requested shot"):
        runner.summarize(rows[:3], 4)
    zeros = runner.summarize([dict(r, logical_failure=False) for r in rows], 4)
    assert zeros["ler"] == 0 and zeros["ler_95_exact_two_sided_low"] == 0
    assert zeros["ler_95_exact_two_sided_high"] == pytest.approx(1 - .025**.25)
    assert zeros["ler_95_one_sided_upper"] == pytest.approx(1 - .05**.25)


def test_dem_adapter_checks_parity_and_rejects_lossy_signatures():
    importorskip_safe("relay_bp")
    H, O, priors = runner._audit_dem(stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 L0"))
    assert H.shape == O.shape == (1, 1)
    assert priors == pytest.approx([.1 * .8 + .2 * .9])
    with pytest.raises(ValueError, match="different-observable"):
        runner._audit_dem(stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0"))
    with pytest.raises(ValueError, match="detector-free"):
        runner._audit_dem(stim.DetectorErrorModel("error(0.1) L0"))


@dataclass(frozen=True)
class _ToyPlan:
    target: str = "Z"


@pytest.fixture
def toy_experiment(monkeypatch):
    class ToyExperiment:
        def __init__(self, code, sequence, *, plans=None, **kwargs):
            self.plans = tuple(plans) if plans is not None else (_ToyPlan(),)

        def build(self):
            return stim.Circuit("R 0\nTICK\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")

        @property
        def noise_schedule(self):
            return self.build()
    monkeypatch.setattr("lightstim.protocols.bb_ppm.BBPPMExperiment", ToyExperiment)
    return ToyExperiment


def test_campaign_publication_plot_contract_and_resume(toy_experiment, tmp_path, monkeypatch):
    importorskip_safe("relay_bp")
    from benchmarks.logical_ops.plot_bb_ppm import read_results
    output, directory = tmp_path / "results.csv", tmp_path / "run"
    arguments = dict(p_values=(.001,), shots=4, workers=1, run_dir=directory, output=output, seed=923)
    rows = runner.run_campaign(**arguments)
    data = read_results(output, shots_by_p={.001: 4}, decoder_params=runner.DEFAULT_DECODER_PARAMS,
                        expected_experiment={"noise_model": "circuit_level_with_idling", "rounds": 10})
    assert data["results"] == rows and data["status"] == "complete"
    samples = directory / "point_000/inputs/detectors.npy"
    before = samples.read_bytes(), samples.stat().st_mtime_ns
    monkeypatch.setattr(runner, "_audit_dem", lambda *_: pytest.fail("A prepared point must not be regenerated"))
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **_: pytest.fail("Completed decode repeated"))
    assert runner.run_campaign(**arguments) == rows
    assert (samples.read_bytes(), samples.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="Run identity changed"):
        runner.run_campaign(**dict(arguments, shots=5))
    with pytest.raises(ValueError, match="another campaign"):
        runner.run_campaign(**dict(arguments, run_dir=tmp_path / "another_run"))


def test_prepared_point_from_another_configuration_is_rejected(toy_experiment, tmp_path):
    importorskip_safe("relay_bp")
    clean = toy_experiment(None, []).build()
    point = tmp_path / "point"
    runner._prepare_point(clean, point, p=.001, shots=3, seed=41, point_index=0, batch_size=2)
    with pytest.raises(ValueError, match="Prepared point differs"):
        runner._prepare_point(clean, point, p=.002, shots=3, seed=41, point_index=0, batch_size=2)
    with pytest.raises(ValueError, match="Prepared point differs"):
        runner._prepare_point(clean, point, p=.001, shots=4, seed=41, point_index=0, batch_size=2)


def test_classical_frame_schedule_preserves_idle_noise_and_removes_markers(tmp_path):
    importorskip_safe("relay_bp")
    from lightstim.noise.config import NoiseConfig
    from lightstim.noise.injector import NoiseInjector
    from lightstim.qec_code.BB_code.ppm.frame import strip_frame_activity

    schedule = stim.Circuit(
        "R 0 1\nTICK\nH 1\nI[noiseless] 0\nTICK\nM 0 1\n"
        "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    clean = strip_frame_activity(schedule)
    point = tmp_path / "point"
    runner._prepare_point(clean, point, p=.001, shots=3, seed=41,
                          point_index=0, batch_size=2, noise_schedule=schedule)
    config = NoiseConfig(p_1q=.001, p_2q=.001, p_meas=.001, p_reset=.001, p_idle=.001)
    injector = NoiseInjector.from_circuit_level_with_idling(config, [0, 1])
    actual = stim.Circuit.from_file(point / "inputs/circuit.stim")
    assert actual == strip_frame_activity(injector.inject_noise(schedule))
    assert actual != injector.inject_noise(clean)
    assert actual.without_noise() == clean
    assert all(instruction.name != "I" for instruction in actual)


def test_stop_after_complete_point_does_not_sample_next_point(toy_experiment, tmp_path, monkeypatch):
    importorskip_safe("relay_bp")
    prepared = []
    original = runner._prepare_point
    def prepare(clean, point, **kwargs):
        prepared.append(kwargs["p"])
        return original(clean, point, **kwargs)
    monkeypatch.setattr(runner, "_prepare_point", prepare)
    def stopped(source, destination, **kwargs):
        destination.mkdir(parents=True)
        runner._atomic_json(destination / "manifest.json", {"status": "complete"})
        return [dict(shot_index=0, converged=True, logical_failure=False, decode_cpu_seconds=0.)], True, True
    monkeypatch.setattr(runner, "decode_saved_inputs", stopped)
    rows = runner.run_campaign(p_values=(.001, .002), shots=1, workers=1,
                              run_dir=tmp_path / "run", output=tmp_path / "out.csv")
    assert len(rows) == 1 and prepared == [.001]
    assert runner._read(tmp_path / "out.manifest.json")["status"] == "partial"


def test_output_reservation_recovers_csv_manifest_crash_and_rejects_other_owner(tmp_path):
    output = tmp_path / "result.csv"
    campaign = {"identity_sha256": "a" * 64, "run_id": "first"}
    runner._claim_output(output, campaign)
    output.write_text("interrupted publication")
    runner._claim_output(output, campaign)  # Correct owner may recover the first publication.
    with pytest.raises(ValueError, match="another campaign"):
        runner._claim_output(output, dict(campaign, run_id="second"))
    with runner._lock(tmp_path / "output.lock"):
        with pytest.raises(OSError):
            with runner._lock(tmp_path / "output.lock"):
                pytest.fail("Duplicate writer acquired the lock")


@pytest.mark.parametrize("arguments", [
    ["--workers", "0"], ["--p-values", "nan"], ["--p-values", "0"],
    ["--shots", "0"], ["--target", "X13"], ["--p-values", ".001", ".001"],
])
def test_cli_rejects_invalid_configuration(arguments):
    with pytest.raises(SystemExit) as error:
        parse_args(arguments)
    assert error.value.code == 2


def test_cli_accepts_both_target_forms_and_explicit_quick_overrides():
    first = parse_args(["--target", "X2 Z7 X12", "--quick", "--shots", "3", "--workers", "1"])
    second = parse_args(["--target", "X2", "Z7", "X12", "--quick", "--shots", "3"])
    assert first.target == second.target == ["X2", "Z7", "X12"]
    assert first.shots == [3, 3]
