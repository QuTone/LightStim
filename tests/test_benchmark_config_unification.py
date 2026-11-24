"""Legacy CLI meanings that differ from the declarative experiment config."""

import json
import sys

import pandas as pd
import pytest

from benchmarks._config import benchmark_decoder, checkpoint_key, checkpoint_rows
from benchmarks.cross_ls import run_cross_ls
from benchmarks.state_injection import run_state_injection


def test_checkpoint_preserves_nearby_64_bit_seeds_and_noise_rates(tmp_path):
    path = tmp_path / "mixed.csv"
    fields = ("seed", "p")
    old = 2**64 - 2
    path.write_text(f"seed,p\n,0.001\n{old},0.0010000000000000002\n")
    saved = {checkpoint_key(row, fields) for row in checkpoint_rows(path)}
    assert checkpoint_key({"seed": old, "p": 0.0010000000000000002}, fields) in saved
    assert checkpoint_key({"seed": old, "p": 0.001}, fields) not in saved
    assert checkpoint_key({"seed": old + 1, "p": 0.0010000000000000002}, fields) not in saved


def test_decoder_names_select_the_historical_gpu_implementation():
    native = benchmark_decoder("nv-qldpc-decoder")
    with pytest.warns(FutureWarning, match="deprecated"):
        alias = benchmark_decoder("gpu_bposd")
    canonical = benchmark_decoder("bposd", "gpu")
    assert alias == canonical
    assert native.backend == "gpu"
    assert native.params == canonical.params


def test_injection_region_and_all_detector_postselection_are_distinct(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    common = ["run_state_injection.py", "--basis", "Z", "--inject-protocols", "corner",
              "--max-shots", "2", "--max-errors", "2", "--batch-size", "2",
              "--distances", "3", "--p-values", "0.001", "--num-workers", "1",
              "--seed", "17", "--output", str(path)]
    for flags in (["--inject-modes", "full_postselection"],
                  ["--mode", "hybrid", "--postselection-region", "injection"],
                  ["--mode", "full_postselection"]):
        monkeypatch.setattr(sys, "argv", common + flags)
        run_state_injection.main()
    rows = pd.read_csv(path)
    assert list(rows["mode"]) == ["hybrid", "full_postselection"]
    assert rows["postselection_region"].iloc[0] == "injection"
    assert list(rows["decoder_name"]) == ["pymatching", "none"]
    assert list(rows["seed"]) == [17, 17]


def test_historical_crossls_checkpoint_uses_only_matching_input_settings():
    old = dict(experiment="sweep", pqrm="1-2-4", d_surf=3, rounds=3, state="Z",
               p_1q=1e-6, p_2q=0.001, ps_mode="hybrid", decoder="mwpf", backend="cpu",
               n_det=88, n_ps=12, shots=50000, kept=40676, errors=10)
    new = dict(old, seed=0, noise_model="circuit_level", noise_custom_params="{}", p_idle=0,
               on_decode_failure="error", mode="hybrid",
               decoder_params=json.dumps({"cluster_node_limit": 50}))
    assert run_cross_ls._ck_key(old) == run_cross_ls._ck_key(new)
    assert run_cross_ls._ck_key(old) != run_cross_ls._ck_key(dict(new, seed=1))
    assert run_cross_ls._ck_key(old) != run_cross_ls._ck_key(dict(new, mode="decode"))


@pytest.mark.parametrize("kind", ["ls", "tg"])
def test_distillation_passes_background_model_seed_and_backend_to_both_stages(kind, monkeypatch):
    import numpy as np
    import stim
    from types import SimpleNamespace
    from lightstim.noise.config import NoiseConfig
    from lightstim.protocols import ls_distillation, tg_distillation

    protocol = ls_distillation if kind == "ls" else tg_distillation
    runs = []
    class CapturePipeline:
        def __init__(self, **kwargs):
            self.settings = kwargs

        def run(self, circuit):
            runs.append((self.settings, str(circuit)))
            return SimpleNamespace(logical_error_rate=0.0)

    monkeypatch.setattr(protocol, "SimulationPipeline", CapturePipeline)
    options = dict(seed=17, backend="gpu", noise_model="phenomenological", decoder_name="bposd",
                   background_noise=NoiseConfig(p_2q=0.1, p_meas=0.05))
    protocol.estimate_p_in(3, 1, p_injected=0, p_background=0.1, **options)
    circuit = stim.Circuit("R 0 1\nCX 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
    analysis = {"ps_obs": [], "target_obs": [0]} if kind == "ls" else {
        "T": np.eye(1), "ps_indices": [], "target_indices": [0],
    }
    protocol.run_simulation(circuit, set(), 0.1, 0, "full", **analysis, **options)
    for settings, noisy_circuit in runs:
        assert settings["seed"] == 17
        assert settings["decoder_config"].backend == "gpu"
        assert "DEPOLARIZE2" not in noisy_circuit
        assert "X_ERROR(0.05)" in noisy_circuit


def test_crossls_code_capacity_includes_surface_data_initialized_in_clean_prefix():
    from lightstim.noise.config import NoiseConfig
    from lightstim.protocols.cross_ls import CrossLSExperiment
    from lightstim.protocols.cross_ls.cross_ls_experiment import _get_surface_data_indices

    exp = CrossLSExperiment([1, 2, 4], 3, rounds=1, noise_model="code_capacity",
                            noise_params=NoiseConfig(p_idle=0.01))
    circuit = exp.build()
    noisy_qubits = {target.value for op in circuit.flattened() if op.name == "DEPOLARIZE1"
                    for target in op.targets_copy()}
    assert set(_get_surface_data_indices(exp.system)) <= noisy_qubits
