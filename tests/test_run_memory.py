"""
Tests for benchmarks/memory/run_memory.py

Three layers:
  1. Unit: build_circuit() for every code/noise-model/basis combination (no simulation)
  2. Unit: decoder config, checkpointing key stability
  3. Integration: CLI end-to-end with tiny shot counts (max_shots=300, max_errors=3)

GPU tests are skipped unless CUDA is available.
All tests are designed to complete in < 2 minutes total on a CPU machine.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "benchmarks" / "memory" / "run_memory.py"
PYTHON = Path(sys.executable)              # the running interpreter, not a hardcoded venv

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "memory"))

from run_memory import (
    COLOR_SE_CIRCUITS,
    _BB_CONFIGS,
    _HGP_CONFIGS,
    _TOPO_CODES,
    _decoder_config,
    _task_key,
    build_circuit,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _has_bposd() -> bool:
    """True if the CPU BP+OSD decoder is registered (stimbposd or ldpc installed).

    Part of the optional ``[decoders]`` extra, so a core-only install skips
    the bposd CLI tests instead of failing — mirroring how the FastAPI tests
    skip when ``fastapi`` is absent.
    """
    try:
        from lightstim.simulation.decoder_backend.registry import list_decoders
        return "bposd" in list_decoders()
    except Exception:
        return False


def _has_mwpf() -> bool:
    try:
        from lightstim.simulation.decoder_backend.registry import list_decoders
        return "mwpf" in list_decoders()
    except Exception:
        return False


# ── Layer 1: circuit building ─────────────────────────────────────────────────

@pytest.mark.parametrize("code,dist", [
    ("rotated_sc",   3),
    ("unrotated_sc", 3),
    ("toric",        3),
    ("color",        3),
    ("xzzx_sc",      3),
])
def test_build_circuit_topo(code, dist):
    circuit, n_data, n_total, k = build_circuit(code, dist, p=1e-2)
    assert circuit.num_qubits > 0
    assert n_data > 0
    assert n_total >= n_data
    assert circuit.num_detectors > 0
    assert circuit.num_observables > 0


@pytest.mark.parametrize("code", sorted(_BB_CONFIGS.keys()))
def test_build_circuit_bb(code):
    d = _BB_CONFIGS[code]["d"]
    circuit, n_data, n_total, k = build_circuit(code, d, p=1e-2)
    assert circuit.num_qubits > 0
    assert k > 1  # BB codes are high-rate


@pytest.mark.parametrize(
    "code_name,expected_n_data,expected_n_total,expected_k",
    [
        ("hgp_13_1_3", 13, 25, 1),
        ("hgp_18_2_3", 18, 36, 2),
        ("hgp_225_9_4", 225, 441, 9),
    ],
)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_build_circuit_hgp(
    code_name,
    expected_n_data,
    expected_n_total,
    expected_k,
    basis,
):
    cfg = _HGP_CONFIGS[code_name]
    circuit, n_data, n_total, k = build_circuit(
        code_name,
        cfg["d"],
        p=1e-2,
        basis=basis,
        se_circuit=cfg["se_circuit"],
    )

    assert n_data == expected_n_data
    assert n_total == expected_n_total
    assert k == expected_k
    assert circuit.num_detectors > 0
    assert circuit.num_observables == k
    dem = circuit.detector_error_model()
    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == k


def test_hgp_rejects_unknown_se_circuit():
    with pytest.raises(ValueError, match="Unknown HGP SE circuit"):
        build_circuit(
            "hgp_225_9_4",
            4,
            p=1e-2,
            se_circuit="not_a_schedule",
        )


def test_build_circuit_xzzx_gets_checkerboard_basis():
    """build_circuit must wire the checkerboard map for xzzx_sc: without it the
    circuit still builds but degenerates to 16 detectors / distance 1."""
    circuit, *_ = build_circuit("xzzx_sc", 3, p=1e-2)
    assert circuit.num_detectors == 24
    assert len(circuit.shortest_graphlike_error()) == 3


@pytest.mark.parametrize("noise_model", ["circuit_level", "phenomenological", "code_capacity"])
def test_build_circuit_noise_models(noise_model):
    circuit, *_ = build_circuit("rotated_sc", 3, p=1e-2, noise_model=noise_model)
    assert circuit.num_qubits > 0


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_build_circuit_bases(basis):
    circuit, *_ = build_circuit("rotated_sc", 3, p=1e-2, basis=basis)
    assert circuit.num_qubits > 0


def test_build_circuit_custom_rounds():
    # rounds != distance should still build without error
    circuit, *_ = build_circuit("rotated_sc", 3, p=1e-2, rounds=5)
    assert circuit.num_qubits > 0


@pytest.mark.parametrize("se_circuit", list(COLOR_SE_CIRCUITS))
def test_build_circuit_color_se_circuits(se_circuit):
    circuit, n_data, n_total, k = build_circuit(
        "color",
        3,
        p=1e-2,
        basis="X",
        se_circuit=se_circuit,
    )
    assert circuit.num_qubits > 0
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1
    assert n_total >= n_data > 0
    assert k == 1


@pytest.mark.parametrize(
    "se_circuit",
    [
        "space_multiplexing",
        "bell_multiplexing",
        "bell_flagging",
        "time_multiplexing",
    ],
)
def test_build_circuit_color_y_basis(se_circuit):
    circuit, *_ = build_circuit(
        "color",
        3,
        p=1e-2,
        basis="Y",
        se_circuit=se_circuit,
    )
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1


def test_build_circuit_middle_out_rejects_y_basis():
    with pytest.raises(ValueError, match="supports basis X, Z"):
        build_circuit(
            "color",
            3,
            p=1e-2,
            basis="Y",
            se_circuit="middle_out",
        )


# ── Layer 2: decoder config + checkpointing ───────────────────────────────────

@pytest.mark.parametrize("name", ["pymatching", "mwpf", "cpu_bposd"])
def test_decoder_config_cpu(name):
    cfg = _decoder_config(name)
    assert cfg.backend == "cpu"
    assert cfg.name is not None


def test_decoder_config_gpu():
    cfg = _decoder_config("gpu_bposd")
    assert cfg.backend == "gpu"


def test_task_key_excludes_result_columns():
    """Result columns must not affect the task key (checkpointing correctness)."""
    base = {
        "code": "rotated_sc", "distance": 3, "p": 1e-3,
        "basis": "Z", "rounds": 3, "noise_model": "circuit_level",
        "se_circuit": "default", "decoder_name": "pymatching",
    }
    with_results = {**base,
                    "shots": 1000, "errors": 10, "logical_error_rate": 0.01,
                    "seconds": 5.0, "n_data": 9, "n_total": 17, "k": 1,
                    "layout": "code_default", "block_class": "code_default"}
    assert _task_key(base) == _task_key(with_results)


def test_task_key_distinguishes_inputs():
    """Different inputs must produce different keys."""
    a = {"code": "rotated_sc", "distance": 3, "p": 1e-3,
         "basis": "Z", "rounds": 3, "se_circuit": "default",
         "noise_model": "circuit_level", "decoder_name": "pymatching"}
    b = {**a, "distance": 5}
    assert _task_key(a) != _task_key(b)


def test_task_key_distinguishes_color_se_circuits():
    a = {
        "code": "color",
        "distance": 3,
        "p": 1e-3,
        "basis": "Z",
        "rounds": 3,
        "se_circuit": "space_multiplexing",
        "noise_model": "circuit_level",
        "decoder_name": "mwpf",
    }
    b = {**a, "se_circuit": "bell_multiplexing"}
    assert _task_key(a) != _task_key(b)


def test_plot_keeps_color_se_circuits_separate():
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("seaborn")
    from plot_memory import plot_ler_vs_p

    rows = []
    for se_circuit, scale in [
        ("space_multiplexing", 1.0),
        ("bell_flagging", 2.0),
    ]:
        for p in [1e-3, 2e-3]:
            rows.append({
                "code": "color",
                "distance": 3,
                "p": p,
                "basis": "Z",
                "rounds": 3,
                "se_circuit": se_circuit,
                "noise_model": "circuit_level",
                "decoder_name": "mwpf",
                "logical_error_rate": p * scale,
            })

    fig, ax = plt.subplots()
    plot_ler_vs_p(pd.DataFrame(rows), ax)
    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {
        "color space_multiplexing d=3 Z mwpf",
        "color bell_flagging d=3 Z mwpf",
    }
    plt.close(fig)


# ── Layer 3: CLI integration ───────────────────────────────────────────────────

def _run_cli(args: list, tmp_out: Path, timeout: int = 90) -> subprocess.CompletedProcess:
    cmd = [str(PYTHON), str(RUNNER)] + args + ["--output", str(tmp_out)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_cli_rotated_sc(tmp_path):
    out = tmp_path / "rotated_sc.csv"
    r = _run_cli(["--codes", "rotated_sc", "--distances", "3",
                  "--p-values", "5e-3", "--max-shots", "300", "--max-errors", "3"], out)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df["code"].iloc[0] == "rotated_sc"
    assert set(["noise_model", "shots", "errors", "logical_error_rate"]).issubset(df.columns)


def test_cli_both_bases(tmp_path):
    out = tmp_path / "both_bases.csv"
    r = _run_cli(["--codes", "rotated_sc", "--distances", "3",
                  "--p-values", "5e-3", "--basis", "Z", "X",
                  "--max-shots", "300", "--max-errors", "3"], out)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert len(df) == 2
    assert set(df["basis"]) == {"Z", "X"}


def test_cli_noise_model_phenomenological(tmp_path):
    out = tmp_path / "phenom.csv"
    r = _run_cli(["--codes", "rotated_sc", "--distances", "3",
                  "--p-values", "5e-3", "--noise-model", "phenomenological",
                  "--max-shots", "300", "--max-errors", "3"], out)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert df["noise_model"].iloc[0] == "phenomenological"


def test_cli_checkpoint_resume(tmp_path):
    """Second identical run must skip all tasks (no duplicate rows)."""
    out = tmp_path / "checkpoint.csv"
    cmd_args = ["--codes", "rotated_sc", "--distances", "3",
                "--p-values", "5e-3", "--max-shots", "300", "--max-errors", "3"]
    r1 = _run_cli(cmd_args, out)
    assert r1.returncode == 0, r1.stderr

    r2 = _run_cli(cmd_args, out)
    assert r2.returncode == 0, r2.stderr
    assert "skipping" in r2.stdout or "0 to run" in r2.stdout

    df = pd.read_csv(out)
    assert len(df) == 1  # no duplicates appended


@pytest.mark.skipif(not _has_mwpf(), reason="mwpf decoder unavailable")
def test_cli_color_se_circuits(tmp_path):
    out = tmp_path / "color.csv"
    r = _run_cli(
        [
            "--codes", "color",
            "--distances", "3",
            "--p-values", "5e-3",
            "--basis", "X",
            "--color-se-circuits",
            "space_multiplexing",
            "bell_multiplexing",
            "--decoder", "mwpf",
            "--max-shots", "30",
            "--max-errors", "1",
            "--batch-size", "10",
            "--num-workers", "1",
        ],
        out,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert len(df) == 2
    assert set(df["se_circuit"]) == {
        "space_multiplexing",
        "bell_multiplexing",
    }
    assert set(df["layout"]) == {"superdense"}
    assert set(df["block_class"]) == {
        "ColorCodeSpaceMultiplexingBlock",
        "ColorCodeBellMultiplexingBlock",
    }


def test_cli_rejects_middle_out_y(tmp_path):
    out = tmp_path / "invalid.csv"
    r = _run_cli(
        [
            "--codes", "color",
            "--distances", "3",
            "--basis", "Y",
            "--color-se-circuits", "middle_out",
        ],
        out,
    )
    assert r.returncode != 0
    assert "middle_out supports only X/Z" in r.stderr
    assert not out.exists()


def test_cli_rejects_more_than_48_workers(tmp_path):
    out = tmp_path / "invalid_workers.csv"
    r = _run_cli(
        ["--codes", "rotated_sc", "--distances", "3", "--num-workers", "49"],
        out,
    )
    assert r.returncode != 0
    assert "--num-workers must be between 1 and 48" in r.stderr
    assert not out.exists()


@pytest.mark.skipif(not _has_bposd(), reason="bposd decoder unavailable; pip install -e '.[decoders]'")
def test_cli_cpu_bposd_surface(tmp_path):
    """cpu_bposd on rotated_sc d=3 — verifies decoder config, fast."""
    out = tmp_path / "bposd_surface.csv"
    r = _run_cli(["--codes", "rotated_sc", "--distances", "3",
                  "--p-values", "5e-3", "--decoder", "cpu_bposd",
                  "--max-shots", "300", "--max-errors", "3"], out)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert df["decoder_name"].iloc[0] == "cpu_bposd"


@pytest.mark.slow
@pytest.mark.skipif(not _has_bposd(), reason="bposd decoder unavailable; pip install -e '.[decoders]'")
def test_cli_cpu_bposd_bb(tmp_path):
    """cpu_bposd on bb_72_12_6 — slow due to OSD precomputation."""
    out = tmp_path / "bb_bposd.csv"
    r = _run_cli(["--codes", "bb_72_12_6", "--p-values", "1e-2",
                  "--decoder", "cpu_bposd",
                  "--max-shots", "300", "--max-errors", "3"], out, timeout=300)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert df["decoder_name"].iloc[0] == "cpu_bposd"


@pytest.mark.slow
@pytest.mark.skipif(not _has_cuda(), reason="No CUDA device available")
def test_cli_gpu_bposd_bb(tmp_path):
    """BB code with GPU decoder — requires CUDA, slow CUDA init."""
    out = tmp_path / "bb_gpu.csv"
    r = _run_cli(["--codes", "bb_72_12_6", "--p-values", "1e-2",
                  "--decoder", "gpu_bposd",
                  "--max-shots", "300", "--max-errors", "3"], out, timeout=300)
    assert r.returncode == 0, r.stderr


def test_cli_multiple_codes_and_p_values(tmp_path):
    """Multiple codes + distances + p values → correct row count."""
    out = tmp_path / "multi.csv"
    r = _run_cli(["--codes", "rotated_sc", "toric",
                  "--distances", "3", "5",
                  "--p-values", "1e-3", "5e-3",
                  "--max-shots", "300", "--max-errors", "3"], out)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    # 2 codes × 2 distances × 2 p = 8 rows
    assert len(df) == 8
