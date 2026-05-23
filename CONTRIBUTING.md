# Contributing to LightStim

## Running the test suite

All Python commands use the project virtualenv:

```bash
# Fast regression — run this before every PR (~2 min, CPU only)
venv/bin/python -m pytest tests/ -v -k "not slow"

# Full suite including slow tests (BPOSD precomputation, etc.)
venv/bin/python -m pytest tests/ -v

# Single module
venv/bin/python -m pytest tests/test_circuit_build.py -v
```

Slow tests are marked `@pytest.mark.slow` and gated on `@pytest.mark.skipif` for GPU tests.
Run them only when you have time and (for GPU tests) a CUDA device available.

## What tests cover

| File | What it tests |
|------|---------------|
| `tests/test_circuit_build.py` | Noiseless circuit build for all experiments (memory, CNOT, LS, injection, GHZ) |
| `tests/test_back_propagated_pauli.py` | SyndromeTracker back-propagation correctness |
| `tests/test_run_memory.py` | `benchmarks/memory/run_memory.py` — unit + CLI integration |
| `tests/test_simulation_backend_quality.py` | LER sanity checks for simulation pipeline |
| `tests/test_mwpf_visualization.py` | MWPF decoder visualization |
| `tests/color_code/test_color_code.py` | Color code geometry, algebra, syndrome extraction |

## Adding tests for a new QEC code

1. **`tests/test_circuit_build.py`** — add a parametrize entry that builds a noiseless memory circuit at the smallest valid distance. This covers the core `QECPatch → CircuitBuilder → SyndromeTracker` pipeline.

2. **`tests/test_run_memory.py`** — if the code is a BB-style LDPC code, add it to `_BB_CONFIGS` in `benchmarks/memory/run_memory.py` and add a `test_build_circuit_bb` parametrize entry.

## Adding tests for core infrastructure changes

If you modify `SyndromeTracker` (`lightstim/ir/tracker.py`) or `CircuitBuilder` (`lightstim/ir/builder.py`), the full `test_circuit_build.py` suite is your first regression check — it exercises every experiment type. `test_back_propagated_pauli.py` specifically targets the back-propagation logic in the tracker.

## Benchmark modules

Each benchmark module under `benchmarks/<module>/` has its own `README.md` with a "Running Tests" section. The pattern is:

- Unit tests: build circuits without running simulation
- CLI integration tests: end-to-end with `--max-shots 300 --max-errors 3` (fast)
- Slow tests: full decoder precomputation or GPU init, marked `@pytest.mark.slow`

## Coding conventions

- All imports use `lightstim.*` (not `src.*`)
- Eval scripts must checkpoint per-task (append CSV on task completion, never batch-save at end)
- GPU experiments: check `nvidia-smi` for free GPU before running; default `num_workers=1` for GPU decoder
