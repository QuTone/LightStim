# State Injection Benchmark

General-purpose state injection benchmark for the unrotated surface code.

Sweeps `inject_state × injection_protocol × post_select_mode × distance × p`. Use the injection options below to run this protocol. Passing `--config FILE` runs the experiment declared in that file through the [shared experiment-config CLI](../../README.md#run-an-experiment-from-json).

```
benchmarks/state_injection/
├── run_state_injection.py   # Run benchmark, output CSV
├── plot_state_injection.py  # Plot LER + PS Rate from CSV
└── results/                 # Generated outputs (gitignored)
```

---

## Quick start

```bash
# Full sweep (all states/protocols/modes, d=3,5,7):
PYTHONPATH=. venv/bin/python benchmarks/state_injection/run_state_injection.py

# Quick smoke test:
PYTHONPATH=. venv/bin/python benchmarks/state_injection/run_state_injection.py --quick

# Plot results (by state, corner, full_postselection):
PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py

# Plot by PS mode (Z state, corner):
PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py \
    --mode modes --inject-state Z --inject-protocol corner
```

---

## `run_state_injection.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--basis`, `--inject-states` | Z X Y | States to inject |
| `--inject-protocols` | corner middle | Injection protocols |
| `--inject-modes` | all three | Historical modes `full_qec`, `full_postselection`, and `hybrid` |
| `--mode` | — | `decode`, `full_postselection`, or `hybrid` |
| `--postselection-region` | `injection` in hybrid mode | Filter the `injection` or `logical_strip` region before decoding |
| `--distances` | 3 5 7 | Code distances |
| `--p-values` | 1e-4 … 1e-2 | Physical error rates |
| `--rounds` | 2 | SE rounds |
| `--noise-model` | circuit_level | Noise model |
| `--decoder` | pymatching | Decoder algorithm |
| `--max-shots` | 1000000 | Shot budget per task |
| `--max-errors` | 100 | Stop after N logical errors |
| `--batch-size` | 10000 | Sampling batch size |
| `--seed` | 0 | Sampling seed |
| `--num-workers` | 4 | Parallel workers |
| `--quick` | off | Smoke test (Z/corner/full_ps, d=3,5, 2 p-values) |
| `--output` | results/state_injection_results.csv | Output CSV path |

The historical `--inject-modes full_postselection` filters the injection region and decodes the survivors. Its equivalent is `--mode hybrid --postselection-region injection`. With `--mode full_postselection`, every detector must be quiet and the runner scores the accepted shots' raw observables.

---

## `plot_state_injection.py` options

Use one noise model and decoder configuration per input CSV. Each curve
compares physical error rates within a state or postselection mode.

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | states | `states` (color by Z/X/Y) or `modes` (color by PS mode) |
| `--inject-protocol` | corner | Protocol shown when `--mode states` |
| `--ps-mode` | full_postselection | PS mode shown when `--mode states` |
| `--inject-state` | Z | State shown when `--mode modes` |
| `--distances` | all | Distances to plot |
| `--input` | results/state_injection_results.csv | Input CSV |
| `--output` | results/state_injection_plot.png | Output PNG |
