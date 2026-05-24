# State Injection Benchmark

General-purpose state injection benchmark for the unrotated surface code.

Sweeps `inject_state × injection_protocol × post_select_mode × distance × p`.

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
| `--inject-states` | Z X Y | States to inject |
| `--inject-protocols` | corner middle | Injection protocols |
| `--inject-modes` | all three | Post-selection modes |
| `--distances` | 3 5 7 | Code distances |
| `--p-values` | 1e-4 … 1e-2 | Physical error rates |
| `--rounds` | 2 | SE rounds |
| `--decoder` | pymatching | Decoder backend |
| `--max-shots` | 1e9 | Shot budget per task |
| `--max-errors` | 100 | Stop after N logical errors |
| `--num-workers` | 8 | Parallel workers |
| `--quick` | off | Smoke test (Z/corner/full_ps, d=3,5, 2 p-values) |
| `--output` | results/state_injection_results.csv | Output CSV path |

---

## `plot_state_injection.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | states | `states` (color by Z/X/Y) or `modes` (color by PS mode) |
| `--inject-protocol` | corner | Protocol shown when `--mode states` |
| `--ps-mode` | full_postselection | PS mode shown when `--mode states` |
| `--inject-state` | Z | State shown when `--mode modes` |
| `--distances` | all | Distances to plot |
| `--input` | results/state_injection_results.csv | Input CSV |
| `--output` | results/state_injection_plot.png | Output PNG |
