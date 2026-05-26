# Logical Circuits Benchmark

Benchmarks for multi-patch logical circuit protocols: Bell-state teleportation,
routing overhead, and magic state distillation (LS and TG 7-to-1).

All experiments share a single unified runner with per-task checkpointing.

## Experiments

| `--experiment` | Protocol | Output CSV |
|---|---|---|
| `bell_tele` | Bell-state teleportation via TG / ZZ-LS / XX-LS | `results/bell_tele_results.csv` |
| `routing` | LER vs routing distance (ZZ-LS and XX-LS, fixed d) | `results/bell_tele_results.csv` |
| `distill_ls` | LS 7-to-1 \|Y⟩ distillation (Steane protocol) | `results/distill_ls_results.csv` |
| `distill_tg` | TG 7-to-1 \|Y⟩ distillation (PQRM hypercube) | `results/distill_tg_results.csv` |
| `all` | All of the above | all three CSVs |

Protocol implementations:
- Bell teleportation: `lightstim/protocols/bell_teleportation.py`
- LS distillation: `lightstim/protocols/ls_distillation.py`
- TG distillation: `lightstim/protocols/tg_distillation.py`

## How to run

```bash
# Quick smoke test (d=3,5, 2 p-values):
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py --quick

# Bell teleportation sweep:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment bell_tele \
    --distances 3 5 7 \
    --p-values 5e-4 1e-3 2e-3 5e-3

# LS 7-to-1 distillation, circuit-level noise:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_ls \
    --distances 3 5 7 \
    --p-values 1e-3 3e-3 5e-3

# LS distillation, injection-only noise (post-selection overhead analysis):
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_ls \
    --noise-mode injection \
    --p-injected 1e-3 5e-3 2e-2

# TG 7-to-1 distillation (GPU recommended for d=7):
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_tg \
    --distances 3 5 7 \
    --p-values 1e-3 3e-3 5e-3 \
    --decoder nv-qldpc-decoder --num-workers 1

# All experiments:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py --experiment all
```

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `--experiment` | `bell_tele` | Experiment to run (see table above) |
| `--distances` | `3 5 7` | Code distances |
| `--p-values` | `5e-4 1e-3 2e-3 5e-3` | Physical error rates |
| `--noise-mode` | `circuit_level` | `circuit_level` or `injection` |
| `--p-injected` | — | Injection noise p (injection mode only) |
| `--decoder` | auto | `pymatching`, `bposd`, `mwpf`, `nv-qldpc-decoder` |
| `--num-workers` | `8` | Parallel CPU workers (use 1 for GPU) |
| `--max-shots` | `1e9` | Max shots per task |
| `--max-errors` | `100` | Stop after N logical errors |
| `--quick` | off | Smoke test: d=3,5, 2 p-values |

> **Decoder guidance**: Bell teleportation (TG protocol) and TG distillation produce
> hyperedge DEMs. Use `mwpf` (CPU) or `nv-qldpc-decoder` (GPU) for these.
> LS-based experiments (ZZ-LS, XX-LS, LS distillation) use `pymatching`.

## Output format

### bell_tele / routing

```
gate, protocol, state, routing_mult, d, rounds, p,
shots, errors, logical_error_rate, decoder, seconds
```

### distill_ls / distill_tg

```
experiment, d, rounds, p_injected, noise_mode, p, p_in,
shots, post_selected_shots, post_selection_rate,
errors, logical_error_rate, decoder, seconds
```

## How to plot

```bash
# Bell teleportation:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_bell_tele.py

# Routing overhead:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_routing_combined.py
```

> Distillation plot scripts are in `distillation/ls_7to1/` and `distillation/tg_7to1/`.
