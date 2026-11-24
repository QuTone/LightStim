# CrossLS Benchmark — Surface ↔ PQRM Lattice Surgery

Benchmarks for the **CrossLS** protocol: cross-code lattice surgery between an
unrotated surface code and a Punctured Quantum Reed-Muller (PQRM) code. This protocol
teleports a non-Clifford logical state from the PQRM block into the surface code via
a joint ZZ measurement, without ever applying the non-Clifford gate to the surface
code directly.

Protocol implementation: `lightstim/protocols/cross_ls/cross_ls_experiment.py`
Notebook walkthrough: `notebooks/CrossLS/cross_ls.ipynb`

## Background

PQRM codes natively support transversal non-Clifford gates (T, T^½, T^¼) but have
high-weight X-stabilizers that preclude standard syndrome extraction. CrossLS resolves
this via a **hybrid SE + post-selection** scheme:

- Z-stabilizers only are measured during the lattice surgery rounds (standard 6-tick SE)
- X-stabilizer violations are detected at final transversal MX readout
- Shots with X-stabilizer violations are discarded (post-selection)

The LightStim framework automatically tracks Pauli evolution across the heterogeneous
boundary and compiles the end-to-end DEM with no manual detector annotation.

### Supported PQRM codes

| Params (rx, rz, m) | Code | T-gate family |
|---|---|---|
| (1, 2, 4) | [[15, 1, 3]] | transversal T |
| (1, 3, 5) | [[31, 1, 5]] | transversal T^½ |
| (1, 4, 6) | [[63, 1, 7]] | transversal T^¼ |

## What this benchmark measures

Two experiments, selectable via `--experiment`:

| Experiment | What it sweeps | Output columns |
|---|---|---|
| `sweep` | PQRM params × d_surf × state (Z/X/Y) × p | `pqrm, d_surf, state, p, ler, ps_rate, ...` |
| `rounds` | rounds per LS cycle for the selected PQRM, distance, state, and p values | `rounds, d_surf, ler, ...` |

The runner checkpoints completed tasks in CSV. Use `--experiment` for CrossLS sweeps. Passing `--config FILE` runs the experiment declared in the file through the [shared experiment-config CLI](../../README.md#run-an-experiment-from-json).

Output: `benchmarks/cross_ls/results/cross_ls_results.csv`

## How to run

```bash
# Quick smoke test (d=3, PQRM(1,2,4), one p, 50k shots):
PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py --quick

# Full LER vs PER sweep (paper figures):
PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py \
    --experiment sweep \
    --pqrm 1,2,4 1,3,5 1,4,6 \
    --distances 3 5 7 \
    --p-values 5e-4 1e-3 2e-3 \
    --decoder mwpf --num-workers 8

# Rounds sweep (LER vs number of SE rounds):
PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py \
    --experiment rounds \
    --pqrm 1,2,4 --distances 3 5 7 \
    --rounds-values 3 5 7 9 --p-values 1e-3

# GPU BP+OSD (for large d or high p):
PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py \
    --experiment sweep --pqrm 1,2,4 1,3,5 \
    --distances 5 7 --p-values 5e-4 1e-3 2e-3 \
    --decoder bposd --backend gpu --num-workers 1
```

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `--experiment` | `all` | `sweep`, `rounds`, or both |
| `--pqrm` | `1,2,4` | PQRM params: `1,2,4` `1,3,5` `1,4,6` (space-separated) |
| `--distances` | `3 5 7` | Surface code distances |
| `--p-values` | `5e-4 1e-3 2e-3` | Physical error rates |
| `--rounds`, `--rounds-values` | distance for `sweep`, `3 5 7` for `rounds` | SE rounds |
| `--basis`, `--states` | `Z X Y` for `sweep`, `Z` for `rounds` | Logical input states |
| `--mode` | `hybrid` | `decode`, `full_postselection`, or `hybrid` |
| `--postselection-region`, `--ps-mode` | `hybrid` | Region for hybrid mode (`hybrid` or `pqrm_only`) |
| `--noise-model` | `circuit_level` | The circuit-level baseline uses `p_1q=1e-6` and `p_idle=0` |
| `--decoder` | `mwpf` | Hypergraph decoder, e.g. `mwpf` or `bposd` |
| `--backend` | `cpu` | `cpu` or `gpu` |
| `--seed` | `0` | Sampling seed |
| `--batch-size` | `10000` | Sampling batch size |
| `--num-workers` | `4` | Parallel workers (use 1 for GPU) |
| `--max-shots` | `1000000` | Max shots per task |
| `--max-errors` | `100` | Stop after N logical errors |
| `--quick` | off | Smoke test mode |

> **Decoder note**: CrossLS circuits contain hyperedges (PQRM X-stab correlations).
> The default `mwpf` and the optional `bposd` decoders support these correlations.

## How to plot

```bash
# LER vs p (sweep experiment):
PYTHONPATH=. venv/bin/python benchmarks/cross_ls/plot_cross_ls.py \
    benchmarks/cross_ls/results/cross_ls_results.csv \
    --experiment sweep

# LER vs rounds:
PYTHONPATH=. venv/bin/python benchmarks/cross_ls/plot_cross_ls.py \
    benchmarks/cross_ls/results/cross_ls_results.csv \
    --experiment rounds
```

## Output CSV columns

```
pqrm, d_surf, rounds, state, p_1q, p_2q, p_meas, p_reset,
decoder, backend, n_det, n_ps, shots, kept, ps_rate,
errors, ler, seconds
```

`ps_rate` is the fraction of accepted shots in the selected simulation mode. The CSV also records the common `accepted`, `acceptance`, `logical_error_rate`, and `sweep_coordinates` columns.
A lower `ps_rate` at higher `p` is expected — PQRM X-errors scale as O(p).
