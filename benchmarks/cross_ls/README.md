# CrossLS Benchmark — Surface ↔ PQRM Lattice Surgery

Benchmarks for the **CrossLS** protocol: cross-code lattice surgery between an
unrotated surface code and a Punctured Quantum Reed-Muller (PQRM) code. This protocol
teleports a non-Clifford logical state from the PQRM block into the surface code via
a joint ZZ measurement, without ever applying the non-Clifford gate to the surface
code directly.

Protocol implementation: `lightstim/protocols/cross_ls.py`
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
| `rounds` | rounds per LS cycle, fixed PQRM=(1,2,4), p=1e-3, Z state | `rounds, d_surf, ler, ...` |

All results use per-task checkpointing — safe to interrupt and resume.

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
| `--experiment` | `sweep` | `sweep` or `rounds` |
| `--pqrm` | `1,2,4` | PQRM params: `1,2,4` `1,3,5` `1,4,6` (space-separated) |
| `--distances` | `3 5 7` | Surface code distances |
| `--p-values` | `5e-4 1e-3 2e-3` | Physical error rates |
| `--rounds-values` | `3 5 7` | SE rounds (rounds experiment only) |
| `--states` | `Z X Y` | Logical input states |
| `--decoder` | `mwpf` | `mwpf`, `bposd` |
| `--backend` | `cpu` | `cpu` or `gpu` |
| `--num-workers` | `8` | Parallel workers (use 1 for GPU) |
| `--max-shots` | `500000` | Max shots per task |
| `--max-errors` | `100` | Stop after N logical errors |
| `--quick` | off | Smoke test mode |

> **Decoder note**: CrossLS circuits contain hyperedges (PQRM X-stab correlations).
> Always use `mwpf` or `bposd` — never `pymatching`.

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

`ps_rate` is the fraction of shots surviving PQRM X-stabilizer post-selection.
A lower `ps_rate` at higher `p` is expected — PQRM X-errors scale as O(p).
