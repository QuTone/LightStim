# Logical Operations Benchmark

General-purpose benchmark for logical gate performance on surface codes.
Sweeps gate types × distances × physical error rates, saves results to CSV, and plots LER vs p.

## What this benchmark does

Measures the logical error rate (LER) of logical operations under
circuit-level noise:

| Gate | Protocol | Sub-experiments |
|------|----------|-----------------|
| `H` | Fold-transversal Hadamard | Z→X and X→Z (2) |
| `S_oneway` | Unrotated fold-transversal S | noisy S, noiseless S†+MX (1) |
| `S_roundtrip` | Unrotated fold-transversal S·S† | S_roundtrip (1) |
| `S_rotated` | Rotated-code mid-cycle dynamical S/S† | noisy roundtrip and two gate-only +Y checks (3) |
| `CNOT_trans` | Transversal CNOT | ZZ_ZZ, ZX_ZX, XZ_XX, XZ_ZZ, XX_XX (5) |
| `CNOT_LS_ZZ_XX` | Lattice Surgery CNOT (ZZ-XX protocol) | same 5 sub-experiments |
| `CNOT_LS_XX_ZZ` | Lattice Surgery CNOT (XX-ZZ protocol) | same 5 sub-experiments |
| `TwoPatchLS_unrotated_XX` | Unrotated surface-code joint XX measurement | LS_XX (1) |
| `TwoPatchLS_unrotated_ZZ` | Unrotated surface-code joint ZZ measurement | LS_ZZ (1) |
| `TwoPatchLS_rotated_XX` | Rotated surface-code joint XX measurement | LS_XX (1, rounds=d) |
| `TwoPatchLS_rotated_ZZ` | Rotated surface-code joint ZZ measurement | LS_ZZ (1, rounds=d) |
| `memory` | X- and Z-basis memory baseline (rounds=d) | memory_X and memory_Z (2) |

All results are written to a single combined CSV with per-task checkpointing —
safe to interrupt and resume.

## How to run

Run the three-patch lattice-surgery CNOT example from the repository root with the [shared experiment-config CLI](../../README.md#run-an-experiment-from-json).

```bash
python benchmarks/logical_ops/run_logical_ops.py --config examples/config/lattice-surgery-cnot.json --quick --output cnot.csv
```

Use the gate options below to run the built-in benchmark sweeps.

```bash
# All gates, default sweep (d=3,5,7; p=5e-4…1e-2):
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py

# Single gate:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py --gate H

# Rotated two-patch lattice surgery with PyMatching:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py \
    --gate TwoPatchLS_rotated_XX TwoPatchLS_rotated_ZZ

# Rotated-code logical S, production grid and historical BP+OSD settings:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py \
    --gate S_rotated --decoder bposd --backend gpu \
    --distances 3 5 7 \
    --p-values 5e-4 1e-3 2e-3 3e-3 5e-3 7e-3 1e-2 \
    --max-shots 1000000000 --max-errors 100 \
    --batch-size 50000 --num-workers 1

# Custom distances and p values:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py \
    --gate CNOT_trans --distances 3 5 7 9 --p-values 1e-4 1e-3 1e-2

# Quick test (2 distances, 2 p values, 100k shots):
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py --quick

# Custom output path:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py \
    --output benchmarks/logical_ops/results/my_run.csv
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--gate` | all | Gate(s) to run; use `--help` for the complete list |
| `--distances` | `3 5 7` | Code distances |
| `--p-values` | `5e-4 1e-3 2e-3 5e-3 1e-2` | Physical error rates |
| `--rounds` | `2` | SE rounds for gate benchmarks (memory and rotated two-patch LS use rounds=d) |
| `--decoder` | `pymatching` for memory and surface-code LS; `bposd` for other gates | Decoder |
| `--noise-model` | `circuit_level` | Noise model applied to the gate circuit |
| `--max-shots` | `1000000` | Max shots per task |
| `--max-errors` | `100` | Stop after this many errors |
| `--num-workers` | `4` | Parallel workers |
| `--seed` | `0` | Sampling seed |
| `--batch-size` | `10000` | Sampling batch size |
| `--quick` | off | Fast test mode |

## How to plot results

```bash
# Default (reads results/logical_ops_results.csv):
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py

# Filter to specific gates:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py --gate H S

# Custom input/output:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py \
    --input benchmarks/logical_ops/results/my_run.csv \
    --output benchmarks/logical_ops/results/my_plot.png
```

Output: `results/logical_ops_plot.png` — one subplot per gate, LER vs p on log-log axes,
one curve per distance.

`S_rotated` is expanded into three panels rather than averaged: the noisy
S→S† full-circuit roundtrip, the gate-only +Y→S†→+X experiment, and the
gate-only +Y→S→−X experiment. The deterministic −X reference removes the need
for a physical logical-Pauli correction. Its single-gate default checkpoint is
`results/rotated_s_results.csv`.

The production sweep uses the existing decoder configuration from
`run_logical_ops.py`: min-sum BP, 1000 iterations, OSD-CS order 10, and dynamic
min-sum scaling (`ms_scaling_factor=0`). The command above uses the seven-point
physical-error grid shared with the unrotated logical-S experiment and stops
after 100 logical errors. `--quick` is only a circuit/pipeline smoke test; its
low-error points are not benchmark data.

## Related benchmarks

### Gross-code Pauli product measurements

The [BB PPM notebook](../../notebooks/LogicalOps/logical_PPM_BB_code.ipynb)
builds the logical `Y1` measurement and plots the recorded full-circuit error
rates. At `p = 0.0005`, `0.001`, and `0.002`, Relay-BP produced 4 failures
in 1,000,000 shots, 19 in 1,000,000, and 116 in 100,000, respectively.
The [recorded counts](reference/bb_ppm/y1_large_20261124/results.csv) and
[run configuration](reference/bb_ppm/y1_large_20261124/results.manifest.json)
are included in the repository.

To run a five-shot check using the same target and noise model, install
`relay-bp[stim]==0.2.2` and run the following from the repository root.

```bash
python benchmarks/logical_ops/run_bb_ppm.py --target Y1 --p-values 0.001 --shots 5 --workers 1 \
    --run-dir benchmarks/logical_ops/results/bb_y1_check \
    --output benchmarks/logical_ops/results/bb_y1_check/results.csv
```

The runner checkpoints completed shots. Use a new run directory when changing
the target, noise points, shot budgets, or decoder configuration.

### State injection

For state injection benchmarks (Z/X/Y state injection into the unrotated surface code),
see [`benchmarks/state_injection/`](../state_injection/README.md).

## Paper figures

For the exact paper figures (Figures 1-6), see `paper_artifact/logical_ops/`:

```bash
# Run all paper figures:
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py

# Plot a specific figure:
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig4.py
```

## Output CSV columns

Each row includes the common shot counts, decoder name and backend, and sweep coordinates. Gate-specific columns include the sub-experiment and bases.

```
gate, sub_experiment, init_basis, measure_basis, d, rounds, p,
shots, post_selected_shots, post_selection_rate,
errors, logical_error_rate, seconds, decoder
```
