# Logical Operations Benchmark

General-purpose benchmark for logical gate performance on the unrotated surface code.
Sweeps gate types × distances × physical error rates, saves results to CSV, and plots LER vs p.

## What this benchmark does

Measures the logical error rate (LER) of six logical operations implemented on the unrotated
surface code under circuit-level noise:

| Gate | Protocol | Sub-experiments |
|------|----------|-----------------|
| `H` | Fold-transversal Hadamard | Z→X and X→Z (2) |
| `S` | Fold-transversal S via S·S† roundtrip | S_roundtrip (1) |
| `CNOT_trans` | Transversal CNOT | ZZ_ZZ, ZX_ZX, XZ_XX, XZ_ZZ, XX_XX (5) |
| `CNOT_LS_ZZ_XX` | Lattice Surgery CNOT (ZZ-XX protocol) | same 5 sub-experiments |
| `CNOT_LS_XX_ZZ` | Lattice Surgery CNOT (XX-ZZ protocol) | same 5 sub-experiments |
| `memory` | Z-basis memory baseline (rounds=d) | memory_Z (1) |
| `pauli` | Logical Pauli: physical application vs Pauli-frame tracking (rounds=d) | P{X\|Z}_{physical\|frame}_L{N} |

All results are written to a single combined CSV with per-task checkpointing —
safe to interrupt and resume.

## How to run

```bash
# All gates, default sweep (d=3,5,7; p=5e-4…1e-2):
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py

# Single gate:
PYTHONPATH=. venv/bin/python benchmarks/logical_ops/run_logical_ops.py --gate H

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
| `--gate` | all | Gate(s) to run: `H S CNOT_trans CNOT_LS_ZZ_XX CNOT_LS_XX_ZZ memory` |
| `--distances` | `3 5 7` | Code distances |
| `--p-values` | `5e-4 1e-3 2e-3 5e-3 1e-2` | Physical error rates |
| `--rounds` | `2` | SE rounds for gate benchmarks (memory and pauli always use rounds=d) |
| `--num-layers` | `0 1 2 4 8` | Pauli layer counts N to sweep (pauli gate only) |
| `--decoder` | `pymatching` for memory/LS CNOT; `bposd` for other gates | Decoder |
| `--max-shots` | `1e9` | Max shots per task |
| `--max-errors` | `100` | Stop after this many errors |
| `--num-workers` | `8` | Parallel workers |
| `--quick` | off | Fast test mode |

## The `pauli` experiment (Handbook §7.1)

Validates that physically applying a logical Pauli costs logical fidelity while
Pauli-frame tracking is free. A memory-like circuit (rounds = d) gets N
consecutive logical Pauli layers inserted at the midpoint:

- **physical** — the Pauli string is applied as real gates; circuit-level noise
  adds single-qubit depolarizing on the string qubits.
- **frame** — the same layers tagged `noiseless` (skipped by noise injection):
  the circuit-model equivalent of a classical Pauli-frame update.

The two arms build gate-for-gate identical clean circuits, so
LER(physical) − LER(frame) isolates exactly the extra error locations of
physical application. Plotting LER vs N gives the per-layer cost as a slope:
positive for physical, zero for frame. X̄ runs in Z-basis memory and Z̄ in
X-basis memory (the anticommuting pairings).

Note: idle noise in this codebase is lumped once per SE round, so the layer
tick carries gate noise on the weight-d string support only — no spectator
idling, no extended duration. The measured per-layer cost is therefore a
lower bound on the real physical cost of applying the operator.

```bash
PYTHONPATH=. python benchmarks/logical_ops/run_logical_ops.py \
    --gate pauli --distances 3 5 7 --num-layers 0 1 2 4 8
```

The plot script writes a dedicated figure for this experiment
(`results/logical_pauli_plot.png`): LER vs p at the largest N, and LER vs N
at the median p — physical solid, frame dashed.

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

## Related benchmarks

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

## Output CSV schema

```
gate, sub_experiment, init_basis, measure_basis, d, rounds, p,
shots, post_selected_shots, post_selection_rate,
errors, logical_error_rate, seconds, decoder
```
