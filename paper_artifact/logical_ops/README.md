# Logical Operations Benchmark — Paper Figures 1–6

This directory reproduces the logical operations benchmark figures for the unrotated surface code.

```
paper_artifact/logical_ops/
├── precomputed/          # Pre-run data (git-tracked). Used as fallback by plot scripts.
├── results/              # Generated outputs (gitignored). Plots land here.
├── run_all.py            # Reproduce raw data for any/all figures
├── plot_fig1.py          # LS CNOT ZZ->XX (Fig 1)
├── plot_fig2.py          # LS CNOT XX->ZZ (Fig 2)
├── plot_fig3.py          # Transversal CNOT (Fig 3)
├── plot_fig4.py          # H gate (Fig 4)
├── plot_fig5.py          # S gate / S_oneway (Fig 5)
└── plot_fig6.py          # Memory baseline (Fig 6)
```

---

## Quick start: regenerate figures from precomputed data

Figures 1–5 have precomputed CSVs. To regenerate those figures without re-running experiments:

```bash
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig1.py
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig2.py
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig3.py
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig4.py
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig5.py
```

**Note: Fig 6 (memory baseline) has no precomputed data.** Run `--figure 6` first:

```bash
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --figure 6
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig6.py
```

Each plot script checks `results/` first, then falls back to `precomputed/`.
If you have run `run_all.py`, your fresh results take priority automatically.

---

## Reproducing the data with `run_all.py`

### Run all six figures

```bash
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py
```

### Run a single figure

```bash
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --figure 1
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --figure 6
```

### Quick smoke-test (2 distances, 2 p-values, 100k shots)

```bash
PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/run_all.py --quick
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--figure N` | all | Run only figure N (1–6) |
| `--gate GATE` | all | Run one gate: `CNOT_LS_ZZ_XX`, `CNOT_LS_XX_ZZ`, `CNOT_trans`, `H`, `S`, `memory` |
| `--quick` | off | 2 distances, 2 p-values, 100k shots — for smoke-testing |
| `--max-shots N` | 1 000 000 000 | Shot budget per task |
| `--max-errors N` | 100 | Stop task after N logical errors |
| `--num-workers N` | 32 | Parallel workers |

**Note:** Always use `venv/bin/python`. The project is run via direct imports and requires the venv.

---

## Per-figure data details

### Fig 1 — LS CNOT ZZ->XX (`fig1_cnot_ls_zz_xx.csv`)

| Property | Value |
|----------|-------|
| Gate | CNOT via Lattice Surgery (ancilla \|+>, ZZ-XX coupling) |
| Sub-experiments | ZZ_ZZ, ZX_ZX, XZ_XX, XZ_ZZ, XX_XX |
| Distances | 3, 5, 7 |
| Decoder | pymatching (CPU) |
| p-values | `[5e-4, 1e-3, 2e-3, 5e-3, 1e-2]` |
| Plot aggregation | mean LER over 5 sub-experiments |

### Fig 2 — LS CNOT XX->ZZ (`fig2_cnot_ls_xx_zz.csv`)

Same as Fig 1 but with XX-ZZ coupling protocol. Same sub-experiments, decoder, and p-values.

### Fig 3 — Transversal CNOT (`fig3_cnot_trans.csv`)

| Property | Value |
|----------|-------|
| Gate | Transversal CNOT (bitwise qubit overlap) |
| Sub-experiments | ZZ_ZZ, ZX_ZX, XZ_XX, XZ_ZZ, XX_XX |
| Distances | 3, 5, 7 |
| Decoder | bposd (CPU) |
| p-values | `[5e-4, 1e-3, 2e-3, 5e-3, 1e-2]` |
| Plot aggregation | mean LER over 5 sub-experiments |

### Fig 4 — H gate (`fig4_h.csv`)

| Property | Value |
|----------|-------|
| Gate | Fold-transversal Hadamard |
| Sub-experiments | H_ZtoX (init Z, meas X), H_XtoZ (init X, meas Z) |
| Distances | 3, 5, 7 |
| Decoder | bposd (CPU) |
| p-values | `[5e-4, 1e-3, 2e-3, 5e-3, 1e-2]` |
| Plot aggregation | mean LER over H_ZtoX and H_XtoZ |

### Fig 5 — S gate (`fig5_s.csv`)

| Property | Value |
|----------|-------|
| Gate | Fold-transversal S (S_oneway: S then noiseless S†, measuring X) |
| Sub-experiments | S_oneway |
| Distances | 3, 5, 7 |
| Decoder | bposd (CPU) |
| p-values | `[5e-4, 1e-3, 2e-3, 5e-3, 1e-2]` |
| Plot aggregation | direct LER (single sub-experiment) |

### Fig 6 — Memory baseline (`fig6_memory.csv`)

| Property | Value |
|----------|-------|
| Gate | Z-basis memory (Unrotated Surface Code) |
| Sub-experiments | Z_memory |
| Distances | 3, 5, 7 |
| Decoder | pymatching (CPU) |
| rounds | d (code distance) |
| p-values | `[5e-4, 1e-3, 2e-3, 5e-3, 1e-2]` |
| Plot aggregation | direct LER |

**No precomputed CSV for Fig 6.** Run `--figure 6` to generate it.

---

## Precomputed data

`precomputed/` contains the data used to generate figures 1–5.
It is git-tracked so reviewers can regenerate those figures without running experiments.

| File | Gate | Description |
|------|------|-------------|
| `fig1_cnot_ls_zz_xx.csv` | LS CNOT ZZ->XX | 5 sub-exps, d=3,5,7, p sweep |
| `fig2_cnot_ls_xx_zz.csv` | LS CNOT XX->ZZ | 5 sub-exps, d=3,5,7, p sweep |
| `fig3_cnot_trans.csv` | Trans CNOT | 5 sub-exps, d=3,5,7, p sweep |
| `fig4_h.csv` | H gate | 2 sub-exps, d=3,5,7, p sweep |
| `fig5_s.csv` | S gate | S_oneway, d=3,5,7, p sweep |

`fig6_memory.csv` is not precomputed — run `run_all.py --figure 6` to generate it.
