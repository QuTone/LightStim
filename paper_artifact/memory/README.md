# Memory Experiment — Paper Figures 1–4

This directory reproduces the four memory benchmark figures in the paper.

```
paper_artifact/memory/
├── precomputed/          # Pre-run data (git-tracked). Used as fallback by plot scripts.
├── results/              # Generated outputs (gitignored). Plots land here.
├── run_all.py            # Reproduce raw data for any/all figures
├── plot_fig1.py          # Surface Code Family (Fig 1)
├── plot_fig2.py          # BB Codes: LER vs PER (Fig 2)
├── plot_fig3.py          # Code Comparison at p=1e-3 (Fig 3)
└── plot_fig4.py          # SE Scheduling (Fig 4)
```

---

## Quick start: regenerate figures from precomputed data

The `results/` PNGs are generated from `precomputed/` CSV files.
To regenerate all four figures without re-running experiments:

```bash
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig1.py
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig2.py
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig3.py
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig4.py
```

Each plot script checks `results/` first, then falls back to `precomputed/`.
If you have run `run_all.py`, your fresh results take priority automatically.

---

## Reproducing the data with `run_all.py`

### Run all four figures (full paper quality)

```bash
# GPU required for Fig 2 (BB codes) and Fig 3 (Color + 4D Hadamard)
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py
```

### Run a single figure

```bash
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --figure 1
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --figure 2
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --figure 3
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --figure 4
```

### Quick smoke-test (few shots, fast)

```bash
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --quick
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--figure N` | all | Run only figure N (1–4) |
| `--quick` | off | 2 p-values, 100k shots — for smoke-testing |
| `--max-shots N` | 1 000 000 | Shot budget per task |
| `--max-errors N` | 200 | Stop task after N logical errors |
| `--num-workers N` | 32 | Parallel workers |
| `--fig2-codes a,b` | all 3 BB | Comma-separated BB code names |
| `--fig2-decoder` | both | `gpu_bposd` or `mwpf` only |
| `--gpu-id N` | auto | Set `CUDA_VISIBLE_DEVICES` |

**Note:** Always use `venv/bin/python`. The GPU decoder (`cudaq_qec`) is only installed in the venv.

---

## Per-figure data details

### Fig 1 — Surface Code Family (`fig1_surface_codes.csv`)

| Code | Distances | Decoder |
|------|-----------|---------|
| Rotated SC | 3, 5, 7 | PyMatching (CPU) |
| Unrotated SC | 3, 5, 7 | PyMatching (CPU) |
| Toric | 3, 5, 7 | PyMatching (CPU) |

- p: `[1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 1.2e-2, 1.5e-2]`
- Noise: circuit-level · Rounds: d · Basis: Z
- max_shots=1e9, max_errors=200 · 63 tasks total

### Fig 2 — BB Codes: LER vs PER (`fig2_bb_codes_<name>_<decoder>.csv`)

| Code | n | k | d | Decoder |
|------|---|---|---|---------|
| [[72, 12, 6]] | 72 | 12 | 6 | GPU BP+OSD + MWPF |
| [[108, 8, 10]] | 108 | 8 | 10 | GPU BP+OSD |
| [[144, 12, 12]] | 144 | 12 | 12 | GPU BP+OSD |

- p: `[1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 7e-4, 5e-4, 3e-4]` (high→low; low-p points extend the extrapolation curve)
- max_shots=1e8, max_errors=100 · **GPU required**

### Fig 3 — Code Comparison at p=1e-3 (single data point per code)

Data is assembled from three sources, all at p=1e-3:

| Source | Codes | Output CSV |
|--------|-------|------------|
| Fig 1 data | Rotated SC, Unrotated SC, Toric (d=3,5,7) | `fig1_surface_codes.csv` |
| Fig 2 data (gpu_bposd) | BB [[72,12,6]], [[108,8,10]], [[144,12,12]] | `fig2_bb_codes_*_gpu_bposd.csv` |
| Fig 3 extra — Color code | Color (6-6-6), d=3,5,7 (MWPF) | `fig3_color_code.csv` |
| Fig 3 extra — 4D Hadamard | [[96, 6, 8]] 4D Geo, p=1e-3 (GPU BP+OSD) | `fig3_4d_hadamard.csv` |

Running `--figure 3` generates the Color code and 4D Hadamard CSVs.
The surface code and BB code data is reused from `--figure 1` / `--figure 2`.

For a standalone Fig 3 run (e.g. if Fig 1/2 data already exists):
```bash
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --figure 3
```

### Fig 4 — SE Scheduling (`fig4_se_scheduling.csv`)

| Schedule | FT? | Description |
|----------|-----|-------------|
| `perpendicular` | Yes | Standard zigzag — full code distance |
| `swapped` | No | Ticks 2/3 swapped — hook errors halve effective distance |

- Code: Rotated SC · d=3,5,7,9 · Basis: Z and X
- p: `[7e-3, 5e-3, 2e-3, 1e-3, 7e-4, 5e-4]`
- max_shots=1e8, max_errors=100 · 96 tasks total · CPU only

---

## Precomputed data

`precomputed/` contains the data used to generate the committed `results/` figures.
It is git-tracked so reviewers can regenerate figures without running experiments.

| File | Description |
|------|-------------|
| `fig1_surface_codes.csv` | Surface code family, d=3,5,7, p sweep |
| `fig2_bb_codes_bb_72_12_6_gpu_bposd.csv` | [[72,12,6]], GPU BP+OSD |
| `fig2_bb_codes_bb_108_8_10_gpu_bposd.csv` | [[108,8,10]], GPU BP+OSD |
| `fig2_bb_codes_bb_144_12_12_gpu_bposd.csv` | [[144,12,12]], GPU BP+OSD |
| `fig3_color_code.csv` | Color (6-6-6), d=3,5,7, p=1e-3, MWPF |
| `fig3_4d_hadamard.csv` | [[96,6,8]] 4D Geo Hadamard, p=1e-3, GPU BP+OSD |
| `fig4_se_scheduling.csv` | SE scheduling comparison, d=3,5,7,9, Z+X basis |
