# State Injection Benchmark — Paper Figures 7–10

This directory reproduces the state injection benchmark figures for the rotated surface code.

```
paper_artifact/state_injection/
├── precomputed/          # Pre-run data (git-tracked). Only covers Z/X states.
├── results/              # Generated outputs (gitignored). Plots land here.
├── run_all.py            # Reproduce raw data (required for Y state figures)
├── plot_fig1.py          # Middle injection, d=3, Z/X/Y states (Fig 7)
├── plot_fig2.py          # Middle injection, d=5, Z/X/Y states (Fig 8)
├── plot_fig3.py          # Corner injection, d=7, Z/X/Y states (Fig 9)
└── plot_fig4.py         # Corner injection, d=7, Z state, all 3 modes (Fig 10)
```

---

## Important: precomputed data only covers Z/X states

The file `precomputed/state_injection.csv` contains data for `inject_state=[X, Z]` only.
**Y state data is missing from precomputed.**

- Figs 7–9 show Z, X, and Y states. With only precomputed data, the Y state line will be skipped.
- Fig 10 shows only the Z state (all 3 modes), so it can be fully plotted from precomputed data.

To generate full figures including Y state, run `run_all.py` first:

```bash
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py --inject-state Y
```

---

## Quick start: regenerate figures from available data

Fig 10 (Z state, all modes) can be fully generated from precomputed data:

```bash
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig4.py
```

Figs 7–9 will plot Z and X state lines from precomputed data (Y line skipped):

```bash
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig1.py
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig2.py
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig3.py
```

Each plot script loads `results/state_injection.csv` first, then merges with `precomputed/state_injection.csv`.
Fresh results take priority automatically via deduplication.

---

## Reproducing the data with `run_all.py`

### Full run (all states, all protocols, all modes)

```bash
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py
```

### Run only Y state (to fill in missing precomputed data)

```bash
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py --inject-state Y
```

### Quick smoke-test (corner only, Z+Y, d=3,5, 3 p-values, 100k shots)

```bash
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/run_all.py --quick
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | off | Corner only, Z+Y, d=3,5, 3 p-values, 100k shots |
| `--inject-state S` | all | Run only one state: Z, X, or Y |
| `--protocol P` | all | Run only one protocol: `corner` or `middle` |
| `--max-shots N` | 100 000 000 | Shot budget per task |
| `--max-errors N` | 200 | Stop task after N logical errors |
| `--num-workers N` | 32 | Parallel workers |

**Note:** Always use `venv/bin/python`. The project requires the venv for correct imports.

---

## Per-figure data details

### Fig 7 — Middle injection, d=3 (`results/fig1_middle_d3.png`)

- Protocol: `middle`, d=3, rounds=2
- Color by state: Z=RUST, X=TEAL, Y=VIOLET
- Mode: `full_postselection`
- Dual y-axis: LER (log, solid lines) + PS survival rate (linear, dashed lines)
- Y state requires running `run_all.py`

### Fig 8 — Middle injection, d=5 (`results/fig2_middle_d5.png`)

Same as Fig 7 but d=5.

### Fig 9 — Corner injection, d=7 (`results/fig3_corner_d7.png`)

- Protocol: `corner`, d=7, rounds=2
- Color by state: Z=RUST, X=TEAL, Y=VIOLET
- Mode: `full_postselection`
- Dual y-axis: LER (log, solid lines) + PS survival rate (linear, dashed lines)

### Fig 10 — Corner injection, d=7, Z state, all modes (`results/fig4_corner_d7_z_modes.png`)

- Protocol: `corner`, d=7, rounds=2, inject_state=Z
- Color by mode: full_postselection=RUST, hybrid=TEAL, full_qec=VIOLET
- Dual y-axis: LER (log, solid lines) + PS survival rate (linear, dashed lines)
- **Fully reproducible from precomputed data**

---

## Precomputed data

| File | Description |
|------|-------------|
| `precomputed/state_injection.csv` | Z/X injection, corner+middle protocols, d=3,5,7, r=2, p sweep |

Schema: `shots, post_selected_shots, post_selection_rate, errors, logical_error_rate, seconds, decoder, injection_protocol, inject_state, post_select_mode, rounds, d, p`

**Y state data is not included.** Run `run_all.py --inject-state Y` to generate it.
