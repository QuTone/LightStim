# paper_artifact/

Evaluation artifact for:

> **LightStim: A Framework for QEC Protocol Evaluation and Prototyping with Automated DEM Construction**  
> https://arxiv.org/abs/2604.21472

This directory contains all scripts and precomputed data needed to reproduce the paper's
figures and tables. The precomputed data is committed to the repository; raw figures are
not (regenerate locally with the plot scripts).

---

## Structure

```
paper_artifact/
├── memory/             Figs 1–4:   Memory experiments (surface, BB, 4D codes)
├── logical_ops/        Figs 5–6:   Logical gate benchmarks (unrotated SC)
├── state_injection/    Figs 7–10:  State injection (rotated SC)
├── cross_ls/           Fig 11:     CrossLS — Surface–PQRM lattice surgery
├── logical_circuits/   Figs 12–14: Bell teleportation & magic-state distillation
└── table/              Table 2:    Correctness validation vs. open-source references
```

Each section has:
- `precomputed/` — pre-run CSV data committed to git (used by plot scripts)
- `results/`     — generated PNG figures (gitignored; run plot scripts to create)
- `run_all.py`   — reproduce the raw data from scratch
- `plot_*.py`    — generate figures from `precomputed/` or `results/`

---

## Quick start: reproduce all figures

```bash
# From repo root, using the precomputed data:
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig1.py
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig2.py
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig3.py
PYTHONPATH=. venv/bin/python paper_artifact/memory/plot_fig4.py

PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig5.py   # (etc.)
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig7.py

PYTHONPATH=. venv/bin/python paper_artifact/cross_ls/plot_cross_ls.py
PYTHONPATH=. venv/bin/python paper_artifact/logical_circuits/plot_bell_tele.py
PYTHONPATH=. venv/bin/python paper_artifact/logical_circuits/plot_distill.py

PYTHONPATH=. venv/bin/python paper_artifact/table/table2_verification.py
```

Figures are saved to `paper_artifact/<section>/results/`.

---

## Rerunning raw data

Each section's `run_all.py` reruns the full numerical simulation.
GPU experiments (BB codes, TG distillation) require CUDA and `cudaq-qec`.

```bash
# Example: rerun memory benchmark (CPU, ~2h on 32 cores)
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py

# Example: rerun BB code figures (GPU required)
PYTHONPATH=. venv/bin/python paper_artifact/memory/run_all.py --figures 2
```

> Raw data outputs land in `paper_artifact/<section>/results/` (gitignored).
> To use them for plotting, replace the `precomputed/` CSV with the new file, or
> pass the path directly to the plot script.

---

## Section summaries

| Section | Figures | What it shows |
|---|---|---|
| `memory/` | 1–4 | LER vs PER for surface codes, BB codes, 4D Hadamard code |
| `logical_ops/` | 5–6 | Transversal CNOT and fold-transversal H/S gate benchmarks |
| `state_injection/` | 7–10 | Non-FT state injection LER and post-selection overhead |
| `cross_ls/` | 11 | CrossLS: LER for Z/X/Y states, distance scaling |
| `logical_circuits/` | 12–14 | Bell teleportation (TG, ZZ-LS, XX-LS); distillation output fidelity |
| `table/` | Table 2 | Correctness validation against Stim reference circuits and PyMatching |
