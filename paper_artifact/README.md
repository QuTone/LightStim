# paper_artifact/

Evaluation artifact for:

> **LightStim: A Framework for QEC Protocol Evaluation and Prototyping with Automated DEM Construction**  
> https://arxiv.org/abs/2604.21472

This directory contains all scripts and precomputed data needed to reproduce the paper's
figures and tables. The precomputed data and figures are committed to the repository, but can be regenerated locally with the scripts.

---

## Structure

```
paper_artifact/
├── memory/             Figs 1–4:   Memory experiments (surface, BB, 4D codes)
├── logical_ops/        Figs 5–6:   Logical gate benchmarks (unrotated SC)
├── state_injection/    Figs 7–10:  State injection (rotated SC)
├── cross_ls/           Fig 11:     CrossLS — Surface–PQRM lattice surgery
├── logical_circuits/   Figs 12–14: Bell teleportation & magic-state distillation
└── table/              Tables 2–3:    Correctness validation, compilation efficiency
```

Each section has:
- `precomputed/` — pre-run CSV data committed to git (used by plot scripts)
- `results/`     — PNG figures committed to git; run plot scripts to regenerate
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

PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig1.py   # (etc.)
PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig1.py   # (etc.)

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

> Raw data outputs land in `paper_artifact/<section>/results/`, overwriting the
> committed figures. To use new data for plotting, replace the `precomputed/` CSV
> with the new file, or pass the path directly to the plot script.

### Rerunning Table 3 compilation benchmarks

The committed Table 3 reference data is stored in
`paper_artifact/table/precompute/table3.json`. It was generated with the
LightStim C++ GF(2) RREF backend, using one timing trial per row.

```bash
# Build the optional C++ RREF extension first. Install pybind11 if needed.
venv/bin/pip install pybind11
venv/bin/python lightstim/utils/cpp/build.py

# Reproduce Table 3 timings with the C++ backend.
PYTHONPATH=. venv/bin/python paper_artifact/table/comprehensive_benchmark.py

# Optional debug baseline using the pure Python/NumPy fallback.
PYTHONPATH=. venv/bin/python paper_artifact/table/comprehensive_benchmark.py --backend python
```

The rerun output is written to `paper_artifact/table/results/table3_<backend>.json`
so the committed reference data is not overwritten accidentally. Use `--trials 3`
to collect a median over three runs, or `--keys <row_key> ...` to rerun selected
rows only.

---

## Section summaries

| Section | Figures | What it shows |
|---|---|---|
| `memory/` | 1–4 | LER vs PER for surface codes, BB codes, 4D Hadamard code |
| `logical_ops/` | 5–6 | Transversal CNOT and fold-transversal H/S gate benchmarks |
| `state_injection/` | 7–10 | Non-FT state injection LER and post-selection overhead |
| `cross_ls/` | 11 | CrossLS: LER for Z/X/Y states, distance scaling |
| `logical_circuits/` | 12–14 | Bell teleportation (TG, ZZ-LS, XX-LS); distillation output fidelity |
| `table/` | Tables 2–3 | Correctness validation against Stim reference circuits and PyMatching; compilation benchmark reference data |
