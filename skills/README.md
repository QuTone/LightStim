# skills/

LightStim Claude Code skills — each skill helps an LLM complete a specific workflow
or design task with the LightStim API.

## Structure

Each skill is a subdirectory with:
- `SKILL.md` — task-oriented instructions Claude loads at the start of a task
- `scripts/template.py` — complete runnable Python example Claude adapts for the user

```
skills/
├── builder-tracker-api/    # Direct use of CircuitBuilder + SyndromeTracker (custom protocols)
├── logical-coupler-design/ # Design a new LogicalCouplerProtocol (lattice surgery)
├── simulate-decode/        # Run SimulationPipeline + get LER
├── custom-noise/           # Configure NoiseConfig + noise models
├── extend-new-code/        # Add a new QEC code (QECPatch + SE_block)
├── notebook-workflow/      # Protocol dev lifecycle: prototype → package → benchmark → demo
└── gotchas/                # Known pitfalls & debugging patterns
```

See also `docs/api/` for formal API reference:
- `docs/api/ir.md` — QECPatch, QECSystem, CircuitBuilder, SyndromeTracker, LogicalCouplerProtocol
- `docs/api/simulation.md` — SimulationPipeline, DecoderConfig, SimulationStats, NoiseConfig

## Skill vs API doc

| | API doc (`docs/api/`) | Skill (`skills/`) |
|---|---|---|
| Organized by | What exists (class hierarchy) | What you want to do (user intent) |
| Coverage | Complete (every parameter) | Curated (task-relevant only) |
| Stance | Neutral | Opinionated — tells you which path to take |
| Failure modes | Not covered | Explicitly covered |

**Use the API doc** when you need a precise method signature or parameter name.  
**Use a skill** when you need to know how to accomplish a goal.

## Installing into Claude Code

The skills in this directory are pre-installed for this project via
`.claude/plugins/lightstim/`. They are active whenever you open this repo
in Claude Code — no manual installation needed.

## Running the template scripts directly

Each `scripts/template.py` is a standalone runnable script (from repo root):

```bash
venv/bin/python skills/builder-tracker-api/scripts/template.py
venv/bin/python skills/logical-coupler-design/scripts/template.py
venv/bin/python skills/simulate-decode/scripts/template.py
venv/bin/python skills/extend-new-code/scripts/template.py
```

---

## Adding a new benchmark

When implementing a new protocol and adding it to `benchmarks/`, follow this structure
so it stays consistent with the rest of the repo:

### 1. Implement the protocol

Write the protocol logic in `lightstim/protocols/<your_protocol>.py`.
This is the reusable core — no CLI, no file I/O, just functions that
build circuits and return results.

### 2. Create the benchmark directory

```
benchmarks/<category>/
├── run_<name>.py     # CLI runner: sweeps parameters, writes CSV
├── plot_<name>.py    # Plotting: reads CSV, writes PNG
└── README.md         # What experiments are supported, how to run them
```

Place under the appropriate category:
- `memory/` — single-patch memory experiments
- `logical_ops/` — single-qubit logical gate benchmarks
- `logical_circuits/` — multi-patch circuits (Bell pair, distillation, …)
- `cross_ls/` — cross-code lattice surgery (CrossLS / PQRM)
- A new top-level folder if the protocol doesn't fit any existing category

### 3. run.py conventions

- Use `argparse` for CLI configuration (distances, p-values, decoder, workers, …)
- Write results to `benchmarks/<category>/results/<name>_results.csv`
- Use **per-task checkpointing**: append one row to CSV immediately after each
  task completes. Never batch-save at the end — the run must be safe to interrupt.
- Include a `--quick` flag for a fast smoke test (2 distances, 2 p-values)
- Print progress to stdout so the user can monitor long runs

### 4. plot.py conventions

- Read from `results/<name>_results.csv` by default; accept a path argument for custom CSVs
- Save figures to `results/fig_<name>.png`
- Use `lightstim.plot.styles.apply_paper_style()` for consistent styling

### 5. README.md

The README describes the benchmark, not the protocol. Focus on:
- What experiments are supported and what they measure
- Exact run commands with common configurations
- CLI option table
- Output CSV column descriptions

Protocol background goes in `lightstim/protocols/` (or a docstring), not here.
