# skills/

LightStim Claude Code skills — each skill helps an LLM complete a specific workflow
or design task with the LightStim API.

## Structure

Each skill is a subdirectory with:
- `SKILL.md` — task-oriented instructions Claude loads at the start of a task
- `scripts/template.py` — complete runnable Python example Claude adapts for the user

```
skills/
├── builder-tracker-api/   # Direct use of CircuitBuilder + SyndromeTracker (custom protocols)
├── logical-coupler-design/ # Design a new LogicalCouplerProtocol (lattice surgery)
├── simulate-decode/        # Run SimulationPipeline + get LER
├── custom-noise/           # Configure NoiseConfig + noise models
├── extend-new-code/        # Add a new QEC code (QECPatch + SE_block)
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
