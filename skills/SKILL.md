# LightStim — Skills Entry Point

Read this first. It orients you to the project and routes you to the right skill.

---

## What is LightStim?

LightStim is a modular QEC framework built on [Stim](https://github.com/quantumlib/Stim).
Its core value is **automatic detector generation**: you define the QEC code and the
syndrome extraction schedule; LightStim computes `DETECTOR` and `OBSERVABLE_INCLUDE`
instructions automatically via symplectic tableau tracking.

Key data flow:
```
QECPatch → QECSystem → CircuitBuilder + SyndromeTracker → stim.Circuit
                                   ↓
                           NoiseInjector → SimulationPipeline → LER
```

---

## Which skill do you need?

| I want to… | Read this skill |
|---|---|
| Build a circuit for a new protocol from scratch | [`builder-tracker-api/`](builder-tracker-api/SKILL.md) |
| Design a new lattice surgery coupler (multi-patch) | [`logical-coupler-design/`](logical-coupler-design/SKILL.md) |
| Run a simulation and get logical error rate | [`simulate-decode/`](simulate-decode/SKILL.md) |
| Configure noise models (circuit-level, phenomenological…) | [`custom-noise/`](custom-noise/SKILL.md) |
| Add a new QEC code (new stabilizer geometry) | [`extend-new-code/`](extend-new-code/SKILL.md) |
| Write or update a protocol notebook | [`notebook-workflow/`](notebook-workflow/SKILL.md) |
| Debug unexpected detector counts, LER≈50%, or tracker errors | [`gotchas/`](gotchas/SKILL.md) |

When in doubt, start with **`builder-tracker-api/`** — it covers the core API that every
other skill builds on.

---

## Key conventions (apply everywhere)

**Imports** — always use `lightstim.*`, never `src.*`:
```python
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
```

**Python environment** — always use `venv/bin/python`, never system Python:
```bash
PYTHONPATH=. venv/bin/python my_script.py
```
Using the wrong Python causes `cudaq_qec` to not be found → LER ≈ 99%.

**Decoder choice** — depends on circuit type:
- Surface/toric/repetition → `pymatching` (fast, correct)
- Color code, BB codes, PQRM → `mwpf` or `bposd` (handles hyperedges)
- GPU → `nv-qldpc-decoder` with `num_workers=1`
- See `gotchas/SKILL.md` §7 for the full decision table

**Benchmark scripts** — must use per-task checkpointing (append one CSV row per task).
See `skills/README.md` → "Adding a new benchmark" for the full convention.

---

## Repository layout (quick reference)

```
lightstim/               Core library
  qec_code/              QEC code definitions (QECPatch subclasses)
  ir/                    CircuitBuilder, SyndromeTracker, QECSystem
  noise/                 NoiseConfig, NoiseInjector, noise rules
  simulation/            SimulationPipeline, decoder backends
  protocols/             Packaged protocol implementations
  plot/                  Paper-style plot utilities

notebooks/               Demo notebooks (one per protocol)
benchmarks/              Large-scale sweep runners + plot scripts
paper_artifact/          Reproducible paper figures (precomputed data + plot scripts)
skills/                  This directory — task-oriented LLM guidance
docs/api/                Formal API reference (class hierarchy)
```

---

## API docs vs skills

| | `docs/api/` | `skills/` |
|---|---|---|
| Organized by | What exists (class hierarchy) | What you want to do |
| Coverage | Complete (every parameter) | Curated (task-relevant) |
| Stance | Neutral | Opinionated — tells you the right path |
| Failure modes | Not covered | Explicitly covered in gotchas |

Use `docs/api/` when you need a precise method signature.
Use a skill when you need to know *how* to accomplish a goal.
