# LightStim Repo Reorg Plan

Three parallel purposes for this repo:
1. **Paper artifact** — reproducible scripts for the submitted paper (Zenodo archive)
2. **Open source** — LightStim as a pip-installable QEC framework
3. **Future extensions** — new codes, protocols, decoders

---

## Overall Strategy

| Layer | What it is | Location |
|-------|-----------|----------|
| Paper scripts | Exact scripts that produced paper figures | `paper_artifact/` (gitignored, Zenodo) |
| General benchmarks | Reusable evaluation engine | `benchmarks/<module>/` |
| Notebooks | Interactive demos per code/protocol | `notebooks/<Module>/` |
| Package | pip-installable library | `lightstim/` |

**Rule:** `benchmarks/` is a general-purpose evaluation engine mirroring `lightstim/protocols/`.
Paper-specific logic (exact figure layout, paper datasets) lives only in `paper_artifact/`.

---

## Per-Module Reorg Pattern

For every benchmark module:

1. **Move paper scripts** → `paper_artifact/<module>/`
   - Scripts that are tightly coupled to paper figures (run_fig*.py, plot_fig*.py, etc.)

2. **Write general runner** → `benchmarks/<module>/run_<module>.py`
   - Sweeps configurable code × distance × p-value × decoder combinations
   - Per-task checkpointing (append-on-complete, never batch-save)
   - Standard CLI: `--codes --distances --p-values --decoder --noise-model --output`
   - Produces a clean CSV with all input params + result columns

3. **Write general plotter** → `benchmarks/<module>/plot_<module>.py`
   - Reads any CSV produced by the runner
   - No hardcoded paper figure logic

4. **Write notebooks** → `notebooks/<Module>/`
   - One notebook per code family
   - Fixed imports: `lightstim.*` (not `src.*` or `experiments.*`)
   - Path setup: `ROOT = Path("../..").resolve()`
   - `.diagram()` kept active only for small circuits (d≤3), commented out for larger

---

## Module Progress

| Module | Paper→artifact | General runner | General plotter | Notebooks |
|--------|---------------|----------------|-----------------|-----------|
| `memory` | ✅ `paper_artifact/memory/` | ✅ `run_memory.py` | ✅ `plot_memory.py` | ✅ 6 notebooks verified |
| `logical_ops` | ⬜ | ⬜ `run_logical_ops.py` exists, review | ⬜ `plot_logical_op.py` exists, review | ⬜ |
| `logical_circuits/distillation` | ⬜ | ⬜ | ⬜ | ⬜ |
| `new_protocol/surface-PQRM-LS` | ⬜ | ⬜ `run_sweep.py` exists, review | ✅ `plot_cross_ls.py` path fixed | ⬜ |
| `LS_distillation` | ⬜ unclear scope | ⬜ | ⬜ | ⬜ |

---

## Benchmark General Runner — Design Spec

All general runners share this structure:

```
benchmarks/<module>/
    run_<module>.py      # CLI runner, CSV output, checkpointing
    plot_<module>.py     # reads any runner CSV, general plots
    results/             # gitignored by default; exceptions added to .gitignore for paper data
```

### CSV Schema Convention
Every runner CSV must include:
- **Input columns**: `code, distance, p, basis, rounds, noise_model, decoder_name`  
  (+ module-specific columns like `rounds_per_op` for logical ops)
- **Result columns**: `shots, errors, logical_error_rate, seconds, n_data, n_total, k`

### Checkpointing
- Append one row per completed task immediately
- On restart: read existing CSV, skip tasks whose input-column key already appears
- Never batch-write at the end

---

## Planned Skills

### Skill 1: `new-code-memory-benchmark`
When a user implements a new `QECPatch` + `ExtractionBlock`, they should be able to run
a memory benchmark with minimal boilerplate.

**Prompt template to give AI:**
```
I have a new QEC code:
  - Code class: <ClassName>(QECPatch) in <path>
  - Extraction block: <BlockClassName> in <path>
  - Constructor params: <param1=..., param2=...>
  - Distance type: parameterized (like surface code) | fixed (like BB code, d=<N>)

Add this code to benchmarks/memory/run_memory.py following the existing pattern:
  1. Add an entry to _BB_CONFIGS (if fixed distance) or _TOPO_CODES (if parameterized)
  2. Add a branch in _make_code() returning (CodeInstance, ExtractionBlock)
  3. Add the import at the top

Reference: benchmarks/memory/run_memory.py
```

**Future home:** `skills/new_code_benchmark.md`

---

### Skill 2: `register-decoder`
Allow the decoder community to plug their decoder into the LightStim ecosystem.

**What's needed:**
- Implement a class inheriting `sinter.Decoder` (for CPU) or the GPU interface
- Call `register_decoder(name, cls, aliases=[...], backend="cpu")` from `lightstim/simulation/decoder_backend/registry.py`
- Add a config branch in the runner's `_decoder_config()` function

**Prompt template to give AI:**
```
I want to register a new decoder called "<name>" in LightStim.

Decoder interface:
  - Backend: cpu | gpu
  - Class: <MyDecoder> (implements sinter.Decoder)
  - Constructor params: <...>
  - Located at: <path>

Steps:
  1. In lightstim/simulation/decoder_backend/decoders/, create <name>.py
     wrapping MyDecoder and calling register_decoder("<name>", ..., backend="cpu")
  2. Import it in lightstim/simulation/decoder_backend/decoders/__init__.py
  3. Add a branch in benchmarks/memory/run_memory.py → _decoder_config("<name>")
     returning DecoderConfig(name="<name>", backend="cpu", params={...})
  4. Add "<name>" to the --decoder choices list

Reference files:
  lightstim/simulation/decoder_backend/decoders/bposd.py  (CPU example)
  lightstim/simulation/decoder_backend/decoders/cudaqx.py (GPU example)
  lightstim/simulation/decoder_backend/registry.py
```

**Future home:** `skills/register_decoder.md`

---

## Import Convention (enforced throughout)

```python
# Correct
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig

# Wrong (legacy, do not use)
from src.ir.qec_system import QECSystem
from experiments.memory import MemoryExperiment
```

Notebook path setup (2 levels from repo root):
```python
ROOT = Path("../..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```
