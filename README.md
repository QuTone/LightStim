# LightStim

LightStim is a modular Quantum Error Correction (QEC) framework built on [Stim](https://github.com/quantumlib/Stim). It provides high-level abstractions for constructing fault-tolerant circuits with **automatic detector generation**, running simulation and decoding pipelines, and comparing logical error rates across codes and protocols.

## What this repo is for

- Build QEC experiments from reusable abstractions (`QECPatch`, `QECSystem`, `CircuitBuilder`, `SyndromeTracker`)
- Support multi-patch workflows (transversal gates, lattice surgery, state injection)
- Inject standardized noise models (`code_capacity`, `phenomenological`, `circuit_level`, `XZ_biased`)
- Decode with a unified backend (PyMatching, BP+OSD CPU/GPU, MWPF)
- Analyze and visualize logical error rates

## Repository layout

```text
LightStim/
├── lightstim/                  # Main library package
│   ├── ir/                     # Core abstractions (QECPatch, QECSystem, CircuitBuilder, SyndromeTracker)
│   ├── qec_code/               # Code implementations (surface, toric, BB, color, PQRM, repetition)
│   ├── noise/                  # Noise config and injectors
│   ├── protocols/              # Experiment orchestration (memory, CNOT, lattice surgery, ...)
│   ├── simulation/             # Decoder backend and simulation pipeline
│   └── plot/                   # Plotting helpers
├── docs/
│   ├── api/                    # API reference (ir.md, simulation.md)
│   ├── getting_started.md      # Installation, quick start, QEC concepts
│   └── vision.md               # Design philosophy and LightStim + AI workflow
├── skills/                     # Claude Code skill definitions
│   ├── builder-tracker-api/    # Direct CircuitBuilder + SyndromeTracker usage
│   ├── logical-coupler-design/ # Design a new LogicalCouplerProtocol
│   ├── simulate-decode/        # Run SimulationPipeline and get LER
│   ├── custom-noise/           # Configure noise models
│   ├── extend-new-code/        # Add a new QEC code
│   └── gotchas/                # Known pitfalls and debugging patterns
├── notebooks/                  # Jupyter notebooks by topic
├── benchmarks/                 # Simulation scripts and paper artifacts
├── requirements.txt
└── Dockerfile
```

## Quick start

### 1) Install

```bash
git clone https://github.com/x8fangQ/LightStim.git
cd LightStim

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU decoder** (`nv-qldpc-decoder`) requires NVIDIA GPU + CUDA 12.x. `cudaq-qec` is included in `requirements.txt` but will fail to install on non-CUDA systems — comment it out if not needed.

### 2) Optional: Jupyter kernel

```bash
python -m ipykernel install --user --name=lightstim --display-name="LightStim"
```

### 3) Run skill templates (self-contained examples)

```bash
venv/bin/python skills/simulate-decode/scripts/template.py
venv/bin/python skills/extend-new-code/scripts/template.py
venv/bin/python skills/builder-tracker-api/scripts/template.py
```

## Usage examples

### Build and simulate a memory experiment

```python
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
)
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# Build circuit
system = QECSystem()
system.add_patch(RotatedSurfaceCode(distance=5), name='main')
tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
builder = CircuitBuilder(tracker, system)
se = RotatedSurfaceCodeExtractionBlock(system)

builder.write_coordinates()
builder.initialize({q: 'Z' for q in system.data_indices}, n=system.num_qubits)
builder.apply_syndrome_extraction(se.circuit, rounds=5)
builder.apply_data_readout({q: 'Z' for q in system.data_indices})

noisy = builder.build_noisy_circuit(
    NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3),
    noise_model='circuit_level',
)

# Simulate and decode
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig('pymatching'),
    max_shots=100_000,
    max_errors=200,
    num_workers=4,
)
stats = pipeline.run(noisy)
print(f"LER: {stats.logical_error_rate:.3e} ± {stats.ler_error_bar():.3e}")
```

### Decoder options

| Decoder | Config | Notes |
|---|---|---|
| PyMatching (MWPM) | `DecoderConfig('pymatching')` | Default. Surface / LS circuits. |
| BP+OSD CPU | `DecoderConfig('bposd')` | LDPC codes, hyperedge circuits. |
| MWPF | `DecoderConfig('mwpf')` | Hyperedges (CrossLS, PQRM, color code). |
| GPU BP+OSD | `DecoderConfig('nv-qldpc-decoder')` | NVIDIA GPU. Large d or high p. |

### Available QEC codes

| Code | Patch class | Notes |
|---|---|---|
| Rotated surface | `RotatedSurfaceCode` | Primary code, most complete |
| Unrotated surface | `UnrotatedSurfaceCode` | Lattice surgery coupler available |
| Toric | `ToricCode` | Periodic boundary conditions |
| Color (6-6-6) | `ColorCode` | Fold-transversal H/S gates |
| Bivariate bicycle | `BBCode` | [[144,12,12]] gross code etc. |
| PQRM | `PQRMPatch` | Transversal T gate; CrossLS protocol |
| Repetition | `RepetitionCode` | Classical benchmark |

All codes are in `lightstim/qec_code/`.

## Try it with your AI coding agent

No tutorial needed. If you use Claude Code, Codex, Cursor, or any AI coding assistant,
paste the prompt below and let it build your first experiment:

> Read `skills/SKILL.md` in this repo first. Then build an unrotated surface code
> ZZ lattice surgery experiment: first patch d=3, second patch d=5 placed at
> offset (0, 10). Prepare the first patch in |+⟩ and the second in |0⟩; measure
> them in Z and X basis respectively. Use circuit-level noise with all error rates
> 1e-3. Choose an appropriate decoder and run a simulation with enough shots to
> get a reliable LER estimate. Print the result.

The skills system routes your AI to the right protocol class, decoder, and
simulation parameters — the same way a collaborator who knows the codebase would.

## Documentation

- `docs/api/ir.md` — `QECPatch`, `QECSystem`, `CircuitBuilder`, `SyndromeTracker`, `LogicalCouplerProtocol`
- `docs/api/simulation.md` — `SimulationPipeline`, `DecoderConfig`, `SimulationStats`, MWPF configuration
- `docs/getting_started.md` — installation, quick start, and QEC concepts
- `docs/vision.md` — design philosophy, key abstractions, and the LightStim + AI workflow
- `lightstim/simulation/README.md` — decoder backend architecture
- `skills/` — task-oriented instructions for AI-assisted development

## License

Apache License 2.0 — see [LICENSE](LICENSE).
