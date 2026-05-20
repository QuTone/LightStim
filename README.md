# LightStim

LightStim is a modular Quantum Error Correction (QEC) framework built on top of [Stim](https://github.com/quantumlib/Stim). It focuses on building fault-tolerant circuits with automatic detector generation, running simulation/decoding pipelines, and comparing logical error rates across codes and protocols.

## What this repo is for

- Build QEC experiments from reusable abstractions (`QECPatch`, `QECSystem`, `CircuitBuilder`, `SyndromeTracker`)
- Support multi-patch workflows (transversal gates, lattice surgery)
- Inject standardized noise models (`code_capacity`, `phenomenological`, `circuit_level`, `XZ_biased`)
- Decode with a unified backend (PyMatching, BP+OSD CPU/GPU, MWPF)
- Analyze and visualize simulation results

## Repository layout

```text
LightStim/
├── docs/                       # Architecture and API documentation
│   ├── user_guide.md
│   └── simulation_pipeline.md
├── lightstim/                  # Main library package
│   ├── ir/                     # Core abstractions (QECPatch, QECSystem, CircuitBuilder, SyndromeTracker)
│   ├── qec_code/               # Code implementations (surface, BB, color, repetition, ...)
│   ├── noise/                  # Noise config and injectors
│   ├── protocols/              # Experiment orchestration (memory, CNOT, lattice surgery, ...)
│   ├── simulation/             # Decoder backend and simulation pipeline
│   │   └── decoder_backend/
│   ├── plot/                   # Plotting helpers
│   └── utils/                  # Utilities
├── skills/                     # Claude Code skill definitions (7 skills)
├── notebooks/                  # Jupyter notebooks
├── tests/
├── benchmarks/
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

### 2) Optional: Jupyter kernel

```bash
python -m ipykernel install --user --name=qec-simulator --display-name="QEC Simulator"
```

### 3) Run skills (self-contained examples)

```bash
python skills/memory-experiment/scripts/template.py
python skills/simulate-decode/scripts/template.py
```

## Minimal usage examples

### Memory experiment

```python
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig

system = QECSystem()
system.add_patch(RotatedSurfaceCode(distance=5), name='main')

experiment = MemoryExperiment(
    qec_system=system,
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=5,
    noise_params=NoiseConfig(p_1q=0.001, p_2q=0.001, p_meas=0.001, p_reset=0.001, p_idle=0.001),
    noise_model='circuit_level',
    basis='Z',
)
circuit = experiment.build()
```

### Simulation and decoding

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig('pymatching'),
    max_shots=100_000,
    max_errors=100,
    num_workers=4,
    print_progress=True,
)
stats = pipeline.run(circuit, json_metadata={'d': 5, 'p': 0.001})
print(f"LER: {stats.logical_error_rate:.3e} ± {stats.ler_error_bar():.3e}")
```

Decoder options:
- PyMatching: `DecoderConfig('pymatching')`
- BP+OSD CPU: `DecoderConfig('bposd', backend='cpu')` — requires `stimbposd`
- BP+OSD GPU: `DecoderConfig('bposd', backend='gpu')` — requires `cudaq_qec` + NVIDIA GPU
- MWPF: `DecoderConfig('mwpf')` — requires `mwpf`

### Available QEC codes

| Code | Patch class | Extraction block |
|---|---|---|
| Rotated surface | `RotatedSurfaceCode` | `RotatedSurfaceCodeExtractionBlock` |
| Unrotated surface | `UnrotatedSurfaceCode` | `UnrotatedSurfaceCodeExtractionBlock` |
| Toric | `ToricCode` | `ToricCodeExtractionBlock` |
| Color | `ColorCode` | `ColorCodeExtractionBlock` |
| Bivariate bicycle (BB) | `BBCode` | `BBCodeExtractionBlock` |
| Repetition | `RepetitionCode` | `RepetitionCodeExtractionBlock` |

All codes are in `lightstim/qec_code/`.

### CircuitBuilder API

```python
builder.initialize(init_dict, n=system.num_qubits)
builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=d)
builder.apply_data_readout(final_measurements=measurements)
```

## More documentation

- Full user guide: `docs/user_guide.md`
- Decoder backend details: `lightstim/simulation/README.md`
- Simulation pipeline guide: `docs/simulation_pipeline.md`

## License

License is not specified yet in this repository.
