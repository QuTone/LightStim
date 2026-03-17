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
├── docs/
│   └── user_guide.md
├── experiments/
│   ├── memory.py
│   ├── CNOT_trans.py
│   ├── two_patch_LS_unrotated.py
│   ├── CNOT_LS.py
│   ├── ghz.py
│   └── state_injection.py
├── notebooks/
│   ├── memory_experiment.ipynb
│   ├── test_decoder.ipynb
│   ├── test_trans_CNOT.ipynb
│   ├── test_LS_two_patch.ipynb
│   ├── test_LS_CNOT.ipynb
│   ├── test_ghz.ipynb
│   ├── test_injection.ipynb
│   └── fold_transversal.ipynb
├── src/
│   ├── ir/                         # Core abstractions and tracking
│   ├── qec_code/                   # Code implementations
│   ├── noise/                      # Noise config and injectors
│   ├── simulation/decoder_backend/ # Decoding pipeline and decoders
│   ├── plot/                       # Plotting helpers
│   └── utils/                      # Utilities
├── tests/
│   └── test_linear_algebra.py
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

### 3) Run examples

- Notebook-first workflow: open notebooks under `notebooks/`
- Script workflow: instantiate an experiment class, call `.build()`, then decode with `SimulationPipeline`

## Minimal usage examples

### Memory experiment

```python
from experiments.memory import MemoryExperiment
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from src.noise.config import NoiseConfig

experiment = MemoryExperiment(
    qec_system=RotatedSurfaceCode(distance=5),
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=5,
    noise_params=NoiseConfig(p_1q=0.001, p_2q=0.005, p_meas=0.001),
    noise_model="circuit_level",
    basis="Z",
)
circuit = experiment.build()
```

### Transversal CNOT experiment

```python
from experiments.CNOT_trans import CNOTTransExperiment
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)

experiment = CNOTTransExperiment(
    code_patch_class=UnrotatedSurfaceCode,
    extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
    code_params_control={"distance": 3},
    code_params_target={"distance": 3},
    offset_target=(6, 0),
    initial_basis_control="Z",
    initial_basis_target="Z",
    measure_basis_control="Z",
    measure_basis_target="Z",
    rounds_before=2,
    rounds_after=2,
)
circuit = experiment.build()
```

### Lattice-surgery CNOT experiment

```python
from experiments.CNOT_LS import CNOTLSExperiment

experiment = CNOTLSExperiment(
    patch_configs={
        "c": {"distance": 3},
        "t": {"distance": 3},
        "a": {"distance": 3},
    },
    offset_ta=(6, 0),
    offset_ca=(0, 6),
    initial_state_dict={"a": "X", "c": "X", "t": "X"},
    measure_state_dict={"a": "Z", "c": "X", "t": "X"},
    rounds=2,
)
circuit = experiment.build()
```

### Builder API (current method names)

```python
builder.initialize(init_dict, n=system.num_qubits)
builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=2)
builder.apply_data_readout(final_measurements=measurements)
```

## Simulation and decoding

```python
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.decoder_backend.config import DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_shots=100_000,
    max_errors=100,
    num_workers=4,
)
stats = pipeline.run(circuit, json_metadata={"d": 3, "p": 0.001})
print(stats.logical_error_rate)
```

Decoder notes:
- PyMatching: `DecoderConfig("pymatching")`
- BP+OSD CPU: `DecoderConfig("bposd", backend="cpu")`
- BP+OSD GPU: `DecoderConfig("bposd", backend="gpu")` (requires `cudaq_qec` + NVIDIA GPU)
- MWPF: `DecoderConfig("mwpf")`

## More documentation

- Full user guide: `docs/user_guide.md`
- Decoder backend details: `src/simulation/README.md`
- Plot API details: `src/plot/README.md`

## License

License is not specified yet in this repository.
