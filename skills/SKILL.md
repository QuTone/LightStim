# skills/

Self-contained executable examples for using the LightStim API.
Each script can be run from the repo root with `python skills/<script>.py`.
Each demonstrates one complete workflow and serves as a reference for LLM-assisted development.

| Script | What it demonstrates |
|--------|----------------------|
| `01_memory_surface_code.py` | Build a rotated surface code Z-memory circuit end-to-end |
| `02_simulate_and_decode.py` | Run SimulationPipeline + PyMatching, read logical error rate |
| `03_transversal_cnot.py` | Two-patch transversal CNOT between unrotated surface codes |
| `04_lattice_surgery_cnot.py` | 3-patch lattice surgery CNOT (control + target + ancilla) |
| `05_state_injection.py` | Inject Z/X/Y logical states with post-selection |
| `06_custom_noise_model.py` | Compare circuit_level / phenomenological / code_capacity noise models |
| `07_extend_new_qec_code.py` | Template for adding a new QEC code (QECPatch + SE_block) |

## Usage

```bash
# From repo root (venv activated)
python skills/01_memory_surface_code.py
python skills/07_extend_new_qec_code.py
```

## Key API entry points

```python
# Build any QEC code
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
from lightstim.ir.qec_system import QECSystem

system = QECSystem()
system.add_patch(RotatedSurfaceCode(distance=3), name='main')

# Run a memory experiment
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig

exp = MemoryExperiment(system, RotatedSurfaceCodeExtractionBlock, rounds=3,
                       noise_params=NoiseConfig(p_2q=1e-3, p_meas=1e-3))
circuit = exp.build()

# Decode + get LER
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

stats = SimulationPipeline(decoder_config=DecoderConfig('pymatching'),
                           max_errors=200, print_progress=False).run(circuit)
print(stats.logical_error_rate)
```
