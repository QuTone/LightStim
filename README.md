# QEC Simulator

A modular Quantum Error Correction (QEC) simulator with automated detector generation, standardized noise injection, and support for complex multi-patch experiments including Lattice Surgery and Transversal Gates.

## 🎯 Project Overview

This project provides a comprehensive framework for building QEC experiments. Users can easily construct experiments using high-level instructions while the framework automatically handles:
- **Automated Detector Generation**: Using Pauli Tableau tracking
- **Multi-Patch System Management**: Coordinate multiple QEC patches in a unified system
- **Logical Operations**: Transversal gates and Lattice Surgery protocols
- **2D Layout Visualization**: All qubits have coordinates for easy debugging
- **Standardized Noise Injection**: Fair comparison infrastructure

## 📁 Project Structure

```
QEC_Simulator/
├── notebooks/                    # Example notebooks and demos
│   ├── memory_experiment.ipynb   # Memory experiment examples
│   ├── test_trans_CNOT.ipynb    # Transversal CNOT gate experiments
│   ├── test_LS_two_patch.ipynb  # Two-patch Lattice Surgery
│   ├── test_LS_CNOT.ipynb       # Lattice Surgery CNOT
│   ├── test_ghz.ipynb           # GHZ state preparation
│   ├── test_qecSys.ipynb        # QEC System testing
│   └── test_qec_code_info.ipynb # Code information testing
│
├── src/
│   ├── ir/                      # Intermediate Representation (Core abstractions)
│   │   ├── qec_patch.py         # Base class for all QEC codes
│   │   ├── qec_system.py        # Multi-patch system management
│   │   ├── tracker.py           # Automated Pauli Tableau tracking
│   │   ├── tableau.py           # Pauli tableau utilities
│   │   ├── builder.py           # High-level circuit builder API
│   │   ├── logical_executor.py  # Logical operation executor
│   │   ├── operation.py         # Logical operation definitions
│   │   ├── coupler.py            # Lattice Surgery coupler protocols
│   │   └── experiment.py        # Abstract base class for experiments
│   │
│   ├── qec_code/                # QEC Code Implementations
│   │   ├── repetition/          # Repetition Code
│   │   │   ├── repetition.py
│   │   │   └── SE_block.py      # Syndrome extraction block
│   │   └── surface_code/        # Surface Code variants
│   │       ├── rotated/          # Rotated Surface Code
│   │       │   ├── code_patch.py
│   │       │   ├── SE_block.py
│   │       │   └── operation.py
│   │       ├── unrotated/        # Unrotated Surface Code
│   │       │   ├── code_patch.py
│   │       │   ├── SE_block.py
│   │       │   ├── operation.py
│   │       │   └── two_patch_coupler.py  # Lattice Surgery coupler
│   │       └── toric/            # Toric Surface Code
│   │           ├── code_patch.py
│   │           └── SE_block.py
│   │
│   ├── experiments/              # Experiment Orchestrators
│   │   ├── memory.py             # Memory experiment
│   │   ├── two_patch_LS_unrotated.py  # Two-patch Lattice Surgery
│   │   ├── CNOT_LS.py           # Lattice Surgery CNOT (3-patch)
│   │   ├── CNOT_trans.py         # Transversal CNOT gate
│   │   └── ghz.py                # GHZ state preparation
│   │
│   ├── noise/                    # Noise Injection System
│   │   ├── config.py             # Noise configuration
│   │   ├── injector.py           # Noise injection logic
│   │   └── rules.py             # Noise rules (depolarizing, measurement errors, etc.)
│   │
│   ├── utils/                    # Utility functions
│   │   └── linear_algebra.py    # Linear algebra utilities
│   │
│   ├── simulation/               # Simulation Infrastructure
│   │   ├── simulator.py
│   │   ├── decoder.py
│   │   └── gpu_worker.py
│   │
│   └── processing/               # Processing and analysis tools
│       └── ...
│
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/x8fangQ/LightStim.git
cd QEC_Simulator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Jupyter Kernel

```bash
# Create Jupyter kernel with the virtual environment
python -m ipykernel install --user --name=qec-simulator --display-name="QEC Simulator"
```

### 3. Run Examples

Open any notebook in `notebooks/`:
- `memory_experiment.ipynb`: Single-patch memory experiments
- `test_trans_CNOT.ipynb`: Transversal CNOT gate between two patches
- `test_LS_two_patch.ipynb`: Two-patch Lattice Surgery
- `test_LS_CNOT.ipynb`: Lattice Surgery CNOT (3-patch experiment)
- `test_ghz.ipynb`: GHZ state preparation using transversal gates

## 📖 Core Concepts

### QEC Patch (IR Layer)

All QEC codes inherit from `QECPatch`, which provides:
- **Geometry**: 2D coordinate mapping for each qubit
- **Physics**: Stabilizers and logical operators as `stim.PauliString`
- **Visualization**: Automatic coordinate assignment for debugging

### QEC System (Multi-Patch Management)

`QECSystem` manages multiple QEC patches in a unified coordinate system:
- **Patch Registration**: Add patches with optional offsets
- **Global Index Mapping**: Automatic mapping from local to global qubit indices
- **Coupler Registration**: Register Lattice Surgery couplers between patches

### Syndrome Tracker (Automation)

`SyndromeTracker` automatically:
- Maintains the Pauli Tableau state
- Generates Detectors when stabilizers commute
- Tracks Logical Observables
- Works for Memory, Lattice Surgery, Transversal Gates, etc.

### Circuit Builder (High-Level API)

Users write simple instructions:
```python
builder.initialize(init_dict, n)          # Initialize qubits
builder.apply_syndrome_extraction(chunk)  # Syndrome extraction rounds
builder.apply_final_readout(measurements) # Final measurement
```

The builder automatically:
- Calls the Tracker to generate detectors
- Updates the tableau state
- Handles circuit construction

### Logical Executor (Logical Operations)

`LogicalExecutor` handles logical operations across patches:
- **Transversal Gates**: CNOT, Pauli gates applied transversally
- **Lattice Surgery**: Multi-patch interactions via couplers
- **Operation Sets**: Register different operation sets for different code types

### Noise Injection (Standardized)

```python
noisy_circuit = builder.build_noisy_circuit(
    noise_params=NoiseConfig(...),
    noise_model='circuit_level'  # or 'code_capacity', 'phenomenological', etc.
)
```

## 🧪 Experiment Types

### Memory Experiment

Single-patch quantum memory with syndrome extraction:

```python
from src.experiments.memory import MemoryExperiment
from src.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
from src.noise.config import NoiseConfig

code = RotatedSurfaceCode(distance=5)
noise_params = NoiseConfig(p_1q=0.001, p_2q=0.005, p_meas=0.002)

experiment = MemoryExperiment(
    qec_system=code,
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=5,
    noise_params=noise_params,
    noise_model='circuit_level',
    basis='Z'
)

circuit = experiment.build()
```

### Transversal CNOT Experiment

Transversal CNOT gate between two surface code patches:

```python
from src.experiments.CNOT_trans import CNOTTransExperiment

experiment = CNOTTransExperiment(
    distance=3,
    offset_target=(6, 0),
    initial_basis_control="Z",
    initial_basis_target="Z",
    measure_basis_control="Z",
    measure_basis_target="Z",
    rounds_before=2,
    rounds_after=2
)

circuit = experiment.build()
```

### Two-Patch Lattice Surgery

Two-patch Lattice Surgery with coupler activation:

```python
from src.experiments.two_patch_LS_unrotated import TwoPatchLSExperiment

experiment = TwoPatchLSExperiment(
    patch1_config={"distance": 3},
    patch2_config={"distance": 3},
    offset=(6, 0),
    interaction_type="XX",  # or "ZZ"
    initial_state_patch1="X",
    initial_state_patch2="X",
    measure_state_patch1="X",
    measure_state_patch2="X",
    rounds=2
)

circuit = experiment.build()
```

### Lattice Surgery CNOT

Three-patch Lattice Surgery CNOT experiment:

```python
from src.experiments.CNOT_LS import CNOTLSExperiment

experiment = CNOTLSExperiment(
    patch_configs={
        "control": {"distance": 3},
        "target": {"distance": 3},
        "ancilla": {"distance": 3}
    },
    offset_ta=(6, 0),
    offset_ca=(0, 6),
    initial_state_dict={"control": "X", "target": "Z", "ancilla": "X"},
    measure_state_dict={"control": "X", "target": "Z", "ancilla": "X"},
    rounds=2
)

circuit = experiment.build()
```

### GHZ State Preparation

GHZ state preparation using transversal CNOT gates:

```python
from src.experiments.ghz import GHZExperiment

experiment = GHZExperiment(
    distance=3,
    offset_patch2=(6, 0),
    offset_patch3=(0, 6),
    initial_basis_patch1="X",  # |+>
    initial_basis_patch2="Z",   # |0>
    initial_basis_patch3="Z",   # |0>
    measure_basis_patch1="Z",
    measure_basis_patch2="Z",
    measure_basis_patch3="Z",
    rounds_before=2,
    rounds_after=2
)

circuit = experiment.build()
```

## 🏗️ Architecture

### Experiment Base Class

All experiments inherit from `QECExperiment`, which provides:
- Common setup (`_setup_experiment()`)
- Noise injection (`_inject_noise()`)
- Unified interface for all experiment types

### Multi-Patch System

The `QECSystem` class enables:
- **Patch Management**: Add multiple patches with spatial offsets
- **Global Coordinates**: Unified coordinate system across patches
- **Coupler Protocols**: Register and activate Lattice Surgery couplers
- **Index Mapping**: Automatic local-to-global qubit index translation

### Logical Operations

The `LogicalExecutor` supports:
- **Transversal Operations**: Apply gates transversally across data qubits
- **Operation Sets**: Different code types can have different logical operations
- **Multi-Patch Operations**: Operations spanning multiple patches

## 📝 Adding a New QEC Code

1. **Create the code class** (inherit from `QECPatch`):
   ```python
   class MyQECCode(QECPatch):
       def _process_params(self):
           # Validate parameters
           pass
       
       def build(self):
           # 1. Register qubit coordinates
           # 2. Define stabilizers
           # 3. Define logical operators
           pass
   ```

2. **Create the syndrome extraction block**:
   ```python
   class MyQECCodeExtractionBlock:
       def __init__(self, system):
           self.system = system
           self.circuit = stim.Circuit()
           self._build_circuit()
       
       def _build_circuit(self):
           # Build the syndrome extraction circuit
           pass
   ```

3. **Create logical operations** (optional):
   ```python
   class MyLogicalOpSet(LogicalOpSet):
       def transversal_cnot(self, control_patch, target_patch):
           # Implement transversal CNOT
           pass
   ```

4. **Use in experiments**:
   ```python
   from src.qec_code.my_code import MyQECCode, MyQECCodeExtractionBlock
   
   code = MyQECCode(...)
   experiment = MemoryExperiment(
       qec_system=code,
       extraction_block_class=MyQECCodeExtractionBlock,
       ...
   )
   ```

## 🔧 Dependencies

- `stim`: Quantum circuit simulation
- `numpy`: Numerical computations
- `jupyter`, `ipykernel`: Notebook support
- `matplotlib`: Visualization (optional)

See `requirements.txt` for complete list.

## 📚 Examples

See notebooks in `notebooks/` for working examples:
- **Memory Experiments**: Single-patch quantum memory
- **Transversal Gates**: CNOT gates between patches
- **Lattice Surgery**: Two-patch and three-patch LS experiments
- **GHZ States**: Multi-patch entanglement preparation

## 🤝 Contributing

When adding new features:
1. Follow the existing architecture patterns
2. Inherit from `QECExperiment` for new experiment types
3. Use `QECSystem` for multi-patch experiments
4. Add examples to `notebooks/`
5. Test with various noise models

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Built on top of [Stim](https://github.com/quantumlib/Stim) for quantum circuit simulation.
