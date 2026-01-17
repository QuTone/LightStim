# QEC Simulator

A modular Quantum Error Correction (QEC) simulator with automated detector generation and standardized noise injection.

## 🎯 Project Overview

This project provides a modular framework for building QEC experiments. Users can easily construct experiments using high-level instructions while the framework automatically handles:
- **Automated Detector Generation**: Using Pauli Tableau tracking
- **2D Layout Visualization**: All qubits have coordinates for easy debugging
- **Standardized Noise Injection**: Fair comparison infrastructure

## 📁 Project Structure

```
QEC_Simulator/
├── notebooks/              # Example notebooks and demos
│   ├── memory_experiment.ipynb    # Memory experiment examples
│   └── test_qec_code_info.ipynb   # Code information testing
│
├── src/
│   ├── ir/                 # Intermediate Representation (Core abstractions)
│   │   ├── qec_patch.py    # Base class for all QEC codes
│   │   ├── tracker.py      # Automated Pauli Tableau tracking
│   │   ├── tableau.py      # Stabilizer tableau utilities
│   │   └── utils.py        # Linear algebra utilities
│   │
│   ├── qec_code/           # QEC Code Implementations
│   │   ├── repetition/     # Repetition Code
│   │   │   └── repetition.py
│   │   └── surface_code/   # Surface Code variants
│   │       ├── rotated.py
│   │       ├── unrotated.py
│   │       └── toric.py
│   │
│   ├── circuit/            # Circuit Construction
│   │   └── builder.py      # High-level circuit builder API
│   │
│   ├── noise/              # Noise Injection System
│   │   ├── config.py       # Noise configuration
│   │   ├── injector.py     # Noise injection logic
│   │   └── rules.py        # Noise rules (depolarizing, measurement errors, etc.)
│   │
│   ├── experiments/        # Experiment Orchestrators
│   │   └── memory.py       # Memory experiment template
│   │
│   └── simulation/         # Simulation Infrastructure (for future)
│       ├── simulator.py
│       ├── decoder.py
│       └── gpu_worker.py
│
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
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

### 3. Run Example

Open `notebooks/memory_experiment.ipynb` in Jupyter and run the cells. The notebook demonstrates:
- Rotated Surface Code memory experiment
- Unrotated Surface Code memory experiment
- Repetition Code memory experiment

## 📖 Core Concepts

### QEC Patch (IR Layer)

All QEC codes inherit from `QECPatch`, which provides:
- **Geometry**: 2D coordinate mapping for each qubit
- **Physics**: Stabilizers and logical operators as `stim.PauliString`
- **Visualization**: Automatic coordinate assignment for debugging

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

### Noise Injection (Standardized)

```python
noisy_circuit = builder.build_noisy_circuit(
    noise_params=NoiseConfig(...),
    noise_model='circuit_level'  # or 'code_capacity', 'phenomenological', etc.
)
```

## 🧪 Creating a Memory Experiment

```python
from src.experiments.memory import MemoryExperiment
from src.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
from src.noise.config import NoiseConfig

# 1. Define code and noise
code = RotatedSurfaceCode(distance=5)
noise_params = NoiseConfig(p_1q=0.001, p_2q=0.005, p_meas=0.002)

# 2. Build experiment
experiment = MemoryExperiment(
    qec_patch=code,
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=5,
    noise_params=noise_params,
    noise_model='circuit_level',
    basis='Z'
)

# 3. Generate circuit
circuit = experiment.build()
```

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

3. **Use in experiments**:
   ```python
   from src.qec_code.my_code import MyQECCode, MyQECCodeExtractionBlock
   
   code = MyQECCode(...)
   experiment = MemoryExperiment(
       qec_patch=code,
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

See `notebooks/memory_experiment.ipynb` for working examples with:
- Rotated Surface Code
- Unrotated Surface Code  
- Repetition Code

## 🤝 Contributing

When adding new QEC codes:
1. Implement the `QECPatch` subclass in `src/qec_code/`
2. Create the corresponding `ExtractionBlock` class
3. Add examples to `notebooks/`
4. Test with various noise models

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Built on top of [Stim](https://github.com/quantumlib/Stim) for quantum circuit simulation.
