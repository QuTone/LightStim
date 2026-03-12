# LightStim User Guide

A comprehensive guide to the LightStim Quantum Error Correction simulator.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation & Setup](#2-installation--setup)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [User Guide](#5-user-guide)
   - [Building QEC Codes](#51-building-qec-codes)
   - [Creating Experiments](#52-creating-experiments)
   - [Multi-Patch Systems](#53-multi-patch-systems)
   - [Noise Configuration](#54-noise-configuration)
   - [Simulation & Decoding](#55-simulation--decoding)
   - [Visualization](#56-visualization)
   - [Circuit Verification](#57-circuit-verification)
6. [API Reference](#6-api-reference)
7. [Extending LightStim](#7-extending-lightstim)

---

## 1. Introduction

**LightStim** is a modular Quantum Error Correction (QEC) simulator built on top of [Stim](https://github.com/quantumlib/Stim), Google's fast stabilizer circuit simulator. LightStim provides a high-level framework for constructing, simulating, and analyzing QEC experiments while Stim handles the low-level circuit simulation.

### What Stim provides

Stim is a high-performance stabilizer circuit simulator that represents quantum circuits as sequences of Clifford gates, measurements, resets, and noise channels. It can sample detector outcomes at millions of shots per second and produce detector error models for decoding.

### What LightStim adds

- **Automated detector generation** via Pauli tableau tracking -- no manual detector annotation required
- **Multi-patch system management** with automatic local-to-global index mapping and coordinate transforms
- **Pluggable QEC codes**: Rotated Surface Code, Unrotated Surface Code, Toric Code, Repetition Code, and Bivariate Bicycle (BB) codes
- **Experiment orchestration**: Memory, Transversal CNOT, Lattice Surgery, GHZ state preparation, and State Injection experiments
- **Standardized noise injection**: Code-capacity, phenomenological, circuit-level, and biased noise models
- **Unified decoder backend**: PyMatching, BP+OSD, and MWPF decoders through a registry-based system
- **Visualization**: LER vs. physical error rate, LER vs. code distance, and configurable custom plots

### Three-Layer Architecture

LightStim separates concerns into three layers:

| Layer | Responsibility | Key abstraction |
|-------|---------------|-----------------|
| **Geometry** | 2D qubit coordinates | `QECPatch.qubit_coords` |
| **Physics** | Stabilizers and logical operators as `stim.PauliString` | `QECPatch.stabilizers`, `QECPatch.logical_ops` |
| **Visualization** | Auto-generated `QUBIT_COORDS` instructions | `CircuitBuilder.write_coordinates()` |

`QECPatch.qubit_coords` is the single source of truth for geometry. Grid maps are derived from it using `GRID_SCALE = 1000` integer rounding.

---

## 2. Installation & Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/x8fangQ/LightStim.git
cd LightStim

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Jupyter kernel (for notebooks)

```bash
python -m ipykernel install --user --name=qec-simulator --display-name="QEC Simulator"
```

### Decoder packages

`requirements.txt` already includes CPU decoder dependencies (`stimbposd`, `mwpf`, `frozendict`, `frozenlist`).

Install extra decoder packages only when needed:

```bash
pip install pymatching  # MWPM decoder (recommended if not already installed)
pip install cudaq_qec   # BP+OSD decoder, GPU (NVIDIA only; nv-qldpc-decoder)
```

---

## 3. Quick Start

Build a distance-3 rotated surface code memory experiment, sample it, and verify noiseless correctness.

```python
import sys, os
sys.path.insert(0, ".")

import stim
import numpy as np
from experiments.memory import MemoryExperiment
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)

# 1. Create a distance-3 rotated surface code patch
code = RotatedSurfaceCode(distance=3)

# 2. Build a Z-memory experiment (3 rounds, no noise)
experiment = MemoryExperiment(
    qec_system=code,
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=3,
    noise_params=None,   # noiseless
    basis="Z",
)
circuit = experiment.build()

# 3. Verify: noiseless circuit should produce zero detector flips
#    and zero logical observable flips
sampler = circuit.compile_detector_sampler()
dets, obs = sampler.sample(shots=1000, separate_observables=True)

assert np.all(dets == 0), "Detector flips found in noiseless circuit!"
assert np.all(obs == 0),  "Logical errors found in noiseless circuit!"
print(f"Verification passed: {circuit.num_detectors} detectors, "
      f"{circuit.num_observables} observables, 1000 shots all clean.")
```

### Adding noise and decoding

```python
from src.noise.config import NoiseConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.decoder_backend.config import DecoderConfig

# Build a noisy circuit
experiment = MemoryExperiment(
    qec_system=RotatedSurfaceCode(distance=3),
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=3,
    noise_params=NoiseConfig(p_1q=0.001, p_2q=0.005, p_meas=0.001),
    noise_model="circuit_level",
    basis="Z",
)
noisy_circuit = experiment.build()

# Decode with PyMatching
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_shots=100_000,
    max_errors=100,
    num_workers=4,
)
stats = pipeline.run(noisy_circuit, json_metadata={"d": 3, "p": 0.001})
print(f"LER = {stats.logical_error_rate:.2e} "
      f"({stats.errors} errors / {stats.post_selected_shots} shots)")
```

---

## 4. Core Concepts

This section provides a brief QEC primer for developers who may not have deep quantum error correction background.

### QEC Patches

A **QEC patch** is a collection of physical qubits arranged in a 2D layout that together encode one or more **logical qubits**. Each patch contains:

- **Data qubits**: Store the logical quantum information
- **Syndrome qubits** (ancillae): Used to measure stabilizers without disturbing the logical state
- **Stabilizers**: Multi-qubit Pauli operators (products of X, Y, Z on subsets of data qubits) that define the code space. Measuring them reveals errors without revealing the encoded data.
- **Logical operators**: Pauli operators that act on the encoded logical qubit(s). They commute with all stabilizers but are not themselves stabilizers.

### Syndrome Extraction Rounds

Each round of **syndrome extraction** (SE) measures all stabilizers once. The circuit for one SE round typically consists of:

1. Reset syndrome qubits
2. Apply entangling gates (CNOTs) between syndrome and data qubits
3. Measure syndrome qubits

The measurement outcomes form a **syndrome**. Changes in the syndrome between consecutive rounds signal errors.

### Noise Models

LightStim supports four noise models, each adding errors at different stages:

| Model | Where noise is added |
|-------|---------------------|
| `code_capacity` | Pauli errors on data qubits only (before measurement) |
| `phenomenological` | Pauli errors on data qubits + measurement errors |
| `circuit_level` | Errors after every gate, measurement, reset, and idle |
| `XZ_biased` | Circuit-level with asymmetric X/Z error rates |

### Detectors and Observables

- A **detector** is a parity check on measurement outcomes that should be deterministic in the absence of errors. When a detector "fires" (flips), it signals that an error occurred.
- A **logical observable** is a parity of measurements whose value encodes the logical qubit state. Decoding errors cause observable flips, which correspond to logical errors.

LightStim generates detectors automatically using **Pauli tableau tracking**: as the circuit evolves, the tracker maintains a tableau of stabilizers and their measurement records, emitting `DETECTOR` instructions whenever a stabilizer is re-measured.

### Decoding

A **decoder** takes the syndrome (detector outcomes) and predicts which logical observables were flipped by errors. The **logical error rate** (LER) is the fraction of shots where the decoder's prediction disagrees with the actual observable.

---

## 5. User Guide

### 5.1 Building QEC Codes

All QEC codes inherit from `QECPatch` and are instantiated with keyword arguments.

#### Rotated Surface Code

The most commonly used code. Uses a rotated lattice for better qubit efficiency.

```python
from src.qec_code.surface_code.rotated import RotatedSurfaceCode

# Square code (distance_z = distance_x = d)
code = RotatedSurfaceCode(distance=5)

# Rectangular code
code = RotatedSurfaceCode(distance_z=5, distance_x=3)

# With coordinate offset
code = RotatedSurfaceCode(distance=3, shift=(10, 0))
```

**Parameters**: `distance` (odd int, sets both), `distance_z` (odd int), `distance_x` (odd int), `shift` (tuple, default `(0,0)`)

#### Unrotated Surface Code

The standard planar surface code. Used for lattice surgery experiments.

```python
from src.qec_code.surface_code.unrotated import UnrotatedSurfaceCode

code = UnrotatedSurfaceCode(distance=3)
code = UnrotatedSurfaceCode(distance_z=5, distance_x=3)
```

**Parameters**: `distance` (int >= 2), `distance_z` (int), `distance_x` (int), `shift` (tuple, default `(0,0)`)

#### Toric Code

Surface code with periodic boundary conditions (on a torus). Encodes 2 logical qubits.

```python
from src.qec_code.surface_code.toric import ToricCode

# Square toric code
code = ToricCode(distance=4)

# Rectangular toric code
code = ToricCode(l_z=4, l_x=3)
```

**Parameters**: `distance` (int >= 2, sets both), `l_z` (int), `l_x` (int), `shift` (tuple, default `(0,0)`)

#### Repetition Code

The simplest QEC code -- a 1D chain of data qubits with Z-stabilizers.

```python
from src.qec_code.repetition.repetition import RepetitionCode

code = RepetitionCode(distance=5)
```

**Parameters**: `distance` (int >= 2), `shift` (tuple, default `(0,0)`)

#### Bivariate Bicycle (BB) Code

High-rate quantum LDPC codes defined by two 3-term polynomials over cyclic groups.

```python
from src.qec_code.BB_code import BBCode

# The [[144,12,12]] Gross code
code = BBCode(
    l=12, m=6,
    A=[[3, 0], [0, 1], [1, 1]],
    B=[[3, 0], [0, 1], [2, 2]],
    d=12,
)
```

**Parameters**: `l` (int), `m` (int), `A` (list of [x_exp, y_exp] pairs, 3 monomials), `B` (same), `d` (int, optional metadata), `shift` (tuple, default `(0,0)`). Optional polynomial logical operator parameters: `f`, `g`, `h`, `alpha`, `beta`.

#### Inspecting a code

Every code provides a `get_info()` method returning a dict with all geometry and physics data:

```python
info = code.get_info()
print(f"Data qubits: {len(info['data_coords'])}")
print(f"X-syndromes: {len(info['syndrome_coords_x'])}")
print(f"Z-syndromes: {len(info['syndrome_coords_z'])}")
print(f"Stabilizers: {len(info['stabilizers'])}")
print(f"Logicals: {len(info['logical_ops'])}")
```

---

### 5.2 Creating Experiments

LightStim provides six experiment classes covering memory, transversal gates, lattice surgery, GHZ preparation, and state injection.

#### MemoryExperiment

Single-patch quantum memory: initialize, run SE rounds, measure.

```python
from experiments.memory import MemoryExperiment
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
)
from src.noise.config import NoiseConfig

experiment = MemoryExperiment(
    qec_system=RotatedSurfaceCode(distance=5),
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=5,
    noise_params=NoiseConfig(p_1q=0.001, p_2q=0.005, p_meas=0.001),
    noise_model="circuit_level",
    basis="Z",           # or "X"
    if_detector=True,     # generate detectors (default)
)
circuit = experiment.build()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `qec_system` | `QECPatch` or `QECSystem` | The code patch or system to use |
| `extraction_block_class` | `Type` | SE block class (e.g. `RotatedSurfaceCodeExtractionBlock`) |
| `rounds` | `int` | Number of SE rounds (default 2) |
| `noise_params` | `NoiseConfig` or `None` | Noise parameters (None = noiseless) |
| `noise_model` | `str` | Noise model name (default `"circuit_level"`) |
| `basis` | `"X"` or `"Z"` | Memory basis (default `"Z"`) |
| `if_detector` | `bool` | Whether to generate detectors (default `True`) |

#### CNOTTransExperiment

Transversal CNOT between two CSS code patches.

```python
from experiments.CNOT_trans import CNOTTransExperiment
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
)

experiment = CNOTTransExperiment(
    code_patch_class=UnrotatedSurfaceCode,
    extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
    code_params_control={"distance": 3},
    code_params_target={"distance": 3},    # defaults to same as control
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

| Parameter | Type | Description |
|-----------|------|-------------|
| `code_patch_class` | `Type[QECPatch]` | Patch class (e.g. `UnrotatedSurfaceCode`) |
| `extraction_block_class` | `Type` | SE block class |
| `code_params_control` | `dict` | kwargs for control patch constructor |
| `code_params_target` | `dict` or `None` | kwargs for target patch (defaults to control) |
| `offset_target` | `tuple` | (dx, dy) offset for target patch |
| `initial_basis_control/target` | `"X"` or `"Z"` | Initial state basis per patch |
| `measure_basis_control/target` | `"X"` or `"Z"` | Measurement basis per patch |
| `rounds_before/after` | `int` | SE rounds before/after the CNOT |
| `noise_params` | `NoiseConfig` or `None` | Noise parameters |
| `noise_model` | `str` | Noise model name |

#### TwoPatchLSExperiment

Two-patch lattice surgery with unrotated surface codes.

```python
from experiments.two_patch_LS_unrotated import TwoPatchLSExperiment

experiment = TwoPatchLSExperiment(
    patch1_config={"distance": 3},
    patch2_config={"distance": 3},
    offset=(6, 0),
    interaction_type="XX",       # or "ZZ"
    initial_state_patch1="X",
    initial_state_patch2="X",
    measure_state_patch1="X",
    measure_state_patch2="X",
    rounds=2,
    rotate_patch1=True,          # rotate patch1 to align logical operators
)
circuit = experiment.build()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `patch1_config/patch2_config` | `dict` | kwargs for `UnrotatedSurfaceCode` |
| `offset` | `tuple` | (dx, dy) offset for patch2 |
| `interaction_type` | `"XX"` or `"ZZ"` | Coupler interaction type |
| `initial_state_patch1/2` | `"X"` or `"Z"` | Initial state basis |
| `measure_state_patch1/2` | `"X"` or `"Z"` | Measurement basis |
| `rounds` | `int` | SE rounds before and after coupler activation |
| `rotate_patch1` | `bool` | Rotate patch1 by pi for alignment (default `True`) |

#### CNOTLSExperiment

Three-patch lattice surgery CNOT using Control, Target, and Ancilla patches.

```python
from experiments.CNOT_LS import CNOTLSExperiment

experiment = CNOTLSExperiment(
    patch_configs={
        "c": {"distance": 3},    # control
        "t": {"distance": 3},    # target
        "a": {"distance": 3},    # ancilla
    },
    offset_ta=(6, 0),            # target relative to ancilla
    offset_ca=(0, 6),            # control relative to ancilla
    initial_state_dict={"a": "X", "c": "X", "t": "X"},
    measure_state_dict={"a": "Z", "c": "X", "t": "X"},
    rounds=2,
)
circuit = experiment.build()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `patch_configs` | `dict` | `{"c": {...}, "t": {...}, "a": {...}}` for each patch |
| `offset_ta` | `tuple` | Target offset relative to Ancilla |
| `offset_ca` | `tuple` | Control offset relative to Ancilla |
| `initial_state_dict` | `dict` | `{"a": "X"/"Z", "c": ..., "t": ...}` |
| `measure_state_dict` | `dict` | Same format for measurement bases |
| `rounds` | `int` | SE rounds per phase |
| `rotate_patches` | `bool` | Auto-rotate for alignment (default `True`) |

#### GHZExperiment

GHZ state preparation using transversal CNOT gates on three surface code patches: `|+>|0>|0>` -> CNOT(1,2) -> CNOT(1,3).

```python
from experiments.ghz import GHZExperiment

experiment = GHZExperiment(
    distance=3,                    # or (d1, d2, d3) tuple
    offset_patch2=(6, 0),
    offset_patch3=(12, 0),
    initial_basis_patch1="X",      # |+>
    initial_basis_patch2="Z",      # |0>
    initial_basis_patch3="Z",      # |0>
    measure_basis_patch1="Z",
    measure_basis_patch2="Z",
    measure_basis_patch3="Z",
    rounds_before=2,
    rounds_after=2,
)
circuit = experiment.build()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance` | `int` or `(int, int, int)` | Distance for all patches (or per-patch tuple) |
| `offset_patch2/3` | `tuple` | Offsets for patches 2 and 3 |
| `initial_basis_patch1/2/3` | `"X"` or `"Z"` | Initial state per patch |
| `measure_basis_patch1/2/3` | `"X"` or `"Z"` | Measurement basis per patch |
| `rounds_before/after` | `int` | SE rounds before/after CNOT gates |

#### StateInjectionExperiment

State injection for rotated surface code using corner or middle protocols.

```python
from experiments.state_injection import StateInjectionExperiment

experiment = StateInjectionExperiment(
    distance=5,
    rounds=3,
    injection_protocol="corner",   # or "middle"
    inject_state="Z",              # "Z" -> |0>, "X" -> |+>
)
circuit = experiment.build()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance` | `int` | Code distance (odd) |
| `rounds` | `int` | SE rounds |
| `injection_protocol` | `"corner"` or `"middle"` | Injection site |
| `inject_state` | `"Z"` or `"X"` | Target logical state (`"Z"` = \|0>, `"X"` = \|+>) |

---

### 5.3 Multi-Patch Systems

For experiments involving more than one code patch (transversal gates, lattice surgery), use `QECSystem` to manage patches in a unified coordinate space.

#### Creating a system and adding patches

```python
from src.ir.qec_system import QECSystem
from src.qec_code.surface_code.unrotated import UnrotatedSurfaceCode

system = QECSystem()

patch1 = UnrotatedSurfaceCode(distance=3)
patch2 = UnrotatedSurfaceCode(distance=3)

# add_patch returns a global-index view of the patch
global_patch1 = system.add_patch(patch1, name="control")
global_patch2 = system.add_patch(patch2, name="target", offset=(6, 0))

print(f"Total qubits: {system.num_qubits}")
print(f"Total logicals: {system.num_logicals}")
```

#### Coordinate transforms

Patches support coordinate transformations before being added to a system:

```python
import numpy as np

patch = UnrotatedSurfaceCode(distance=3)

# Rotate by angle (radians)
patch.rotate_coords(np.pi)
patch.reset_rotation_angle()    # clear rotation tracking

# Transpose coordinates (swap x, y)
patch.transpose_coords()
patch.reset_transposition()

# Shift coordinates
patch.shift_coords(5, 5)
```

#### Coupler registration (lattice surgery)

```python
from src.qec_code.surface_code.unrotated import UnrotatedTwoPatchCoupler

coupler_protocol = UnrotatedTwoPatchCoupler()
system.register_coupler(
    coupler_protocol,
    patch_names=["control", "target"],
    name="coupler_c_t",
    interaction_type="XX",        # or "ZZ"
)
```

#### Key system properties

```python
system.num_qubits            # total qubit count (all patches + couplers)
system.num_logicals           # total logical qubit count
system.data_indices           # set of all data qubit global indices
system.data_coords            # list of all data qubit coordinates
system.index_map              # coord -> global index mapping
system.grid_map               # grid_key -> global index mapping
system.index_to_owner_map     # global index -> patch name mapping
system.local_to_global_map    # patch_name -> {local_idx: global_idx}
system.patches                # {name: (patch, offset)} dict
system.coupler_patches        # {name: coupler_patch} dict
system.active_syndrome_indices     # list of active syndrome qubit indices
system.active_syndrome_indices_x   # X-type syndrome indices
system.active_syndrome_indices_z   # Z-type syndrome indices
system.active_stabilizers_x        # list of active X-stabilizer dicts
system.active_stabilizers_z        # list of active Z-stabilizer dicts
```

---

### 5.4 Noise Configuration

Noise is specified via a `NoiseConfig` dataclass and a model name string.

#### NoiseConfig fields

```python
from src.noise.config import NoiseConfig

noise = NoiseConfig(
    p_1q=0.001,         # depolarizing after 1-qubit gates (H, S, ...)
    p_2q=0.005,         # depolarizing after 2-qubit gates (CNOT, CZ, ...)
    p_meas=0.001,       # measurement outcome flip probability
    p_reset=0.001,      # reset state flip probability
    p_idle=0.0005,      # depolarizing on idle qubits during TICK
)
```

For biased noise, use `custom_params`:

```python
noise = NoiseConfig(
    p_1q=0.001,
    p_2q=0.005,
    p_meas=0.001,
    custom_params={
        "p_1q_x": 0.0001,
        "p_1q_z": 0.001,
    }
)
```

#### Available noise models

| Model | String | Description |
|-------|--------|-------------|
| Code capacity | `"code_capacity"` | Pauli errors on data qubits only |
| Phenomenological | `"phenomenological"` | Data qubit errors + measurement errors |
| Circuit level | `"circuit_level"` | Full gate-level noise (most realistic) |
| XZ biased | `"XZ_biased"` | Circuit-level with asymmetric X vs Z rates |

#### Passing noise to experiments

```python
experiment = MemoryExperiment(
    qec_system=code,
    extraction_block_class=SEBlock,
    rounds=5,
    noise_params=NoiseConfig(p_1q=0.001, p_2q=0.005, p_meas=0.001),
    noise_model="circuit_level",
    basis="Z",
)
circuit = experiment.build()   # returns noisy circuit
```

Pass `noise_params=None` for a noiseless circuit (useful for verification).

---

### 5.5 Simulation & Decoding

The `SimulationPipeline` handles sampling, optional post-selection, decoding, and statistics collection.

#### Basic single-circuit simulation

```python
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.decoder_backend.config import DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_shots=1_000_000,
    max_errors=100,
    num_workers=4,
)

stats = pipeline.run(circuit, json_metadata={"d": 3, "p": 0.001})
print(f"LER: {stats.logical_error_rate:.2e}")
print(f"Shots: {stats.shots}, Errors: {stats.errors}")
print(f"Time: {stats.seconds:.1f}s")
```

#### Batch simulation

Run multiple circuits (e.g., sweeping distance or error rate) and collect results into a DataFrame:

```python
from src.simulation.decoder_backend.pipeline import ExperimentTask

tasks = []
for d in [3, 5, 7]:
    for p in [1e-3, 3e-3, 5e-3]:
        exp = MemoryExperiment(
            qec_system=RotatedSurfaceCode(distance=d),
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            rounds=d,
            noise_params=NoiseConfig(p_1q=p, p_2q=p, p_meas=p),
            noise_model="circuit_level",
            basis="Z",
        )
        circ = exp.build()
        tasks.append(ExperimentTask(circ, json_metadata={"d": d, "p": p}))

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_shots=100_000,
    max_errors=100,
    num_workers=4,
    output_dir="data/results",    # auto-save results
)
df = pipeline.run_batch(tasks)
print(df[["d", "p", "logical_error_rate", "shots", "errors"]])
```

#### Available decoders

| Decoder | Config name | Backend | Install | Best for |
|---------|------------|---------|---------|----------|
| **PyMatching** | `"pymatching"` | `"cpu"` | `pip install pymatching` | Surface codes (MWPM) |
| **BP+OSD** | `"bposd"` | `"cpu"` | `pip install stimbposd` | LDPC / BB codes (CPU) |
| **BP+OSD GPU** | `"bposd"` or `"nv-qldpc-decoder"` | `"gpu"` | `pip install cudaq_qec` | LDPC / BB codes (GPU) |
| **MWPF** | `"mwpf"` | `"cpu"` | `pip install mwpf frozendict frozenlist` | General codes |

Aliases: `"mwpm"` -> `"pymatching"`, `"bp_osd"` -> `"bposd"`.

Requesting `backend="gpu"` when `cudaq_qec` is not installed raises `ImportError` immediately.

```python
# PyMatching (default)
decoder = DecoderConfig("pymatching")

# BP+OSD CPU with unified parameters
decoder = DecoderConfig("bposd", backend="cpu", params={
    "max_iterations": 1000,
    "osd_order": 10,
    "osd_method": "osd_cs",
    "bp_method": "min_sum",
    "ms_scaling_factor": 0,
})

# BP+OSD GPU (requires cudaq_qec + NVIDIA GPU)
decoder = DecoderConfig("bposd", backend="gpu", params={
    "max_iterations": 1000,   # same unified names work on GPU
    "osd_order": 10,
    "osd_method": "osd_cs",
})

# MWPF
decoder = DecoderConfig("mwpf")
```

Both CPU and GPU bposd backends accept the same unified parameter names:

| Unified param | CPU maps to | GPU maps to | Default |
|---|---|---|---|
| `max_iterations` | `max_bp_iters` | `max_iterations` | `1000` |
| `bp_method` | `bp_method` (`'minimum_sum'`) | `bp_method` (`1`) | `'min_sum'` |
| `ms_scaling_factor` | `ms_scaling_factor` | `scale_factor` | `0` |
| `osd_order` | `osd_order` | `osd_order` | `10` |
| `osd_method` | `osd_method` (`'osd_cs'`) | `osd_method` (`3`) | `'osd_cs'` |
| `use_osd` | *(ignored; always on)* | `use_osd` | `True` |

#### SimulationPipeline constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decoder_config` | `DecoderConfig` | `DecoderConfig("pymatching")` | Decoder configuration |
| `max_shots` | `int` | `1_000_000` | Maximum number of shots |
| `max_errors` | `int` | `100` | Stop after this many errors |
| `batch_size` | `int` | `10_000` | Shots per sampling batch |
| `num_workers` | `int` | `4` | Parallel worker processes |
| `post_select_detector_indices` | `list[int]` or `None` | `None` | Detectors to post-select on |
| `output_dir` | `str` or `None` | `None` | Directory to save results |
| `output_format` | `str` | `"csv"` | Output format (`"csv"`, `"json"`, `"parquet"`) |
| `print_progress` | `bool` | `True` | Print progress updates |

#### SimulationStats fields

| Field | Type | Description |
|-------|------|-------------|
| `shots` | `int` | Total shots sampled |
| `post_selected_shots` | `int` | Shots remaining after post-selection |
| `errors` | `int` | Logical errors detected |
| `seconds` | `float` | Wall-clock time |
| `decoder` | `str` | Decoder name used |
| `json_metadata` | `dict` | Metadata passed in |
| `logical_error_rate` | `float` (property) | `errors / post_selected_shots` |
| `post_selection_rate` | `float` (property) | `post_selected_shots / shots` |

---

### 5.6 Visualization

The plot module provides preset and configurable plotting functions. Input is a `pandas.DataFrame` from `SimulationPipeline.run_batch()`.

#### LER vs. physical error rate

```python
from src.plot import plot_ler_vs_p

plot_ler_vs_p(df, hue="d", x_col="p1")  # falls back to "p" if "p1" is absent
```

#### LER vs. code distance

```python
from src.plot import plot_ler_vs_distance

plot_ler_vs_distance(df, hue="decoder", x_col="d")
```

#### Generic simulation results plot

```python
from src.plot.plotter import plot_simulation_results

plot_simulation_results(
    df,
    x="p",
    y="logical_error_rate",
    hue="d",
    x_scale="log",
    y_scale="log",
    save_path="plots/ler_vs_p.png",
)
```

#### Custom plot with PlotConfig

```python
from src.plot.plotter import plot_custom
from src.plot.config import PlotConfig

cfg = PlotConfig(
    x="p",
    y="logical_error_rate",
    hue="d",
    x_scale="log",
    y_scale="log",
    palette="distance",
    title="LER vs Physical Error Rate",
    x_label="Physical Error Rate (p)",
    y_label="Logical Error Rate (LER)",
    error_bars=True,
    figsize=(8, 6),
    marker="o",
    linewidth=2.5,
)
plot_custom(df, cfg, save_path="plots/custom.png")
```

#### PlotConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `x` | `str` | (required) | DataFrame column for x-axis |
| `y` | `str` | (required) | DataFrame column for y-axis |
| `hue` | `str` or `None` | `None` | Column for color grouping |
| `style` | `str` or `None` | `None` | Column for line style |
| `facet_col` | `str` or `None` | `None` | Column for facet columns |
| `facet_row` | `str` or `None` | `None` | Column for facet rows |
| `x_scale` | `"linear"` or `"log"` | `"log"` | X-axis scale |
| `y_scale` | `"linear"` or `"log"` | `"log"` | Y-axis scale |
| `palette` | `dict` or `str` or `None` | `None` | Color palette (`"distance"` for built-in) |
| `title` | `str` or `None` | `None` | Plot title |
| `x_label` | `str` or `None` | `None` | X-axis label |
| `y_label` | `str` or `None` | `None` | Y-axis label |
| `error_bars` | `bool` | `True` | Show error bars |
| `figsize` | `tuple` | `(7, 5)` | Figure size |
| `marker` | `str` or `None` | `"o"` | Marker style |
| `linewidth` | `float` | `2.5` | Line width |

#### Stim circuit diagrams

You can also use Stim's built-in circuit visualization:

```python
# Interactive SVG diagram with detector slices and operations
circuit.diagram("detslice-with-ops-svg")

# Timeline diagram
circuit.diagram("timeline-svg")
```

---

### 5.7 Circuit Verification

Before running expensive simulations with noise, verify that your noiseless circuit is correct. The noiseless sampling pattern is:

```python
import numpy as np

# Build noiseless circuit (noise_params=None)
circuit = experiment.build()

# Sample with detector sampler
sampler = circuit.compile_detector_sampler()
dets, obs = sampler.sample(shots=1000, separate_observables=True)

# All detectors should be zero (no flips)
assert np.all(dets == 0), f"Detector flips found! Failing detectors: {np.where(dets.any(axis=0))[0]}"

# All observables should be zero
assert np.all(obs == 0), f"Observable flips found!"

print("Circuit verification passed.")
```

You can also verify with the detector error model:

```python
# This will raise an error if detectors are malformed
dem = circuit.detector_error_model()
print(f"DEM has {dem.num_detectors} detectors, {dem.num_observables} observables")
```

---

## 6. API Reference

### 6.1 IR Module (`src/ir/`)

#### QECPatch (`src/ir/qec_patch.py`)

Abstract base class for all QEC codes.

```python
class QECPatch(ABC):
    def __init__(self, **kwargs)
```

**Constructor**: All keyword arguments are stored in `self.params` and processed by `_process_params()`.

**Abstract methods** (subclasses must implement):

| Method | Description |
|--------|-------------|
| `_process_params()` | Validate and extract constructor parameters |
| `build()` | Construct geometry (coords), stabilizers, and logical operators |

**Key properties and attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `qubit_coords` | `dict[int, tuple]` | `{local_index: (x, y)}` -- single source of truth |
| `data_indices` | `set[int]` | Set of data qubit local indices |
| `data_coords` | `list[tuple]` | List of data qubit coordinates |
| `syndrome_indices_x` | `set[int]` | X-type syndrome qubit indices |
| `syndrome_indices_z` | `set[int]` | Z-type syndrome qubit indices |
| `syndrome_coords` | `list[tuple]` | All syndrome qubit coordinates |
| `stabilizers` | `list[dict]` | Stabilizer definitions |
| `logical_ops` | `list[dict]` | Logical operator definitions |
| `index_map` | `dict[tuple, int]` | Coordinate to local index mapping |
| `num_logicals` | `int` | Number of logical qubits |

**Public methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_info()` | `-> dict` | Returns comprehensive patch information |
| `get_grid_key()` | `(coord) -> tuple` | Quantize coordinate to grid key |
| `create_stim_stabilizer()` | `(target_dict, syn_coord=None, type=None)` | Register a stabilizer |
| `create_stim_logical()` | `(target_dict, op_type)` | Register a logical operator |
| `rotate_coords()` | `(angle: float)` | Rotate all coordinates by angle (radians) |
| `transpose_coords()` | `()` | Swap x and y coordinates |
| `shift_coords()` | `(dx: float, dy: float)` | Translate all coordinates |
| `reset_rotation_angle()` | `()` | Clear rotation tracking |
| `reset_transposition()` | `()` | Clear transposition tracking |
| `snap_coord()` | `(coord) -> tuple` | Snap coordinate to nearest grid point |

---

#### QECSystem (`src/ir/qec_system.py`)

Multi-patch manager with global coordinate space.

```python
class QECSystem:
    def __init__(self)
```

**Public methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_patch()` | `(patch, name, offset=(0,0)) -> QECPatch` | Add a patch; returns global-index view |
| `register_coupler()` | `(protocol, patch_names, name, **params)` | Register a lattice surgery coupler |
| `register_tracker()` | `(tracker)` | Register a SyndromeTracker |
| `register_builder()` | `(builder)` | Register a CircuitBuilder |

**Key properties**:

| Property | Type | Description |
|----------|------|-------------|
| `num_qubits` | `int` | Total qubit count |
| `num_logicals` | `int` | Total logical qubit count |
| `data_indices` | `set[int]` | All data qubit global indices |
| `data_coords` | `list[tuple]` | All data qubit coordinates |
| `index_map` | `dict[tuple, int]` | Global coord -> index map |
| `grid_map` | `dict[tuple, int]` | Grid key -> global index |
| `qubit_coords` | `dict[int, tuple]` | Global index -> coordinate |
| `index_to_owner_map` | `dict[int, str]` | Global index -> patch name |
| `local_to_global_map` | `dict[str, dict]` | `{patch_name: {local: global}}` |
| `patches` | `dict` | `{name: (patch, offset)}` |
| `coupler_patches` | `dict` | `{name: coupler_patch}` |
| `active_syndrome_indices` | `list[int]` | Active syndrome global indices |
| `active_syndrome_indices_x` | `list[int]` | Active X-syndrome indices |
| `active_syndrome_indices_z` | `list[int]` | Active Z-syndrome indices |
| `active_stabilizers_x` | `list[dict]` | Active X-stabilizer info dicts |
| `active_stabilizers_z` | `list[dict]` | Active Z-stabilizer info dicts |

---

#### SyndromeTracker (`src/ir/tracker.py`)

Pauli tableau-based automated detector generation.

```python
class SyndromeTracker:
    def __init__(self, num_qubits: int, expected_num_logicals: int = 0)
```

**Public methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `process_initialization()` | `(init_tableau)` | Register initial stabilizers |
| `process_mid_measurement()` | `(circuit, back_propagated_paulis, syn_coords)` | Process one syndrome round and emit detectors |
| `process_final_measurement()` | `(circuit, final_paulis, idx_to_coord_map)` | Process final readout and emit detectors/observables |
| `process_unitary_block()` | `(unitary_block)` | Update tableau through a unitary circuit block |

**Key attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `stabilizer_tableau` | `PauliTableau` | Current stabilizer tableau |
| `logical_tableau` | `PauliTableau` | Current logical operator tableau |
| `circuit` | `stim.Circuit` | Accumulated detector/observable instructions |
| `measurement_count` | `int` | Running measurement counter |

---

#### CircuitBuilder (`src/ir/builder.py`)

High-level circuit construction API.

```python
class CircuitBuilder:
    def __init__(self, tracker: SyndromeTracker, system_config, if_detector: bool = True)
```

**Public methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `write_coordinates()` | `()` | Emit `QUBIT_COORDS` instructions |
| `initialize()` | `(init_dict: dict, n: int)` | Initialize data qubits (`{qubit_idx: "X"/"Z"}`) |
| `apply_syndrome_extraction()` | `(circuit_chunk, rounds)` | Apply SE rounds, generate detectors |
| `apply_data_readout()` | `(final_measurements: dict)` | Final measurement + detectors + observables |
| `apply_unitary_block()` | `(unitary_block: stim.Circuit)` | Apply unitary and update tableau |
| `activate_coupler()` | `(coupler_name: str)` | Activate a registered coupler |
| `deactivate_coupler()` | `(coupler_name: str)` | Deactivate a coupler |
| `build_noisy_circuit()` | `(noise_params, noise_model) -> stim.Circuit` | Inject noise into the clean circuit |

**Key attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `circuit` | `stim.Circuit` | The accumulated clean circuit |
| `tracker` | `SyndromeTracker` | Associated syndrome tracker |

---

#### QECExperiment (`src/ir/experiment.py`)

Abstract base class for experiments.

```python
class QECExperiment(ABC):
    def __init__(self,
                 extraction_block_class: Type,
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = "circuit_level",
                 if_detector: bool = True)
```

**Abstract methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `build()` | `stim.Circuit` | Construct the full experiment circuit |

**Helper methods** (for subclasses):

| Method | Description |
|--------|-------------|
| `_setup_experiment()` | Create tracker, builder, executor from `self.system` |
| `_inject_noise(circuit)` | Wrap circuit with noise if `noise_params` is set |

---

#### LogicalExecutor (`src/ir/logical_executor.py`)

Routes logical operations to code-family-specific operation sets.

```python
class LogicalExecutor:
    def __init__(self, builder: CircuitBuilder)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_op_set()` | `(patch_type, op_set)` | Register an `LogicalOpSet` for a patch type |
| `apply_logical_operation()` | `(op_name, patches, **kwargs)` | Dispatch operation by name via reflection |

---

#### LogicalOpSet / CSSLogicalOpSet (`src/ir/operation.py`)

```python
class LogicalOpSet(ABC):
    def __init__(self, name: str = "QECCode")

class CSSLogicalOpSet(LogicalOpSet):
    def __init__(self)
```

`CSSLogicalOpSet` provides universal operations for CSS codes:

| Method | Signature | Description |
|--------|-----------|-------------|
| `transversal_cnot()` | `(builder, control_patch, target_patch)` | Transversal CNOT between two CSS patches |
| `prepare_logical_z()` | `(builder, patch)` | Prepare logical \|0> (stub) |
| `prepare_logical_x()` | `(builder, patch)` | Prepare logical \|+> (stub) |

---

#### LogicalCouplerProtocol (`src/ir/coupler.py`)

Abstract factory for lattice surgery coupler patches.

```python
class LogicalCouplerProtocol(ABC):
    EXPECTED_PATCH_COUNT: Optional[int] = None

    def __init__(self, name_prefix: str = "coupler")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_coupler_patch()` | `(patches, name, **params) -> QECPatch` | Factory: create coupler patch from interacting patches |
| `_build_coupler_geometry()` | `(coupler_patch, patches, **params)` | **Abstract**: subclass fills coupler geometry |

---

#### PauliTableau (`src/ir/tableau.py`)

Binary matrix (M x 2N) Gottesman-Knill representation for stabilizer tracking.

```python
class PauliTableau:
    def __init__(self, num_qubits: int)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_stabilizers()` | `(paulis: ndarray, new_records=None)` | Batch add (K, 2N) stabilizer rows |
| `update_row()` | `(target_idx, source_idx)` | XOR row[target] with row[source] |
| `update_row_from_external()` | `(target_idx, external_pauli, external_record)` | XOR with external data |
| `replace_row()` | `(idx, new_pauli, new_record)` | Replace a row entirely |
| `remove_rows()` | `(indices: list[int])` | Delete rows by index |
| `expand()` | `(delta: int)` | Add delta new qubits (identity columns) |
| `get_record()` | `(idx) -> list[int]` | Get measurement record for row |

**Properties**: `count` (number of rows), `num_qubits`, `matrix` (ndarray), `records` (list of lists).

---

### 6.2 QEC Code Module (`src/qec_code/`)

#### RotatedSurfaceCode (`src/qec_code/surface_code/rotated/code_patch.py`)

```python
class RotatedSurfaceCode(QECPatch):
    def __init__(self, *, distance=None, distance_z=None, distance_x=None, shift=(0,0), **kwargs)
```

**Additional methods**: `shift_logical_operators(op_type, offset)`, `get_info()`

**SE block**: `RotatedSurfaceCodeExtractionBlock(system)` -- 4-tick zigzag CNOT schedule.

**Logical ops**: `RotatedSurfaceCodeLogicalOpSet(LogicalOpSet)` -- stubs for Hadamard.

---

#### UnrotatedSurfaceCode (`src/qec_code/surface_code/unrotated/code_patch.py`)

```python
class UnrotatedSurfaceCode(QECPatch):
    def __init__(self, *, distance=None, distance_z=None, distance_x=None, shift=(0,0), **kwargs)
```

**Additional methods**: `shift_logical_operators(op_type, offset)`, `get_info()`

**SE block**: `UnrotatedSurfaceCodeExtractionBlock(system)` -- 6-tick CNOT schedule.

**Logical ops**: `UnrotatedSurfaceCodeLogicalOpSet(LogicalOpSet)` -- stubs for fold-transversal Hadamard and S gate.

**Coupler**: `UnrotatedTwoPatchCoupler(LogicalCouplerProtocol)` -- two-patch lattice surgery coupler.

---

#### ToricCode (`src/qec_code/surface_code/toric/code_patch.py`)

```python
class ToricCode(QECPatch):
    def __init__(self, *, distance=None, l_z=None, l_x=None, shift=(0,0), **kwargs)
```

Encodes 2 logical qubits. All weight-4 stabilizers (periodic boundaries).

**SE block**: `ToricCodeExtractionBlock(system)` -- 6-tick schedule with coordinate wrapping.

---

#### RepetitionCode (`src/qec_code/repetition/repetition.py`)

```python
class RepetitionCode(QECPatch):
    def __init__(self, *, distance, shift=(0,0), **kwargs)
```

1D chain, Z-stabilizers, transversal X logical.

**SE block**: `RepetitionCodeExtractionBlock(system)` -- 2-tick CNOT schedule.

---

#### BBCode (`src/qec_code/BB_code/code_patch.py`)

```python
class BBCode(QECPatch):
    def __init__(self, *, l, m, A, B, d=None, shift=(0,0),
                 f=None, g=None, h=None, alpha=None, beta=None, **kwargs)
```

CSS code on 2D torus with weight-6 stabilizers. Builds logical operators either from polynomial parameters (`f`, `g`, `h`, `alpha`, `beta`) or numerically via GF(2) kernel computation.

**SE block**: `BBCodeExtractionBlock(system)` -- 7-tick CNOT schedule matching Bravyi et al. 2024.

---

### 6.3 Noise Module (`src/noise/`)

#### NoiseConfig (`src/noise/config.py`)

```python
@dataclass
class NoiseConfig:
    p_1q: float = 0.0        # 1-qubit gate depolarizing
    p_2q: float = 0.0        # 2-qubit gate depolarizing
    p_meas: float = 0.0      # measurement flip
    p_reset: float = 0.0     # reset flip
    p_idle: float = 0.0      # idle depolarizing
    custom_params: Dict[str, float] = field(default_factory=dict)

    def get(self, param_name: str, default: float = 0.0) -> float
```

#### NoiseInjector (`src/noise/injector.py`)

```python
class NoiseInjector:
    def __init__(self, model: NoiseConfig)
    def add_rule(self, rule: NoiseRule)
    def inject_noise(self, circuit: stim.Circuit, active_qubits=None) -> stim.Circuit
```

**Factory class methods** (create pre-configured injectors):

| Factory | Description |
|---------|-------------|
| `NoiseInjector.from_code_capacity(config, data_qubit_indices)` | Code capacity noise |
| `NoiseInjector.from_phenomenological(config, data_qubit_indices)` | Phenomenological noise |
| `NoiseInjector.from_circuit_level(config, data_qubit_indices)` | Circuit-level noise |
| `NoiseInjector.from_XZ_biased(config, data_qubit_indices)` | XZ-biased noise |

#### Noise rules (`src/noise/rules.py`)

| Rule class | Description |
|-----------|-------------|
| `DepolarizeAfterGate` | Add depolarizing channel after specified gates |
| `GeneralPauliAfterGate` | Add arbitrary Pauli channel after gates |
| `FlipBeforeMeasurement` | Flip measurement outcomes with probability |
| `FlipAfterReset` | Flip reset state with probability |
| `TaggedIdling` | Add idling noise on data qubits during `TICK` (uses `"SE_start"` tag) |

---

### 6.4 Simulation Module (`src/simulation/decoder_backend/`)

#### DecoderConfig (`src/simulation/decoder_backend/config.py`)

```python
@dataclass
class DecoderConfig:
    name: str                                     # "pymatching", "bposd", "mwpf"
    backend: Literal["cpu", "gpu", "fpga"] = "cpu"
    params: Dict[str, Any] = field(default_factory=dict)
```

#### PipelineConfig (`src/simulation/decoder_backend/config.py`)

```python
@dataclass
class PipelineConfig:
    max_shots: int = 1_000_000
    max_errors: int = 100
    batch_size: int = 10_000
    num_workers: int = 4
    decoder: Optional[DecoderConfig] = None       # defaults to pymatching
    post_select_detector_indices: Optional[List[int]] = None
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None
    output_format: Literal["csv", "json", "parquet"] = "csv"
    save_resume_filepath: Optional[str] = None
    print_progress: bool = True
```

#### SimulationStats (`src/simulation/decoder_backend/config.py`)

```python
@dataclass
class SimulationStats:
    shots: int
    post_selected_shots: int
    errors: int
    seconds: float
    decoder: str
    json_metadata: Dict[str, Any]

    @property
    def post_selection_rate(self) -> float    # post_selected_shots / shots
    @property
    def logical_error_rate(self) -> float     # errors / post_selected_shots
```

#### ExperimentTask (`src/simulation/decoder_backend/pipeline.py`)

```python
class ExperimentTask:
    def __init__(self, circuit: stim.Circuit, json_metadata: Optional[Dict] = None)
```

#### SimulationPipeline (`src/simulation/decoder_backend/pipeline.py`)

```python
class SimulationPipeline:
    def __init__(self,
                 decoder_config=None,        # DecoderConfig
                 max_shots=1_000_000,
                 max_errors=100,
                 batch_size=10_000,
                 num_workers=4,
                 post_select_detector_indices=None,
                 output_dir=None,
                 output_filename=None,
                 output_format="csv",
                 save_resume_filepath=None,
                 print_progress=True)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(circuit, json_metadata=None) -> SimulationStats` | Run single circuit |
| `run_batch()` | `(tasks: list) -> pd.DataFrame` | Run multiple tasks; return DataFrame |

When no post-selection is needed and `backend="cpu"`, `run()` delegates to `sinter.collect` for maximum performance. GPU backends always use the custom sampling loop (sinter's adaptive batching is too slow for GPU kernel launch overhead).

#### Registry functions (`src/simulation/decoder_backend/registry.py`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `register_decoder()` | `(name, cls, aliases=[], backend="cpu")` | Register a decoder class under a name and backend |
| `get_decoder()` | `(name, backend="cpu", **params)` | Get a decoder instance; raises `ImportError` if the requested backend has no registration |
| `list_decoders()` | `() -> list[str]` | List registered decoder names |

---

### 6.5 Plot Module (`src/plot/`)

#### PlotConfig (`src/plot/config.py`)

```python
@dataclass
class PlotConfig:
    x: str                          # (required) x-axis column
    y: str                          # (required) y-axis column
    hue: Optional[str] = None
    style: Optional[str] = None
    facet_col: Optional[str] = None
    facet_row: Optional[str] = None
    x_scale: Literal["linear", "log"] = "log"
    y_scale: Literal["linear", "log"] = "log"
    palette: Optional[Dict | str] = None    # "distance" for built-in
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    error_bars: bool = True
    figsize: tuple = (7, 5)
    marker: Optional[str] = "o"
    linewidth: float = 2.5
```

#### Plotting functions (`src/plot/plotter.py`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `plot_ler_vs_p()` | `(df, hue="d", x_col="p1", save_path=None, **kwargs)` | LER vs physical error rate |
| `plot_ler_vs_distance()` | `(df, hue="decoder", x_col="d", save_path=None, **kwargs)` | LER vs code distance |
| `plot_simulation_results()` | `(df, x="p1", y="logical_error_rate", hue="d", x_scale="log", y_scale="log", save_path=None, **kwargs)` | Generic x-y plot |
| `plot_custom()` | `(df, cfg: PlotConfig, save_path=None)` | Fully configurable plot |

All functions return a `matplotlib.figure.Figure`.

---

### 6.6 Experiments Module (`experiments/`)

#### MemoryExperiment (`experiments/memory.py`)

See [Section 5.2](#52-creating-experiments). Does **not** inherit from `QECExperiment` -- standalone class.

```python
class MemoryExperiment:
    def __init__(self, qec_system, extraction_block_class, rounds=2,
                 noise_params=None, noise_model="circuit_level",
                 basis="Z", if_detector=True)
    def build(self) -> stim.Circuit
```

#### CNOTTransExperiment (`experiments/CNOT_trans.py`)

Inherits from `QECExperiment`.

```python
class CNOTTransExperiment(QECExperiment):
    def __init__(self, code_patch_class, extraction_block_class,
                 code_params_control, code_params_target=None,
                 offset_target=(6,0),
                 initial_basis_control="Z", initial_basis_target="Z",
                 measure_basis_control="Z", measure_basis_target="Z",
                 rounds_before=2, rounds_after=2,
                 noise_params=None, noise_model="circuit_level",
                 if_detector=True)
    def build(self) -> stim.Circuit
```

#### TwoPatchLSExperiment (`experiments/two_patch_LS_unrotated.py`)

Standalone class (does not inherit `QECExperiment`).

```python
class TwoPatchLSExperiment:
    def __init__(self, patch1_config, patch2_config, offset,
                 interaction_type="XX", coupler_protocol=UnrotatedTwoPatchCoupler(),
                 initial_state_patch1="X", initial_state_patch2="X",
                 measure_state_patch1="X", measure_state_patch2="X",
                 extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                 rounds=2, noise_params=None, noise_model="circuit_level",
                 if_detector=True, rotate_patch1=True)
    def build(self) -> stim.Circuit
```

#### CNOTLSExperiment (`experiments/CNOT_LS.py`)

Standalone class. Three-patch lattice surgery CNOT.

```python
class CNOTLSExperiment:
    def __init__(self, patch_configs, offset_ta, offset_ca,
                 initial_state_dict={"a": "X", "c": "X", "t": "X"},
                 measure_state_dict={"a": "Z", "c": "X", "t": "X"},
                 extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                 rounds=2, noise_params=None, noise_model="circuit_level",
                 if_detector=True, rotate_patches=True)
    def build(self) -> stim.Circuit
```

#### GHZExperiment (`experiments/ghz.py`)

Inherits from `QECExperiment`.

```python
class GHZExperiment(QECExperiment):
    def __init__(self, distance=3,
                 offset_patch2=(6,0), offset_patch3=(12,0),
                 initial_basis_patch1="X", initial_basis_patch2="Z",
                 initial_basis_patch3="Z",
                 measure_basis_patch1="Z", measure_basis_patch2="Z",
                 measure_basis_patch3="Z",
                 rounds_before=2, rounds_after=2,
                 extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                 noise_params=None, noise_model="circuit_level",
                 if_detector=True)
    def build(self) -> stim.Circuit
```

#### StateInjectionExperiment (`experiments/state_injection.py`)

Standalone class. Rotated surface code state injection.

```python
class StateInjectionExperiment:
    def __init__(self, distance=3, rounds=2,
                 injection_protocol="corner", inject_state="Z",
                 extraction_block_class=RotatedSurfaceCodeExtractionBlock,
                 noise_params=None, noise_model="circuit_level",
                 if_detector=True)
    def build(self) -> stim.Circuit
```

---

## 7. Extending LightStim

### Adding a new QEC code

1. **Create a patch class** inheriting from `QECPatch`:

```python
# src/qec_code/my_code/code_patch.py
from src.ir.qec_patch import QECPatch

class MyCode(QECPatch):
    def _process_params(self):
        self.distance = self.params["distance"]
        # validate parameters...

    def build(self):
        # 1. Register qubit coordinates
        idx = 0
        for x in range(self.distance):
            for y in range(self.distance):
                self.qubit_coords[idx] = (x, y)
                self.data_indices.add(idx)
                idx += 1
        # ... add syndrome qubits, stabilizers, logicals
        self.create_stim_stabilizer(targets, "Z")
        self.create_stim_logical(targets, "X")
```

2. **Create an SE block**:

```python
# src/qec_code/my_code/SE_block.py
import stim

class MyCodeExtractionBlock:
    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _build_circuit(self):
        # Reset, entangle, measure syndrome qubits
        self.circuit.append("R", syndrome_indices)
        self.circuit.append("TICK", tag="SE_start")
        # ... CNOT schedule ...
        self.circuit.append("M", syndrome_indices)
```

3. **Use with existing experiments**:

```python
code = MyCode(distance=5)
experiment = MemoryExperiment(
    qec_system=code,
    extraction_block_class=MyCodeExtractionBlock,
    rounds=5, basis="Z",
)
```

### Adding a new experiment

Inherit from `QECExperiment` and implement `build()`:

```python
from src.ir.experiment import QECExperiment

class MyExperiment(QECExperiment):
    def __init__(self, my_params, **kwargs):
        super().__init__(**kwargs)
        self.my_params = my_params

    def build(self) -> stim.Circuit:
        # 1. Create patches, build QECSystem
        self.system = QECSystem()
        self.system.add_patch(...)

        # 2. Setup tracker, builder, executor
        self._setup_experiment()

        # 3. Build circuit
        self.builder.write_coordinates()
        self.builder.initialize(init_dict, n)
        se_block = self.extraction_block_class(self.system)
        self.builder.apply_syndrome_extraction(se_block.circuit, self.rounds)
        self.builder.apply_data_readout(measurements)

        # 4. Inject noise
        return self._inject_noise(self.builder.circuit)
```

### Adding a new decoder

Create a decoder module and register it:

```python
# src/simulation/decoder_backend/decoders/my_decoder.py
from ..registry import register_decoder

class MyDecoder:
    """Must implement sinter.Decoder interface:
    - compile_decoder_for_dem(dem) -> CompiledDecoder
    - decode_shots_bit_packed(bit_packed_detection_event_data) -> predictions
    """
    pass

register_decoder("my_decoder", MyDecoder, aliases=["my_alias"])
```

The decoder will then be available via `DecoderConfig("my_decoder")`.

### Adding a new coupler protocol

Subclass `LogicalCouplerProtocol`:

```python
from src.ir.coupler import LogicalCouplerProtocol

class MyCoupler(LogicalCouplerProtocol):
    EXPECTED_PATCH_COUNT = 2  # or None for variable

    def __init__(self):
        super().__init__(name_prefix="my_coupler")

    def _build_coupler_geometry(self, coupler_patch, patches, **params):
        # Analyze patch boundaries and populate coupler_patch with:
        # - qubit_coords, data_indices, syndrome_indices
        # - stabilizers, logical operators
        pass
```

Register with `QECSystem.register_coupler(MyCoupler(), patch_names=[...], name="...")`.
