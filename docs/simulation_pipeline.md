# SimulationPipeline Usage Guide

## Overview

`SimulationPipeline` is LightStim's unified simulation backend. It handles the full loop:

```
Stim Circuit → DEM → Sampling → Post-Selection → Decoding → LER Stats
```

**Key design**: one custom loop for all paths (CPU/GPU, with/without post-selection, single/multi-process). No `sinter.collect` dependency.

## Quick Start

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# 1. Configure decoder
decoder_config = DecoderConfig(
    name="pymatching",   # decoder algorithm
    backend="cpu",       # "cpu" or "gpu"
)

# 2. Create pipeline
pipeline = SimulationPipeline(
    decoder_config=decoder_config,
    max_shots=1_000_000,
    max_errors=100,
    batch_size=10_000,
    num_workers=4,       # parallel workers (CPU cores or GPU count)
)

# 3. Run
stats = pipeline.run(circuit)  # circuit: stim.Circuit (with noise)
print(f"LER: {stats.logical_error_rate:.2e} ± {stats.ler_error_bar():.2e}")  # 95% CI
```

## Environment

**GPU decoding requires the `venv` environment:**
```bash
# System python does NOT have cudaq_qec
python -c "import cudaq_qec"  # ImportError

# Use the project venv
venv/bin/python -c "import cudaq_qec"  # OK
```

Always use `venv/bin/python` when running GPU decoder experiments.

## Available Decoders

| Name | Backend | Package | Best For |
|------|---------|---------|----------|
| `pymatching` | cpu | pymatching (via sinter) | Surface codes, topological codes |
| `bposd` | cpu | stimbposd | qLDPC codes (small-medium) |
| `bposd` or `nv-qldpc-decoder` | gpu | cudaq_qec | qLDPC codes (large, high throughput) |
| `mwpf` | cpu | mwpf | Color codes, codes with hyperedges |

### PyMatching (MWPM)
```python
DecoderConfig(name="pymatching", backend="cpu")
```
- No extra params needed. Default decoder.
- Only handles degree-2 edges (graph-like errors). For codes with hyperedges (color code, qLDPC), use BP+OSD or MWPF.

### BP+OSD (CPU)
```python
DecoderConfig(
    name="bposd",
    backend="cpu",
    params={
        "max_iterations": 1000,
        "osd_order": 10,
        "bp_method": "min_sum",      # or "product_sum"
        "osd_method": "osd_cs",      # or "osd_0", "osd_e"
    },
)
```

### BP+OSD (GPU)
```python
DecoderConfig(
    name="nv-qldpc-decoder",   # or name="bposd", backend="gpu"
    backend="gpu",
    params={
        "max_iterations": 100,
        "osd_order": 10,
        "bp_method": "min_sum",           # recommended default
        "ms_scaling_factor": 0,           # recommended default (disables min-sum scaling)
        "osd_method": "osd_cs",
        "use_osd": True,
    },
)
```

**GPU-specific notes:**
- `num_workers` = number of GPUs. Worker `i` is assigned to GPU `i`.
- `bp_method` is translated: `"min_sum"` → int 1, `"product_sum"` → int 0 (cudaq_qec convention).
- Requires `cudaq_qec` package (only in venv).

### MWPF
```python
# Surface code (default c=50)
DecoderConfig(name="mwpf", backend="cpu")

# Color code / qLDPC (use c=200 for complex hypergraph structures)
DecoderConfig(name="mwpf", backend="cpu", params={"cluster_node_limit": 200})
```
- Natively handles hyperedges — **do NOT use `decompose_errors=True`** (pipeline defaults correctly to False).
- `cluster_node_limit` (`c`) is the key tuning knob: `c=50` for surface codes, `c=200` for color/qLDPC codes.
- See `docs/mwpf_user_guide.md` for full parameter reference.

## bp_method: min_sum vs product_sum

**This is code-dependent. There is no universal winner.**

| Code Family | Better bp_method | Why |
|-------------|-----------------|-----|
| 4D Toric / Hadamard [[96,6,8]] | `min_sum` | Sparse Tanner graph, high girth; min_sum avoids numerical underflow |
| PQRM-Surface CrossLS (MagicCross) | `product_sum` | Dense, low-girth Tanner graph; exact BP handles short-loop correlations better |
| Surface code (as qLDPC) | `min_sum` | Sparse, regular structure |

**Recommended default**: `bp_method="min_sum"` + `ms_scaling_factor=0`

This combination is the default in both CPU and GPU backends. `ms_scaling_factor=0` disables the min-sum scaling correction, which generally gives better OSD convergence. This works well for most code families.

Try `product_sum` (without scaling factor) only if LER seems too high — especially for codes with weight-4 stabilizers or short cycles in the Tanner graph (e.g., PQRM-Surface CrossLS).

## Post-Selection

Post-selection filters out shots where specified detectors fired (indicating a detected error that should be discarded, e.g., in distillation or state injection).

```python
pipeline = SimulationPipeline(
    decoder_config=decoder_config,
    max_shots=5_000_000,
    max_errors=100,
    # Option 1: Explicit detector indices
    post_select_detector_indices=[100, 101, 102],
    # Option 2: Auto-detect from circuit tags (if circuit uses tagged detectors)
    # Leave post_select_detector_indices=None to auto-detect
)
```

**Post-select on observables** (discard shots where specific observables flip):
```python
pipeline = SimulationPipeline(
    ...,
    post_select_observable_indices=[0, 1],  # discard if L0 or L1 flips
)
```

**Target specific observables** (only count errors on a subset):
```python
pipeline = SimulationPipeline(
    ...,
    target_observable_indices=[3],  # only measure LER on L3
)
```

## Batch Mode

Run multiple circuits (e.g., distance sweep) in sequence:

```python
from lightstim.simulation.decoder_backend import ExperimentTask

tasks = []
for d in [3, 5, 7]:
    circuit = build_memory_circuit(distance=d, p=1e-3)
    tasks.append(ExperimentTask(circuit, {"distance": d, "p": 1e-3}))

df = pipeline.run_batch(tasks)  # returns pandas DataFrame
print(df[["distance", "p", "shots", "errors", "logical_error_rate"]])
```

## Architecture

```
SimulationPipeline.run(circuit)
  │
  ├─ _resolve_post_select_indices(circuit)
  │     └─ from config or auto-detect from circuit tags
  │
  └─ _run_custom(circuit, meta, post_indices)
       │
       ├─ num_workers <= 1: _run_custom_single()
       │     └─ Single-thread loop: sample → post-select → decode → count errors
       │
       └─ num_workers > 1: spawn multiprocessing.Process workers
             └─ Each worker: _decode_worker_cpu(worker_id, gpu_id=worker_id if GPU)
                   └─ Same loop: sample → post-select → decode → count errors
                   └─ Shared counters via mp.Value (lock-protected)
```

**sinter dependency**: Only the `sinter.Decoder` / `sinter.CompiledDecoder` abstract interface is used. The sampling loop, post-selection, parallelism, and progress reporting are all custom — no `sinter.collect()`.

## Output & Progress

**Progress reporting** (prints every ~10s):
```
shots=100,000 kept=95,000 errors=12 LER=1.26e-04 elapsed=45.2s ETA=3m12s
```

**Save results to file:**
```python
pipeline = SimulationPipeline(
    ...,
    output_dir="results/",
    output_filename="memory_d5.csv",   # or .json, .parquet
    output_format="csv",
)
```

## Complete Example: 4D Hadamard GPU

```python
from lightstim.qec_code.four_d_geo_code import FourDGeoCode, FourDGeoCodeExtractionBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# Build circuit
code = FourDGeoCode(
    L=[[1, 1, 1, 1], [0, 2, 0, 2], [0, 0, 2, 2], [0, 0, 0, 4]],  # [[96,6,8]]
    d=8,
)
system = QECSystem()
system.add_patch(code, name="4d_hadamard")

noise = NoiseConfig(p_idle=1e-3, p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3)
exp = MemoryExperiment(
    qec_system=system,
    extraction_block_class=FourDGeoCodeExtractionBlock,
    rounds=8,
    noise_params=noise,
    noise_model="circuit_level",
    basis="Z",
)
circuit = exp.build()

# Decode with GPU BP+OSD
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(
        name="nv-qldpc-decoder",
        backend="gpu",
        params={
            "max_iterations": 100,
            "osd_order": 10,
            "bp_method": "min_sum",   # use min_sum for this code
            "osd_method": "osd_cs",
            "use_osd": True,
        },
    ),
    max_shots=2_000_000,
    max_errors=200,
    batch_size=10_000,
    num_workers=2,         # 2 GPUs
    print_progress=True,
)

stats = pipeline.run(circuit)
print(f"LER: {stats.logical_error_rate:.2e}")  # expect ~6e-05
```

**Run with venv:**
```bash
venv/bin/python my_script.py
```
