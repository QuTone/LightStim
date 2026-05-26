---
name: simulate-decode
description: >
  Run a LightStim circuit through the simulation pipeline and extract logical
  error rates. Use this skill whenever the user asks to simulate a circuit, get
  LER vs physical error rate, run PyMatching or BPOSD or MWPF decoding, sweep
  noise parameters, benchmark a code's threshold, plot logical error rates,
  configure post-selection (for state injection or distillation), or target
  specific logical observables in a multi-qubit circuit.
user-invocable: true
---

# Simulate and Decode

Takes a noisy `stim.Circuit` and runs it through `SimulationPipeline`.

## Step 0: Get a noisy circuit

```python
# From CircuitBuilder (custom protocol):
noisy = builder.build_noisy_circuit(NoiseConfig(p_2q=1e-3, p_meas=1e-3), "circuit_level")

# From any experiment class:
noisy = experiment.build()  # already returns noisy circuit
```

## `SimulationPipeline` API

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),  # see decoder table below
    max_errors=200,      # primary stopping criterion: stop after N logical errors
    max_shots=1_000_000, # hard cap on total shots
    batch_size=10_000,   # shots per batch
    num_workers=4,       # parallel CPU processes
    print_progress=False,
)
stats = pipeline.run(noisy_circuit)

print(stats.logical_error_rate)    # errors / post_selected_shots
print(stats.ler_error_bar())       # 95% Wilson CI half-width (z=1.96)
print(stats.ler_error_bar(z=1.0))  # 1-sigma half-width
print(stats.shots)                 # total shots attempted
print(stats.post_selected_shots)   # shots surviving post-selection (= shots if no PS)
```

## Decoder options

| Name | Backend | When to use |
|---|---|---|
| `"pymatching"` | cpu | Default. MWPM. Always available. Fast for surface codes. |
| `"bposd"` | cpu | BP+OSD via `stimbposd`. Better for LDPC (BB, PQRM). |
| `"mwpf"` | cpu | Minimum-weight parity factor. Required for PQRM/CrossLS (hyperedges). |
| `"nv-qldpc-decoder"` | gpu | GPU BP+OSD via `cudaq_qec`. Use `batch_size ≥ 50_000`, `num_workers=1`. |

**Decoder selection rule:**
- Surface codes (rotated, unrotated, toric): `pymatching`
- LDPC codes (BB, PQRM) with no hyperedges: `bposd`
- CrossLS / PQRM (X-stabs weight > 2 → hyperedges): `mwpf`

## Post-selection

Use post-selection for state injection (discard shots with injection errors) or
distillation (post-select on magic state purity).

```python
# Option A: Auto-detect from circuit tags (detectors tagged 'post-select')
# The SyndromeTracker inserts this tag when post_select_detector_coords is set.
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_errors=200,
    # post_select_detector_indices auto-discovered from circuit tags
)

# Option B: Post-select on decoder-corrected observable (distillation output)
# First, identify which observable index corresponds to the output qubit:
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix, identify_distillation_observables
)
matrix, patch_names = build_obs_patch_matrix(circuit, system)
_, target_obs, ps_obs = identify_distillation_observables(
    matrix, patch_names, output_patches=["W4"]
)

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("bposd"),
    post_select_corrected_observable_indices=ps_obs,  # discard failed distillations
    target_observable_indices=target_obs,             # measure LER on output qubit only
    max_errors=50,
)
stats = pipeline.run(circuit)
print(f"Post-selection rate: {stats.post_selection_rate:.3f}")
print(f"LER on output qubit: {stats.logical_error_rate:.4e}")
```

## Multi-observable circuits

For circuits with k > 1 logical observables (transversal CNOT, Bell pairs, distillation),
specify which ones count as errors:

```python
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    target_observable_indices=[0],   # only check logical 0 (ZZ observable of CNOT)
    max_errors=200,
)
# Without target_observable_indices, any observable flip counts as an error.
```

## Threshold sweep pattern

```python
import numpy as np
from lightstim.noise.config import NoiseConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_errors=200, max_shots=500_000, print_progress=False,
)

results = []
for d in [3, 5, 7]:
    for p in np.logspace(-3, -1, 8):
        circuit = build_circuit(d=d, p=p)   # your build function
        stats = pipeline.run(circuit)
        results.append({
            "d": d, "p": p,
            "ler": stats.logical_error_rate,
            "eb":  stats.ler_error_bar(),
        })
```

## Working examples

- `benchmarks/memory/run_memory.py` — full threshold sweep with checkpointing, argparse, CSV output
- `benchmarks/logical_circuits/run_logical_circuits.py` — multi-experiment runner pattern
