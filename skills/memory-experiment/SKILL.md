---
name: memory-experiment
description: >
  Build a QEC memory experiment in LightStim — initialize a logical state,
  run syndrome extraction rounds, and measure a logical observable. Use this
  skill whenever the user asks to run a memory experiment, test a QEC code,
  build a circuit for X or Z memory, simulate error correction, or check that
  a code patch builds correctly. Also use it when the user just says "I want
  to try [code name]" without specifying an experiment type — memory is the
  natural starting point.
user-invocable: true
---

# Memory Experiment

A memory experiment is the simplest end-to-end LightStim workflow and the
template for almost everything else. It:
1. Creates a code patch and wraps it in a `QECSystem`
2. Initializes logical data in a basis (Z or X)
3. Runs `rounds` of syndrome extraction
4. Measures data qubits and checks the logical observable

## How to help the user

1. Ask (or infer from context): which QEC code, which distance/parameters,
   which basis (Z or X), how many rounds, and whether they want noise.
2. Read `scripts/template.py` — it shows the full pattern for rotated surface
   code. Adapt it by swapping the patch class and extraction block class.
3. Show the adapted code. Offer to run a noiseless check (0 detection events
   expected) to confirm correctness.
4. If they want simulation results, hand off to the `simulate-decode` skill.

## Code family reference

| Code | Patch class | Extraction block | Notes |
|---|---|---|---|
| Rotated surface | `RotatedSurfaceCode(distance=d)` | `RotatedSurfaceCodeExtractionBlock` | Default |
| Unrotated surface | `UnrotatedSurfaceCode(distance=d)` | `UnrotatedSurfaceCodeExtractionBlock` | |
| Toric | `ToricCode(distance=d)` | `ToricCodeExtractionBlock` | |
| Color | `ColorCode(distance=d)` | `ColorCodeExtractionBlock` | |
| BB code | `BBCode(l=6, m=6, A=..., B=...)` | `BBCodeExtractionBlock` | CSS LDPC |
| Repetition | `RepetitionCode(distance=d)` | `RepetitionCodeExtractionBlock` | Z-only |

All imports: `from lightstim.qec_code.<family> import <PatchClass>, <BlockClass>`

## Noiseless sanity check (always do this first)

```python
noiseless = exp.builder.circuit
sampler = noiseless.compile_detector_sampler()
det, _ = sampler.sample(shots=20, separate_observables=True)
assert det.sum() == 0  # must be zero — any failures indicate a build bug
```

## Reference script

Read `scripts/template.py` for a complete working example.
