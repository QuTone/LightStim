---
name: transversal-cnot
description: >
  Build a transversal CNOT gate between two CSS code patches in LightStim.
  Use this skill whenever the user asks about transversal gates, transversal
  CNOT, fault-tolerant two-qubit gates between patches, physical CX between
  matching qubits, or wants to test a CSS code's transversal gate set.
user-invocable: true
---

# Transversal CNOT

Applies physical CX gates between matching qubits on control and target patches.
Fault-tolerant when both patches use the same code and layout.

**Protocol:** `rounds_before` SE rounds → transversal CX → `rounds_after` SE rounds → readout

## How to help the user

1. Ask: which code (unrotated surface code is the default for transversal CNOT),
   distance, initial bases for control and target, how many rounds before/after.
2. Read `scripts/template.py` — it shows the full `CNOTTransExperiment` pattern.
3. Key parameter: `offset_target` places the target patch relative to control.
   For unrotated surface code d=3: use `(8, 0)` as a safe separation.
4. Note: `num_observables` depends on the basis combination. ZZ observable
   requires both patches initialized and measured in Z.

## CNOTTransExperiment constructor

```python
from lightstim.protocols.cnot_trans import CNOTTransExperiment

exp = CNOTTransExperiment(
    code_patch_class=UnrotatedSurfaceCode,
    extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
    code_params_control={'distance': 3},
    code_params_target={'distance': 3},  # defaults to control params if omitted
    offset_target=(8, 0),
    initial_basis_control='Z',  # 'Z' or 'X'
    initial_basis_target='Z',
    measure_basis_control='Z',
    measure_basis_target='Z',
    rounds_before=2,
    rounds_after=2,
    noise_params=NoiseConfig(...),
    noise_model='circuit_level',
)
circuit = exp.build()
```

## Reference script

Read `scripts/template.py` for a complete working example.
