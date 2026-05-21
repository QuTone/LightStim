---
name: lattice-surgery-cnot
description: >
  Build a logical CNOT via lattice surgery using 3 surface code patches in
  LightStim. Use this skill whenever the user asks about lattice surgery,
  logical CNOT via joint measurements, XX/ZZ coupler activations, multi-patch
  logical operations, or merge-split protocols between surface code patches.
user-invocable: true
---

# Lattice Surgery CNOT

Implements logical CNOT by temporarily activating joint stabilizer measurements
(XX and ZZ couplers) between an ancilla and the control/target patches.

**Layout:**
```
  Ancilla (A)
     |  ZZ coupler
  Control (C)  ── XX coupler ──  Target (T)
```

**Protocol (two surgery rounds):**
1. Prepare A in |+⟩; measure ZZ(C,A) and XX(T,A); measure A in Z
2. Prepare A in |0⟩; measure XX(T,A) and ZZ(C,A); measure A in X

## How to help the user

1. Ask: distance for each patch (can all be the same), offset layout, rounds,
   initial and final state for each patch.
2. `offset_ta` = position of target relative to ancilla (e.g. `(8, 0)`)
3. `offset_ca` = position of control relative to ancilla (e.g. `(0, 8)`)
4. `initial_state_dict` and `measure_state_dict` control logical init/readout
   for each patch. Default: init all in X (`{'a':'X','c':'X','t':'X'}`).
5. Read `scripts/template.py` for the full pattern.

## CNOTLSExperiment constructor

```python
from lightstim.protocols.cnot_ls import CNOTLSExperiment

exp = CNOTLSExperiment(
    patch_configs={'c': {'distance': 3}, 't': {'distance': 3}, 'a': {'distance': 3}},
    offset_ta=(8, 0),
    offset_ca=(0, 8),
    initial_state_dict={'a': 'X', 'c': 'X', 't': 'X'},
    measure_state_dict={'a': 'Z', 'c': 'X', 't': 'X'},
    rounds=2,
    noise_params=NoiseConfig(...),
)
circuit = exp.build()
```

## Reference script

Read `scripts/template.py` for a complete working example.
