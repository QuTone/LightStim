---
name: state-injection
description: >
  Inject a logical Z, X, or Y state into a surface code patch using the state
  injection protocol in LightStim. Use this skill whenever the user asks about
  state injection, preparing a specific logical state fault-tolerantly, magic
  state preparation, T-gate resource states, Y-state injection, post-selection
  on injection events, or comparing full_qec vs hybrid vs full_postselection
  decoding modes.
user-invocable: true
---

# State Injection

Encodes a physical qubit state directly into a logical code block, then grows
it into a full distance-d patch through SE rounds. Avoids the overhead of a
full encoding circuit.

## How to help the user

1. Ask: which state to inject (Z / X / Y), which code (rotated or unrotated
   surface code), distance, rounds, and post-selection mode.
2. Read `scripts/template.py` — it shows all three states on rotated SC.
3. Y injection is the most complex: it uses a noiseless S_DAG transversal
   step during readout (unencode protocol).

## Post-selection modes

| Mode | Description | When to use |
|---|---|---|
| `full_postselection` | Discard any shot with a detection event | Low noise, high purity needed |
| `full_qec` | Decode and correct all errors | Standard QEC simulation |
| `hybrid` | Post-select injection detectors, decode the rest | Best LER/overhead tradeoff |

## StateInjectionExperiment constructor

```python
from lightstim.protocols.state_injection import StateInjectionExperiment

exp = StateInjectionExperiment(
    code_patch_class=RotatedSurfaceCode,
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    op_set_class=RotatedSurfaceCodeLogicalOpSet,
    distance=3,
    rounds=2,
    inject_state='Z',         # 'Z', 'X', or 'Y'
    protocol='corner',         # 'corner' or 'middle'
    post_select_mode='full_postselection',
    noise_params=NoiseConfig(...),
)
circuit = exp.build()
```

For unrotated surface code, swap in `UnrotatedSurfaceCode`,
`UnrotatedSurfaceCodeExtractionBlock`, `UnrotatedSurfaceCodeLogicalOpSet`.

## Reference script

Read `scripts/template.py` for Z, X, and Y injection with noiseless verification.
