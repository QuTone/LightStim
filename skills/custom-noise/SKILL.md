---
name: custom-noise
description: >
  Configure noise models and error rates for LightStim experiments. Use this
  skill whenever the user asks about noise models, setting physical error rates,
  choosing between circuit_level and phenomenological and code_capacity noise,
  biased noise, custom error rates, or wants to understand how noise is applied
  to a circuit. Also trigger when the user asks "what p should I use?" or
  "how do I add noise to my circuit?"
user-invocable: true
---

# Custom Noise Model

LightStim separates circuit construction from noise injection. Build the
noiseless circuit first, then wrap it with a noise model via `NoiseConfig`.

## NoiseConfig fields

```python
from lightstim.noise.config import NoiseConfig

noise = NoiseConfig(
    p_1q=1e-3,    # depolarizing after single-qubit gates (H, S, ...)
    p_2q=1e-3,    # depolarizing after two-qubit gates (CX, CZ, ...)
    p_meas=1e-3,  # measurement flip probability
    p_reset=1e-3, # state-prep flip probability
    p_idle=1e-3,  # depolarizing on idle qubits between SE ticks
    custom_params={'p_z': 0.01, 'p_x': 0.001},  # arbitrary extras
)
```

Set unused fields to 0 (default). Use `custom_params` for biased or
non-standard rates — access them in a custom `NoiseInjector` rule via
`noise.get('p_z')`.

## Noise model strategies

| Strategy | What it injects | Use for |
|---|---|---|
| `circuit_level` | Errors after every gate + meas/reset flip | Realistic hardware simulation |
| `phenomenological` | Meas errors + data errors between rounds | Simplified threshold analysis |
| `code_capacity` | Data errors only (idle errors on data qubits) | Code distance/threshold studies |

Pass as `noise_model=` to any experiment constructor or `MemoryExperiment`.

## How to help the user

1. Ask what hardware model they're targeting — this determines which strategy
   and which parameters matter.
2. For superconducting: use `circuit_level` with `p_2q ≈ 10× p_1q`.
3. For threshold searches: start with `phenomenological` (faster) then
   re-confirm with `circuit_level`.
4. Read `scripts/template.py` for a side-by-side comparison of all three
   strategies at the same physical error rate.

## Reference script

Read `scripts/template.py` for a complete comparison across noise models.
