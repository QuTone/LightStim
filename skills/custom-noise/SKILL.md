---
name: custom-noise
description: >
  Configure noise models and error rates for LightStim experiments. Use this
  skill whenever the user asks about noise models, setting physical error rates,
  choosing between circuit_level / phenomenological / code_capacity / XZ_biased
  noise, biased noise for neutral atom or trapped ion hardware, or wants to
  understand how noise is applied to a circuit. Also trigger when the user asks
  "what p should I use?" or "how do I add noise to my circuit?"
user-invocable: true
---

# Custom Noise Model

LightStim separates circuit construction from noise injection. Build the
noiseless circuit first (via `builder.build_noisy_circuit`), then inject noise
as a post-processing step. The clean circuit remains available in `builder.circuit`.

## The injection API

```python
noisy_circuit = builder.build_noisy_circuit(
    noise_params,         # NoiseConfig instance
    noise_model,          # str — strategy name, see table below
)
# Returns a new stim.Circuit. Does NOT modify builder.circuit in place.
```

## `NoiseConfig` fields

```python
from lightstim.noise.config import NoiseConfig

noise = NoiseConfig(
    p_1q=1e-3,    # Depolarizing after single-qubit gates (H, S, ...)
    p_2q=1e-3,    # Depolarizing after two-qubit gates (CX, CZ, ...)
    p_meas=1e-3,  # Measurement bit-flip probability
    p_reset=1e-3, # State-preparation bit-flip probability
    p_idle=0,     # Depolarizing on data qubits at SE_start tick (between rounds)
    custom_params={},  # Arbitrary extra rates for XZ_biased or custom rules
)
```

Unused fields default to 0. Set only the fields the target noise model uses.

## Noise model strategies

| Strategy | Injects | Relevant params | Use for |
|---|---|---|---|
| `circuit_level` | Depolarizing after every gate, flip before meas/after reset | `p_1q`, `p_2q`, `p_meas`, `p_reset` | Superconducting qubits, realistic simulation |
| `phenomenological` | Data errors at SE_start tick + measurement flips | `p_idle`, `p_meas` | Fast threshold analysis, ignores gate structure |
| `code_capacity` | Data errors only (no gate/meas noise) | `p_idle` | Ideal measurements, pure code distance study |
| `XZ_biased` | Independent X and Z channels after each gate | See below | Biased noise (neutral atoms, trapped ions) |

## Decision guide

- **Superconducting hardware**: `circuit_level` with `p_2q ≈ 10× p_1q`, `p_meas ≈ p_2q`.
- **Threshold plots (fast)**: `phenomenological` first, then cross-check key points with `circuit_level`.
- **Code distance study only**: `code_capacity` — fastest, no gate noise.
- **Biased hardware** (neutral atoms η ≫ 1, trapped ions η ≲ 1): `XZ_biased`.

## XZ-biased noise model

For hardware with asymmetric X/Z error rates, use `XZ_biased` with the
`compute_XZ_biased_params` helper to convert physical rates + bias ratio η:

```python
from lightstim.noise.injector import NoiseInjector

# eta = p_X / p_Z: >1 means X-biased, <1 means Z-biased
biased_noise = NoiseInjector.compute_XZ_biased_params(
    p_1q=1e-3, p_2q=2e-3, p_meas=1e-3, p_reset=1e-3,
    eta=0.01,   # strongly Z-biased (neutral atom regime)
)
noisy = builder.build_noisy_circuit(biased_noise, noise_model='XZ_biased')
```

`compute_XZ_biased_params` fills `custom_params` with:
`p_1q_x`, `p_1q_z`, `p_2q_x`, `p_2q_z` — the per-axis rates
derived from the total rates and the bias η.

## Noiseless gate phases

Some protocol phases (e.g. state injection initialization, encoding circuits)
should not receive noise. Pass `noiseless=True` to suppress noise on specific calls:

```python
builder.initialize(init_dict, n, noiseless=True)            # RX/R tags 'noiseless'
builder.apply_syndrome_extraction(se.circuit, noiseless=True) # all gates tagged
builder.apply_unitary_block(gate, noiseless=True)           # same
```

The noise injector skips all instructions tagged `'noiseless'`.
This is how state-injection protocols avoid noising the unencoded region.

## Reference script

Read `scripts/template.py` for a side-by-side comparison of all four strategies
at the same physical error rate, plus the XZ-biased example.
