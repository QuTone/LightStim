---
name: simulate-decode
description: >
  Run a LightStim circuit through the simulation pipeline and extract logical
  error rates. Use this skill whenever the user asks to simulate a circuit, get
  LER vs physical error rate, run PyMatching or BPOSD or MWPF decoding, sweep
  noise parameters, benchmark a code's threshold, or plot logical error rates.
  Also trigger when the user has a built circuit and asks "how do I know if it's
  working?" or "what's the error rate?"
user-invocable: true
---

# Simulate and Decode

Takes a built `stim.Circuit`, runs it through `SimulationPipeline`, and
returns `SimulationStats` with `logical_error_rate`, `shots`, and `errors`.

## How to help the user

1. Confirm they have (or help them build) a noisy circuit — see
   `memory-experiment` skill if needed.
2. Ask which decoder they want. Default is `pymatching` (always available).
   `bposd` and `mwpf` require optional dependencies.
3. Ask how many errors to collect (`max_errors`). 100–200 is fast; 1000+ gives
   tight error bars.
4. Read `scripts/template.py` and adapt for their circuit and parameters.
5. If they want a threshold plot (LER vs p for multiple distances), show the
   sweep loop pattern from the template.

## SimulationPipeline API

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(name='pymatching'),  # or 'bposd', 'mwpf'
    max_errors=200,      # stop after this many logical errors
    max_shots=1_000_000, # hard cap on total shots
    print_progress=False,
)
stats = pipeline.run(circuit)
print(stats.logical_error_rate)  # errors / post_selected_shots
print(stats.shots)
print(stats.post_selected_shots) # shots that survived post-selection (if any)
```

## Decoder options

| Name | Backend | Notes |
|---|---|---|
| `pymatching` | cpu | Always available, fast, approximate for hyperedges |
| `bposd` | cpu / gpu | Requires `stimbposd`; better for LDPC codes |
| `mwpf` | cpu | Requires `mwpf`; exact, handles hyperedges |
| `nv-qldpc-decoder` | gpu | Requires `cudaq-qec`; GPU BP+OSD |

## Reference script

Read `scripts/template.py` for a complete sweep over (distance, p) pairs.
