# Quantinuum color-code memory

A d=3 circuit reconstruction of [Lavasani et al., arXiv:2610.24077v1](https://arxiv.org/abs/2610.24077v1):
**Bell-flag preparation -> 20 SWAP-superdense rounds -> readout**, averaged over
six signed logical Pauli inputs. Flags are decoded, not postselected; software
SWAPs are noiseless and take no extra moment. No ideal terminal SE round is added.

[Open the executed notebook](memory.ipynb) for the circuit, detector slices and
simulation. `circuit.py` builds the protocol; `run.py` samples it through
LightStim's `SimulationPipeline` with Tesseract and plots using `lightstim.plot.styles`.

## Noise and Scope

Both models use `p_1q = p_2q = p_meas = p_reset = 0.001`:

| Model | Idle noise |
| --- | --- |
| `circuit_level` | None (`p_idle=0`) |
| `circuit_level_with_idling` | `p_idle=0.001` per inactive qubit per TICK-delimited moment |

Per-moment idling replaces the legacy `SE_start` data-idle channel; it does not
add to it. These are uncalibrated Pauli models, without leakage, repump, transport
or crosstalk. Neither rates nor decoder settings are fitted to hardware.

**Known limitation:** X/Z circuits have fault distance three; the reconstructed
Y-readout boundary has a weight-two witness. Y remains in the average. This is
not a calibrated Helios emulator, a full fault-tolerance claim, or evidence of
an error in the hardware paper. d=5/7 and logical 2QRB are outside this example.

![20-round comparison](comparison.png)

## Run

From a LightStim checkout ([installation notes](../../README.md#quick-start)):

```sh
python -m pip install -e '.[dev]' tesseract-decoder
python -m examples.quantinuum_color_code_memory.run --shots 1000
```

This writes counts and a figure to `build/examples/quantinuum_color_code_memory/`.
The saved reference uses `--shots 50000 --workers 8`: 600,000 shots total.
`results.json` retains counts, shared configuration, versions and source/circuit
hashes. Its original source snapshot is `fb3dfd7`, based on main `96535e8`, with
reference Pauli/tracker backends. Multi-worker seeds are process-dependent, so
reruns need not give identical counts. The notebook's small run is not used in
the reference figure.

The estimator is `failures / shots / 20.5`, averaged equally over six inputs.
Hardware uses Table 1's 178/15,840. Intervals are 95% (the paper reports 68%):
exact binomial bounds for hardware, and averaged simultaneous Bonferroni bounds
for simulation, without assuming independent random streams across inputs.
Tests check all 127 deterministic relations, fault distances, noise placement
and a small optional decoding run; CI does not run the production sweep.
