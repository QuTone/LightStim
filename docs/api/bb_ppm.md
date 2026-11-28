# Gross-code Pauli product measurements

`BBPPMExperiment` builds a Stim circuit for an ordered sequence of logical
Pauli product measurements on one `[[144, 12, 12]]` gross-code module. Logical
indices run from 1 through 12 in the basis used by [Tour de gross](https://arxiv.org/abs/2506.03094).

## Build and use a circuit

```python
from lightstim.protocols.bb_ppm import BBPPMExperiment, BBPPMStep
from lightstim.qec_code.BB_code.ppm import build_tour_de_gross_code

experiment = BBPPMExperiment(
    build_tour_de_gross_code(),
    [BBPPMStep(((1, "Y"),))],
    rounds=10,
    initial_states="X",
    final_measure_states="X",
)
circuit = experiment.build()
records = circuit.compile_sampler(seed=0).sample(shots=32)
y1_outcomes = experiment.ppm_outcomes[0].evaluate(records)
```

Each `BBPPMStep` requests one joint measurement. For example,
`BBPPMStep(((2, "X"), (7, "Z"), (12, "X")))` measures the product
`X2 Z7 X12`. Each logical index may occur once per step. Steps execute in the
order supplied and may measure noncommuting products. Native measurements
can act on logical qubit 1. General measurement synthesis uses qubit 1 as a
disposable pivot and qubits 2 through 12 as data.

`initial_states` and `final_measure_states` specify uniform physical-data
preparation and readout in the X or Z basis. The initial basis defaults to Z
and the final basis defaults to the initial basis. `rounds` sets the number
of extraction rounds for each native measurement. The initial and final QEC
boundaries are noiseless by default.

After `build()`, `ppm_outcomes[i].evaluate(records)` returns the measured
parity for step `i`, including its classical Pauli frame. A zero bit means
eigenvalue +1 and a one bit means −1. Use `experiment.build_noisy_circuit(noise_params)`
with a LightStim `NoiseConfig` to apply noise to the built schedule, including
idle locations.

## Logical error rate

The [worked example](../../notebooks/LogicalOps/logical_PPM_BB_code.ipynb)
displays the full `Y1` circuit, checks the noiseless detectors and observables,
and plots the three recorded noise points. Running the diagram cell with the
notebook dependencies (`pip install -e ".[dev]"`) writes the full Stim circuit,
its SVG diagram, and a PNG preview under `benchmarks/logical_ops/results/bb_ppm_code_demo/`.
With `RUN_SAMPLING = False` it
reads and validates the saved counts. To sample a different measurement,
edit `ppm_sequence` in section 1 and enable `RUN_SAMPLING` in section 4.

The benchmark sets the one-qubit, two-qubit, measurement, reset, and idle
error probabilities to the same value `p` in LightStim's
`circuit_level_with_idling` model. A shot fails if any logical observable
predicted by Relay-BP differs from the sampled observable. The reported rate
is the failure count divided by all shots. A decoder correction that fails
the syndrome check is also counted under nonconvergence. The graph shows
two-sided 95% Clopper–Pearson intervals.

The [recorded configuration](../../benchmarks/logical_ops/reference/bb_ppm/y1_large_20261124/results.manifest.json)
includes the frozen Relay-BP parameters and fixed shot budgets. For a short
command-line run, see the [benchmark instructions](../../benchmarks/logical_ops/README.md#gross-code-pauli-product-measurements).
