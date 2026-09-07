# Bacon-Shor code

`BaconShorCode(distance=d)` implements the square `[[d², 1, (d−1)², d]]`
subsystem code for integer `d >= 2`. The implementation follows the subsystem
construction introduced by Dave Bacon, [*Operator quantum error-correcting
subsystems for self-correcting quantum memories*, Physical Review A **73**,
012340 (2006)](https://doi.org/10.1103/PhysRevA.73.012340)
([arXiv:quant-ph/0506023](https://arxiv.org/abs/quant-ph/0506023)).

```python
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode

experiment = MemoryExperiment(
    qec_patch=BaconShorCode(distance=3),
    basis="Z",  # X memory is also supported
    rounds=5,
)
circuit = experiment.build()
detections, observables = circuit.compile_detector_sampler().sample(
    256, separate_observables=True,
)
assert not detections.any()
assert not observables.any()
```

For data qubits `q[r,c]`, with `0 <= r,c < d`, this implementation uses:

- **Gauge group G:** horizontal `X[r,c] X[r,c+1]` and vertical
  `Z[r,c] Z[r+1,c]` generators, stored in `patch.gauges`.
- **Stabilizer center S:** products of each horizontal XX generator over every
  row, and products of each vertical ZZ generator over every column. These
  `2(d−1)` checks are stored in `patch.stabilizers`, with `syn_idx=None`.
- **Protected logicals:** X along the first full column and Z along the first
  full row. There is one protected logical qubit and `(d−1)²` gauge qubits.

The layout places data at `(2c, 2r)`, X-gauge ancillas at `(2c+1, 2r)`, and
Z-gauge ancillas at `(2c, 2r+1)`. It uses `2d(d−1)` dedicated measurement
ancillas, giving `d² + 2d(d−1)` physical qubits. Gauge qubits are subsystem
degrees of freedom, distinct from these physical ancillas.

The default `GenericCSSGaugeExtractionBlock` measures all X gauges and then
all Z gauges, with a separate physical measurement block for each basis.
`se_block_kwargs={"basis_order": ("Z", "X")}` reverses this order; single-basis
and repeated-basis sequences are also accepted. The tracker infers the current
gauge constraints and their record parities from the measurements while S and
G remain fixed. A final memory readout requires adequate preparation of the
code and protected logical state.

Within each basis, bipartite edge coloring avoids simultaneous CNOT collisions.
This generic schedule has tests for gauge measurement flows, protected logical
flows, detector relations, and noisy DEM extraction. These checks do not establish
its circuit-level distance, decoding threshold, or an optimized fault-tolerant
schedule.
