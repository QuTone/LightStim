# SHYPS memory integration

`SHYPSCode(r=3)` implements the subsystem hypergraph-product simplex code
with 49 data qubits, 9 protected logical qubits and code distance 4.
`r=4` implements the 225-data-qubit, 16-logical-qubit, distance-8 instance.
Other values of `r` are currently rejected.

The construction follows Malcolm et al.,
[Computing Efficiently in QLDPC Codes, §§VIII.4–VIII.5](https://arxiv.org/html/2502.07150v2).
The classical circulant checks use `1 + x² + x³` for `r=3` and the primitive
polynomial `1 + x + x⁴` for `r=4`. The canonical generator `C` spans the
kernel of this circulant `H`. The quantum gauges are `H ⊗ I` and `I ⊗ H`;
the centre generators are `H ⊗ C` and `C ⊗ H`. Redundant gauge checks are
retained. Logical representatives are paired bare operators constructed
using the canonical simplex pivots.

```python
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.shyps import SHYPSCode

experiment = MemoryExperiment(
    qec_patch=SHYPSCode(r=3),
    rounds=3,
    basis="Z",  # "X" is also supported.
    noise_params=NoiseConfig(p_2q=0.001, p_meas=0.001),
)
circuit = experiment.build()
dem = circuit.detector_error_model()
assert circuit.num_observables == 9
```

The patch declares its centre and gauges once. The default
`GenericCSSGaugeExtractionBlock` measures all X gauges followed by all Z
gauges; it is passed to `MemoryExperiment` automatically. A different
`basis_order` can be supplied through `se_block_kwargs`.

This realization uses a separate ancilla for every gauge generator:
147 total physical qubits for `r=3`, or 675 for `r=4`. Coordinates provide
a display layout, with no nearest-neighbor hardware constraint. The
generic extraction schedule and noise example are integration baselines;
they do not reproduce the paper's optimized circuit, Clifford compilation,
decoder, single-shot results or performance claims. The code distance is
an algebraic property of the code, not a measured circuit fault distance.

The declaration's logical id `a*r+b` corresponds to bare X support `P[a] ⊗ C[b]` and bare Z
support `C[a] ⊗ P[b]`. Here `P` contains unit vectors at the canonical
simplex pivots and satisfies `P @ C.T = I`. The physical data index is
`row*(2**r-1)+column`; its initial display coordinate is `(column,row)`.
`MemoryExperiment` tracks a basis of the protected logical-state constraints;
its observable indices do not promise this particular choice of representatives.
