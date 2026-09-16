# SHYPS memory integration

The [memory notebook](../../../notebooks/Memory/memory_shyps.ipynb) presents
the dedicated schedule and the resulting native LightStim memory circuit.

`SHYPSCode(r=3)` implements the subsystem hypergraph-product simplex code
with 49 data qubits, 9 protected logical qubits and code distance 4.
`r=4` implements the 225-data-qubit, 16-logical-qubit, distance-8 instance.
`r=5` implements the 961-data-qubit, 25-logical-qubit, distance-16 instance.
Other values of `r` are currently rejected.

The construction follows Malcolm et al.,
[Computing Efficiently in QLDPC Codes, §§VIII.4–VIII.5](https://arxiv.org/html/2502.07150v2).
The classical circulant checks use `1 + x² + x³` for `r=3` and the primitive
polynomial `1 + x + x⁴` for `r=4`, and `1 + x² + x⁵` for `r=5`.
The canonical generator `C` spans the
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
`SHYPSCodeExtractionBlock` measures all X gauges followed by all Z gauges;
it is passed to `MemoryExperiment` automatically. A different `basis_order`
can be supplied through `se_block_kwargs`, including single or repeated bases.

Each basis uses three CNOT layers, in the order given by `code.gauge_offsets`:

| `r` | Layer 1 | Layer 2 | Layer 3 |
| --- | --- | --- | --- |
| 3 | 0 | 2 | 3 |
| 4 | 0 | 1 | 4 |
| 5 | 0 | 2 | 5 |

With `m = 2**r - 1`, X ancilla `(i,j)` controls CNOTs onto data
`((i+offset) % m,j)`; Z ancilla `(i,j)` receives CNOTs from data
`(i,(j+offset) % m)`. These are semantic product indices, independent of
coordinate shifts, rotations or transpositions. Each offset gives a matching,
so each layer is collision-free. A complete X/Z cycle has six CNOT layers;
reset and measurement time are additional. Each basis is passed separately
to the existing Builder/Tracker as a measurement block, which generates the
detectors and observables automatically.

The generic coloration implementation remains available by explicitly passing
`extraction_block_class=GenericCSSGaugeExtractionBlock` to `MemoryExperiment`
(import it from `lightstim.qec_code.generic_css`). Different CNOT orderings can
have the same ideal measurement instrument but different fault propagation.

This realization uses a separate ancilla for every gauge generator:
147 total physical qubits for `r=3`, 675 for `r=4`, or 2883 for `r=5`.
Coordinates still provide a display layout with the ancillas below and to
the right of the data. The fixed cyclic offsets specify the interaction
schedule, not a compiled atom-movement trajectory or a planar wrap-around
routing solution. Code distance describes the algebraic code; circuit fault
distance requires a separate analysis of the chosen schedule and fault model.

The r=5 patch algebra and dedicated SE are checked independently. Full r=5
memory/DEM validation is still pending: the existing Tracker's matrix
processing is expensive at this size, including with optional C++ RREF.
Use the r=3 notebook-scale example for an interactive memory review.

The declaration's logical id `a*r+b` corresponds to bare X support `P[a] ⊗ C[b]` and bare Z
support `C[a] ⊗ P[b]`. Here `P` contains unit vectors at the canonical
simplex pivots and satisfies `P @ C.T = I`. The physical data index is
`row*(2**r-1)+column`; its initial display coordinate is `(column,row)`.
`MemoryExperiment` tracks a basis of the protected logical-state constraints;
its observable indices do not promise this particular choice of representatives.

## Memory benchmark CLI

Use the shared [memory runner](../../../benchmarks/memory/README.md):

```bash
python benchmarks/memory/run_memory.py \
    --codes shyps_49_9_4 --basis Z X --p-values 0.001 \
    --decoder cpu_bposd --num-workers 1 \
    --max-shots 64 --max-errors 65 --batch-size 64
```

The other registered instances are `shyps_225_16_8` and `shyps_961_25_16`.
The code fixes its distance, and `--rounds` defaults to that distance in complete
XZ cycles. The notebook uses two cycles for a compact diagram. CLI noise rates
all default to the swept `p`; add `--p-idle 0 --p-1q 0` to match the notebook's
noise settings. Final data readout is noisy in both cases.

All automatic X/Z detectors and all `r**2` protected logical observables are
retained. The runner reports a failure if any tracked logical observable is
decoded incorrectly. Its CPU/GPU BP+OSD, Relay-BP, MWPF, plain BP and MLE-ILP
interfaces consume the full DEM. Optional decoder packages must be installed;
select one visible CUDA device when using `gpu_bposd`. PyMatching requires a
graphlike decomposition, which the standard noisy SHYPS memory does not admit.

`--decoder-params '{"pre_iter":20,"num_sets":4,"set_max_iter":20}'` configures
the `relay-bp` backend, for example. Effective parameters are written to the
shared CSV and included in the checkpoint key. These are full-graph decoder
interfaces; a sliding-window BP+LSD backend is not part of this integration.
