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

The default `BaconShorCodeExtractionBlock` in `SE_block.py` measures all X
gauges and then all Z gauges, with a separate physical measurement block for
each basis. It uses a fixed geometric schedule in each patch's local frame:

| Basis | Ancilla-to-data neighbor directions | CNOT direction |
| --- | --- | --- |
| X | left `(-1, 0)`, then right `(+1, 0)` | ancilla → data |
| Z | negative y `(0, -1)`, then positive y `(0, +1)` | data → ancilla |

In the row-index layout these Z directions are top, then bottom. Patch
rotations/transpositions transform the schedule with the patch. Neighbor lookup
uses global coordinates and only accepts data in the declared gauge support.
Each basis takes two parallel CNOT layers, so an XZ pair has CNOT depth four,
independent of distance. Reset and measurement time are additional.

`se_block_kwargs={"basis_order": ("Z", "X")}` reverses this order; single-basis
and repeated-basis sequences are also accepted. The tracker infers the current
gauge constraints and their record parities from the measurements while S and
G remain fixed. A final memory readout requires adequate preparation of the
code and protected logical state.

The generic edge-colored implementation remains explicitly selectable:

```python
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock

generic_circuit = MemoryExperiment(
    qec_patch=BaconShorCode(distance=3),
    extraction_block_class=GenericCSSGaugeExtractionBlock,
    basis="Z",
    rounds=3,
).build()
```

Both schedules have CNOT depth four and the same ideal gauge-measurement
instrument, but they need not use the same layer assignments. In particular,
the generic schedule mixes positive/negative-y interactions within a Z layer.
Identical ideal measurements do not by themselves imply identical noisy
circuits or logical error rates.

Tests cover signed measurement/logical flows, transformed multi-patch geometry,
compressed memory relations, and noisy DEM extraction. The review in
[`playground/subsystem/bacon_shor_schedule/`](../../../playground/subsystem/bacon_shor_schedule/README.md)
also saves distance bounds and a decoding comparison for both schedules. Its
distance checks apply to the specified X/Z memories and Pauli fault models;
they do not establish a threshold or an atom-movement implementation.

## Memory decoding

The recommended baseline for this XZ-alternating memory is **CPU PyMatching
(MWPM), using detectors of the memory basis**. The complete same-basis DEM in
the reviewed noise models is graphlike. Physical locality alone does not imply
this property: the full XZ-detector DEM has hyperedges, and needs a decoder
that handles them or a justified projection/decomposition.

The review notebook explicitly selects original, automatically generated
Z-record detectors for Z memory before compiling the decoder. It preserves
all physical operations and the logical observable. This drops complementary
syndrome information, so it is a baseline rather than an optimality claim.
X memory uses the corresponding X-record selection.

[`playground/subsystem/bacon_shor_decoder_review.ipynb`](../../../playground/subsystem/bacon_shor_decoder_review.ipynb)
compares CPU BP+OSD and MWPM on identical samples from the dedicated circuit.
BP `serial`/`parallel` describes message updates; neither configuration uses
a GPU. The full XZ BP+OSD configurations are reported separately. The public
`MemoryExperiment` continues to generate the complete detector set; decoder
selection and detector projection are explicit choices at the experiment level.

The regular [Bacon–Shor memory demo](../../../notebooks/Memory/memory_bacon_shor.ipynb)
shows the dedicated memory circuit diagram and runs a native-noise CPU MWPM baseline. Its final
data readout is noisy; the historical review above uses ideal final readout.
