# `[[6, 2, 2]]` H-code patch

`HSixCode()` implements the smallest member of the **H-code family**, the
self-dual `[[6, 2, 2]]` CSS code used as the base code of Quantinuum's Magic-H6
`|H>` magic-state distillation protocol
([arXiv:2506.14688](https://arxiv.org/abs/2506.14688); reference implementation
[CQCL/Magic-H6](https://github.com/CQCL/Magic-H6)). It encodes `k = 2` logical
qubits with distance `d = 2` (error-*detecting*).

```python
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_six import HSixCode

experiment = MemoryExperiment(
    qec_patch=HSixCode(),
    basis="Z",          # X memory is also supported
    rounds=3,
)
circuit = experiment.build()
detections, observables = circuit.compile_detector_sampler().sample(
    256, separate_observables=True,
)
assert not detections.any()
assert not observables.any()
assert circuit.num_observables == 2      # k = 2
```

## Canonical logical convention

Data-qubit labels `0..5`, in the Magic-H6 ordering. The stabilizer group and a
canonical logical basis are

```
S^X_1 = X0 X1 X2 X3      S^Z_1 = Z0 Z1 Z2 Z3
S^X_2 = X2 X3 X4 X5      S^Z_2 = Z2 Z3 Z4 Z5

X0_L  = X0 X2 X4         Z0_L  = Z0 Z2 Z4
X1_L  = X1 X3 X5         Z1_L  = Z1 Z3 Z5
```

Logical operators are registered as `(X0, Z0, X1, Z1)` — logical qubit 0's pair,
then logical qubit 1's. The X and Z checks share support, so a **transversal
Hadamard** maps the stabilizer group to itself and swaps `X_i_L <-> Z_i_L` on
both logical qubits; a transversal `S` is logical `S`. That self-duality is what
makes the code useful for distilling `|H>` states, and it is exposed through
`HSixLogicalOpSet` (transversal `H`, `S`, `S_DAG`, `X`, and two-patch `CNOT`,
`CZ`, `CY`).

`HSixCode.get_stabs(bits)` and `.get_logicals(bits)` are direct ports of the
same-named helpers in the reference `Code614.py`, so classical parities computed
here match the raw-Stim port bit-for-bit.

## Coordinate convention

The code has no 2D locality; coordinates only feed plotting and data/syndrome
role inference. Data qubit `i` sits at `(2*i, 0)`. X ancillas are placed on the
row `y = +1`, Z ancillas on `y = -1`, each centred over its 4-qubit support.
`HSixCode(h_check_ancillas=2)` adds two bare `role="syndrome"` ancillas on
`y = 3` for the Magic-H6 Bell-pair "H-check" (`HSixLogicalXCheckBlock`);
`HSixExtractionBlock` ignores them. A `shift=(dx, dy)` kwarg offsets every
qubit for multi-block layouts.

`from lightstim.utils.tanner import draw_tanner_graph` — `draw_tanner_graph(HSixCode())`
renders the bipartite Tanner graph in this convention (X checks above the data
line, Z checks below, mirror images of each other). It is the generic
`QECPatch` Tanner drawer; `matplotlib` is an optional dependency, imported
lazily.

## Syndrome extraction

The code has no natural geometric CNOT schedule, so `HSixExtractionBlock`
is LightStim's generic CSS coloration block
(`GenericCSSColorationExtractionBlock`) — it colours the X and Z Tanner graphs
independently and emits one CNOT layer per colour. This is the same choice the
Kasai codes make. `MemoryExperiment` picks it up automatically as the patch's
`default_extraction_block_class`.

## Low-weight fault audit

`single_fault_audit(circuit)` (in `lightstim.utils.fault_audit`, a code-agnostic
helper) inserts every single-qubit Pauli fault after every operation of an
annotated circuit and classifies it as *detected*, *undetected-harmless*, or
*undetected-logical*. A circuit with no undetected-logical single fault has
circuit fault distance `>= 2`, i.e. an `O(p^2)` conditional logical-error rate
under detector post-selection.

The baseline `MemoryExperiment` circuit above **passes this audit in both bases**
(`result.has_circuit_fault_distance_two is True`); see
`tests/test_H_six_fault_audit.py`, which also checks the `O(p^2)` slope by
Monte-Carlo. The non-fault-tolerant `get_dist_circ` encoder path does *not* pass
it — a single encoder fault can flip a logical undetected — which is why the
flag-verified encoder (`get_ft_init_circ`, Fig. 5 of arXiv:2506.14688) and the
Bell-pair H-check exist.

## Roadmap / follow-ups

This patch is stage 1–2 of a longer roadmap; later stages are separate PRs:

1. **(this PR)** H-code patch with the canonical conventions above, generic-CSS
   memory baseline, Tanner visualisation, and the low-weight fault audit
   establishing `O(p^2)` conditional scaling for the memory experiment.
2. **Flag-verified state preparation and syndrome extraction, natively.**
   `get_ft_init_circ` is ported, but wiring flag ancillas through
   `CircuitBuilder` / `SyndromeTracker` so their detectors are auto-generated
   needs a non-destructive mid-circuit ancilla-measurement gadget the builder
   does not yet expose.
3. **Level-1 Magic-H6 Clifford proxy** (`[[6, 2, 2]]`, `k = 2`) and its expected
   `O(p^2)` output suppression.
4. **Level-2 concatenated `[[36, 4, 4]]`** (`k = 4`) and its expected `O(p^4)`
   suppression.
5. **A true non-Clifford validation path** (`|H>` states + controlled-H), e.g.
   via Clifft, to compare the real protocol against the Clifford proxy.

Stages 3–4 exist in draft form on the `feat/magic-h6-protocol` branch
(`lightstim/protocols/magic_h6_benchmark.py`).
