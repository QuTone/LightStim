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

`lightstim.utils.fault_audit.low_weight_fault_audit(circuit)` (code-agnostic)
inserts low-weight Pauli faults into an annotated circuit and classifies each as
*detected*, *undetected-harmless*, or *undetected-logical*. It first **atomises**
the circuit (splits merged gate layers so `stim` cannot collapse a whole encoder
into one `CX`), then sweeps

* **weight 1** — every single-qubit Pauli after every physical gate;
* **weight 2 on two-qubit gates** — every `P_a ⊗ P_b` after every two-qubit gate
  (the correlated hook faults a `DEPOLARIZE2` channel would sample).

No undetected-logical fault at either weight ⇒ circuit fault distance `>= 2` ⇒
`O(p^2)` conditional logical-error rate under detector post-selection. The
atomisation makes this verdict agree with a Monte-Carlo slope fit.
`single_fault_audit` is the weight-1-only variant.

| circuit | audit | conditional LER |
|---|---|---|
| bare-qubit `MemoryExperiment` (both bases) | distance `>= 2` | `O(p^2)` (MC slope ≈ 2.0) |
| `encoded_memory_circuit(encoder="zero_zero")` — `\|00>_L` | distance `>= 2` | `O(p^2)` (MC slope ≈ 2.0) |
| `encoded_memory_circuit(..., flag_verified=True)` | distance `>= 2` | `O(p^2)` |
| `get_dist_circ` `\|++>_L` + SE + Bell-pair H-check | **distance 1** | `O(p)` — roadmap stage 3 |

Under generic-CSS extraction the `|00>_L` encoded Z-memory is already circuit
fault distance `>= 2` — flag verification is not needed for it. The genuinely
distance-1 circuit is the `|++>_L` distillation path (`get_dist_circ` has weight-1
hook errors on its control-qubit spine, and so does the H-check ancilla prep);
closing that is roadmap stage 3. See `tests/test_H_six_fault_audit.py`.

### Flag-verified preparation

`encoded_memory_circuit(encoder="zero_zero", flag_verified=True)` runs the Fig. 5
(arXiv:2506.14688) flag-verified `|00>_L` encoder **natively through the
builder**: two ancillas are entangled with the encoder's control qubits (0, 2)
before the data-CX core and disentangled + measured after; the caller
post-selects on the two flag `DETECTOR`s (indices in
`info["flag_detector_indices"]`). SE rounds, readout and observables are
tracker-generated as usual; only the flag `R`/`M`/`DETECTOR` are appended
directly (the tracker rejects a measurement block that follows a data-entangling
unitary, so the encoder is split into per-gate unitary blocks with the flag
gadget spliced between).

## Roadmap / follow-ups

Stages 1–2 are in this PR; later stages are separate PRs.

1. **(done)** H-code patch with the canonical conventions above, generic-CSS
   memory baseline, Tanner visualisation, `low_weight_fault_audit`, and
   `encoded_memory_circuit` (builder-native `|00>_L` encoded memory). Circuit
   fault distance `>= 2` with `O(p^2)` conditional scaling (exact audit +
   Monte-Carlo).
2. **(done)** Flag-verified state preparation and syndrome extraction, natively;
   `low_weight_fault_audit` for low-weight fault audits; `O(p^2)` conditional
   logical-error scaling verified under post-selection for the bare, encoded and
   flag-verified circuits.
3. **Level-1 Magic-H6 Clifford proxy** (`[[6, 2, 2]]`, `k = 2`) and its expected
   `O(p^2)` output suppression — against a dedicated **input `|H>` infidelity**
   channel `p_in` (cf. `tg_distillation.estimate_p_in`), not circuit-level `p`
   (under which the level-1 proxy is only `O(p)`).
4. **Level-2 concatenated `[[36, 4, 4]]`** (`k = 4`) and its expected `O(p^4)`
   suppression in `p_in`.
5. **A true non-Clifford validation path** (`|H>` states + controlled-H), e.g.
   via Clifft, to compare the real protocol against the Clifford proxy.

Stages 3–4 exist in draft form on the `feat/magic-h6-protocol` branch
(`lightstim/protocols/magic_h6_benchmark.py`).
