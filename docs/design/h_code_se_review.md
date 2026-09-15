# H-family extraction: depth, provenance, and fault-distance boundary

This review compares the current LightStim `GenericCSSColorationExtractionBlock`
with `HCodeExtractionBlock`. It does not compare against every possible
edge-coloring algorithm. The H-family default remains the dedicated unflagged
block; experiments can choose either class explicitly.

## Depth comparison

Both circuits use four check ancillas and `2n+4` CNOTs for the four checks of
`HCode(n)`. The current generic block separately edge-colors the X and Z Tanner
graphs, then executes X followed by Z. Each graph has maximum degree `n-2`,
so this implementation uses `2(n-2)` CNOT layers. The dedicated circuit
pipelines X and Z in `n+2` layers. The saving is exactly `n-6` CNOT layers.

| n | Generic CNOT depth | Dedicated CNOT depth | Saved layers |
|---:|---:|---:|---:|
| 6 | 8 | 8 | 0 |
| 8 | 12 | 10 | 2 |
| 12 | 20 | 14 | 6 |
| 16 | 28 | 18 | 10 |
| 32 | 60 | 34 | 26 |
| 64 | 124 | 66 | 58 |

Counting reset, initial H, final H, and measurement as four additional stages
gives operation depths `2n` and `n+6`. These are layer counts, not measured
hardware times. The CNOT-depth ratio tends to 1/2 at large n; neither optimal
finite-size depth nor a uniformly better logical-error coefficient is claimed.
With idle noise disabled, the gate counts are identical despite the different
depths. Fault-aware order still matters independently of depth.

## Where the dedicated schedule comes from

The `n+2` schedule was derived during this LightStim integration. It is not a
copied Quantinuum circuit and has no claimed paper citation for this exact
ordering. [Jones, arXiv:1210.3388, Section II](https://arxiv.org/pdf/1210.3388)
is the reference for the H-code family. Section IV assumes perfect operations
except the input magic states, so its distillation scaling is not a proof for
our noisy extraction gates.

Our checks XA, XB, ZA, ZB touch shared data 2 in layers 0,1,2,3, respectively.
They next touch their private data in ascending order, then shared data 3 in
layers n-2,n-1,n,n+1. This has two useful properties:

- Interactions in every layer are disjoint. Every X interaction on a data qubit
  precedes every Z interaction on that same qubit. Only commuting gates are
  interleaved relative to serial X-then-Z extraction; exact full-tableau tests
  verify the ideal unitary, including logical action.
- Shared data 2 and 3 bracket each ancilla's interactions. A proper nonempty
  propagated ancilla-error tail contains exactly one shared data qubit, giving
  a nontrivial syndrome against the other opposite-basis check at the data
  boundary. Whether that syndrome has already been measured matters below.

## What Quantinuum actually used

[Dasu et al., arXiv:2506.14688, Fig. 1 and Sections I–II](https://arxiv.org/html/2506.14688v1)
describe a Magic-H6 protocol with an H check and syndrome postselection. The
experimental benchmarking circuit has a QED_X stage and destructive data
readout; it is not a generic four-check H-family memory round.

The exact public source inspected is commit
`0106aefc537257d77af737dfef00d5d335014b4c` of
[Quantinuum/Magic-H6](https://github.com/Quantinuum/Magic-H6/tree/0106aefc537257d77af737dfef00d5d335014b4c).
The QASM in
[FaultTolerantMSBenchmark.ipynb](https://github.com/Quantinuum/Magic-H6/blob/0106aefc537257d77af737dfef00d5d335014b4c/MagicStateBenchmarking/FaultTolerantBenchmarking/FaultTolerantMSBenchmark.ipynb)
explicitly measures the X check on `(0,1,2,3)`, then the X check on `(2,3,4,5)`.
Each weight-four check uses a syndrome ancilla and a flag ancilla, with this
CNOT order (a: syndrome, f: flag, d0..d3: that check's ordered data):

```text
a→d0, a→f, a→d1, a→d2, a→f, a→d3
```

Ancilla H gates and measurements surround those CNOTs. The six CNOTs in one
check share the syndrome ancilla. The QASM acceptance function checks both X
syndromes and both flags, plus the other protocol checks and final data
parities. Thus neither the generic coloring schedule nor our four-ancilla
pipeline should be presented as their experimental scheduling. The source's
instruction order also does not specify every compiled H1 hardware timestamp.

The reference
[Code614.py](https://github.com/Quantinuum/Magic-H6/blob/0106aefc537257d77af737dfef00d5d335014b4c/Stim/ConcatenatedMSProtocolSim/Code614.py)
contains Clifford-proxy distillation and flagged initialization primitives;
it is not a general-n unflagged syndrome-extraction implementation.

## Closed memory: exact single-fault audit

The audit scans **every even n from 6 through 64**, both X/Z memory bases,
1/2/3 extraction rounds, and two idle-noise settings, for both schedules:
**720 circuits** in Stim 1.15.0. Set `p_1q=p_2q=p_meas=p_reset=0.001`, with
`p_idle=0` or `0.001`. Acceptance means every detector is zero, including final
readout checks. Failure means any logical observable flips. No decoder is used.

For each circuit, the undecomposed DEM is checked for logical-only single-fault
signatures, including hyperedges. A graphlike witness maps to physical noise
locations; two-fault witnesses are checked for independent locations. Thus an
absent single-fault logical signature plus a physical two-fault witness proves
distance two for that circuit. Graphlike search alone would not establish the
lower bound.

| Sizes tested | Generic memory fault distance | Dedicated memory fault distance |
|---|---:|---:|
| n=6 | 2 | 2 |
| every even n=8,...,64 | 1 | 2 |

Results hold for both bases, all tested round counts, and both idle settings.
[Full results](h_code_se_comparison.csv) and the
[reproduction script](../../benchmarks/memory/audit_h_code_se.py) are included:

```bash
PYTHONPATH=. venv/bin/python benchmarks/memory/audit_h_code_se.py --max-n 64 --output /tmp/h-code-se-comparison.json
```

This finite scan does not prove noisy distance two for every unbounded n.
Nor is a bad result for this deterministic generic ordering a theorem that
edge-coloring-based methods cannot produce fault-tolerant schedules.

### Independent H8 counterexample to the generic ordering

In the current generic circuit, XB visits data `(6,5,4,2,3,7)`. A single X
fault on XB after `CX(XB,5)` propagates X to `(4,2,3,7)`. This is equivalent,
modulo the XB stabilizer, to **X5 X6**. It commutes with all checks but acts
nontrivially on logicals. No later syndrome round can detect that logical.
This is one allowed Pauli outcome of a faulty two-qubit gate. An independent
Pauli-propagation regression test verifies the fault, all four ancilla outcomes,
and its nontrivial logical action, separately from the DEM audit.

## An SE output boundary is a different claim

The dedicated schedule also has a useful counterexample to overclaiming:
a **Z on ZA after CX(data 0, ZA)** propagates to **Z1 Z3**. All four ancilla
measurements in that round can remain unchanged. The residual is weight two
and cannot be reduced to weight one modulo the stabilizer group. However, it
has a nontrivial XB syndrome at the output data boundary. Subsequent checks or
final X readout can detect it. This is verified by independent propagation at
n=6,8,12,64.

This violates the stronger requirement that one internal fault leave at most
one undetected data error modulo stabilizers. It does **not** contradict the
closed-memory distance-two result, and is **not** an undetected logical error
at that output boundary. It establishes why the memory result must not be
advertised as a composable, flag-free fault-tolerant extraction gadget.
A future H6 flag block needs its own input/output error and acceptance contract.

## Default configuration recommendation

Keep the dedicated block as the **H-family memory baseline**: it has a general
ideal schedule, smaller depth for n>6, a reproducible closed-memory audit, and
one consistent implementation at n=6 and larger sizes. At H6 alone, the current
generic circuit ties it in depth and audited distance. Keep generic coloration
available as an explicit comparison/fallback, without promising distance two.
For Magic-H6 protocol reproduction, implement the H6-only flag block from the
reference and validate the complete protocol's boundaries separately. The
notebook names its extraction class explicitly so its physical configuration
remains visible during review.
