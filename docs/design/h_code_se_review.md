# H-family extraction: depth, provenance, and fault-distance boundary

This review compares the current LightStim `GenericCSSColorationExtractionBlock`
with `HCodeExtractionBlock`. It does not compare against every possible
edge-coloring algorithm. The currently registered H-family default is the
dedicated unflagged block; experiments can choose either class explicitly.
This registration does not certify a fault-tolerant SE output boundary.

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

These CNOT-only Pauli checks are **separate from the controlled-H check in
Fig. 1e**. The latter checks the magic-state protocol's logical H observable;
it is not an X stabilizer measurement. A change of basis relating H to a Pauli
does not make that physical gadget an ordinary H-code syndrome measurement.
An X/Z memory experiment does not need the controlled-H gadget.

For an independent published reference for flagged Pauli measurement, see
[Chao and Reichardt, arXiv:1705.02329, Fig. 2(c)](https://arxiv.org/pdf/1705.02329)
and [Hilder et al., PRX 12, 011032, Fig. 1(c)](https://doi.org/10.1103/PhysRevX.12.011032).
The latter implements a flagged weight-four Pauli parity check. These support
the individual measurement primitive; a complete H6 X/Z extraction round still
needs its own composition and acceptance audit.

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

## Can final readout checks determine acceptance?

Yes, for a declared **destructive memory/readout benchmark**. For example, Z
readout supplies the parities on data `(0,1,2,3)` and `(2,3,4,5)` for H6.
Comparing these to the preceding Z syndromes closes the final detectors.
Those are stabilizer consistency checks, not a demand that the logical
measurement have its desired value. Logical failures among accepted shots must
still be counted separately, and the acceptance probability must be reported.

The Quantinuum notebook's `checks` function rejects nontrivial final `meas1`
stabilizer parities; its separate `success` function scores a logical parity.
Section II.1 of the paper also explicitly says its non-FT comparison retains
syndrome information inferred from measuring out the code blocks.

Such retrospective acceptance does **not** certify a quantum output heralded
before destructive readout. For that claim, acceptance must use information
available at the stated output boundary. Detectable residual data errors and
undetected logical errors must be distinguished; distance two does not supply
a decoder that corrects every arbitrary weight-one residual.

Conversely, discarding the final checks while retaining noisy data readout
does not isolate SE fault tolerance. A single terminal bit flip can flip the
reported logical parity without affecting any earlier syndrome measurement,
regardless of how good the preceding SE was. A regression test inserts just
this fault after an otherwise noiseless H6 memory circuit: in both bases and
both schedules, Stim gives `error(0.001) D6 L0 L1`, where D6 is a final-readout
detector. Including D6 rejects the event; accepting on the earlier D0..D5 alone
does not. It is a readout error, not evidence of an undetected logical Pauli on
the encoded quantum output.

A supplementary simulation of H6, two SE rounds, 400,000 shots per point,
`p=0.002,0.004,0.008,0.016`, gate/SPAM noise as above and zero idle noise gives
these log-log slopes for **conditional logical-readout error**:

| Schedule / basis | All detectors | SE detectors only |
|---|---:|---:|
| Dedicated / Z | 1.995 | 1.016 |
| Dedicated / X | 1.965 | 1.018 |
| Coloration / Z | 2.061 | 1.011 |
| Coloration / X | 1.982 | 1.003 |

Both acceptance rules are applied to the same samples at each point. These
finite-p fits illustrate the boundary effect; the single-fault counterexample
establishes why the second metric has a first-order term.

The SE-only acceptance column still scores raw logical readout; it does not
decode the final detectors. Early postselection followed by final-detector
decoding is a different experiment and requires a separate fault/scaling audit.

## Chosen default for the first asset

The maintainer chose the dedicated `HCodeExtractionBlock` as the default for
both `HCode(n)` and `HSixCode()`. Generic coloration remains an explicit
configuration for comparison. The acceptance contract of the audited memory
baseline includes all detectors, including final readout; this choice does
not certify the open-output SE gadget under the stronger residual-error
condition above. The dedicated schedule is locally derived, while the current
generic H8 ordering has the undetected single-fault logical counterexample.

The current scope keeps memory SE unflagged. The next H6 protocol experiment
will handle the joint logical-H check and its Clifford proxy separately;
neither that check nor Fig. 5's flagged |00> preparation is a four-stabilizer
extraction round. General-family flagged extraction is outside this scope.

If flagged H6 SE is revisited, the separately sourced weight-four Pauli checks
provide a starting point; X and Z can be related by Clifford conjugation.
A complete-round audit must still test the chosen quantum output boundary,
including whether a single accepted internal fault leaves at most a weight-one
residual modulo stabilizers, and whether a fault-free round detects a
weight-one input error. Concurrent scheduling and ancilla reuse require their
own validation. The memory experiment must also declare its preparation,
readout, and acceptance rules.
