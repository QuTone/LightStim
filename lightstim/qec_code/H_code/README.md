# H-code family: [[n, n-4, 2]]

`HCode(n=6)` is H6 (also called C6). The family accepts even `n >= 6`, with
`n` data qubits, four syndrome ancillas, `n-4` logical qubits, and distance two.
`HSixCode()` is a compatibility subclass that delegates to `HCode(n=6)`.
The original `lightstim.qec_code.H_six` import paths remain supported.

The initial LightStim H6 assets were contributed by **Maggie Bao (Infleqtion)**
in [PR #98](https://github.com/QuTone/LightStim/pull/98). The general family is
from [Jones, arXiv:1210.3388, Section II](https://arxiv.org/pdf/1210.3388).
For Magic-H6, see [Dasu et al.](https://arxiv.org/html/2506.14688v1) and
[Quantinuum's reference implementation](https://github.com/Quantinuum/Magic-H6).

## Run and inspect memory

```python
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode, HSixCode

patch = HCode(n=8)  # [[8,4,2]]; HSixCode() preserves the old H6 convention.
circuit = MemoryExperiment(qec_patch=patch, basis="Z", rounds=3).build()
dets, obs = circuit.compile_detector_sampler(seed=98).sample(
    256, separate_observables=True
)
assert circuit.num_observables == 4
assert not dets.any() and not obs.any()
```

Use `basis="X"` for X memory. The executed
[memory notebook](../../../notebooks/Memory/memory_H_code.ipynb) keeps the original PR's compact memory workflow: choose n and the extraction
class, run X/Z memory, inspect detector slices, and check single faults.
The detailed [SE review](../../../docs/design/h_code_se_review.md) records
depth comparisons, paper/source provenance, and the output-boundary limitation.

## Canonical checks and logical operators

For zero-based data labels `0..n-1`, define

```text
A = (0, 1, 2, 3)
B = (2, 3, ..., n-1)
SxA = X(A), SxB = X(B), SzA = Z(A), SzB = Z(B).
```

For each `q=4,...,n-1`, logical slot `j=q-4` uses support `(0,2,q)` for even
`q`, and `(1,3,q)` for odd `q`, for both X and Z. Logical representatives are
registered as `(XL0, ZL0, XL1, ZL1, ...)`. Each X/Z pair anticommutes, and
different logical slots commute. The parity helpers `get_stabs(bits)` and
`get_logicals(bits)` infer `n` from the string length; H6 helpers require six
bits. For n=6 this reproduces the original PR exactly:

```text
SxA = X0 X1 X2 X3        SzA = Z0 Z1 Z2 Z3
SxB = X2 X3 X4 X5        SzB = Z2 Z3 Z4 Z5
XL0 = X0 X2 X4          ZL0 = Z0 Z2 Z4
XL1 = X1 X3 X5          ZL1 = Z1 Z3 Z5
```

This convention is equivalent to Jones's family after relabeling data and
choosing a canonical logical basis. Every weight-one Pauli is detected;
`X0 X1` commutes with the checks and acts logically, so the distance is two.
Identical X/Z supports make transversal physical H act as H on each logical
slot. A single HCode(n=36) is [[36,32,2]], distinct from the concatenated H6
[[36,4,4]] construction.

## Coordinates

Data `i` sits at `(2*i,0)`. X-check ancillas sit at `(3,1)` and `(n+1,1)`;
Z-check ancillas sit at `(3,-1)` and `(n+1,-1)`. Each ancilla is centered over
its support. `shift=(dx,dy)` offsets all coordinates and composes with later
patch/system shifts. The n=6 coordinates remain unchanged.

These are Tanner-graph drawing coordinates, without a nearest-neighbor
hardware assumption. There are four check ancillas, with no flag or protocol
ancillas in the patch.

## Dedicated concurrent syndrome extraction

`HCodeExtractionBlock` is the default for both constructors;
`HSixExtractionBlock` aliases it. All four ancillas reset together, X ancillas
receive H, the CNOT pipeline runs, X ancillas receive H again, and all four
ancillas are measured together. X CNOTs point ancilla-to-data; Z CNOTs point
data-to-ancilla.

The zero-based CNOT-layer schedule is:

| Check | First interaction | Private-data interactions | Last interaction |
|---|---|---|---|
| XA | data 2 at t=0 | data 0,1 at t=1,2 | data 3 at t=n-2 |
| XB | data 2 at t=1 | data 4,...,n-1 at t=2,...,n-3 | data 3 at t=n-1 |
| ZA | data 2 at t=2 | data 0,1 at t=3,4 | data 3 at t=n |
| ZB | data 2 at t=3 | data 4,...,n-1 at t=4,...,n-1 | data 3 at t=n+1 |

X and Z checks overlap in time on different data qubits. Every X interaction
on a given data qubit still precedes every Z interaction on that qubit. Thus
interleaving reorders only mutually commuting gates: the entire ideal Clifford
unitary equals serial X-then-Z extraction, without residual ancilla coupling
or logical action. Tests compare full Stim tableaus, beyond noiseless memory
samples. This is a dedicated LightStim schedule, not a claimed published
canonical extraction circuit.

There are **n+2 CNOT layers** and **2n+4 CNOTs** per round. Counting reset, two H
layers, and measurement as one layer each gives **n+6 operation layers**;
actual time depends on gate durations. `x_layers` and `z_layers` share the
same time axis. Their occupied-layer counts `depth_x` and `depth_z` overlap;
use `cnot_depth` for total CNOT depth. Independent patches run in parallel
using system-global qubit indices. Only active, unmodified H-code checks are
supported by this dedicated block; other extraction needs an explicit block.

`GenericCSSColorationExtractionBlock` remains available as an explicit
`MemoryExperiment(extraction_block_class=...)` override. Its X-then-Z schedule
has `2(n-2)` CNOT layers, and its ordering does not in general preserve circuit
fault distance two as n grows. The dedicated ordering places shared data 2 and
3 at each check's ends: every proper nonempty hook tail contains exactly one
shared data qubit and can be detected by the other opposite-basis check.
The full noisy circuit still needs its own audit, including boundaries.

`MemoryExperiment` constructs detectors and all `n-4` observables through the
existing `CircuitBuilder`/`SyndromeTracker`. Bare product-state initialization
projects the initially unfixed checks in the first round. Later rounds compare
check outcomes; final readout supplies the matching boundary checks. With
`r` extraction rounds, this memory has `4r` detectors in either basis.

## Validation and postselection semantics

Tests cover n=6,8,12,20,32,64; X/Z memory; 1,2,3 extraction rounds; independent
Pauli gate/SPAM noise `p_1q=p_2q=p_meas=p_reset=p`, with both `p_idle=0` and
`p_idle=p`. In every tested circuit:

1. The **undecomposed** DEM has no single-fault signature that flips a logical
   observable without firing a detector, including hyperedges.
2. A two-fault graphlike witness maps to independent physical noise locations.

Together these establish circuit fault distance two for the tested circuits.
Default graphlike search alone skips hyperedges and cannot supply (1).
The family algebra and ideal schedule are general; these finite-size noisy
checks are not a proof of arbitrary protocol composition or open-output fault
tolerance. In fact, a Z fault on ZA after its data-0 interaction can leave
Z1 Z3 without an alarm in that round; later checks/final X readout detect it.
The [SE review](../../../docs/design/h_code_se_review.md) records this boundary
counterexample and the expanded 720-circuit comparison against generic coloring.

Acceptance requires **every detector zero**, including final readout checks;
block failure means **any of the n-4 logical observables flips**:

```python
accepted = ~dets.any(axis=1)
acceptance = accepted.mean()
conditional_ler = obs[accepted].any(axis=1).mean()
```

The slow tests sample conditional block error at four values of p for H6 and
H10, with 400,000 shots per point and no idle noise. This checks the quadratic
small-p behavior suggested by the fault audit. There is no decoder or special
flag postselection tag. These are destructively read out memory experiments,
not Magic-H6 output-state fidelity measurements.

The next flag implementation will be a **separate H6-only extraction block**,
with its own fault and acceptance tests. General-family flagged extraction is
outside this milestone. See the
[integration walkthrough](../../../docs/design/h6_core_integration.md).

## Reproduce

From the repository root, using the LightStim virtual environment:

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/test_H_code.py tests/test_H_six_code.py tests/test_H_code_fault_distance.py tests/test_protocols.py -m "not slow" --timeout=90
PYTHONPATH=. venv/bin/python -m pytest tests/test_H_code_fault_distance.py -m slow -s --timeout=90
```
