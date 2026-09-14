# H6: the [[6, 2, 2]] CSS code

`HSixCode` is the six-data-qubit H6 (also called C6) code. This integration
provides its patch, generic CSS syndrome extraction, and bare X/Z memory
experiments. It implements this fixed code, rather than a parameterized family.

The initial LightStim H6 assets were contributed by **Maggie Bao (Infleqtion)**
in [PR #98](https://github.com/QuTone/LightStim/pull/98). The physics reference is
[Dasu et al., arXiv:2506.14688, Section I](https://arxiv.org/html/2506.14688v1#S1),
with [Quantinuum's reference code](https://github.com/Quantinuum/Magic-H6).

## Run the baseline

```python
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_six import HSixCode

circuit = MemoryExperiment(qec_patch=HSixCode(), basis="Z", rounds=3).build()
dets, obs = circuit.compile_detector_sampler(seed=98).sample(
    256, separate_observables=True
)
assert not dets.any() and not obs.any()
assert circuit.num_observables == 2
```

Use `basis="X"` for X memory. The executed
[memory notebook](../../../notebooks/Memory/memory_H_six.ipynb) includes a Tanner
graph, detector slices, a circuit fault audit, and a small postselection sample.

## Canonical logical convention

Data labels are `0..5`, with four independent stabilizers:

```text
Sx0 = X0 X1 X2 X3        Sz0 = Z0 Z1 Z2 Z3
Sx1 = X2 X3 X4 X5        Sz1 = Z2 Z3 Z4 Z5

XL0 = X0 X2 X4          ZL0 = Z0 Z2 Z4
XL1 = X1 X3 X5          ZL1 = Z1 Z3 Z5
```

Logical representatives are registered in the order `(XL0, ZL0, XL1, ZL1)`.
Each X/Z pair anticommutes; different logical qubits commute.
The parity helpers `get_stabs(bits)` and `get_logicals(bits)` follow this order
for six-bit strings measured in either the X or Z basis.

There are six data qubits and four independent checks, leaving two logical
qubits. Every single-qubit Pauli anticommutes with a check. A weight-two Pauli
such as `X0 X1` commutes with all checks and acts nontrivially on the logicals,
so the code distance is two despite the weight-three representatives above.

Matching X/Z supports make physical transversal H swap each logical X/Z pair.
Gate APIs and their phase conventions require separate logical-action tests;
they are outside this memory integration.

## Coordinates and extraction

Data qubit `i` sits at `(2*i, 0)`. The two X ancillas sit at `(3, 1)` and
`(7, 1)`; the Z ancillas sit at `(3, -1)` and `(7, -1)`.
These coordinates describe the Tanner graph layout, without imposing a
nearest-neighbor hardware constraint. `shift=(dx, dy)` offsets the whole patch.
Only the four check ancillas belong to this patch; later protocols allocate
their additional ancillas separately.

`HSixExtractionBlock` aliases `GenericCSSColorationExtractionBlock`.
The X and Z Tanner graphs each have maximum degree four, giving four parallel
CNOT layers per basis, eight total. X checks use ancilla-to-data CNOTs; Z checks
use data-to-ancilla CNOTs. No flags are used. `MemoryExperiment` discovers this
default automatically and constructs detectors and both logical observables
through `CircuitBuilder` and `SyndromeTracker`.

A bare product-state initialization is not an encoded-state preparation.
The first extraction projects the unfixed stabilizers; subsequent rounds
compare their outcomes. Only the checks fixed by the preparation basis can
produce first-round boundary detectors. Readout provides the matching final
boundary.

## Exact meaning of the validation

For independent circuit-level Pauli noise with
`p_1q = p_2q = p_meas = p_reset = p` and `p_idle = 0`, the tests cover X/Z
memory at 1, 2, and 3 extraction rounds:

1. Noiseless detectors and both logical observables are deterministic.
2. The **undecomposed** DEM contains no single-fault signature that flips a
   logical observable without firing a detector, including hyperedge errors.
3. A two-fault graphlike witness maps to independent physical noise locations.

Together, (2) and (3) establish circuit fault distance two for these circuits.
`shortest_graphlike_error()` alone is insufficient: by default it skips
ungraphlike errors. Setting `ignore_ungraphlike_errors=False` attempts
decomposition; it is not a general exhaustive hyperedge search.

The acceptance rule is **all detector outcomes zero**, including the final
readout checks. A failure is **either of the two logical observables flipped**:

```python
accepted = ~dets.any(axis=1)
acceptance = accepted.mean()
conditional_ler = obs[accepted].any(axis=1).mean()
```

The slow tests sample this conditional block error rate at four values of
`p`. The single-fault exclusion and two-fault witness explain the small-`p`
quadratic scaling; finite-shot fits are an empirical check. This acceptance
rule is explicitly applied to samples; the baseline declares no special
flag postselection tags.

These results concern a destructively read out memory experiment. They do not
establish a reusable fault-tolerant encoder, flagged extraction, or Magic-H6
output-state fidelity. See the
[integration walkthrough](../../../docs/design/h6_core_integration.md)
for the original assets and the follow-up milestones.

## Reproduce

From the repository root, using the LightStim virtual environment:

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/test_H_six_code.py tests/test_H_six_fault_distance.py tests/test_protocols.py -m "not slow" --timeout=90
PYTHONPATH=. venv/bin/python -m pytest tests/test_H_six_fault_distance.py -m slow -s --timeout=90
```
