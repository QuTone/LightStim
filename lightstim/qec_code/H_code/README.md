# H-code family: [[n, n-4, 2]]

`HCode(n)` supports even `n >= 6`, with `n` data qubits, four syndrome
ancillas, and `n-4` logical qubits. `HSixCode()` delegates to `HCode(n=6)`
and preserves the H6 logical convention. Both are imported from
`lightstim.qec_code.H_code`.

The initial H6 implementation was contributed by **Maggie Bao (Infleqtion)**
in [PR #98](https://github.com/QuTone/LightStim/pull/98).
The family is defined by [Jones, Section II](https://arxiv.org/pdf/1210.3388).
The H6 encoder follows [Dasu et al., Fig. 1(d)](https://arxiv.org/html/2506.14688v1);
see also [Quantinuum's Magic-H6 implementation](https://github.com/Quantinuum/Magic-H6).

## Memory experiment

```python
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode, HCodeExtractionBlock, HSixCode
from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

patch = HCode(n=8)
circuit = MemoryExperiment(
    qec_patch=patch, basis="Z", rounds=3,
    extraction_block_class=HCodeExtractionBlock,  # Default; coloration is also available.
).build()
```

Use `basis="X"` for X memory. Each extraction round measures both X and Z
checks. The builder generates all `n-4` logical observables and `4*rounds`
detectors, including final data-readout checks. `patch.get_info()` returns
the stabilizers, logical operators, coordinates, and code parameters.
The [memory notebook](../../../notebooks/Memory/memory_H_code.ipynb) shows
both bases and a detector-slice view; set `N=6` for H6.

## Logical and coordinate conventions

For zero-based data labels, the two supports are `A=(0,1,2,3)` and
`B=(2,3,...,n-1)`, giving checks `X(A), X(B), Z(A), Z(B)`.
For `q=4,...,n-1`, logical slot `j=q-4` uses support `(0,2,q)` for even
`q` and `(1,3,q)` for odd `q`, for both X and Z. Representatives are
registered as `(XL0,ZL0,XL1,ZL1,...)`. For H6:

```text
XL0 = X0 X2 X4          ZL0 = Z0 Z2 Z4
XL1 = X1 X3 X5          ZL1 = Z1 Z3 Z5
```

Each logical X/Z pair anticommutes; different slots commute. Every weight-one
Pauli is detected, while `X0 X1` is a weight-two logical operator.
`get_stabs(bits)` and `get_logicals(bits)` return same-basis parities in
the registered order; the HSix helpers require six bits.

Data qubit `i` sits at `(2*i,0)`. X ancillas sit at `(3,1)`, `(n+1,1)`;
Z ancillas sit at `(3,-1)`, `(n+1,-1)`. `shift=(dx,dy)` translates the
patch. These coordinates describe a Tanner layout without imposing local
hardware connectivity.

## Syndrome extraction and postselection

`HCodeExtractionBlock` is the default concurrent, unflagged schedule;
`HSixExtractionBlock` is its compatibility alias. It places shared data
qubits 2 and 3 at each check's ends. X interactions precede Z interactions
on each data qubit, giving the same ideal unitary as serial X-then-Z extraction.
The exact schedule is documented in [SE_block.py](SE_block.py).

| Per-round depth | Dedicated | Generic CSS coloration |
|---|---:|---:|
| CNOT layers | `n+2` | `2*(n-2)` |
| Including two H layers | `n+4` | `2*n-2` |
| Including reset and measurement | `n+6` | `2*n` |

The dedicated block uses system-global indices and supports independent
H-code patches with unmodified active checks. Select
`GenericCSSColorationExtractionBlock` explicitly to use coloration.
Its ordering can produce undetected single faults; the regression suite
includes an n=8 example.

[Fault-distance tests](../../../tests/test_H_code_fault_distance.py) check
the dedicated memory at n=6,8,12,20,32,64, both X/Z bases, 1–3 rounds, and
independent Pauli gate/reset/measurement noise with and without idle noise.
The undecomposed detector error model detects every single-fault logical
signature, including hyperedges; independent two-fault witnesses establish
distance two for those tested circuits.

Full postselection accepts only **all-zero detectors, including final readout**.
Conditional block LER counts **any logical observable flip** among accepted
shots. The dedicated schedule is a LightStim construction: its validated
memory boundary includes final readout. An individual round can leave an
unflagged weight-two data error, so these tests do not establish fault tolerance
for a state that will be passed to another protocol without readout.
Flagged preparation and logical-H measurement are separate protocol work.

## Logical operations and H6 encoding

`HCodeLogicalOpSet.transversal_hadamard(builder, patch)` applies logical H
to **all n-4 logical slots together**. It inherits inter-patch CNOT from
`CSSLogicalOpSet`. Operations require the global patch view returned by
`system.add_patch()` and use the builder to update detector tracking.

`HSixLogicalOpSet.encode` prepares two independent +1 X/Y/Z eigenstates
using the unflagged Fig. 1(d) encoder. It accepts `HSixCode()` or `HCode(n=6)`
and requires fresh data qubits:

```python
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.H_code import HSixLogicalOpSet

system = QECSystem()
patch = system.add_patch(HSixCode(), name="h6")
builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
HSixLogicalOpSet().encode(builder, patch, logical_bases=("X", "Y"))
builder.stabilizer_canonicalization()
builder.apply_syndrome_extraction(HCodeExtractionBlock(system).circuit, rounds=2)
```

`prepare_logical_xx/yy/zz` select equal bases for the two logical slots.
The single-letter `prepare_logical_x/y/z` names remain whole-patch API aliases.
`noiseless=True` uses the existing noise-injector tags.
The encoder has no fault-tolerant preparation guarantee.
`MemoryExperiment` uses product-state initialization and does not insert it
automatically.

## Benchmark and tests

Use the [unified memory runner](../../../benchmarks/memory/README.md#h-family-full-postselection-memory)
and its existing plotter:

```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes h_code --h-n 6 8 --mode full_postselection \
    --basis Z X --p-values 0.004 0.008 --max-shots 1000
venv/bin/python -m pytest tests/test_H_code.py tests/test_H_six_code.py \
    tests/test_H_code_ops.py tests/test_H_code_fault_distance.py -m "not slow"
```

The slow fault-distance tests additionally sample H6/H10 conditional LER
scaling under full detector postselection.
