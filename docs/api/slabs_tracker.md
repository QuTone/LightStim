# Bare/dressed S, L, and L_absorbed tracking

`SyndromeTracker(...)` uses this tracker by default. The existing mode name is
`logical_tracking="slabs"`; ordinary experiments need no mode argument.
`logical_tracking="legacy"` is an explicit historical comparison control.

This document describes the v2 algorithm and its integration with LightStim's
physical measurement blocks. It specifies the intended role spaces and the
implemented row operations. Test coverage and CI status must be reported for a
particular revision; this specification does not assert that every benchmark has
passed or constitute a general correctness proof.

## State, records, and the current absorbed quotient

| Symbol | Storage | Meaning |
|---|---|---|
| `S_bare` | Rows of `tracker.stabilizers` without the dressed marker | Known constraints assigned error-detection roles by the current code. |
| `S_dressed` | The marked rows of that same S tableau | Remaining S directions carrying protected information. |
| `L` | `tracker.logicals` | Live logical-state constraints. |
| `L_absorbed` | `tracker.absorbed_logicals`, alias `absorbed_ops` | Representatives of the **current** dressed quotient modulo `S_bare` and L. |

Thus S remains one tableau, partitioned into bare and dressed rows. There is no
additional physical-state tableau. All tables use binary `[X | Z]` Pauli rows
and absolute measurement-record lists. Multiplying two rows XORs both their
Paulis and their record lists. For example, multiplying `Z1 Z2 {m0}` by
`Z2 {m1}` gives `Z1 {m0 XOR m1}`.

A row's empty record list is an initialized value, not evidence that it is bare.
The implementation does not store general constant Pauli phases in these binary
rows. Signed Stim flows, reference behavior, and physical fault responses are
therefore part of validation, in addition to binary span checks.

Let `span(...)` refer to Pauli row span over GF(2), and let K be the active
code's role space in the current physical frame. At a classification boundary,
using **known** rows only:

```text
span(S_bare) = span(S) intersect K
span(S)     = span(S_bare, S_dressed)

span(S_bare, L, L_absorbed) = span(S_bare, L, S_dressed)
rank(L_absorbed modulo span(S_bare,L)) = number of L_absorbed rows
```

`L_absorbed` is rebuilt by selecting current `S_dressed` rows independent modulo
`span(S_bare,L)`. Their current records are copied with them. It does not retain
an older eigenvalue history when that direction becomes bare, and it is never a
source for detector or observable emission. In particular, a fully established
merged-code check can be bare with **no absorbed row**, even though its first
measurement generated a logical observable.

Previously discovered record-only logical relations live in `logical_history`.
This is an output ledger, not a quantum tableau and not part of the current
protected rank. It preserves those relations independently of later changes to
S, L, and `L_absorbed`.

Some preparation interfaces retain unknown code placeholders with record `-1`.
They cannot supply an eigenvalue or an emitted relation. They are excluded from
known-span intersections; multiplying two unknown placeholders must not cancel
their uncertainty merely because both use the same sentinel.

## Classification by active checks

K supplies **roles, not values**. An active declaration outside the known state
must not become an empty-record state constraint. The classifier computes the
whole intersection, including combinations absent from the supplied row list:

```text
H = reduce S modulo K
C = kernel(H transpose)
B = C S                           # span(B) = span(S) intersect K
records(B) = C records(S)          # the same XOR coefficients
```

Preserve every original S row already in K, in its existing order, including
redundant local checks. Append independent rows of B only to complete any missing
bare directions. Then select a complement from the original S rows as
`S_dressed`. Explicitly selected postselection rows are retained with their
exact record expressions. Rank, rather than number of stored S rows, defines a
subspace's dimension.

For example, with `S=[Z1Z2{a}, Z2{b}]` and `K=span(Z1)`, neither input row is
individually bare, but their product is. The bare constraint is `Z1{a XOR b}`;
K cannot supply a fictitious empty record expression for Z1.

A dressed Boolean flag is not combined with OR during multiplication. Products
of dressed representatives can be bare. Reconstructing the intersection and
complement determines the roles, while the elimination coefficients transport
records and row metadata.

## The five atomic operations

### 1. Initialization and reset

Builder emits reset gates and constructs their preparation rows in X, Y, Z
order: `RX -> X{}`, `RY -> Y{}`, `R -> Z{}`.

Reset first forgets the previous state on the touched qubits: joint elimination
on `[S;L;L_absorbed]` retains combinations with zero X and Z support there.
Fresh preparation rows are then added to S. This is a partial trace followed by
preparation, not a measurement whose outcome can be substituted into records.
Surviving spectator constraints retain their record expressions.

Fresh physical preparation is not automatically an encoded logical state. The
existing code-preparation boundary classifies initialized constraints against the
code, establishing L and code checks. Auxiliary syndrome and zero-logical-coupler
preparations remain auxiliary. A code declaration alone does not initialize an
unknown check.

### 2. Mid-measurement: A, B, C, D

A physical measurement block has an input boundary, Clifford evolution, and a
terminal commuting measurement collection. Its measured qubits are either
retained in the tracked quantum state or disposable syndrome ancillas. These
are physical block properties, not inferred patch/corridor ownership.

**A — Back propagation.** Builder derives the input-frame measured Paulis M,
the forward Clifford map F, and any actual input reset rows R. Known reset
factors are removed when deriving M. A measurement reducing to identity is a
flag with a deterministic record relation.

**B — Projection and relation extraction.** Operate on
`[S_bare; S_dressed; L; L_absorbed]` with that pivot priority. Mid-measurement
keeps the established stable empty-record-first ordering within each S role;
this is a representative preference, not a role definition.

For an anticommuting measured Pauli, multiply the pivot into every other
anticommuting row, carrying record XORs, then replace the pivot by the measured
Pauli with its fresh record. All three stored tableaus participate. An
`L_absorbed`-only pivot is an invariant violation: its Pauli is redundant in S/L.

For a commuting measurement `M_i`, test these known spans in order:

```text
span(S_bare, M_0, ..., M_(i-1)) -> detector relation
span(S_bare, S_dressed, L)       -> logical relation, archived in logical_history
otherwise                      -> no deterministic relation established
```

The previous measurements carry their actual fresh records. Their inclusion is
necessary even if the first random measurement replaced an L row. For example,
two consecutive measurements of initially random ZZ produce the detector
`J0 XOR J1`; the second must not become a new logical relation merely because
the first temporarily occupied an L slot. A known identity measurement is
handled directly. Unknown placeholders never provide missing values.

Within the detector space, prefer an exact known bare check, then an exact
earlier measurement, before general decomposition. This preserves the local
adjacent-round parity of an existing check, including redundant checks. Merely
preserving the total relation span can otherwise replace local detectors by
large products and make an existing graphlike decoder incompatible.

**C — Refresh once in the input frame.** RREF the projected L within its own
span, carrying the full row transform into its records. Then select an ordered
independent complement exactly once:

```text
basis:    M
candidates, in priority order: [canonical L; projected S]
```

Selected L rows remain L; selected old S rows remain S. For retained measured
qubits, S starts with an independent subset of M carrying the new outcomes,
followed by the selected old S rows. For disposable syndrome ancillas, temporary
S starts with the reset rows R, followed by the selected old S rows. R means the
actual `RX/RY/R` preparation constraints, not random seeds or another tableau.
Projected `L_absorbed` rows remain redundant during the frame update; they do not
supply extra physical directions to this selection.

**D — Forward propagation and output classification.** Apply F to S, L, and
`L_absorbed`. For disposable ancillas, remove each measured Pauli factor and XOR
its terminal measurement record into every affected row. Remove reset-derived
identity rows whose flag relation was already handled.

At the declared code boundary, classify S by the output-frame K, then rebuild
`L_absorbed` from the current dressed quotient. D does **not** select surviving
L a second time. The C complement selection and the output role classification
answer different questions: which physical constraints survive the measurement,
and which of those constraints belong to the continuing code.

Builder owns the code-boundary policy. Single ordinary extraction rounds can
promote established unclassified preparation directions; multiple disposable
blocks rebase at the complete round boundary. Retained-data extraction may need
to identify a code frame inside the Clifford block. Subsystem schedules retain
their existing gauge-aware classification between physical blocks.

### 3. Destructive readout: B, C

Builder emits the requested single-qubit measurements in X, Y, Z order. Within
one declared tracker stage there is one joint projection, not a hidden sequence
of corridor and data operations.

The low-level readout input must contain independent commuting Pauli rows.
Builder's single-qubit measurements on distinct qubits satisfy this condition.
Repeated or dependent check collections belong to mid-measurement; destructive
readout rejects them before updating state or allocating measurement records.

**B — Project, then extract the full closure spaces.** Use the same
bare-before-dressed-before-L pivot priority, updating `L_absorbed` jointly but
excluding it from relation extraction. A pivot replaced by this readout is
marked destroyed and cannot close its old relation.

For the surviving physical rows `T=[S_bare;S_dressed;L]`:

```text
D_close = span(S_bare) intersect span(M)
C_close = span(T)      intersect span(M)
O_close = a complement of D_close in C_close
```

Carry coefficients on both T and M to construct the record parities. Preserve
original readable bare rows first, including redundant local checks, then add
any missing intersection directions. Observable representatives prefer readable
original dressed/L rows before completing the quotient with products.

Testing each original row for membership in M is insufficient. A native example
is two Z-initialized unrotated surface-code patches followed by transversal
CNOT B→A. L can contain `ZA ZB` and `ZB`; neither individually closes when only A
is Z-read, but their product `ZA` does. Full intersection emits that logical
readout immediately.

Pauli-zero combinations can still carry nonempty record relations. Close them
before dropping dependent state rows. Relations already emitted or archived
must keep their established role and must not be counted again. Explicit
postselection acceptance relations are preserved even when they also occur as
logical observables.

**C — Eliminate measured support and classify the remainder.** If a row changes
from p to `p_folded` with `p XOR p_folded = c M`, XOR `c records(M)` into its
records. Do this consistently for surviving physical constraints and selected
postselection constraints. Remove closed/zero physical directions, preserve the
selected constraint subspace, and classify remaining S using the continuing
role space. Rebuild `L_absorbed` from the resulting current `S_dressed` and L.
There is no absorbed-release operation.

Use the **pre-readout roles** for B. The continuing K supplied to C is not an
instruction to relabel all input checks before emission: at complete final
readout the continuing K is empty, which must not turn all final detectors into
observables. Builder obtains continuing declared combinations with zero support
on retired qubits by a kernel calculation. The tracker also reduces that role
space modulo the measured Pauli span for independent commuting readout inputs.

### 4. Unitary blocks

Conjugate S, L, and `L_absorbed` by the same Clifford symplectic map. Their
measurement records do not change under this frame update. The cached role
space is conjugated into the same frame. Register expansion must widen both
halves of every `[X|Z]` matrix, including the role-space cache.

### 5. Coupler activation, deactivation, and explicit readout stages

Activation/deactivation changes the declared code and its next classification
boundary. It does not itself measure a logical value or invent a stabilizer
value. The physical protocol must state where a split occurs.

A protocol can declare multiple tracker stages inside one physical readout
layer:

```python
from lightstim.ir.builder import DataReadoutStage

builder.apply_staged_data_readout(
    final_measurements,
    stages=(
        DataReadoutStage(tuple(corridor_qubits), ("AB",)),
        DataReadoutStage(tuple(patch_data_qubits)),
    ),
)
```

The stages must partition the supplied qubits. The first stage explicitly
deactivates AB and tracks its selected measurements; the next tracks patch
readout in the resulting state. Builder emits the complete dictionary's original
X/Y/Z gate order once, without adding a stage TICK or moving physical gates.
Absolute measurement indices connect the physical order to the declared tracker
order. The tracker does not infer a corridor or a coupler from measurement
ownership. An unstaged call is a single declared readout stage.

A held merge followed by another SE round therefore compares current active
checks and produces adjacent-measurement detectors. A declared split followed
by a new merge is another logical readout. These are different protocol
boundaries even when some physical measurement gates can share one layer.

## Postselection and subsystem codes

Postselection denotes a physical signed constraint subspace, not an OR-combined
row attribute. Projection keeps the part of this subspace commuting with the
measurement collection. Row changes, forward propagation, measured-factor
substitution, and reset tracing carry its records with the same coefficients.
Readout computes its complete closure intersection and preserves the selected
parity space across later stages. A selected acceptance relation can coexist
with a logical observable; tagging acceptance must not erase the observable's
protected role.

The row-space helpers retain exact selected S representatives and return
lineage for metadata transport. Mixed selected/unselected ancestry is not enough
to mark an output row selected. Coordinate-based protocol selection remains a
separate supported input.

For subsystem codes, the active role space includes the declared gauge group G,
not only the stabilizer center. The existing classifier constructs known gauge
constraints from the physical state intersected with G, and protected logical
constraints from the state commuting with G. Preparation constraints remain
explicit until the center is established. A changing gauge frame is not a new
protected logical direction. K itself need not be a simultaneously known state;
only its intersection with the actual known state supplies bare constraints.

## Dynamic rank and a small logical example

The live protected rank is

```text
k_live = rank(span(S_bare, S_dressed, L)) - rank(span(S_bare))
```

The compatibility field `expected_num_logicals` exposes this dynamic rank in v2.
It is not a conserved initial logical-qubit budget and is not the number of
historical logical observables. Code preparation retains the separately declared
logical count for its own preparation checks. Only when L is independent modulo
`S_bare` does `L.count + rank(L_absorbed)` equal `k_live` by simple row counting.

For two logical X preparations, followed by ZZ measurement J, an explicit split,
and final Z readouts:

| Boundary | Bare/dressed ZZ role | L | L_absorbed |
|---|---|---|---|
| Before ZZ | No measured ZZ constraint | `XA`, `XB` | Empty |
| Merged code after ZZ | `S_bare` contains `ZA ZB {J}` | Surviving `XA XB` representative | Empty |
| After split | `S_dressed` contains `ZA ZB {J}` | `XA XB`, with any physical split-frame records | Current dressed ZZ representative with record J |
| Final Z readout | ZZ closes | X-type L is destroyed | Not used for emission |

The result is exactly one protected relation, `zA XOR zB XOR J`. For ZX input,
`zA=0` supplies a second independent relation. For ZZ input, the first measurement
also supplies the historical relation `J=0`, giving three protected relations
across the protocol. Equivalent emitted bases may differ by detector or other
already recorded relation combinations; the physical quotient is the contract.

## Implementation and validation boundary

The relevant implementation is split by responsibility:

- [`tracker.py`](../../lightstim/ir/tracker.py): physical initialization/reset,
  mid-measurement B/C/D and Clifford propagation; `_v2_refresh_complement` is the
  single C complement selection.
- [`tracker_roles.py`](../../lightstim/ir/tracker_roles.py): bare/dressed roles,
  current absorbed representatives, and dynamic rank.
- [`tracker_algebra.py`](../../lightstim/ir/tracker_algebra.py): full intersections,
  retained local representatives, and exact row/record lineage.
- [`tracker_readout.py`](../../lightstim/ir/tracker_readout.py): joint projection,
  closure extraction, postselection transport, and measured-support elimination.
- [`builder.py`](../../lightstim/ir/builder.py): physical block analysis, code
  boundaries, role-space declarations, and explicit staged readout.

Regression coverage includes [tableau operations](../../tests/test_tracker_v2_algebra.py),
[projection and reset](../../tests/test_tracker_v2_projection.py),
[readout and postselection](../../tests/test_tracker_v2_readout.py),
[native protocol outputs](../../tests/test_tracker_v2_native.py), and
[physical readout staging](../../tests/test_staged_data_readout.py).
Existing protocol tests also exercise state injection, distillation, CrossLS,
and subsystem codes with the default tracker.
