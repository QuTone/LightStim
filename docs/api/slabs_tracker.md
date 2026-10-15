# S, L, and L_absorbed tracking: algorithm and proof boundary

`SyndromeTracker(..., logical_tracking="slabs")` is an experimental extension of
the existing tracker. The default remains `"legacy"`. This document specifies the
implemented convention, the linear identities it uses, and unresolved correctness
questions. It is **not a proof of general detector/logical classification**.

## State and records

| Name | Implementation | Meaning |
|---|---|---|
| `S` | `tracker.stabilizers` | Current state constraints assigned stabilizer roles. |
| `L` | `tracker.logicals` | Live protected logical representatives. |
| `L_absorbed` | `tracker.absorbed_logicals` | Protected constraints absorbed into `span(S,L)`; aliases `absorbed_ops`. |

All three tables store a binary Pauli row and a list of absolute measurement
indices. `P{m0,m3}` means that P's eigenvalue depends on `m0 XOR m3`, with a fixed
Pauli phase omitted by the unsigned tracker. Row multiplication XORs both the
Pauli bits and the record sets. `P{}` has no measurement-dependent record parity;
its fixed phase is still treated separately.
These are the original tableau records, not another state representation.

The defining redundancy of `L_absorbed` concerns **Pauli support**, not the
augmented `[Pauli | records]` row. For example, `S=[ZZ{J}]` and
`L_absorbed=[ZZ{}]` have the same Pauli span, but their product gives the classical
relation `J`. Erasing one merely because its Pauli is dependent loses its role.
`logical_history` stores already discovered record relations; it is an output
ledger, not a fourth quantum tableau or a contribution to the live logical census.

Binary row identities prove deterministic supports modulo fixed phases. Signed
relations need separate phase-aware checks, such as Stim flow checks; reference
samples supply affine offsets when sampling detector and observable errors.

## Implemented A/B/C/D updates

The input state must be complete for the supported Clifford measurement block.
Paulis compared or combined must be in the same frame. A commuting measurement
outside the tracked state span is rejected. Unknown record sentinels cannot be
silently discarded when emitting a logical parity.

Mid-measurement / syndrome-extraction block:

```text
A. Compute back-propagated measurements M and the forward Clifford map F.
B. Project the joint rows [S; L; L_absorbed] through each measurement.
   Use the first anticommuting S row if any exists, otherwise L, then L_absorbed.
   Multiply it into every other anticommuting row, carrying record XORs.
   Replace the pivot by the measurement and its fresh record.
   For a commuting dependency: S-only -> DET;
   L/L_absorbed involvement -> logical_history.
C. RREF projected L within its own span, applying the full transform to records.
   Retain zero rows through this transform; projected L need not have full rank.
   Select surviving L modulo M; absorb omitted L along with old L_absorbed.
   Choose absorbed rows independent modulo surviving L.
   Refresh S from measured rows, or reset-seeded extraction rows, as appropriate.
D. Propagate all tables by F and substitute measured-ancilla records.
   Complete C's normalization in this common output frame:
     preserve the whole intersection span(L_absorbed) intersect span(S);
     eliminate surviving-L components from the complementary absorbed rows;
     choose representatives from existing ordered S rows for those directions;
     reconstruct their records from absorbed history and an S complement.
```

The last step does not copy the fresh record of a selected S row. Its record
expression follows the row combination that represents the protected constraint.
It changes only absorbed directions requiring L elimination and preserves the
whole intersection, including combinations absent from the original row list.

Readout of one supplied collection:

```text
A. Form its measured Pauli span M; obtain the active stabilizer declaration K.
B. Project the entire collection jointly with the same S-first rule.
C. Expose span(L,L_absorbed) intersect M before reducing support modulo M.
   In the quotient by M, select independent survivors, giving L priority.
   Keep absorbed combinations in span(K modulo M, surviving L), modulo L;
   release the complementary absorbed directions into L.
   Reconstruct known active S combinations from old S and its records.
   Preserve closing S combinations and explicitly postselected S rows.
D. Decompose the closing S/L rows into M and emit DET/OBS respectively.
   Fold remaining measured support with the same measurement-record XORs.
```

Builder derives K from declared active-check combinations with zero X/Z support
on retired or measured qubits, using a kernel calculation. The tracker does not
partition the input into corridor and patch measurements. K supplies code roles,
never an eigenvalue.
Consequently, an absorbed direction can remain protected inside a merged code
while a repeated SE round yields a detector; splitting can instead release it.
Reset uses joint elimination to retain only combinations with zero support on
reset qubits, then applies the original preparation path.

Implementation: [`tracker.py`](../../lightstim/ir/tracker.py), especially
`_slabs_absorb_projected`, `_slabs_normalize_absorbed`, and
`_slabs_release_projected`; Builder passes the active declaration in
[`apply_data_readout`](../../lightstim/ir/builder.py).

## What the linear algebra establishes

1. An invertible row transform T sends `(P,R)` to `(TP,TR)` and preserves its
   augmented binary row span. This justifies L-only RREF and record transport.
2. Replacing an absorbed row `a` by `a+l`, with the same record XOR, preserves
   `span(L,L_absorbed)` as augmented rows while L is retained.
3. For a measurement with commutation functional `chi` and pivot p,
   `h -> h + chi(h)*p` is the usual projection update for the other constraints.
   The full conditional physical stabilizer group is independent of the chosen
   valid pivot; this fact alone says nothing about its S/L classification.
4. Span intersection enumerates closing combinations even when no original row
   closes. If `p+p_folded = c*M`, its records must change by `c*m` as well.

These identities explain the implemented row operations. A general proof of
detector/logical classification is outside the scope of this experimental PR.

## Logical-count bookkeeping and a distance compatibility change

**Constructed-state logical-count check.** This is a low-level fixture that
directly populates the three tracker tableaus, not a reproduction of an existing
benchmark. Prepare `|00>`, with
`S=[ZI{}]` explicitly postselected, `L=[IZ{}]`, `L_absorbed=[ZI{}]`.
Read Z1: its logical closes, while release leaves `S=[ZI{}], L=[ZI{}]`.
Next mid-measurement X0 uses the S pivot, turning L into `I{}`. A final X0
readout gives complete parity rank 2, but the expected logical census remains
1 although both protected tables are empty. A prior Z0 round gives `I{m0}`;
that relation is already a detector, and the same census issue remains.
The zero-row tests preserve parity behavior; they do not claim this count is fixed.
Native state-injection, Cross-LS, and LS-distillation postselection are checked
separately by the protocol compatibility tests.
The same manually constructed overlapping S/L state also exposes the existing
legacy count assumption: replacing an S pivot can remove the coincident L row
without decrementing `expected_num_logicals`. Slabs release can expose that state
while preserving an explicitly selected S row. This is a bookkeeping integration
boundary; it is not evidence that native postselection protocols have regressed.
Cross-LS marks row-based postselection only before its terminal readout; state
injection uses detector-coordinate selection. Neither audited path performs the
constructed continuation after preserving a selected S/L overlap.

**Checkerboard: intentional promotion of joint consistency.** The compiler's distance-3
`checkerboard_3x3_ppm.json` prepares nine patches, performs distant ZZ, XX, and
XZX PPMs, then reads each patch in its preparation basis. With one SE round,
legacy gives 397 detectors, 9 observables, and graphlike distance 3; slabs gives
394 detectors, 12 observables, and distance 1. With three rounds, both have
distance 3. The physical noisy circuit is identical between tracker modes.

For this protocol, the independent logical oracle resolves the classification:
nine ideal qubits prepared in `ZXZZXXZZX`, the three PPMs, and matching final
readout give 12 measurements and 12 independent deterministic relations. This
matches the requested rule that ZZ initialization, ZZ PPM, and final ZZ readout
have three protected relations. The three directions removed from the legacy
detector space are exactly final joint-consistency relations. Writing J for the
protocol's `records_post_split` parity, the current annotation identities are:

| PPM | Legacy final detector | Exact parity in slabs output coordinates |
|---|---|---|
| ZZ on q00, q02 | D395 | J0 XOR O1 |
| XX on q01, q12 | D396 | J1 XOR O3 |
| XZX on q11, q20, q22 | D394 | J2 XOR O8 |

All three legacy detectors are emitted by final patch readout Step D. Slabs
releases absorbed constraints at the split's Step C and closes them as logical
outputs at final Step D. For example, the fully annotated identity is
`D395_legacy = D37_slabs XOR D38_slabs XOR D40_slabs XOR D41_slabs XOR O1_slabs XOR O9_slabs`,
where O9 is the first PPM's logical-history parity.

An encoded X on q00 immediately after the first split, before the second PPM,
is `X1 X7 X13` and commutes with all 72 active checks. It triggers legacy D395
and O0, but no slabs detector and slabs O0/O1. The independent ideal fault flips
only the final q00 result. Thus legacy D395 is not a pure detector under the
logical-stage response criterion used by the acceptance tests. Its promotion
is consistent with the requested logical-output semantics.

This is not solely an added-target effect: retaining slabs detectors but replacing
all outputs with the exact nine legacy observable parities still gives distance 1.
The one-fault witness is `X214` at coordinates `(6,2)` after 8 TICKs; it flips the
first logical-history target (legacy L6 / slabs L9) without a slabs detector event.
It triggers legacy D395. The old distance used a joint-consistency check as a
detector. Under the requested roles, one SE round leaves this logical measurement
fault unprotected; three rounds restore distance 3 in the audited circuit.
Both annotation sets have valid, complete physical parity spaces, so that check
alone would not have resolved their different roles.
See [reproduction evidence](../evidence/slabs_checkerboard_distance.json) and the
[passing semantic regression](../../tests/test_slabs_distance_contract.py).

## Exit criteria

An **experimental engineering milestone** requires the complete original test
suite and declared benchmark/protocol checks on the PR head, green required CI,
and explicit regression coverage for the seven examples, split versus held
merges, overlapping three-patch sequences, initialization/final-basis variants,
partial/combined readout, reset, and postselection. Validation must check full
record-space rank, signed/reference behavior, logical fault response spaces,
and the census where supported—not merely observable counts or noiseless DEMs.
CI results are reported separately from this algorithm specification. The
default must remain legacy.

A **general classification claim** additionally requires a precise invariant
defining detector and protected roles, preservation under equivalent row bases
and permitted measurement regroupings, correct zero-Pauli record handling and
PS/census updates, and an inductive proof for every A/B/C/D transition. The
explicit checkerboard distance change must remain documented; it is not repaired
by restoring a check with logical response. Protocol-matrix coverage is empirical
validation and does not replace such a proof.

Relevant executable checks include [`acceptance`](../../tests/test_slabs_acceptance.py),
[`absorb normalization`](../../tests/test_slabs_absorb_normalization.py),
[`partial readout`](../../tests/test_slabs_partial_readout.py), and
[`three-patch oracles`](../../tests/test_slabs_three_patch_matrix.py),
[`unrotated matrix`](../../tests/test_slabs_unrotated_matrix.py), and
[`strict known-limit checks`](../../tests/test_slabs_known_limits.py).
