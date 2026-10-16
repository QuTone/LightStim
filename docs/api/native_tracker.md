# Native algebra for the bare/dressed Tracker

The optional native backend accelerates the existing `SyndromeTracker`.
Its detector rules, record lineage, bare/dressed classification, logical
history, postselection, and staged destructive readout remain those described
in [the Tracker specification](slabs_tracker.md).

## Installation and selection

```bash
git submodule update --init third_party/stim
pip install -e .
LIGHTSTIM_PAULI_BACKEND=native python your_experiment.py
# Opt into persistent physical-state and record tracking:
LIGHTSTIM_PAULI_BACKEND=native LIGHTSTIM_TRACKER_BACKEND=persistent python your_experiment.py
```

Building the inverse-tableau kernel requires a C++20 compiler and the pinned
Stim source submodule. `pip` installs pybind11 into its isolated build
environment. Source distributions include the required Stim sources and license.
Both native extensions are optional: without a compiler or the Stim checkout,
the ordinary Python installation remains usable.

A wheel built with the extensions contains the compiled backend, so its users
do not need a compiler, pybind11, or a Stim source checkout. The repository does
not yet publish a complete platform/Python wheel matrix; source installs need
the build tools above to obtain acceleration. Public Python Stim is still a
normal dependency, and no private ABI is shared with its installed extension.

`LIGHTSTIM_PAULI_BACKEND` supports:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use the native algebra when installed, otherwise the reference implementation. |
| `reference` | Use the original decomposition and NumPy propagation. The existing GF(2) C++ RREF may still be used. |
| `native` | Require the native extension; accelerate general binary subspaces, lineage, records and Pauli propagation. Greedy coefficient-weight reduction retains its reference policy. |

The default build uses the compiler's portable architecture target. A local
build can opt into `LIGHTSTIM_NATIVE_ARCH=native`. Do not distribute such a
wheel to machines with weaker instruction-set support. Set
`LIGHTSTIM_BUILD_JOBS` to control compilation concurrency (default two).

This is an algebra backend, not the `LIGHTSTIM_DETECTOR_BACKEND=record_tableau`
annotator from the original PR. That older annotator is not part of this
integration.

`LIGHTSTIM_TRACKER_BACKEND=reference` (default) uses the existing execution
structure with whichever algebra backend is selected. `persistent` requires
the new native extension and uses a persistent inverse frame for the slabs
Tracker's physical state and record expressions. It raises a clear import
error if unavailable; it does not silently disable the engine. Explicit legacy
logical tracking retains its existing Python implementation.

## Persistent state and records

The opt-in `PersistentRecordState` adapts PR #110's inverse-frame evolution,
generator re-basing, and record-parity arithmetic to the merged Tracker's
policies. Clifford gates update the same inverse tableau. Measurement queries
use its coordinates; projection updates source-row coordinates and records in
C++, using the pivot chosen by the current Tracker. Redundant rows retain their
individual records, and unknown placeholders cannot cancel into known values.
Measured ancilla factors are replaced by their records in the native engine.

The current Tracker specifies bare/dressed classification, exact-check
preference, logical-history capture, postselection subspaces and staged readout.
Their native execution uses packed source coordinates and lineage. A role or
gauge declaration can be noncommuting, so it uses the full inverse-frame X/Z
coordinates; it is never inserted as an extra conditional-state constraint.
Intersections are constructed from dependencies of source residues, retaining
the original-source coefficients. DET, OBS and postselection closures share
the same measurement basis instead of independently reducing physical matrices.

The inverse engine is authoritative during native measurement blocks.
Projection, relation queries, logical canonicalization, basis refresh,
ancilla-record folding and row selection execute without a dense Python-state
round trip. Classification and absorbed-quotient reconstruction apply source
lineages directly to generator coordinates. Destructive readout computes its
closures there too; physical detector supports are reconstructed from measured
rows. Native packed physical views remain necessary where the public contract
chooses a particular physical-column representative, such as logical RREF and
destructive support reduction.

Public `PauliTableau.matrix` and `records` remain writable. Native-backed banks
materialize those views on demand; exposed or replaced data is reconciled at
the next native boundary. Unexposed banks can reuse the engine directly.
Python still manages Builder boundaries, role metadata, history and annotation
emission. The preserved semantics do not require the previous repeated dense
elimination, and this implementation removes that work rather than attributing
it to a semantic lower bound.

Measurement and destructive-readout transitions clone the native inverse frame
and records before projection, then commit the completed output. These
transaction snapshots preserve the reference path's temporary-tableau behavior:
a later record error cannot expose a partial projection through public banks.
They copy the existing frame directly, without re-importing physical matrices
or rebuilding its basis. Persistent tracking therefore describes the retained
frame and source lineage, not one immutable Python object identity.

Deep copies have independent inverse frames and records. Register expansion
creates an engine for the new width on the next native operation. Diagnostic
`tracker._record_state.stats` counts constructions, reconciliations, native
gates, queries, projections, folds, coordinate operations and physical exports
for the current engine. A zero persistent `decompositions` count does not mean
the engine is idle: exact checks and bare-plus-measurement-prefix relations
have their own native paths and counters.

Neither backend changes Builder's declared measurement blocks, classification
boundaries, or repeat compression. Differential tests compare intermediate
state/records, role metadata and history as well as exact emitted circuits.

## Computation and preserved contracts

For a stateless commuting basis, the engine anchors the earliest independent original
rows as generators `S_i = C Z_i C†`. Querying `C† P C` returns anticommutation
coordinates and decomposition coordinates. A target belongs to the supplied
span only when its X coordinates vanish **and** every used Z generator is
known. In the persistent engine, classification and row recombination can
replace those individual generators with products. Membership then requires
that the target's Z coordinates lie in the span of the actual retained source
coordinate rows; generator presence alone is insufficient. For example, a
known `Z1*Z2` product does not make `Z1` and `Z2` separately known. Completing an
underconstrained tableau never supplies a new constraint.
The construction is unsigned, so redundant binary inputs such as `XX, ZZ, YY`
are valid even though their positive signed versions cannot all stabilize a
state.

Coefficients refer to the same original independent basis rows as the
reference solver. Dependent original rows remain available to the Tracker's
exact local-check preference. New target directions preserve the public
solver's existing component order. General noncommuting spaces use packed
binary bases with the same earliest-source coefficient convention.
`reduce_weight=True` retains the original representative-selection policy.
Native lineage transformations apply the same record parity rules, including
the distinct treatment of unknown placeholders and Pauli-zero record relations.

Pauli frame propagation uses a packed XOR product instead of dense unsigned
integer matrix multiplication. This includes stabilizers, live logicals,
absorbed representatives, active role spaces, and selected postselection
constraints. Exact-check and known-row lookups are reused within a measurement
block and invalidated after every projection, without changing which original
row wins.

Inverse bases are cached by normalized matrix **contents**, including shape,
with limits of eight entries and 64 MiB of accounted storage. In-place tracker
updates cannot reuse stale decompositions. Oversized bases are used without
being retained. Unsupported bases are remembered within the same budget.
When a new basis strictly extends a cached row prefix, the engine copies that
tableau and anchors only the appended rows. Existing cache entries remain
immutable, so branches with different suffixes cannot contaminate one another.

The general packed-basis cache likewise reuses growing source prefixes while
keeping earlier branches immutable. Native Tracker operations use persistent
coordinate handles directly whenever possible. The mathematical operations
and representative choices remain those of the current Tracker; their storage
and execution have changed. Performance comparisons must still use equivalent
constructors, outputs, native GF(2) baselines and timing scopes.

## Repeated rounds and measurement boundaries

Both algebra backends preserve the same Builder scheduling and Tracker
transition boundaries; the persistent implementation executes those
transitions through native coordinate operations. Each
declared physical measurement block receives its own tracker update; Builder
classifies/rebases the state after the complete syndrome-extraction round.
Subsystem classification also runs at each physical block boundary. Selecting
`native` preserves these boundaries and the existing Stim `REPEAT` packaging.

Builder supplies the original Clifford circuit chunk for every native
measurement transition. Direct calls to `process_mid_measurement` that supply
only a symplectic matrix and omit `circuit_chunk` retain the compatible matrix
execution path, while using the selected algebra backend.

Builder can advance a verified single-logical steady state after evaluating
two probe rounds. This shortcut requires stable detector and state transitions,
no logical history or postselection, and no accumulated logical-record delta.
When the shortcut is unavailable, Builder executes every remaining round and
may still pack byte-identical emitted bodies into `REPEAT`. Output compression
therefore does not imply that repeated Tracker computation was skipped. In
particular, nonempty logical-record deltas retain explicit updates so a later
history relation can access the complete record expression.

The original PR #110 annotator used a separate `_repeat` path: it replayed every
iteration on its record tableau, then repacked identical annotated bodies or
periodic detector differences. That annotator is not used here. Reference and
native measurements should use the same round counts and measurement grouping;
memory cases with verified state advancement do not establish the speedup for
logical protocols that must execute each round.

## Validation

`tests/test_pauli_backend.py` compares exact coefficients and extension indices
against the reference solver, including incomplete and redundant bases, cache
invalidation, arbitrary Pauli propagation, and machine-word boundaries.
Protocol differential tests compare emitted circuits and logical history.
`test_native_role_algebra.py` compares exact role/readout lineage on arbitrary
subspaces and under invertible changes of coordinates. Frozen #124 oracle
fixtures in `test_full_tracker_semantic_oracle.py` compare complete boundary
states and emitted circuits, independently of the modified reference path.
CI runs reference algebra, native algebra and the persistent Tracker, including
the bare/dressed, logical-history, postselection and staged-readout regressions.

```bash
LIGHTSTIM_PAULI_BACKEND=reference python -m pytest tests/ -m 'not slow' --timeout=90
LIGHTSTIM_PAULI_BACKEND=native python -m pytest tests/ -m 'not slow' --timeout=90
LIGHTSTIM_PAULI_BACKEND=native LIGHTSTIM_TRACKER_BACKEND=persistent python -m pytest tests/ -m 'not slow' --timeout=90
python benchmarks/tracker/benchmark_pauli_backend.py --validate --output /tmp/pauli-backend.json
```

The following is a **historical kernel-only run**, before the persistent
role/closure engine, and is not a performance result for the current source.
That local run (Python 3.10, Python Stim 1.15, native Stim 1.16, portable build,
C++ RREF enabled in both modes) produced the following cold-cache medians of
three trials. Each uses unrotated memory with `rounds=d`; all emitted circuits
were byte-identical and passed DEM/noiseless checks.

| Distance | Reference | Native | Speedup |
| --- | --- | --- | --- |
| 3 | 15.4 ms | 13.7 ms | 1.12× |
| 7 | 186 ms | 115 ms | 1.62× |
| 11 | 2.24 s | 0.711 s | 3.15× |
| 15 | 10.94 s | 3.31 s | 3.31× |

Both modes include the shared per-block lookup optimization. These measure
construction plus circuit building, not sampling/decoding throughput, and are
not a reproduction of PR #110's older Table 3 baseline. Re-run the benchmark
for the hardware and protocol of interest.

## Attribution

The inverse-tableau query and generator re-basing are adapted from
**Hou Yuewen (`isaachyw`, `isaachyw@umich.edu`)**, who contributed
[PR #110: refactor: add stim record-tableau detector backend](https://github.com/QuTone/LightStim/pull/110).
The source contribution is commit
[`37df61c`](https://github.com/QuTone/LightStim/commit/37df61c), with PR head
`b3a949f`. The native build setup also derives from that contribution. Both are
used under this repository's Apache-2.0 license; vendored Stim retains its own
license.

The integration adapts those computational primitives to the Tracker merged
in PR #124. Preserve the Hou Yuewen co-author trailer and PR #110 attribution
when squashing or merging the integration.
