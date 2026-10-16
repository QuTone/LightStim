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
```

Building the inverse-tableau kernel requires a C++20 compiler and the pinned
Stim source submodule. `pip` installs pybind11 into its isolated build
environment. Source distributions include the required Stim sources and license.
Both native extensions are optional: without a compiler or the Stim checkout,
the ordinary Python installation remains usable.

`LIGHTSTIM_PAULI_BACKEND` supports:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use the native algebra when installed, otherwise the reference implementation. |
| `reference` | Use the original decomposition and NumPy propagation. The existing GF(2) C++ RREF may still be used. |
| `native` | Require the native extension; raise an import error if it is unavailable. Noncommuting bases and greedy weight reduction still use the general solver. |

The default build uses the compiler's portable architecture target. A local
build can opt into `LIGHTSTIM_NATIVE_ARCH=native`. Do not distribute such a
wheel to machines with weaker instruction-set support. Set
`LIGHTSTIM_BUILD_JOBS` to control compilation concurrency (default two).

This is an algebra backend, not the `LIGHTSTIM_DETECTOR_BACKEND=record_tableau`
annotator from the original PR. That older annotator is not part of this
integration.

## Computation and preserved contracts

For a commuting basis, the engine anchors the earliest independent original
rows as generators `S_i = C Z_i C†`. Querying `C† P C` returns anticommutation
coordinates and decomposition coordinates. A target belongs to the supplied
span only when its X coordinates vanish **and** every used Z generator is
known. Completing an underconstrained tableau never supplies a new constraint.
The construction is unsigned, so redundant binary inputs such as `XX, ZZ, YY`
are valid even though their positive signed versions cannot all stabilize a
state.

Coefficients refer to the same original independent basis rows as the
reference solver. Dependent original rows remain available to the Tracker's
exact local-check preference. New target directions preserve the public
solver's existing component order. General noncommuting spaces and
`reduce_weight=True` retain the original solver, including its representative
choices. The native engine never receives measurement records.

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

The backend still constructs an inverse basis when the supplied space changes
without a reusable prefix.
Role intersections, readout closures, and logical canonicalization retain their
existing algebra. This integration therefore does not imply the original
record-tableau annotator's reported speedups: performance must be measured
against the current Tracker with equivalent outputs.

## Validation

`tests/test_pauli_backend.py` compares exact coefficients and extension indices
against the reference solver, including incomplete and redundant bases, cache
invalidation, arbitrary Pauli propagation, and machine-word boundaries.
Protocol differential tests compare emitted circuits and logical history.
CI runs the existing non-slow test suite with both backend settings, including
the bare/dressed, logical-history, postselection, and staged-readout regressions.

```bash
LIGHTSTIM_PAULI_BACKEND=reference python -m pytest tests/ -m 'not slow' --timeout=90
LIGHTSTIM_PAULI_BACKEND=native python -m pytest tests/ -m 'not slow' --timeout=90
python benchmarks/tracker/benchmark_pauli_backend.py --validate --output /tmp/pauli-backend.json
```

One local run (Python 3.10, Python Stim 1.15, native Stim 1.16, portable build,
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
