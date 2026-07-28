# Tracker Measurement Semantics

## Status

Implemented on 2026-07-28. The retained-data measurement path and the
Builder/Tracker boundary are covered by Color Code and cross-protocol
regressions.

## Required Boundaries

LightStim should distinguish these concepts explicitly:

1. **Physical measurement block**: input resets followed by a Clifford circuit
   ending in one commuting measurement layer. This is the atomic input to a
   tracker state update.
2. **Syndrome-extraction round**: a Builder-level protocol grouping of one or
   more physical measurement blocks.
3. **Tracker measurement update**: infer measurement-record relations and
   update the full stabilizer/logical state from the physical block. It must not
   decide whether a protocol-level round has ended.
4. **Canonicalization checkpoint**: an optional Builder-requested change of
   tableau basis. It is a representation/compression operation, not a
   prerequisite for measurement correctness.

The tracker may own the GF(2) implementation of canonicalization, but the
Builder owns the decision to invoke it.

## Implemented Boundary

`SyndromeTracker.process_mid_measurement(...)` now always processes one
physical measurement block. It returns the exact stabilizer rows that Gaussian
elimination found eligible for later logical classification, but it does not
decide whether to classify them.

At the end of its declared measurement-block group, `CircuitBuilder` chooses
one of three explicit actions:

- a single disposable-ancilla block promotes the eligible rows with
  `promote_stabilizer_rows_to_logicals(...)`;
- a multi-block disposable-ancilla round requests the record-preserving
  `rebase_stabilizers_onto_code_basis(...)`;
- a retained-data round keeps its physical state basis and validates the
  logical count after the block group.

The old `finalize_logicals` argument and
`finalize_composite_syndrome_measurement(...)` protocol-shaped Tracker method
have been removed.

The measurement-block implementation now distinguishes:

- terminal measurement targets;
- qubits explicitly reset before the physical readout;
- QEC-patch syndrome-ancilla roles;
- measured data representatives whose projected eigenvalue is carried by a
  prior measurement record.

Only QEC-patch syndrome ancillas are discarded from the tracked output state.
Measured data representatives remain part of that state. A terminal
measurement target is not automatically a removable syndrome-ancilla factor.

The implemented retained-data path applies input resets, projects the terminal
measurements, and forward-propagates the resulting full state. Middle-Out
representative-qubit preparation belongs to the extraction block, so memory
initialization prepares only passive data qubits. Builder-side code-frame
discovery finds a physical Clifford prefix where all active code stabilizers
enter the tracked span, requests record-preserving canonicalization there, and
maps the classified state back to the measurement-block input. This keeps
intermediate code declarations out of `QECPatch`.

Repeated-round compression explicitly analyzes two rounds after the first two
physical rounds. When a valid detector basis remains tied to fixed historical
records, the Builder multiplies corresponding adjacent-round detectors to
remove those anchors. It emits a `REPEAT` body only after the resulting
detector body and tableau transition are periodic.

## Verified Criteria

- Tracker measurement updates contain no SE-round finalization policy.
- Builder explicitly groups physical blocks and owns state classification,
  canonicalization, and repeated-round compression.
- Existing surface-code, lattice-surgery, Bell-multiplexing, Bell-flagging,
  time-multiplexing, and Middle-Out regressions retain their validated
  behavior.
- Middle-Out remains inferred from physical measurement-boundary blocks
  without an intermediate `QECPatch` declaration.
