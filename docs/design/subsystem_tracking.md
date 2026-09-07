# Subsystem gauge-state inference

An alternating gauge schedule previously reached the correct physical
conditional state, then failed during code-state classification. A round
ending in Z gauges was rebased onto the patch's static centre S; surviving
gauge-fixed constraints were counted as additional protected logicals.
Declaring all noncommuting gauges as stabilizers instead made that rebase
demand X constraints destroyed by the Z measurements.

The physical measurement update already projects anticommuting Paulis and
forward-propagates the state. This extension changes its classification
boundary and adds an explicit gauge declaration. It keeps the existing
QECPatch, QECSystem, SE circuits, measurement blocks and atomic operations.

## Fixed declaration and derived state

A subsystem patch declares once:

- `stabilizers`: generators for the centre S;
- `gauges`: generators for G, which can be redundant or noncommuting;
- `num_logicals`: the number k of protected qubits, excluding gauge qubits;
- the existing geometry and optional logical representatives.

Registration checks S is the full centre of G and
`k = n - (rank(G) + rank(S))/2`, using data qubits only. These are Pauli
spaces modulo phase. A declaration does not assert a known eigenvalue;
initialization and the actual Stim circuit establish that information.

The SE circuit still specifies what is measured and in which order. Each
commuting physical measurement block drives the existing Pauli update. No
per-round active-stabilizer list, transition object or manual outcome mapping
is required. `system.active_gauges` exposes the fixed available gauge
inventory, including operators that were not measured in the current phase.

Let T be the row space of the complete tracked conditional state, combining
the tracker stabilizer and logical tables. The known gauge-fixed constraints
are **A = T ∩ G**. This is an intersection of spans: a product of two rows
can lie in G even when neither row individually does. Each inferred row
retains its coefficients over T; those coefficients XOR its absolute
measurement records as well.

Protected logical-state constraints must commute with every gauge. The
classifier therefore finds **B = T ∩ G^perp** on the data register and picks
independent representatives modulo A. Once the centre is known,
`A ∩ B = S`, so this is the known bare-logical space `B/S`. This distinction
matters for logical/gauge correlations: T/A alone is not always a collection
of initialized protected logical states. The classifier preserves valid
existing logical representatives where possible. Observable IDs follow the
resulting tracker basis; declaration `logical_id` metadata does not itself
choose the emitted observable basis.

During initialization, T may not yet contain all of S. Remaining physical
preparation constraints stay in the stabilizer bank and continue through
the next measurements. No unmeasured stabilizer placeholders are inserted.
Once all of S is known, the classifier validates k protected logical-state
constraints. Final data readout requires the centre to be established.
`infer_gauge_fixed_stabilizers(system)` returns A as a read-only derived
snapshot; during preparation, it can be smaller than the full stabilizer bank.

For a standard stabilizer code, mathematically G=S. In the implementation,
an empty `patch.gauges` selects the original pipeline unchanged. In a system
containing both kinds of patches, ordinary active stabilizers are included
in the effective gauge span.

## Integration and validation

`GenericCSSGaugeExtractionBlock` builds separate X/Z measurement blocks
from the declared gauges, with bipartite edge coloring within each basis.
`basis_order` accepts single, repeated and reversed bases. The existing
`MemoryExperiment` discovers this default extractor for:

- `BaconShorCode(distance=d)`, square nearest-neighbor XX/ZZ gauges;
- `SHYPSCode(r=3)` and `SHYPSCode(r=4)`, subsystem hypergraph-product
  simplex constructions. See the [construction reference](../../lightstim/qec_code/shyps/README.md).

The executable [memory notebook](../../notebooks/Memory/memory_subsystem.ipynb)
uses the public pipeline for both families. Regression tests cover declaration
algebra, signed Stim flows for measured gauges and bare logicals, partial and
omitted gauge rounds, preparation, both memory bases and phase orders,
noiseless sampling, and noisy detector error models. Bacon–Shor repeated-round
tests compare the complete affine detector space and logical parities modulo
detectors against explicit execution. Subsystem rounds use explicit execution
whenever verified periodic compression is unavailable.

## Scope

This integration establishes fixed-algebra CSS subsystem memory with
automatically inferred gauge fixing. General mixed protected states,
logical/gauge-entangled inputs, pending absorbed logical relations and
row-index post-selection metadata require additional handling and fail
explicitly in the classifier. The legacy `z_only` readout shortcut assumes
one directly measured ancilla per stabilizer and is rejected for subsystem
gauges; the full detector pipeline supports both X and Z memories.

Gauge-bearing coupler lifecycle, code-algebra changes between measurement
boundaries, and general spacetime logical creation are not established by
these integrations. The existing retained-data measurement engine and
Middle-Out examples remain available on their original paths. A dynamically
changing physical circuit does not by itself require a new patch abstraction;
future examples must identify which code-boundary or logical-lifecycle
assumption they actually exceed.

The supplied generic schedules are functional integration baselines. Their
noise DEMs do not establish circuit fault distance, an optimized decoder,
single-shot performance or a threshold.
