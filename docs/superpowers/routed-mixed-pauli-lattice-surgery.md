# Routed Mixed-Pauli Lattice Surgery — Design Record

This document records the correct approach reached in this discussion. The goal is to
support mixed X/Z logical product measurements such as `ZZZX` and `X1Z2`.

## Core rule

For every data patch, two questions must be answered first:

1. Does the target want to measure `X` or `Z` on this patch?
2. What is the native interface basis (`X` or `Z`) of the edge that actually
   connects to the ancillary path / bus?

Whether a logical H is needed is decided by comparing these two bases:

| Interface basis at the ancilla | Target Pauli | Logical H needed? |
| --- | --- | --- |
| X | X | No |
| X | Z | Yes |
| Z | Z | No |
| Z | X | Yes |

So it is wrong to simply assume "if the target contains an `X`, convert it to `Z`
first". If a patch already connects to the ancilla through an `X` edge and the
target also measures `X`, the original frame should be kept — no extra H.

The implementation should process in this order:

1. For each participating patch's selected edge, infer (or accept explicitly) the
   native interface Pauli.
2. Compare the native interface Pauli against the target Pauli product term by term.
3. Apply a logical H only on the mismatched patches.
4. If the original logical frame is wanted back after the measurement, apply H
   again before readout.

## Geometry rules

The ancilla region is not restricted to a purely vertical or purely horizontal
corridor. Following the long-range multi-target CNOT figures of Litinski /
von Oppen, the better interpretation is: the bending bus is itself one complete
long ancillary surface-code patch.

Therefore the primary geometric rule for a full-width route is NOT "color every
coordinate X/Z by its nearest interface", but:

1. The ancilla bus has a fixed, code-distance-scale width.
2. The data/syndrome checkerboard extends continuously across the whole bent patch.
3. 90° corners are treated like ordinary surface-code patch corners, naturally
   allowing weight-2 / weight-3 boundary stabilizers.
4. Adjacent patch-sized route blocks must include seam rows/columns so that the
   whole bus is equivalent to one standard unrotated rectangular / bent patch —
   not a chain of small patches that each restart the checkerboard phase.

The data patch's selected interface window merges with this long ancilla patch.
The merge-check product may leave one logical boundary factor of the long ancilla;
that factor is fixed by the ancilla's initialization/measurement frame, not by
reading out individual ancilla data qubits.

## Stabilizer template rules

The paper-style long ancilla's stabilizers are generated as an unrotated
surface-code patch, but the X/Z sheet junction must use mixed / domain-wall
stabilizers:

- Bulk plaquettes remain the standard weight-4 X/Z checkerboard.
- Near the seam/corner where the X/Z route label changes, use mixed stabilizers;
  do not force these positions to stay pure CSS X/Z checks.
- Weight-2 / weight-3 stabilizers appear naturally on outer boundaries, inner
  boundaries, interface windows, and corners.
- A bend does not re-initialize the checkerboard phase; it is just a corner of the
  same patch.
- What enters the logical product is the merge-check syndrome product on the
  data–ancilla contact window. The ancilla's internal stabilizers keep the bus a
  code patch and must not be interpreted as independent data readouts.

A mixed X/Z stabilizer template is only required where an X boundary and a Z
boundary genuinely meet through a domain wall / twist at the same local position.
The code keeps `mixed_stabilizers=True` as an experimental scaffold, but the
Figure-10-style bent long ancilla prefers ordinary CSS patch stabilizers.

## Examples

For `ZZZX`:

- If the four attached interfaces are interpreted as `Z, Z, Z, Z`, then only the
  final `X`-target patch needs a logical H; perform a routed `ZZZZ` measurement,
  and H back to the original frame afterwards if needed.
- If the four attached interfaces are natively `Z, Z, Z, X`, then no H at all is
  needed. What is genuinely required then is that the route itself supports mixed
  X/Z local stabilizers, generating the corresponding mixed checks at the
  `X`-region / `Z`-region junction.

For `X1Z2`:

- If patch 1 attaches through an `X` interface and patch 2 through a `Z`
  interface, this is a native mixed X/Z measurement.
- Converting `X1` to `Z1` by default is then incorrect. The right direction is
  mixed-boundary stabilizer generation, together with outcome / tracker support
  for the corresponding logical product.

## Current implementation status

Implemented:

- `UnrotatedRoutedMultiPatchCoupler` supports explicitly selecting each patch's
  attachment edge and connects those edges under arbitrary layouts via Manhattan
  routing.
- The direct routed coupler defaults to a full ancillary-patch route with
  `route_width = 2 * code_distance - 1`. For a d=3 unrotated patch this span is 5
  integer lattice coordinates, not the logical distance 3 itself.
- The full ancillary-patch route no longer allows data patches at arbitrary
  physical coordinates. Every data/obstacle patch participating in routing must
  sit on one coarse grid: each patch occupies exactly one
  `route_width × route_width` coordinate block, but the origin pitch of adjacent
  coarse cells is `route_pitch = route_width + 1 = 2d`, not `route_width`. The
  extra integer coordinate is the seam row/column shared when two standard patch
  blocks are stitched; without that seam the boundary weight-3/weight-4
  stabilizers misalign. For d=3, `route_width=5`, `route_pitch=6`, so common safe
  same-direction patch spacings are 12, 24, 36, ... integer coordinates.
- The routed coupler first runs Manhattan BFS on this coarse grid, avoiding
  coarse cells occupied by data patches; each route cell is then expanded into a
  full `route_width × route_width` ancillary block, with seam rows/columns
  explicitly inserted between adjacent route cells. The terminal regions of
  selected interfaces and the intermediate bends/corridors are therefore
  assembled from standard patch blocks — no longer irregular shapes inflated
  from a thin skeleton. Two horizontally adjacent d=3 blocks connect into 6 data
  qubits on even data rows, i.e. a standard `distance_z=6` unrotated rectangular
  patch.
- A route cell is a geometric coarse block, not a small patch that independently
  re-initializes the checkerboard phase. The data/X-syndrome/Z-syndrome parity of
  the full ancillary region must be continuous over the whole merged lattice; if
  each 5×5 cell restarted its own phase, the picture would still look like
  blocks, but the Pauli-product algebra of the boundary merge would break.
- The interface basis can be inferred automatically from the selected edge or
  passed explicitly.
- The routed Pauli-product helper compares the target Pauli against the native
  interface Pauli and inserts logical H only at mismatched positions.
- The `mixed_stabilizers=True` full-width routed coupler is the notebook's
  current paper-style long-ancillary-patch path: it generates X/Z checks in the
  bulk, mixed checks on the X/Z sheet seam, and low-weight boundary stabilizers
  appear naturally at corners.
- `solve_routed_pauli_product_long_ancilla` compresses the leftover ancillary
  boundary support of the syndrome product into a single `AncillaLogicalTerm`.
  This represents the long ancilla patch's known logical boundary factor, not
  independent ancillary data readout terms.
- The long-ancilla helper now uses the paper-style geometric product: it selects
  the stabilizers belonging to the target product sheet along the whole connected
  ancillary bus, instead of using a minimum-weight linear-algebra solution that
  picks a few stabilizers only near the data-patch interfaces. The X/Z
  plaquettes highlighted in the notebook are exactly the stabilizers entering
  the syndrome product.
- `mixed_stabilizers=False` only suits pure-CSS / same-basis routed checks; for a
  mixed-interface bus like `ZZZX` it produces incorrect pure X/Z stabilizers at
  seams/corners.
- The mixed-template path algebraically suspends original patch stabilizers that
  anticommute with the routed checks, keeping the tested active stabilizer set
  commuting.
- The mixed-template path also prunes locally inside the coupler: pure coupler
  checks on the X/Z seam that anticommute with a mixed check are replaced by the
  mixed/twist template; where a naive CSS checkerboard generates a pair of
  anticommuting pure X/Z corner checks at a bend, one of the pair is removed by a
  fixed local rule. This pruning introduces no high-weight closure.
- `solve_routed_pauli_product_syndromes` can automatically solve and verify the
  outcome decomposition of the target logical product. Proper routed/mixed
  lattice surgery should succeed with `include_ancilla_readout_terms=False`; a
  failure means the current local stabilizer template does not cancel all
  ancillary Paulis in the product.
- The previously attempted high-weight closure syndrome has been removed. It can
  only patch the residual algebraically and is not a sound local
  lattice-surgery stabilizer design.
- The mixed-interface boundary no longer arbitrarily recolors the data patch's
  own boundary syndromes. A boundary syndrome may be reused only when its native
  type is the complementary stabilizer type of that logical interface — e.g.
  reuse X boundary checks on a Z interface and Z boundary checks on an X
  interface. For one selected edge at d=3 that is `d-1 = 2` boundary checks, not
  every syndrome along the whole geometric boundary.
- `routed_coupler_data_basis` generates the physical basis map from the routed
  ancillary region's local `route_coord_basis`:
  - `mode="opposite"` for initialization: `Z` route regions prepare in `X`, `X`
    route regions prepare in `Z`.
  - `mode="same"` for readout: `Z` route regions read `Z`, `X` route regions
    read `X`.
- `build_zz_circuit` in `multi_patch_LS_straight_unrotated.ipynb` (formerly
  `multi_patch_LS.ipynb`) initializes non-coupler/data-patch qubits in the `Z`
  basis by default; only coupler ancillary data is separately initialized in `X`
  for ZZ surgery. Any full-ancillary-patch visualization (the role formerly
  played by `routed_ZZZX_LS.ipynb`, removed in the branch cleanup) should follow
  the same convention rather than defaulting ordinary data patches to `X`.
- `solve_routed_pauli_product_syndromes` still separates the two kinds of
  ancillary diagnostic terms:
  - `selected_ancilla_known_terms` are deterministic +1 eigenvalues supplied by
    the chosen ancillary initialization basis — not readouts.
  - `selected_ancilla_terms` are the residual ancillary data readout terms; for
    the target native full-width mixed measurement they should be 0.
- The interface that truly corresponds to the "green-dot stabilizer product" is
  `solve_routed_pauli_product_merge_checks`. It now uses a basis-aware no-trim
  diagnostic: it selects complete coupler stabilizers geometrically, but also
  filters to the product sheet by the local route label. In `Z`-labeled regions
  only `Z` checks are multiplied, in `X`-labeled regions only `X` checks, and
  `MIXED` seam checks are kept; the opposite interleaved checks do not belong to
  that logical-product sheet and must not be multiplied in. Only original
  data-patch stabilizers are allowed as code-space equivalence terms. If the
  product still leaves an ancillary/data residual Pauli, it returns
  `verified=False` and lists the `residual_terms`. This avoids reporting success
  after artificially truncating boundary Paulis that never canceled naturally,
  and avoids multiplying in X/Z checks from the wrong sheet.
- The product decomposer still accepts `ancilla_readout_bases` for residual
  diagnostics: when syndrome-only fails, it reports which ancillary Paulis were
  not canceled by local checks. This is diagnostic only, not the final
  implementation path.

Verified:

- The `ZZZX` product algebra of the paper-style long ancillary patch passes: in
  the d=3 example, the mixed-domain-wall bent ancilla's syndrome product plus one
  `AncillaLogicalTerm` verifies as the target `ZZZX`. The logical factor is the
  long ancilla's known boundary logical, not a set of independent ancillary data
  readouts.
- The Z-normalized `ZZZX` final-readout path passes detector-error-model
  verification. In that mode the helper is equivalent to measuring routed
  `ZZZZ`, with H applied only on the `X`-target patch. This legacy validated
  helper currently uses `route_width=1` explicitly, to keep the tracker/DEM
  verification closed.
- The mixed-template scaffold has tests for mixed-check generation, the presence
  of non-weight-4 checks, and pairwise commutation of the active stabilizer set.
- Mixed-check extraction now runs in compatible stabilizer batches; each mixed
  syndrome uses an X-basis ancilla, `Z` terms via `CZ(data, syndrome)`, `X`
  terms via `CNOT(syndrome, data)`, with layer coloring of entangling edges
  inside a batch so `detslice-with-ops-svg` does not degenerate into a long
  string of single-check thin-line diagrams.
- The native `X1Z2` product algebra passes: with coupler checks plus active
  patch-stabilizer corrections, the solver verifies syndrome-only that the final
  product equals `X1Z2`.
- The earlier conclusion that the four-patch native-interface `ZZZX` mixed full
  ancillary patch "verified with 20 measured local merge checks" was wrong: that
  path trimmed some ancillary boundary Paulis out of the local checks. The later
  "select all coupler checks" strict diagnostic was also incorrect, because it
  multiplied in interleaved X checks on the Z side and interleaved Z checks on
  the X side. The current paper-style long-ancilla path first filters to the
  correct product sheet and subtracts code-space equivalences with original
  patch stabilizers; in the d=3 example what remains is one known logical
  boundary factor of the long ancillary patch, not extra data readouts.

Not yet claimed complete:

- The patch-span routed ancillary region is used for direct mixed coupler
  geometry, but patch-span native mixed tracker/observable and the
  detector-error-model schedule have not been verified end to end, so the
  notebook no longer claims the full mixed `detslice-with-ops-svg` as a final
  DEM diagram.
- Mixed-check extraction in the current unrotated SE block is no longer
  single-check serial; it batches compatible mixed checks and layers CNOT edges.
  It is still not a schedule claimed fault-tolerant for mixed-boundary lattice
  surgery.
- What has been verified are these specific routed mixed geometries and the
  unrotated d=3 examples; more complex multi-branch layouts, other distances,
  and other route orders should each be verified with the product decomposition,
  tracker observable, and detector error model.
