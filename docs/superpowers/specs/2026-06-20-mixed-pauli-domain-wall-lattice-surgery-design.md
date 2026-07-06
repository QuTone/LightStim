# Mixed-Pauli Domain-Wall Lattice Surgery — Design Document

- Date: 2026-06-20
- Scope: **correctness of the static stabilizer construction** (no circuits / DEM /
  fault-tolerant schedule)
- First: the **minimal two-patch `X1Z2` case**; generalize to `ZZZX` and full-width
  only once that is right
- Supersedes: the mixed-product verification part of
  `docs/superpowers/routed-mixed-pauli-lattice-surgery.md` (the `AncillaLogicalTerm`
  path in that record is judged incorrect by this design)

---

## 1. Goal

Support mixed-type (any X/Z combination) logical product measurements between
**data patches at different positions**, typical examples `X1Z2`, `ZZZX`. This stage
only requires the **static construction** to be correct: the ancilla bus, as a
stabilizer code, must have a check set that strictly satisfies the four hard
requirements below algebraically — with no after-the-fact rescue such as a "known
logical factor".

## 2. Background: why the current implementation is wrong

The current notebook (`routed_ZZZX_LS.ipynb`) uses `mixed_stabilizers=True` +
`solve_routed_pauli_product_long_ancilla`: it guesses a type per syndrome qubit,
adds boundary candidates, ad-hoc prunes anticommuting pairs, and finally bundles
the un-cancelable residue into an `AncillaLogicalTerm` declared as a "known
boundary logical". Measured:

| Case | ancilla data | ancilla stab | expected stab (#data−1) | residual support |
| --- | --- | --- | --- | --- |
| ZZZX | 206 | 212 | 205 | 69 qubits (55 Z + 14 X) |
| X1Z2 | 48 | 52 | 47 | 20 qubits |

Two core errors:

1. **Degrees of freedom ≠ 1**: more stabilizers than data qubits — not a clean
   single-logical-qubit code at all.
2. **False-positive verification**: in `solve_routed_pauli_product_long_ancilla`,
   `verified = all(term.owner == coupler_name for term in residual_terms)` —
   it reports success as long as the residual Pauli lands on the coupler's own
   qubits, **without ever checking that the residual actually equals the target
   logical**. The actual residual is a 69-qubit blob, swept under the rug as a
   "logical factor".

## 3. Physical core: why the domain wall is necessary

Lattice surgery **never measures a data patch's logical operator in isolation**.
Merging along one boundary measures
`(data logical) × (ancilla operator on that boundary)`:

- p1 end (X merge): `X1 · X_anc^left`
- p2 end (Z merge): `Z2 · Z_anc^top`

Target `X1Z2` = product of the two = `X1 Z2 · (X_anc^left · Z_anc^top)`. The
ancilla leftover in parentheses must be deterministically +1.

- **Same basis (ZZ)**: the two ends are the two halves of one ancilla Z-string;
  their product is one ancilla stabilizer (+1) → cancels automatically. This is
  why ordinary surgery is simple.
- **Mixed basis (XZ)**: `X_anc^left` and `Z_anc^top` are the ancilla's own two
  **anticommuting** logicals (`X̄_anc`, `Z̄_anc`); their product is `Ȳ_anc ≠ I`,
  not a deterministic value → the leftover cannot be canceled. This is the root
  of the current "leftover blob".

**The domain wall welds the two anticommuting end operators into one bent
string**, so their product becomes `(same operator)² = I` → cancels
automatically → a clean `X1Z2`. The mixed checks on the wall are the "weld
spots". Hence "mixed stabilizers in the middle" is not optional — it is a
physical necessity.

Rejected alternatives:

- **Corner construction** (joining an X boundary and a Z boundary at a convex
  corner): the corner's `X̄/Z̄` are still two anticommuting logicals; the ancilla
  would need preparation in a Y eigenstate — expensive; not adopted.
- **H-trick** (logical H on the X-target patch → all-Z measurement): DEM-verified
  and the simplest, but it **violates requirement ③** (all Z) and is not native
  mixed; not adopted this stage (kept as legacy baseline).
- **Twist defect**: when a wall endpoint lands in the bulk it degenerates into a
  twist (weight-5) — general but too heavy; this design **avoids** twists by
  terminating both wall ends on the bus boundary.

## 4. Hard requirements (formalized as assertions)

For the ancilla bus (counted as a standalone patch):

1. **Degrees of freedom = 1**
   - `len(bus.data) - 1 == len(bus.stabilizers)`
   - `gf2_rank(symplectic(bus.stabilizers)) == len(bus.data) - 1`
   - all checks pairwise commute
2. **Product = measured logical**
   - There exists a subset of checks whose product × (original data-patch
     stabilizers, as code-space equivalence) **exactly equals** the target
     product's symplectic vector, with **residual == 0**
   - No "known logical factor / ancilla readout leftover" is accepted as a pass
     condition
3. **Type mixing**: `{'X','Z','MIXED'} ⊆ set(check types)` (must not be all X or
   all Z)
4. **Wall connectivity**: the MIXED checks form one connected domain wall (not
   scattered points)

## 5. Geometry

Under the unrotated convention **left/right = X boundaries, top/bottom = Z
boundaries** (see `code_patch.py`: logical X is the x=0 vertical X-string,
logical Z is the y=0 horizontal Z-string), an X-target attaches through a
vertical edge and a Z-target through a horizontal edge, so the `X1Z2` bus is
naturally **L-shaped** (horizontal arm against p1's vertical X boundary,
vertical arm against p2's horizontal Z boundary).

```
   p1 ──X-interface──►[ X-sector ]══ WALL(mixed) ══[ Z-sector ]
                                                        │
                                                   Z-interface
                                                        ▼
                                                       p2
```

- X-sector: the bus's long edge is Z-type; the X-string enters from p1;
- Z-sector: the bus's long edge is X-type; the Z-string runs to p2;
- WALL: a single column of mixed checks crossing one straight arm, **both ends
  on the bus boundary** (→ no twist).

Minimization: use the smallest workable `route_width` and closely placed p1/p2 so
the bus is only a few dozen qubits — easy to hand-check and assert strictly. The
L shape (a bend) is adopted as confirmed by the user; no forcing into a straight
line.

## 6. Construction algorithm: Hadamard-region domain wall (verified on a real lattice)

Core principle: **domain wall = transversal-Hadamard conjugation of one whole
connected subregion R of the bus (the X-sector)**. For every stabilizer, swap
X↔Z on the qubits inside R (leave those outside R unchanged):

- pure checks entirely inside R → pure-type flip (X↔Z);
- checks entirely outside R → unchanged;
- **checks straddling ∂R → automatically become mixed**.

Because Hadamard is unitary, **conjugation preserves commutation**, so the mixed
checks obtained this way are **guaranteed** to commute with all neighbors, and
the generator count, rank, and logical count are unchanged →
`#data − 1 = #stab` holds automatically. This turns "commutation" from a
constraint to be hand-tuned into a **mathematical guarantee**.

Implementation: add a **clean, new domain-wall construction method** to
`UnrotatedRoutedMultiPatchCoupler`, dedicated to the mixed case, leaving the
legacy `_init_stabilizers` untouched. Steps:

1. First build the bus as a **valid CSS unrotated patch** (all-commuting, one
   logical, `#data−1` checks).
2. Pick H-region `R` = the connected subregion meant to become the X-sector (the
   arm near p1). `∂R` is the wall.
3. Conjugate every stabilizer X↔Z on the qubits inside R; checks on `∂R`
   automatically become mixed.
4. **Boundary merge checks**: generate the checks connecting bus and patch at
   p1's X interface and p2's Z interface (reusing the
   `_probe_and_create_stabilizer` approach); the type is decided by that side's
   sector.
5. Choose `R` so the bus's logical string presents as X at the p1 end and Z at
   the p2 end (verified: a pure horizontal Z-string conjugated over the left
   half R becomes left-half X + right-half Z).

**Verification record** (`scratch_domain_wall_check.py`, 33-qubit bus with
d_z=7 / d_x=3):

| Construction | data/stab | rank | anticommuting pairs | types |
| --- | --- | --- | --- | --- |
| baseline CSS | 33/32 | 32 | 0 | X:18, Z:14 |
| isolated mixed check | +1 | — | **4 (collapses)** | — |
| Hadamard-region wall | 33/32 | 32 | **0 (all commute)** | X:15, Z:12, MIXED:5 |

The mixed checks land on a single wall column (connected); the logical Z-string
becomes a left-X / right-Z mixed string.

> **Note**: the wall shape is `∂R` and may be a **straight line (vertical or
> horizontal)** — no diagonal needed; as long as `∂R` terminates on the bus
> boundary there is no twist. "Trimming" is **not needed** on a clean patch (the
> counts come out right automatically); it may only be needed when the
> construction is squeezed into the existing over-wide coarse-grid geometry and
> extra qubits appear — and then it is handled as "delete same-type boundary
> checks together with their qubits", which does not affect commutation
> (equivalent to choosing R more tightly).

## 7. Verification harness (test entry point)

Provide `verify_mixed_bus(system, coupler_name, patch_names, target_paulis)`
returning a structured result asserted by tests:

- `dof_ok`: requirement ① of §4 (counts + rank + commutation)
- `product_ok`: requirement ② (residual == 0, strict)
- `types_ok`: requirement ③
- `wall_connected`: requirement ④
- On failure, emit diagnostics (which checks are extra/missing, which qubits the
  residual lands on)

Add these assertions to `tests/test_protocols.py` (minimal `X1Z2` case first).

## 8. Implementation order

1. **Minimal `X1Z2` bus**: implement the domain-wall construction method, TDD
   until all four assertions pass.
2. **Four-patch `ZZZX`**: generalize to a tree-shaped multi-arm bus (Z trunk +
   one X arm carrying the wall).
3. **Full-width (`route_width = 2d−1`)**: attach the construction to the
   existing coarse-grid + seam geometry, keeping all assertions passing.
4. Switch the notebook to the new construction and the strict
   `verify_mixed_bus`; delete the `AncillaLogicalTerm` false-positive path.

## 9. Non-goals (this stage)

- Syndrome-extraction circuits, detector error model, fault-tolerant schedule.
- Twist-based (weight-5) general mixed measurements; Y-type products.
- Changes to the existing legacy `_init_stabilizers` / H-trick paths (kept as
  baselines).

## 10. Risks

- ~~Exact lattice form / commutation of the wall~~ **verified**: the
  Hadamard-region construction guarantees full commutation and unchanged counts;
  the wall can be a straight line, and `∂R` ending on the bus boundary avoids
  twists.
- **Requirement ② (merge product = X1Z2 with residual 0) not yet verified end to
  end**: p1/bus/p2 must actually be connected with merge checks added, then
  re-checked. This is implementation step one, driven by the assertions of
  §4/§7.
- **Interaction between merge checks and the wall**: if the H-region boundary is
  too close to the p1/p2 interfaces, merge checks may land on conjugated qubits.
  Mitigation: place the wall (`∂R`) on an arm segment far from both interfaces.
- **Count drift after generalizing to full-width / coarse grid**: the
  interaction of seam rows/columns with R, and any trimming, require re-checking
  the assertions.

## 11. Final implementation (verified — horizontal `X1Z2`, no bend)

After comparison against the rotated case (Fowler–Gidney FIG. 9) and hand-drawn
standard unrotated mixed stabilizers, the construction that finally landed is
**simpler and more robust than building a mixed merge from scratch**:

> **Method**: use the codebase's **already-working same-basis `XX` merge**
> (`UnrotatedMultiPatchCoupler`, two standard patches side by side
> horizontally), then apply a **transversal Hadamard to all data qubits of p2**
> (= a logical rotation of p2, X2↔Z2).

Key points:
- Hadamard is unitary → **conjugation preserves commutation and the logical
  count**, so DOF, commutation, and twist-freedom are **mathematical
  guarantees** — no hand-tuning of parities/counts (this avoids the pitfalls hit
  repeatedly when building a mixed merge from scratch).
- The merged operator `X1·X2` becomes **`X1·Z2`** after H conjugation; the
  coupler checks spanning the two patches automatically become **mixed
  domain-wall checks** (X on p1-side qubits, Z on p2-side qubits) — exactly the
  standard unrotated mixed stabilizer.
- This is the paper's "rotate the logical qubit" method realized on the
  unrotated code; the Hadamard absorbs the half-cell dislocation entirely, the
  seam is straight, and no explicit step is needed.

**Verification (d=3, `routed_ZZZX_LS.ipynb` ran end to end)**:

| Requirement | Result |
| --- | --- |
| ① `#data − rank = 38 − 37 = 1` | ✓ DOF=1 |
| ① all commute | ✓ 0 anticommuting pairs |
| ② joint `X1·Z2` measured; none of `X1/X2/Z2` measured alone | ✓ true joint |
| ③ pure-X / pure-Z / MIXED all present | ✓ MIXED=5 |
| ④ domain wall, no twist | ✓ 5 MIXED checks, 0 containing Y |

Measured logicals: `X̄1 = X on p1 @ {(0,0),(0,2),(0,4)}`,
`Z̄2 = Z on p2 @ {(10,0),(10,2),(10,4)}`, annotated in the notebook figure.

**Still not done**: the FT schedule at the circuit / detector-error-model level;
bent buses; `ZZZX`. The analysis of the domain wall / Hadamard region in
§§6–10 above still stands; this section is its simplest landing in the
two-patch horizontal case.
