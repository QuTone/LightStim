# Parameterized Rotated Bent (XZ) Joint-Measurement Generator — Design

**Date:** 2026-06-26
**Status:** Implemented (coordinate-aware).
**Revision (2026-06-26):** The generator is now genuinely **coordinate-aware** — `PatchSpec.origin`
drives the geometry. Both patches are placed as real `d×d` rotated patches at their origins; the
corner **bus** is derived from their positions (`bus = p1.cols × p2.rows`); data / CSS + mixed
stabilizers / readout regenerate when an origin moves. Alignment is validated (`p2.left = p1.right+2`,
`p1.bottom = p2.top+2`, i.e. patches exactly `2d` apart on each axis; any common translation is fine);
misaligned placements raise `BentLayoutError` with the concrete reason — never a silent hard-coded
fallback. Boundary-keep uses a commuting + GF(2)-independent + **logical-preserving** greedy (no
position-specific tiebreak). Type phase is chosen by trying both parities and validating, so the
construction is translation-correct.
**Revision (2026-06-26b) — routed (extended) bus, both axes:** the generator now supports a **routed
L-bus** that extends horizontally AND/OR vertically.  `_arrange` builds a horizontal Z-band
(`p1.left..p2.right` across `p2`'s rows) plus a vertical bus column (at `p1`'s columns, down to the
X-patch), and accepts `hgap = p2.left-p1.right = 2+2k` and `vgap = p1.bottom-p2.top = 2+2m`
(`k,m>=0`; `0` = minimal, larger routes the Z-patch further right / the X-patch further down).  The
mixed seam stays at the X-arm width (`#mixed = d` for any `k,m`).  Only genuinely impossible placements
raise `BentLayoutError`: a gap that is zero/negative (overlap) or **odd** (patches on incompatible
odd-lattice parities).  Verified valid for the minimal, horizontal-only, vertical-only, and diagonal
(`k=m`) routed buses (e.g. `(1,11)/(11,1)` → 38 data, 3 mixed, all 8 checks pass).  The updated
`rotated.png` (a horizontally-extended bus) and placements like `(1,11)/(11,1)` were previously rejected
only because the generator hard-coded the minimal gaps — an implementation limit, not a theoretical one.

**Note on the golden:** because the generator uses *real full patches*, the canonical d=3 layout is
**26 data / 25 checks** — the figure-grounded golden (23 data) **plus the Z-patch's far column**, which
the hand-transcribed golden had trimmed. The golden remains a valid *sub*-layout (kept as a fixture);
the d=3 regression is now "valid + figure-consistent (same logicals, 3 mixed seam checks, golden ⊆
data)" rather than byte-identical.
**Goal:** Replace the hand-coded d=3 layout in `notebooks/LogicalOps/rotated_bent_XZ_LS.ipynb` with a
**parameterized generator**: given the two logical patches that participate in the joint measurement
(plus the bend/seam geometry), automatically build the data qubits, ordinary CSS stabilizers, mixed
(XZ) domain-wall stabilizers, the boundary trimming/replacement, the readout chain, and the no-MPP
gate-level syndrome-extraction circuit — then self-verify. The current validated d=3 layout becomes a
**golden regression fixture**; the generator must reproduce it exactly at d=3 and then scale to d=5,7.

The generator reuses the existing rotated-code coordinate/stabilizer convention (it does **not** invent
a new coordinate system).

---

## Background — what the golden d=3 layout actually is

(Computationally verified against `rotated_bent_XZ_LS.ipynb` cell 1: commutation, `#data−rank=1`,
readout-chain product, partition.)

The golden is **not three independent patches**. It is **two logical patches joined by an auto-generated
bent bus**:

- **p1 = X-patch** — origin `(1,7)`, d=3, `measured_logical="X"`, `orientation="X_horizontal"`; the
  measured `X̄₁` sits on its bus-facing row (`row 7`).
- **p2 = Z-patch** — origin `(7,1)`, d=3, `measured_logical="Z"`, `orientation="X_horizontal"` (so its
  `Z̄` is vertical); the measured `Z̄₂` sits on its bus-facing column (`col 7`). In the golden, p2 is
  **reduced from a full d×d**: its bus-facing weight-2 boundary checks are dropped/replaced by the CSS
  junction, and it spans only cols 7,9. The exact trimmed shape is **not hand-specified — it is pinned by
  the d=3 regression** and follows from the seam/bus generator's trim rule (principle 2/4 below).
- **bus / bend region** — the L-shaped connector (cols 1–5, rows 1–5). **Not a `PatchSpec`.** It is
  Z-type, carries the bend, and has an **inner-corner truncation** (data `(1,1)` and its corner X-check
  removed; the corner Z-plaquette downgraded to weight-3).
- **domain wall (mixed XZ seam)** — the p1↔bus interface on `row 6` (3 mixed checks). Each mixed check
  is uniform-Pauli per side and **alternates with the seam index** (even j → p1-side Z / bus-side X;
  odd j → p1-side X / bus-side Z); the end check is weight-3. This is the only place mixed checks occur.
- **CSS junction** — the bus↔p2 interface on `col 6`: ordinary weight-4 checks, **no** mixed.
- **readout chain** — a Z-comb across the bus plus the even-index mixed checks; its GF(2) product is
  exactly `X̄₁·Z̄₂`.

**Library convention reused** (verified in `rotated/code_patch.py`, `ir/qec_patch.py`): data on
`(odd,odd)`, ancillas on `(even,even)`, spacing 2; corner data at `(1,1)`; placement via `shift`;
default **X̄ vertical (x=1), Z̄ horizontal (y=1)** with orientation flipped post-hoc via
`transpose_coords()`; stabilizer dicts via `create_stim_stabilizer`; coord↔index via
`index_map`/`grid_map`/`get_grid_key`/`snap_coord`. Boundary-check replacement reuses the coupler's
`conflicting_stabilizer_coords` (keyed by `syn_coord`, Pauli-type agnostic).

---

## Design principles (from the spec discussion)

1. `PatchSpec` describes **only** the logical patches that contribute a measured logical (here p1, p2).
2. A patch need **not** stay a full `d×d` in the final layout: when it meets the bus, the generator
   **trims its bus-facing boundary** (delete/replace the near-seam weight-2 stabilizers).
3. The **bus / bend / routing region is not a `PatchSpec`** — it may be an irregular connector
   (triangles, cut corners, stubs).
4. The **seam/bus generator decides** which boundary checks are deleted and which are replaced by mixed
   (XZ) stabilizers.
5. Mixed (XZ) stabilizers are generated automatically from the two patches' positions, orientations, the
   target joint, and the bend/seam geometry.
6. The current validated **d=3 layout is the golden reference**; the generator must reproduce it exactly
   at d=3, with input = the two measured patches + bend/seam geometry (not three full patches).

---

## Interface

```python
@dataclass(frozen=True)
class PatchSpec:
    name: str
    origin: tuple[int, int]      # bus-facing corner data coord, in the library (1,1)-corner convention
    distance: int                # odd
    measured_logical: str        # "X" | "Z"  — which logical this patch contributes to the joint
    orientation: str             # "X_horizontal" | "X_vertical" — direction of this patch's own X̄ string

def build_rotated_bent_xz_layout(
    patches: list[PatchSpec],    # exactly the logical patches (here [p1(X), p2(Z)])
    bend="auto",                 # bend/bus geometry; "auto" infers the L-connector from patch positions,
                                 # or an explicit elbow/seam descriptor for disambiguation
    joint_type: str = "XZ",
    readout_rule: str = "auto",  # "auto" = GF(2)-solve the stabilizer subset whose product == joint logical
) -> "BentLayout": ...
```

`BentLayout` exposes:
- `.data: list[(col,row)]`, `.checks: list[{'syn','type','pauli','corners'}]` (same dict schema the
  notebook/`bent_joint_se` already consume; `type ∈ {'X','Z','M'}`),
- `.x_logical`, `.z_logical` (the measured supports), `.readout_chain: set[syn]`,
- `.build_circuit(rounds, p)` → `stim.Circuit` (no-MPP gate-level SE; reuses/generalizes
  `RotatedBentJointMeasurement`),
- `.verify()` → dict of the eight acceptance checks (below).

`build_rotated_bent_xz_layout([p1, p2])` with the golden's p1=`(1,7)`/p2=`(7,1)` at d=3 must return the
golden `data`/`checks`/`x_logical`/`z_logical`/`readout_chain` **exactly**.

---

## Construction algorithm

A pipeline of small, independently testable stages (each consumes/produces the coord-keyed dicts):

1. **Place patches.** For each `PatchSpec`, build `RotatedSurfaceCode(distance=d, shift=origin)`, apply
   `transpose_coords()` if the orientation requires X̄ horizontal; extract `data_coords`, CSS
   stabilizers (coord-keyed), and the measured-logical support on the bus-facing boundary.
2. **Infer the bus region.** From the two patches' positions + orientations + `bend`, compute the
   L-shaped connector region (width d) and its data qubits (rotated convention).
3. **Bus CSS checks.** Emit the bus's ordinary X/Z plaquettes (rotated `build()` rules over the bus
   region).
4. **Boundary trim + interface checks.** At each patch↔bus interface decide, per the joint geometry:
   - **CSS junction** (matching boundary types) → fuse facing weight-2 checks into weight-4 CSS; mark the
     replaced originals via `conflicting_stabilizer_coords` (by `syn_coord`).
   - **mixed domain wall** (X-side meets Z-side) → delete the facing weight-2 checks and emit the **d
     mixed (XZ) checks**, uniform-Pauli per side, alternating with seam index (even j → patch-side Z /
     bus-side X; odd j → patch-side X / bus-side Z); end check weight-3.
5. **Inner-corner truncation.** Cut the bend's inner-corner data + its corner check; downgrade the
   adjacent plaquette to weight-3, as required for commutation.
6. **Readout chain (`auto`).** GF(2)-solve (via the existing `solve_linear_decomposition` /
   `logical_pauli_product_vector`) for the stabilizer subset whose symplectic product equals the joint
   `X̄₁·Z̄₂`; that subset is the readout chain.
7. **SE circuit.** Feed `.data`/`.checks` to a generalized `RotatedBentJointMeasurement` to emit the
   no-MPP gate-level SE circuit (pure X/Z on the perpendicular schedule; mixed in own H…H blocks).

The exact trim/replace and corner rules are **pinned by the d=3 regression** (Stage-by-stage the
generated artifact must match the golden) and **gated at d=5,7 by `.verify()`**.

---

## Verification (`.verify()` — the eight acceptance checks)

Run automatically for every generated layout/circuit:

1. all stabilizers commute (symplectic);
2. joint `X̄₁·Z̄₂` is measured (in the stabilizer span);
3. single `X̄₁` **not** measured and single `Z̄₂` **not** measured;
4. no `Y` / no twist (every check is pure-X, pure-Z, or X∪Z with no qubit carrying both);
5. `#data − rank == 1` (one logical);
6. DEM valid (`detector_error_model` detectors/observables consistent; noiseless-deterministic);
7. no `MPP` in the circuit;
8. no tick collision (no qubit touched by two gates in the same tick / one clean op-type per slice).

`peek_observable_expectation` is used (per the 2026-06-20 lesson) to confirm the *actually measured*
joint is `X̄₁·Z̄₂` and not a single logical — never trusting the geometric label.

---

## d=3 golden regression fixture

The generator at d=3 (`p1=PatchSpec("p1",(1,7),3,"X","X_horizontal")`,
`p2=PatchSpec("p2",(7,1),3,"Z","X_horizontal")`, `bend="auto"`) must reproduce **exactly**:

- **DATA (23):** (1,3)(1,5)(1,7)(1,9)(1,11)(3,1)(3,3)(3,5)(3,7)(3,9)(3,11)(5,1)(5,3)(5,5)(5,7)(5,9)(5,11)(7,1)(7,3)(7,5)(9,1)(9,3)(9,5)  — note `(1,1)` absent.
- **CHECKS (22):** 9 X + 10 Z + 3 mixed, identical syn/type/pauli to the current notebook cell 1
  (mixed at (2,6),(4,6),(6,6) with the alternating per-side Paulis; weight-3 checks at (2,2) and (6,6)).
- **X̄₁** = {(1,7),(3,7),(5,7)}, **Z̄₂** = {(7,1),(7,3),(7,5)}.
- **readout_chain** = {(0,4),(2,2),(2,6),(4,0),(4,4),(6,2),(6,6)}.
- Circuit identical to the current `build_circuit` (45 qubits / 62 detectors / 1 observable at rounds=3,
  noiseless) and the four hard requirements + acceptance pass.

The hand-coded table stays in the repo **only** as this regression fixture (e.g. a JSON/py fixture under
tests), not as the notebook's source of truth.

---

## Generalization to d=5,7 and LER

Same generator, `distance=5` and `7`. Invariants: `#mixed == d`; `#stab == #data − 1`; the eight checks
pass. The exact bus width, corner-truncation, and readout-chain length scale with d; correctness is
asserted by `.verify()` at each d (a failure means the generalization rule is wrong → iterate). Then a
distance sweep with circuit-level noise + MWPM produces the **LER-vs-p suppression curves** for
d ∈ {3,5,7}.

---

## File layout

| File | Responsibility |
|---|---|
| `lightstim/qec_code/surface_code/rotated/bent_layout.py` (new) | `PatchSpec`, `build_rotated_bent_xz_layout`, `BentLayout` (construction + `.verify()`). Reuses `RotatedSurfaceCode`, `QECPatch` helpers, coupler conflict machinery, `solve_linear_decomposition`. |
| `lightstim/qec_code/surface_code/rotated/bent_joint_se.py` (modify) | generalize `RotatedBentJointMeasurement` to take a `BentLayout` (already coord-dict based). |
| `lightstim/qec_code/surface_code/rotated/__init__.py` (modify) | export the new symbols. |
| `tests/test_rotated_bent_layout.py` (new) | **d=3 golden regression** (generated == fixture) + the eight checks at d=3,5,7. |
| `notebooks/LogicalOps/rotated_bent_XZ_LS.ipynb` (modify) | replace the hand-coded cell 1 with `layout = build_rotated_bent_xz_layout([p1,p2]); data, checks = layout.data, layout.checks`; keep viz/acceptance/detslice; LER sweep over d=3,5,7. |

---

## Risks / open points

- **Bend/bus inference** is the hard part: the auto L-connector + which boundary is CSS-fused vs
  mixed-walled must reproduce the golden and generalize. Mitigation: d=3 regression pins it; `.verify()`
  gates d=5,7; `bend` can take an explicit descriptor if "auto" is ambiguous.
- **Mixed-seam alternation + corner truncation at larger d**: the parity rule and the single inner-corner
  cut must keep commutation at d=5,7. Mitigation: `.verify()` (commute + no-twist + joint) is the oracle.
- **Readout-chain auto**: GF(2) solve may return a non-unique subset; pick the minimal-weight / canonical
  one and confirm its product == joint logical and (at d=3) == the golden chain.
- **Orientation/transpose bookkeeping**: `transpose_coords` must keep stabilizer `syn_coord` metadata
  consistent — reuse the library transform, do not re-derive.
- No `MPP` and no tick-collision are explicit acceptance gates (checks 7,8), not assumed.

---

## Implementation phases

1. **`PatchSpec` + patch placement** (stage 1) — build/orient/extract a single rotated patch; unit-test
   coords + CSS checks + logical support against the library d=3 patch.
2. **Bus inference + CSS** (stages 2–3) — L-connector region + its CSS checks; test geometry.
3. **Interface trim + mixed seam + corner truncation** (stages 4–5) — the novel core; **d=3 golden
   regression** (generated checks == fixture) gates this.
4. **Readout auto + `.verify()`** (stages 6 + checks) — GF(2) readout; the eight checks pass at d=3.
5. **SE circuit generalization** — `RotatedBentJointMeasurement(layout)`; circuit == golden at d=3.
6. **Scale + LER** — d=5,7 pass `.verify()`; notebook swapped to the generator; LER d∈{3,5,7} curves.

Each phase verifies before the next; the d=3 regression (phase 3) and `.verify()` (phase 4) gate
everything downstream.
