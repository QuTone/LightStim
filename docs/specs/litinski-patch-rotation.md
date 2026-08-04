# Spec — Litinski patch rotation (step 0: protocol, pinned to the paper)

Status: **draft for review**. No code written yet.
Source: Litinski, *A Game of Surface Codes* (arXiv:1808.02892), Fig 11a, Fig 40b, Fig 40c.

## 0. Why a new operation at all

LightStim already has `QECPatch.rotate_90` / `QECSystem.rotate_patch`. That is a **coordinate
permutation** of the patch contents — it needs the physical qubits to be rearranged
(neutral-atom style), and it **flips every stabilizer type at fixed positions**:

| | weight-4 | weight-2 |
|---|---|---|
| `rotate_90` (existing) | all 4 flip type | all 4 change |
| Litinski Fig 45 (observed) | **unchanged** | only these move |

So the two are different operations. The paper's one deforms the patch **on a fixed qubit
lattice**; the existing one moves qubits. This spec covers the paper's one.

## 1. What the paper actually says (verbatim)

**Fig 11a caption** — "Patches can be rotated in 3⊙ to change whether the X or Z operator is
adjacent to the compact block's ancilla region." (⊙ = one time step = *d* code cycles, so a
rotation is 3 steps.)

**Fig 40c, moving boundaries / qubit movement** — "Extending the patch via its Z boundary in
the second step is the same operation as a Z⊗Z lattice surgery between the patch and a
rectangular |+⟩ ancilla qubit to the right. This needs to be done for *d* code cycles to
account for measurement errors. Finally, the patch is shortened again by measuring the left
two thirds of physical qubits in the X basis."

**Fig 40b, moving corners** — "The movement of corners of a surface-code patch … corresponds
to a change of boundary stabilizers. In order to account for measurement errors of the newly
measured stabilizers, this requires *d* code cycles. The top left physical qubit in the second
step of Fig. 40b is removed from the patch via an X measurement."

**The point I got wrong twice:** every one of these runs for *d* code cycles and always
*grows redundancy first, then removes it*. Nothing is switched instantaneously.

## 2. Facts verified against LightStim / GF(2) before writing this

1. **Rectangular patches already exist.** `RotatedSurfaceCode(distance_x=…, distance_z=…)`
   builds them, and **both distances must be ODD** — so a grow step must add **2 columns
   (or rows) at a time**, not 1. Verified: `dx=3,dz=5` → 15 data, 14 stabilizers, k=1,
   Z̄ weight 5 (horizontal), X̄ weight 3 (vertical).
2. **Growing is well-anchored.** Growing 3×3 → 5×3 keeps **7 of the 8** old stabilizers
   *verbatim*; only the weight-2 on the boundary being extended through changes.
   → detectors stay anchored across the grow step, no blind round.
3. **The target state exists and preserves the logical.** Keeping all weight-4 bulk fixed and
   replacing only the four weight-2 boundary stabilizers (each edge admits exactly **one**
   type that commutes with the fixed bulk) gives a valid code: mutually commuting, 8
   independent stabilizers, distance 3, and **X̄/Z̄ swap orientation**. Both logicals survive
   with weight-5 L-shaped representatives that commute with the new stabilizer set.
4. **Doing (3) instantaneously fails.** Deactivating the old weight-2 and activating the new
   ones in one step gives **distance 1** (control, no switch: distance 3). Confirmed *not* a
   missing-detector artifact: the set of new-check products that commute with all old checks
   is **empty**, so no bridging detector exists. This is why the grow/shrink scaffolding in
   §3 is mandatory.

## 2b. The protocol read off Fig 45 (q2's rotation, steps 4→7)

Fig 45 walks a real two-qubit device through the rotation, and the text pins it:
*"Qubit |q2⟩ is rotated in steps 5-8 using the protocol in Fig. 11a."* Four panels =
three transitions = **3⊙**, matching Fig 11a's cost. Reading the panels:

| transition | q2's shape | operation |
|---|---|---|
| 4 → 5 | small patch (top-right) → **tall bar** | **GROW** — extends into the free space below, from `d` to `2d-1` rows |
| 5 → 6 | tall bar → tall bar (**same footprint**, boundary/checkerboard changes) | **MOVE CORNERS** |
| 6 → 7 | tall bar → small patch (**bottom**-right) | **SHRINK** — the top part is measured out |
| 7 → 8 | small patch (bottom) → **tall bar** again | **GROW** — back up |
| 8 → 9 | tall bar → small patch (**top**-right, its original tile) | **SHRINK** — the bottom part is measured out |

**The full operation is steps 4→9, not 4→7.** Steps 4→7 leave the patch *rotated but
displaced* (it has slid from the top tile to the bottom tile). Steps 7→9 are a plain **qubit
movement** (Fig 40c: grow, then shrink from the other end) that carries it **back to its
original tile**. Without them the patch would not be where the rest of the schedule expects it.

Note this needs **no new primitive**: the move-back reuses the very same `grow_patch` /
`shrink_patch` atoms as the first half.

### Verified geometry — the growth is `d → 2d-1`

**Counted off Fig 45 step 5:** for d = 3 the grown patch is **5 rows** of data qubits, not 7.
So the patch extends by `d-1` into the free space, reaching `2d-1` — odd for odd d, so the
rotated-surface-code builder accepts it. (An earlier draft of this spec said `2d+1`, reasoning
from LightStim's seam pitch `2d+2` = tile + seam column + tile. That is the spacing between two
*separate* patches; growing a *single* patch into empty space abuts directly and gives `2d-1`.
`2d+1` is wrong and does not match the paper.)

The rule is general — nothing about the stabilizer pattern is hardcoded. `grow_patch` takes the
enlarged geometry as a parameter. The geometry source is **`RuleBasedRectPatch`** (not the plain
builder — its boundary pattern does not match the paper): the caller passes as `forced` the live
patch's surviving checks **plus the destination sub-patch's far-side boundary checks** (see below).

**Verified facts — boundary emergence and destination anchoring (2026-07-25).** With corner
anchoring restricted to corners not already covered by a selected check, the grown-bar boundary
of BOTH published references emerges from bulk + `forced` + the lexicographic sweep and vetoes
alone — including the same-type corner pair at the far corner of the growth:
- Litinski Fig 45 step 5 (d=3, grow down, `2d-1`): 14/14 checks equal, full GF(2) exhaustive
  distance 3, 7/8 verbatim anchoring;
- Kishony-Fowler, *Surface code off-the-hook* Fig 5b (d=5, conjugate orientation, grow right,
  `2d+1`): 54/54 checks equal (ground truth machine-extracted from the PDF vector graphics).
The free edge's mid-edge topological corner is a **protocol input**: both papers pin it to the
boundary of the destination sub-patch that the later shrink leaves behind (visible at d=5;
at d=3 the vetoes fix it). The caller expresses it via `forced` — no per-paper hardcoding.
Note the growth amount is itself a per-protocol convention: Litinski uses `2d-1`, K&F `2d+1`.
(`tests/test_rule_based_patch.py`)

Verified:

| d | grown to | data qubits | rows | old checks kept verbatim | p=0 |
|---|---|---|---|---|---|
| 3 | 5 = 2d−1 | 9 → 15 (+6) | **5** ✅ matches Fig 45 step 5 | **7 / 8** | deterministic |
| 5 | 9 = 2d−1 | 25 → 45 (+20) | 9 | **22 / 24** | deterministic |

Only the weight-2 checks on the boundary being extended through change; everything else is
anchored, which is what keeps the detector history alive across the growth.

## 3. The protocol (3 steps, d code cycles each)

Start: `d × d` patch, X̄ vertical, Z̄ horizontal.
Goal: same footprint class, boundary types swapped (X̄ horizontal, Z̄ vertical) — i.e. the
conjugate convention — with the logical state carried through.

| step | op | cycles | what happens |
|---|---|---|---|
| 1 | **GROW** | d | initialise a fresh ancilla region on the free side (\|+⟩ for a Z-boundary extension), merge it by lattice surgery → enlarged patch (`d` → `2d-1` in that direction). 7/8 old checks survive verbatim. |
| 2 | **MOVE CORNERS** | d | on the *enlarged* patch, change the boundary stabilizer set. The extra space is what supplies the redundancy that the in-place version lacked. |
| 3 | **SHRINK** | d | measure the no-longer-needed qubits out in the appropriate basis (X for a Z-boundary extension per Fig 40c) → back to `d × d`, boundaries swapped, but sitting in the **far** tile. |
| 4 | **GROW** | d | grow back toward the original tile (same atom as step 1). |
| 5 | **SHRINK** | d | measure out the far end → patch is back in its **original tile**, rotated. |

Steps 4–5 are Fig 40c's *qubit movement*; they exist purely to return the patch to the tile
the rest of the schedule expects it in.

## 4. Atomic operations to add

All three live on `QECSystem` (that is where live-patch mutation already lives, next to
`rotate_patch` / `retire_measured_patch`).

### 4.1 `grow_patch(name, new_patch, offset) -> (new_data, new_syn)`   ✅ IMPLEMENTED
Signature as built: ``new_patch`` is the enlarged reference geometry from
``RuleBasedRectPatch`` (see §2b), so no stabilizer pattern is hardcoded anywhere.
- initialise the new data qubits **in the basis of the logical being extended** — growing
  through the Z boundary extends Z̄ → all-|0⟩ (RESOLVED 2026-07-25; the spec's original
  "Z boundary → |+⟩" line was wrong and self-contradictory with this section's own
  Fig-41 trivial-outcome detector rule; confirmed three ways: Fig 40d transposed, the
  Kishony-Fowler reference implementation's ``reset="z"`` grow block, and measured full
  graphlike distance).  Every new Z check is then first-round deterministic, so the retiring
  growth-edge lobes' enlarged successors deterministically continue their banked values —
  these bridging detectors are what span the handoff window (under |+⟩ the measured Z-memory
  distance was 1);
- syndrome extraction for deformation steps uses the **K&F diagonal schedule**
  (``DiagonalSurfaceCodeExtractionBlock``): the deformed patch's logical representatives
  bend, and an exhaustive N/Z-variant sweep shows every zigzag orientation caps the Z-memory
  distance at 2, while diagonal hooks never align with any logical direction — measured full
  distance for both bases at d = 3 and 5 (handoff §3.4);
- extend the patch's stabilizer records to the enlarged geometry (reuse
  `RotatedSurfaceCode(distance_x=…, distance_z=…)` as the reference geometry);
- run **d** SE rounds;
- detector rule: newly introduced checks whose value is a product of the initialised state and
  previously measured checks are deterministic on their first round (Fig 41's "trivial
  outcomes"); the genuinely new ones get no detector until their second round.
- **reuse:** the corridor path already initialises fresh data columns
  (`builder.initialize(init_dict=…)`) and merges them — same machinery.

### 4.2 `shrink_patch(name, new_patch, offset)`   ✅ IMPLEMENTED
Signature as built mirrors the other two atoms: ``new_patch`` is the SMALLER reference
geometry (`RuleBasedRectPatch` with the shrunk boundary as ``forced``).  The caller first
destructively measures the removed region through the builder
(``apply_data_readout({removed: 'X'})`` — X basis per Fig 40c and the K&F reference), then
calls this method:
- checks fully inside the surviving footprint keep their uid; checks touching the removed
  region deactivate; cut-line checks reappear TRUNCATED (support = old ∩ surviving), and
  their value continuity is automatic — ``tracker.retire_qubits`` folds the readout records
  into the truncated stabilizer rows and logical rows, so the first post-shrink measurement
  is deterministic (bridging detector);
- the §3.2 detection-coverage criterion is a runtime assertion: every surviving data qubit
  must keep both-type coverage via kept-uid checks or truncated successors, else raise;
- removed qubits return to the dormant pool (cells reusable, indices masked).
Verified (tests/test_shrink_patch.py): panel-7 layout equality, rotated logical orientation
(X̄ vertical / Z̄ horizontal), and full-distance p=0-clean start→grow→move→shrink chains at
d = 3 and 5, both bases.

### 4.3 `move_corners(name, new_patch, offset)`   ✅ IMPLEMENTED
Signature as built mirrors `grow_patch`: ``new_patch`` is a same-footprint reference
geometry (`RuleBasedRectPatch` with the full new boundary as ``forced`` — the boundary
choice IS the operation, so nothing is left to the sweep) with one long edge flipped X↔Z,
which slides the same-type corner wrap to the other end of the bar (Fig 45 panel 5→6:
left edge X→Z; verified check-by-check against panel 6).
- unchanged checks keep their uid (detector continuity); retiring lobes deactivate and
  their ancillas drop out of the SE round automatically;
- the replacement lobes' first outcomes gauge-fix against the retired anticommuting
  lobes (no first-round detector — expected);
- NO data qubits added/removed and NO initialisation (Fig 40b's corner-qubit X-removal
  is specific to its example; this movement keeps all 15/45 data qubits);
- caller runs **d** SE rounds (diagonal schedule).
Verified (tests/test_move_corners.py): p=0 both bases, 12/14 resp. 40/44 anchoring, and
graphlike distance = d through the full start→grow→move sequence at d = 3 and 5.

## 5. Acceptance gates (every step, not just the end)

1. **p = 0 determinism** — no detector ever fires; the observable is deterministic.
2. **Full distance** — `shortest_graphlike_error() == d` at **d = 3 and d = 5**.
3. **No blind round** — at every switch point, check that bridging detectors exist
   (the null-space test of §2.4). A switch with an empty null space is rejected outright.
4. **Logical preserved** — GF(2): both X̄ and Z̄ have representatives surviving each step;
   circuit level: the prepared eigenvalue is returned deterministically.

## 6. Then: wire into the odd-cycle path

Replace the current `rotate_90`-based convention flip with this operation, and keep the
existing `rotate_90` for its original job (shortening the bus path):

- run the paper rotation (conjugation, with rotation as a side effect);
- if the resulting orientation makes the next PPM's bus path **shorter** → keep it;
- if it makes it **longer** → apply the existing `rotate_90` to turn it back, so the net
  effect is only the weight-2 (boundary) change.

## 7. Resolved questions

1. **Which direction / how far to grow.** — RESOLVED.
   *Direction:* the paper states no general rule; in Fig 45 q2 simply grows into the free
   space below it. Rule adopted: **grow toward whichever side is free** (if above and below
   are occupied, grow left or right).
   *Amount:* **`d → 2d-1`** (counted off Fig 45 step 5: 5 rows of data qubits for d = 3).
   Odd for odd d, valid, and 7/8-anchored (table in §2b).
   Note the contrast with Fig 40c's *qubit movement*, which grows to three tiles and then
   measures out "the left two thirds" — that is the MOVE protocol, not the ROTATION.
2. **Is `move_corners` separate?** — RESOLVED: **yes, it is its own operation.** Fig 45
   step 5→6 keeps the footprint fixed and changes only the boundary, and Fig 40b gives it
   its own protocol with its own *d* code cycles.
3. **Routing impact.** — RESOLVED as a non-issue. The rotation runs **between two PPMs**,
   never concurrently with one, and the enlarged footprint is **transient** (the patch is back
   in its original tile by the end, §2b). Routing therefore only ever sees the patch in its
   normal `d × d` footprint, so `route_and_build` is untouched.
   *(Deliberately NOT done in this version: overlapping a rotation with a concurrent PPM would
   be faster but would require reserving the growth space against that PPM's corridor. Out of
   scope by decision — revisit only if rotation cost shows up as a bottleneck.)*
