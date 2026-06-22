# Rotated Surface Code Lattice Surgery — Design

**Date:** 2026-06-20
**Revised:** 2026-06-22 (corrected the X̄X̄/Z̄Z̄ → geometry mapping; see [Correction](#correction-the-merge-geometry-mapping-is-the-transpose-of-the-naive-same-edge-intuition))
**Status:** Implemented & verified — both stages shipped on `lattice-surgery`.
**Goal:** Add lattice-surgery logical Clifford operations (handbook §7.2) to the **standard rotated CSS surface code**, reusing the existing LS machinery. Delivered in two stages: (1) two-patch joint X̄X̄ / Z̄Z̄ measurement, (2) three-patch LS CNOT, each producing an LER-vs-p result.

## Key finding (why this is small)

In LightStim, the hard parts of lattice surgery are **code-agnostic** and already implemented:
- joint logical eigenvalue readout — `tracker.py` (Gaussian elimination over measurement records)
- merge/split stabilizer activation — `qec_system.activate_coupler/deactivate_coupler` (via `conflicting_stabilizer_coords`)
- 3-measurement CNOT + Pauli frame — `protocols/cnot_ls.py`
- coupler plug-in interface — `LogicalCouplerProtocol._build_coupler_geometry`

The **only** rotated-specific piece is the coupler **geometry**. So the work was: one new `RotatedTwoPatchCoupler` class + parameterizing the experiment classes (defaults stay unrotated → zero regression). This finding held: `CNOTLSExperiment` and `TwoPatchLSExperiment` build a fully-working rotated LS by swapping in `code_patch_class` / `coupler_class` / `extraction_block_class`, with `rotate_patches=False`.

## §7.2 → rotated mapping (corrected)

Rotated CSS patch (verified from `rotated/code_patch.py`): data at odd coordinates; **X̄ runs vertical** (the `x=1` column, terminating on the top/bottom X-boundaries), **Z̄ runs horizontal** (the `y=1` row, terminating on the left/right Z-boundaries). `distance_x` is the **vertical** dimension (height), `distance_z` is the **horizontal** dimension (width).

| §7.2 operation | rotated realization (verified) |
|---|---|
| measure **Z̄₁Z̄₂**, intermediate \|+⟩, split-measure X | stack patches **vertically** (offset in y); `interaction_type="ZZ"`; merged `distance_x` grows |
| measure **X̄₁X̄₂**, intermediate \|0⟩, split-measure Z | place patches **side-by-side** (offset in x); `interaction_type="XX"`; merged `distance_z` grows |
| CNOT = Z̄꜀Z̄ₐ → X̄ₐX̄ₜ → Z̄ₐ + Pauli frame | unchanged — `cnot_ls.py` runs as-is (`offset_ca` vertical, `offset_ta` side-by-side) |

### Correction: the merge-geometry mapping is the transpose of the naive same-edge intuition

The original (2026-06-20) version of this table had **ZZ and XX swapped**. It assumed the natural rule *"to measure Ō₁Ō₂, merge along the boundaries where Ō terminates"* — i.e. measure X̄X̄ by merging the X-boundaries (vertical stack), measure Z̄Z̄ by merging the Z-boundaries (side-by-side). That rule is **wrong for this construction**, and shipping it caused the `RuntimeError: Logical Count Mismatch` in rotated LS-CNOT.

What actually happens (read off the constructed seam, and confirmed by `stim.TableauSimulator.peek_observable_expectation`):

- **Vertical stack** *fuses the facing X-boundary stabilizers* into weight-4 and inserts a **row of new Z-type seam stabilizers**; the product of those Z-seam stabilizers is **Z̄₁Z̄₂**. So a vertical stack measures **Z̄Z̄**, not X̄X̄.
- **Side-by-side** *fuses the facing Z-boundary stabilizers* and inserts a **column of new X-type seam stabilizers** whose product is **X̄₁X̄₂**. So side-by-side measures **X̄X̄**.

The operational rule is therefore: **merging along a T-type boundary measures the *opposite* logical's joint operator.** Equivalently, to measure Ō₁Ō₂ you stack the patches **perpendicular** to the direction Ō runs (Z̄ is horizontal → stack vertically; X̄ is vertical → place side-by-side). This is the transpose of the original intuition.

The fix was localized entirely to `RotatedTwoPatchCoupler` (the `interaction_type → geometry` map and merged-dimension growth); `tracker.py` / `process_mid_measurement` / the CNOT sequence were never wrong — the tracker faithfully reported the *actually measured* joint operator, which is what surfaced the mislabel.

**Lesson (now enforced in the notebooks):** do not trust the `"ZZ"`/`"XX"` label or the geometric picture — verify the actual measured joint with `peek_observable_expectation` (determination-trajectory: step the noiseless circuit, find where each candidate joint goes undetermined→determined; the merge that *measures* it is the transition point).

## Coupler approach: B1 pure constructive (diff against merged code)

The merge of two rotated CSS patches **is** a single larger rotated CSS code. So `_build_coupler_geometry`:

1. From `interaction_type` + patch bounds, pick the merge axis and the merged code's dimensions:
   - **ZZ (vertical stack):** `M = RotatedSurfaceCode(distance_z=d_z, distance_x=d_xA + d_xB + 1)` — one intermediate data **row**.
   - **XX (side-by-side):** `M = RotatedSurfaceCode(distance_z=d_zA + d_zB + 1, distance_x=d_x)` — one intermediate data **column**.

   `M` is built at patch A's origin so its top/left block overlays A exactly. (Patches must share the transverse distance: ZZ requires equal `distance_z`, XX requires equal `distance_x`.)
2. **Diff** `M` against `A ∪ B`:
   - `new data qubits  = M.data  − (A.data ∪ B.data)` → `add_qubit(role='data')` (the §7.2 intermediate qubits)
   - `new syndrome qubits = M.syndrome − (A.syndrome ∪ B.syndrome)` → `add_qubit(role='syndrome_x'|'syndrome_z')` per M's classification
   - for each `M` stabilizer, by `syn_coord`:
     - syn_coord on a **new** syndrome qubit → emit coupler stabilizer (seam syndrome)
     - syn_coord on an **existing** patch syndrome qubit but support **differs** (boundary fused weight-2 → weight-4) → emit coupler stabilizer with M's extended support **and** add syn_coord to `conflicting_stabilizer_coords`
     - support **identical** → skip (kept by the original patch)
3. Emit in the established coupler contract: `stabilizers.append({'pauli': {coord: type}, 'type': type, 'syn_coord': coord})` (coordinate-keyed; `qec_system._translate_record` resolves coords system-wide and masks conflicts by syn_coord).

**Safety net (constructive self-check):** assert `A.data ∪ B.data ⊂ M.data` and the remainder is exactly the expected intermediate row/column; otherwise raise `ValueError("…patches are misaligned for this interaction…")`. This is what rejects a wrong/swapped offset, so the geometry is uniquely pinned. Correctness of the seam follows from `M` being a valid code.

### Verified worked examples (d=3, extracted from the shipped coupler)

**ZZ merge = vertical stack** — B at offset `(0, 8)` (= `(0, 2d+2)`):
- intermediate data **row** `(1,7) (3,7) (5,7)`; merged `M = RotatedSurfaceCode(distance_z=3, distance_x=7)`.
- `conflicting_stabilizer_coords = {(4,6), (2,8)}` — A's & B's facing **X**-boundary stabs, fused to weight-4 (e.g. `(4,6)` → `(3,5)(3,7)(5,5)(5,7)`).
- new **Z**-type seam syndromes at `(0,8) (2,6) (4,8) (6,6)`; the merge measures the joint **Z̄₁Z̄₂** (confirmed by `peek_observable_expectation`).

**XX merge = side-by-side** — B at offset `(8, 0)` (= `(2d+2, 0)`):
- intermediate data **column** `(7,1) (7,3) (7,5)`; merged `M = RotatedSurfaceCode(distance_z=7, distance_x=3)`.
- `conflicting_stabilizer_coords = {(6,2), (8,4)}` — facing **Z**-boundary stabs, fused to weight-4.
- new **X**-type seam syndromes at `(6,0) (6,4) (8,2) (8,6)`; the merge measures the joint **X̄₁X̄₂** (confirmed by `peek_observable_expectation`).

Placement needs only translation (no rotate/transpose); patches keep `rotation_angle = 0`.

## Changes (as built)

| File | Change |
|---|---|
| `lightstim/qec_code/surface_code/rotated/two_patch_coupler.py` | `RotatedTwoPatchCoupler(LogicalCouplerProtocol)`, `EXPECTED_PATCH_COUNT=2`. **Corrected** `interaction_type → geometry`: `ZZ` = vertical stack / grow `distance_x`; `XX` = side-by-side / grow `distance_z` |
| `lightstim/qec_code/surface_code/rotated/__init__.py` | export `RotatedTwoPatchCoupler` |
| `lightstim/protocols/two_patch_ls.py` | `code_patch_class` / `coupler_protocol` / `extraction_block_class` params; `rotate_patch1=False` for rotated |
| `lightstim/protocols/cnot_ls.py` | `code_patch_class` / `coupler_class` / `extraction_block_class` params; `rotate_patches=False` for rotated (already code-agnostic) |
| `tests/test_rotated_lattice_surgery.py` | coupler geometry == `RotatedSurfaceCode(merged)` diff (XX & ZZ, d=3/5); circuit-level two-patch LS noiseless + DEM-valid |
| `benchmarks/rotated_ls/run_and_plot.py` | LER-vs-p sweep with corrected offsets (`ZZ=(0,2d+2)`, `XX=(2d+2,0)`) |
| `notebooks/LogicalOps/two_patch_LS_rotated.ipynb`, `two_patch_LS_unrotated.ipynb` | two-patch examples + `peek_observable_expectation` joint-operator verification |
| `notebooks/LogicalOps/logical_CNOT_LS_rotated.ipynb` | **new** — rotated LS-CNOT, 5 sub-experiments, merge-joint / nobs / LER verification cells |

Defaults remain unrotated → existing behavior and precomputed data unaffected.

## Validation (results)

- **Stage 1 (two-patch):** coupler geometry == `RotatedSurfaceCode(merged)` diff for XX & ZZ (d=3,5); circuit-level rotated X̄X̄/Z̄Z̄ noiseless-deterministic and DEM-valid; `peek_observable_expectation` confirms `ZZ→Z̄₁Z̄₂`, `XX→X̄₁X̄₂`; LER vs p shows distance suppression. **7 rotated tests pass.**
- **Stage 2 (CNOT):** rotated LS-CNOT over the 5 sub-experiments (`ZZ_ZZ, ZX_ZX, XZ_XX, XZ_ZZ, XX_XX`):
  - all build **clean** (noiseless detectors/observables silent) and **DEM-valid**;
  - observable counts **match unrotated exactly** `{2, 2, 1, 1, 2}` (rotated is more compact: 65 qubits / 152 detectors vs 85 / 222 at d=3);
  - merge order verified `ZZ(c,a)→Z̄꜀Z̄ₐ` then `XX(a,t)→X̄ₐX̄ₜ` (ZZ_XX protocol, the required sequence);
  - **distance suppression** at p=1e-3, rounds=d: comprehensive LER 1.4e-2 (d=3) → 1.5e-3 (d=5), ≈9× — and the swapped offset is rejected by the coupler's self-check, so the result is not a coincidence.

## Implementation phases (TDD) — complete

1. ✅ `RotatedTwoPatchCoupler` + unit test (geometry diff, XX & ZZ, d=3 and d=5).
2. ✅ Parameterize `TwoPatchLSExperiment`; circuit-level test; rotated LS LER (`benchmarks/rotated_ls/`).
3. ✅ Parameterize `CNOTLSExperiment`; rotated LS-CNOT; LER + verification (`notebooks/LogicalOps/logical_CNOT_LS_rotated.ipynb`).
