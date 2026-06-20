# Rotated Surface Code Lattice Surgery — Design

**Date:** 2026-06-20
**Status:** Approved (user delegated implementation)
**Goal:** Add lattice-surgery logical Clifford operations (handbook §7.2) to the **standard rotated CSS surface code**, reusing the existing LS machinery. Deliver, in two stages: (1) two-patch joint X̄X̄ / Z̄Z̄ measurement, (2) three-patch LS CNOT, each producing an LER-vs-p figure.

## Key finding (why this is small)

In LightStim, the hard parts of lattice surgery are **code-agnostic** and already implemented:
- joint logical eigenvalue readout — `tracker.py` (Gaussian elimination over measurement records)
- merge/split stabilizer activation — `qec_system.activate_coupler/deactivate_coupler` (via `conflicting_stabilizer_coords`)
- 3-measurement CNOT + Pauli frame — `protocols/cnot_ls.py`
- coupler plug-in interface — `LogicalCouplerProtocol._build_coupler_geometry`

The **only** rotated-specific piece is the coupler **geometry**. So the work is: one new `RotatedTwoPatchCoupler` class + parameterizing the experiment classes (defaults stay unrotated → zero regression).

## §7.2 → rotated mapping

Rotated CSS patch (verified from `rotated/code_patch.py`): data at `(2x+1, y)` on odd rows; **top/bottom edges = weight-2 X-stabilizers** (support X̄, vertical, x=1 column); **left/right edges = weight-2 Z-stabilizers** (support Z̄, horizontal, y=1 row).

| §7.2 | rotated realization |
|---|---|
| measure X̄₁X̄₂, merge along X-edges, intermediate \|0⟩, split measure Z | stack patches **vertically** (A top, B bottom); X-boundaries face; `interaction_type="XX"` |
| measure Z̄₁Z̄₂, merge along Z-edges, intermediate \|+⟩, split measure X | place patches **side-by-side**; Z-boundaries face; `interaction_type="ZZ"` |
| CNOT = Z̄cZ̄a → X̄aX̄t → Z̄a + Pauli frame | unchanged — `cnot_ls.py` runs as-is |

## Coupler approach: B1 pure constructive (diff against merged code)

The merge of two rotated CSS patches **is** a single larger rotated CSS code. So `_build_coupler_geometry`:

1. From `interaction_type` + patch bounds, pick merge axis and the merged code's dimensions:
   - XX (vertical stack): `M = RotatedSurfaceCode(d_z=d_z, d_x=d_xA + d_xB + 1)` — one intermediate data row.
   - ZZ (side-by-side): `M = RotatedSurfaceCode(d_z=d_zA + d_zB + 1, d_x=d_x)` — one intermediate data column.
   `M` is built at patch A's origin so its top/left block overlays A exactly.
2. **Diff** `M` against `A ∪ B`:
   - `new data qubits  = M.data  − (A.data ∪ B.data)` → `add_qubit(role='data')` (the §7.2 intermediate qubits)
   - `new syndrome qubits = M.syndrome − (A.syndrome ∪ B.syndrome)` → `add_qubit(role='syndrome_x'|'syndrome_z')` per M's classification
   - for each `M` stabilizer: by `syn_coord`,
     - syn_coord on a **new** syndrome qubit → emit coupler stabilizer (seam syndrome)
     - syn_coord on an **existing** patch syndrome qubit but support **differs** from the patch's stabilizer (boundary fused weight-2 → weight-4) → emit coupler stabilizer with M's extended support **and** add syn_coord to `conflicting_stabilizer_coords`
     - support **identical** → skip (kept by the original patch)
3. Emit in the established coupler contract: `stabilizers.append({'pauli': {coord: type}, 'type': type, 'syn_coord': coord})` (coordinate-keyed; `qec_system._translate_record` resolves coords system-wide and masks conflicts by syn_coord).

**Safety net (constructive self-check):** assert `A.data ∪ B.data ⊂ M.data` and the remainder is exactly the expected intermediate row/column; otherwise raise with a clear message. Correctness of the seam follows from `M` being a valid code.

### Verified worked example (d=3, XX merge)
- A data rows y=1,3,5; B placed at +8 in y (data rows y=9,11,13); intermediate data row y=7 → `(1,7)(3,7)(5,7)`.
- M=(d_z=3, d_x=7): X-stab @ (4,6) becomes weight-4 `(3,5)(5,5)(3,7)(5,7)`; X-stab @ (2,8) becomes weight-4 `(1,7)(3,7)(1,9)(3,9)`; new Z-syndromes @ (2,6),(6,6),(0,8),(4,8).
- `conflicting_stabilizer_coords = {(4,6),(2,8)}` (A's & B's fused boundary X-stabs). A's/B's bulk stabs are identical to M's → untouched.
- Placement needs only translation (same-type boundaries face) — no rotate/transpose.

## Changes

| File | Change |
|---|---|
| `lightstim/qec_code/surface_code/rotated/two_patch_coupler.py` | **new** `RotatedTwoPatchCoupler(LogicalCouplerProtocol)`, `EXPECTED_PATCH_COUNT=2` |
| `lightstim/qec_code/surface_code/rotated/__init__.py` | export it |
| `lightstim/protocols/two_patch_ls.py` | add `code_patch_class` param (already has `coupler_protocol`, `extraction_block_class`); replace hardcoded `UnrotatedSurfaceCode(...)`; factor patch alignment into `_align_patch1()` hook |
| `lightstim/protocols/cnot_ls.py` | add `code_patch_class` / `coupler_class` / `extraction_block_class` params; replace 3 hardcoded refs; factor `_align_patches()` (rotated = translation only) |
| `tests/` | unit: coupler geometry == `RotatedSurfaceCode(merged)` diff (XX & ZZ); integration: circuit-level two-patch LS + CNOT decode |

Defaults remain unrotated → existing behavior and precomputed data unaffected.

## Validation

- **Stage 1 (two-patch):** unit test on coupler geometry; circuit-level rotated X̄X̄/Z̄Z̄ — noiseless joint observable deterministic; LER vs p shows threshold + distance suppression (d=3,5,7); sanity-compare magnitude to rotated memory baseline.
- **Stage 2 (CNOT):** rotated LS CNOT over the 5 sub-experiments → LER-vs-p figure (= §7.2 on rotated); overlay rotated vs unrotated LS CNOT for the handbook.

## Implementation phases (TDD)

1. `RotatedTwoPatchCoupler` + unit test (geometry diff, XX & ZZ, d=3 and d=5).
2. Parameterize `TwoPatchLSExperiment`; circuit-level test; first rotated LS LER point.
3. Parameterize `CNOTLSExperiment`; rotated LS CNOT; LER sweep + plot.
