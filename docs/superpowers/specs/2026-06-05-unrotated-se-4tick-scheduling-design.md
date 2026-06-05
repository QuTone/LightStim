# Unrotated & Toric Surface Code — Selectable 4-tick / 6-tick SE Scheduling

**Date:** 2026-06-05
**Status:** Approved design, pending implementation
**Scope:** `UnrotatedSurfaceCodeExtractionBlock` and `ToricCodeExtractionBlock` (rotated block unchanged)

## Problem

`UnrotatedSurfaceCodeExtractionBlock` ([lightstim/qec_code/surface_code/unrotated/SE_block.py](../../../lightstim/qec_code/surface_code/unrotated/SE_block.py))
and `ToricCodeExtractionBlock` ([lightstim/qec_code/surface_code/toric/SE_block.py](../../../lightstim/qec_code/surface_code/toric/SE_block.py))
both hardcode a single 6-tick CNOT schedule (Li's paper). There is no way to select
a shorter 4-tick schedule. The sibling `RotatedSurfaceCodeExtractionBlock` already
supports multiple schedules via a `scheduling` string parameter and a `SCHEDULES`
class dict; the unrotated and toric blocks should follow the same pattern.

## Goal

Add a `scheduling` parameter to **both** `UnrotatedSurfaceCodeExtractionBlock` and
`ToricCodeExtractionBlock` that selects between:

- `'6tick'` — the existing Li's-paper schedule (**default**, backward compatible).
- `'4tick'` — a new minimal-depth schedule supplied by the user.

The toric block uses the **same delta tables** as the unrotated block (toric is the
unrotated code with periodic boundaries); only its neighbor lookup differs (it wraps
coordinates). Per the user's decision, each block keeps its **own copy** of the
`SCHEDULES` dict (consistent with the current codebase style, which already
duplicates the 6-tick table in the toric block). Accepted trade-off: the two copies
could drift; mitigated by tests on both blocks.

## Non-Goals (out of scope)

- `RotatedSurfaceCodeExtractionBlock` — already 4-tick; left untouched.

## Schedule definitions

Tuple format matches the existing convention: **`(dx_x, dx_z)`**, where the first
element is the X-stabilizer offset (ancilla → data, syndrome is control) and the
second is the Z-stabilizer offset (data → ancilla, syndrome is target). `(0, 0)`
means "this stabilizer type does nothing this tick".

```python
SCHEDULES = {
    '6tick': [                    # existing — Li's paper — DEFAULT
        ((0, 0), (-1, 0)),   # Tick 1
        ((0, 0), (+1, 0)),   # Tick 2
        ((0, +1), (0, +1)),  # Tick 3
        ((0, -1), (0, -1)),  # Tick 4
        ((-1, 0), (0, 0)),   # Tick 5
        ((+1, 0), (0, 0)),   # Tick 6
    ],
    '4tick': [                    # new — user-supplied
        ((0, -1), (0, -1)),  # Tick 1
        ((+1, 0), (0, +1)),  # Tick 2
        ((-1, 0), (+1, 0)),  # Tick 3
        ((0, +1), (-1, 0)),  # Tick 4
    ],
}
```

Confirmed with user: for the `4tick` schedule the left value of each pair is `dx_x`
(X-stabilizer) and the right value is `dx_z` (Z-stabilizer).

Note both stabilizer types visit each of their 4 axis-aligned neighbors
`{(0,±1), (±1,0)}` exactly once across the 4 ticks, with X and Z following different
orderings (hook-error orientation).

## Design

Mirror the rotated block's structure. Changes are confined to two files, each
getting the same treatment.

**`lightstim/qec_code/surface_code/unrotated/SE_block.py`:**

1. Add a class-level `SCHEDULES` dict (both schedules above).
2. `__init__(self, system, scheduling='6tick')`:
   - validate `scheduling` against `SCHEDULES`; raise `ValueError` with a clear
     message listing valid keys on unknown input.
   - store `self.scheduling`.
3. In `_build_circuit`, replace the hardcoded `canonical_tick_deltas` list with
   `self.SCHEDULES[self.scheduling]`. The rest of the loop is unchanged — including
   the `if dx_x != (0, 0)` / `if dx_z != (0, 0)` guards, which are correct for both
   schedules (the 4-tick schedule simply never hits the `(0, 0)` branch).
4. Update the class docstring to document the `scheduling` argument and both
   variants.

**`lightstim/qec_code/surface_code/toric/SE_block.py`:**

Same four steps. The toric block keeps its **own copy** of the `SCHEDULES` dict
(identical delta values to the unrotated block). Its `_build_circuit` already wraps
neighbor coordinates via `_wrap_coord` for periodic boundaries — that logic is
untouched; only the source of `canonical_tick_deltas` changes to
`self.SCHEDULES[self.scheduling]`. `ToricCodeExtractionBlock.__init__` gains the same
`scheduling='6tick'` parameter and validation.

**Backward compatibility:** default `'6tick'` preserves current behavior for both
blocks. Existing constructors (`UnrotatedSurfaceCodeExtractionBlock(system)` in
[two_patch_ls.py](../../../lightstim/protocols/two_patch_ls.py),
[ghz.py](../../../lightstim/protocols/ghz.py); `ToricCodeExtractionBlock(system)`)
are unaffected. Protocols that accept `extraction_block_kwargs` (see
[memory.py](../../../lightstim/protocols/memory.py)) can pass
`{'scheduling': '4tick'}` once this lands.

## Data flow

```
UnrotatedSurfaceCodeExtractionBlock(system, scheduling='4tick')
  └─ _build_circuit()
       └─ deltas = SCHEDULES['4tick']
            └─ for (dx_x, dx_z) in deltas:   # 4 iterations
                 build CNOT layer + TICK      # identical logic to before
```

## Error handling

- Unknown `scheduling` value → `ValueError` listing valid keys (`'6tick'`, `'4tick'`).
- Conflict-freeness (no data qubit driven twice in one tick) is enforced
  automatically by stim: a duplicate target inside a single `CNOT` instruction
  raises at circuit-build time. The 4-tick schedule must satisfy this; the test
  suite exercises it.

## Testing (TDD)

New test module(s) following the existing `tests/` layout. Each of the checks below
is run for **both** `UnrotatedSurfaceCodeExtractionBlock` and
`ToricCodeExtractionBlock`:

1. **Default unchanged:** constructing with no `scheduling` arg uses the 6-tick
   schedule — the generated circuit equals the pre-change output (regression
   guard). Assert 6 TICKs in the CNOT region.
2. **4-tick builds & is conflict-free:** `scheduling='4tick'` constructs without
   error (stim would raise on any same-tick double-drive). Assert 4 TICKs in the
   CNOT region.
3. **Fault tolerance / distance:** build a `d=3` (and ideally `d=5`) memory circuit
   with the 4-tick schedule and confirm the code distance via stim's
   detector-error-model graphlike-error analysis (matches the 6-tick distance for
   the same `d`). For toric, verify the periodic-boundary memory circuit decodes
   correctly under the 4-tick schedule.
4. **Coverage:** both schedules touch every data neighbor of every bulk stabilizer
   exactly once (each ancilla ends up entangled with its full support).
5. **Invalid input:** unknown `scheduling` raises `ValueError` on both blocks.
6. **Cross-block consistency:** the unrotated and toric `SCHEDULES` copies hold the
   same delta tables (guards against the accepted drift risk).

## Risks

- The user-supplied 4-tick schedule must be conflict-free and distance-preserving
  for this lattice's coordinate convention. Test #2 (stim build) and test #3
  (distance) are the gates that catch a bad schedule before merge.
