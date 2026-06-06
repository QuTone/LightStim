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
    '4tick': [                    # minimal-depth; X and Z mirror on the vertical ticks
        ((+1, 0), (+1, 0)),  # Tick 1: both East
        ((0, +1), (0, -1)),  # Tick 2: X North, Z South
        ((0, -1), (0, +1)),  # Tick 3: X South, Z North
        ((-1, 0), (-1, 0)),  # Tick 4: both West
    ],
}
```

Tuple order is `(dx_x, dx_z)` (X-stabilizer offset first, Z second).

### Why this `4tick` schedule (and not the originally-proposed one)

The schedule originally proposed used *different* X and Z offsets per tick (the
rotated-surface-code style, where X and Z follow different rotational orders to set
hook-error orientation):

```
((0, -1), (0, -1)), ((+1, 0), (0, +1)), ((-1, 0), (+1, 0)), ((0, +1), (-1, 0))
```

That does **not** work on this codebase's unrotated lattice. Geometry of the patch:

- Data qubits sit at `(even, even)` and `(odd, odd)`; X-ancillas at `(odd, even)`,
  Z-ancillas at `(even, odd)`.
- An `(even, even)` data qubit is reached by **X-ancillas via horizontal offsets**
  `(±1, 0)` and by **Z-ancillas via vertical offsets** `(0, ±1)`. For `(odd, odd)`
  data the roles flip.
- Therefore a tick whose `dx_x` and `dx_z` have **different orientations** (one
  horizontal, one vertical) makes an X-CNOT and a Z-CNOT land on the *same* data
  qubit → a same-tick conflict. The proposed schedule's ticks 2 and 4 are exactly
  these mixed-orientation ticks (measured: 8 conflicts at d=3).
- Conflict-freeness alone is not sufficient: when an X-ancilla and Z-ancilla share a
  data qubit, the *order* of their CNOTs must be globally consistent, or the ancillas
  stay entangled and the measured operator is not a clean stabilizer (stim/tracker:
  "measurement commutes with all rows but is linearly independent").

A brute-force search over orientation-consistent depth-4 schedules (96 candidates)
found 16 valid ones; all reach full graphlike distance (`d` at d=3 and d=5). The
**binding constraint** is that `dx_x` and `dx_z` share the same orientation on every
tick (both horizontal or both vertical); then X acts on the `(even, even)` sublattice
while Z acts on `(odd, odd)` — disjoint qubits, globally consistent order, no hook
penalty.

The adopted `4tick` schedule shares the horizontal ticks (`dx_x == dx_z`) and
**mirrors X vs Z on the vertical ticks** (`dx_x = (0, ±1)`, `dx_z = (0, ∓1)`). This
fixes a definite hook-error orientation, closer in spirit to a hook-oriented
("perpendicular"-style) schedule than the symmetric `dx_x == dx_z` variant. Both
variants were verified to preserve **full code distance in both bases** (Z-memory and
X-memory: graphlike distance == d at d = 3, 5); the choice between them only affects
hook orientation, which matters for biased-noise / sub-threshold logical error rate,
not for distance. A fully symmetric `dx_x == dx_z` schedule
(`E, N, S, W` for both X and Z) is an equally valid alternative.

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

- A 4-tick schedule must be conflict-free *and* distance-preserving for this
  lattice's coordinate convention. This risk materialized during implementation: the
  originally-proposed schedule conflicted (see "Why this `4tick` schedule" above).
  The conflict-free check (test #2) and the graphlike-distance check (test #3) caught
  it; the adopted schedule passes both at d = 3 and d = 5. Note stim does **not**
  auto-reject a same-tick double-drive (it applies repeated targets sequentially), so
  the explicit per-layer uniqueness assertion in test #2 is load-bearing.
