# Rotated Surface Code — Bent (XZ) Joint Lattice Surgery Measurement — Design

**Date:** 2026-06-25
**Status:** Designed — not yet implemented.
**Goal:** A new notebook `notebooks/LogicalOps/rotated_bent_XZ_LS.ipynb` that performs a **mixed-Pauli joint measurement `M(X̄₁·Z̄₂)`** on the **rotated** CSS surface code, with a **bent (L-shaped) domain wall**, exactly as drawn in `rotated.png`. It mirrors the structure of the existing unrotated, straight-seam notebook `notebooks/LogicalOps/routed_ZX_LS.ipynb`, but the stabilizers are re-derived for the rotated lattice and the bent seam.

The single authoritative source for the layout is the figure **`/nvme2n1/yuehan_zhang/resource_analsis/rotated.png`**. See [Hard requirement](#hard-requirement-the-mixed-checks-come-from-the-figure).

---

## What the figure shows (the ground truth)

Drawing convention (verified by zooming into every region of `rotated.png`):

- **Black filled circle = data qubit** (at tile corners).
- **White open circle = ancilla / measure qubit** (at tile centers).
- **Red tile = X stabilizer, blue tile = Z stabilizer**; **semicircle tile = weight-2 boundary check**.
- **Purple "staircase" region = the bent domain wall**; each purple tile is a **mixed-XZ stabilizer**. The `X`/`Z` letters printed at a tile's corners give the Pauli that check applies to each corner data qubit.
- **Green circles** sit on ancilla positions and form a connected chain from the `X̄` arm, through the purple seam, to the `Z̄₂` arm. Interpretation (confirmed with the user): they mark the **readout chain** — the set of stabilizers whose product equals the measured joint logical `X̄₁·Z̄₂` (the rotated analog of the gold-highlighted checks in `routed_ZX_LS.ipynb` cell 4).

Macroscopic structure: an **L-shaped (bent) merged region** of two rotated patches.

- **Patch 1** — lower-left **horizontal** arm, carrying logical **X̄** drawn as the horizontal dark-red bar.
- **Patch 2** — upper-right **vertical** arm, carrying logical **Z̄₂** drawn as the vertical dark-blue line.
- The two arms meet through the **purple bent seam** (the domain wall between an X-region and a Z-region). The seam runs vertically and then **turns** at the bottom (diagonal collapse) to connect into the horizontal arm — this turn is the "bending".
- The measured joint operator `X̄₁·Z̄₂` is supported on an **L-shaped path**: horizontal along `X̄`, then turning upward along `Z̄₂`.

**Orientation flip (subtlety, not a contradiction).** The library default rotated patch (verified in `rotated/code_patch.py` and the 2026-06-20 spec) has **X̄ vertical** (`x=1` column) and **Z̄ horizontal** (`y=1` row). The figure has the opposite: **X̄ horizontal, Z̄₂ vertical**. This is reconciled by the construction, not assumed:
- Patch 2's measured `Z̄₂` lives on patch 2's **logical-X support** (a vertical column in the default rotated frame) but carries **Z** because of the X↔Z re-typing — giving a vertical `Z̄₂` with no patch rotation.
- Patch 1 is oriented (rectangular `distance_z`/`distance_x` and/or `rotate_coords`) so its `X̄` is horizontal.
- The actual orientations/offsets that reproduce the figure are **pinned by cross-checking against the figure and verified** with `peek_observable_expectation` — never assumed from the `"XX"`/`"ZZ"` labels (per the 2026-06-20 lesson).

---

## Key finding (why custom code is needed)

A 3-agent sweep of the rotated code in LightStim found that **the mixed + bent pieces do not exist** for the rotated code:

| Capability | Status for rotated code |
|---|---|
| `RotatedSurfaceCode` patch (data/stab/logical layout) | ✅ exists, reusable (`rotated/code_patch.py`) |
| `RotatedTwoPatchCoupler` | ⚠️ **straight seam, pure-CSS checks only** — no bend, no mixed |
| `RotatedSurfaceCodeExtractionBlock` | ⚠️ **no MIXED-check extraction** (only X- and Z-type scheduling) |
| Rotated multi-patch / bent / mixed coupler | ❌ does not exist (repo-wide grep: none) |
| MIXED stabilizer creation + extraction | exists **only for the unrotated** code (`unrotated/multi_patch_coupler.py`, `unrotated/SE_block.py`) |

Code-agnostic machinery that **is** reusable as-is: `ir/qec_system.py` (`add_patch`, `register_coupler`, `activate_coupler`, stabilizer/index bookkeeping), `ir/builder.py` (`initialize`, `apply_syndrome_extraction`, `apply_data_readout`, `build_noisy_circuit`), `ir/tracker.py` (joint readout via Gaussian elimination), and the algebra helpers in `protocols/routed_multi_patch_ls.py` (`logical_pauli_product_vector`, `_coupler_on_active_stabilizer_uids`).

So the work is: (1) **figure-grounded** geometry + mixed-check generator for the bent seam, and (2) a rotated **mixed-check SE schedule** (CNOT on X-side data, CZ on Z-side data), both implemented **inside the notebook** (no library files modified unless the user approves).

---

## Hard requirement: the mixed checks come from the figure

**The mixed (XZ) bending stabilizers MUST be transcribed from `rotated.png` and cross-checked tile-by-tile. They are NOT free-generated by a generic rule.**

The re-typing trick (below) is used only to (a) assign X vs Z to each data qubit *within the tiles the figure already shows*, and (b) prove the construction is algebraically correct. Any check whose support or per-corner Pauli disagrees with the figure is a bug, not an alternative valid construction. Concretely, implementation **step 1** produces an explicit enumeration table — for every X / Z / MIXED stabilizer: its syndrome coordinate, its data-qubit coordinates, and the Pauli on each — and reconciles it against the figure before any circuit is built. The figure is read independently by multiple agents and the readings are reconciled (disagreements escalated, not silently averaged).

---

## Construction approach: algebraic re-typing, figure-pinned

Two complementary views, same as `routed_ZX_LS.ipynb`:

### A. Algebraic view (cells: construct → verify → visualize)

Model the merged L-region as **one rotated CSS region + a bent domain-wall cut**, then re-type:

1. Lay out the L-shaped data-qubit grid (rotated convention, data at odd coords), as read from the figure, distance-parameterized.
2. Enumerate all plaquette checks over the L-region by the **rotated** rules (red=X / blue=Z, weight-4 bulk, weight-2 boundary semicircles) — matching the figure's tiles.
3. Partition data qubits into the **X-side** (patch 1, holds X̄) and the **Z-side** (patch 2, holds Z̄₂) across the bent cut.
4. **Re-type:** swap X↔Z on every Z-side data qubit. Then:
   - checks entirely on the X-side stay pure (X/Z);
   - checks entirely on the Z-side become the dual (still pure, type-swapped) — this only swaps the checkerboard parity, which is why both arms still look like normal red/blue checkerboards in the figure;
   - checks **straddling the bent cut become MIXED** (X on X-side corners, Z on Z-side corners) — these are exactly the purple tiles. No qubit receives Y → **no twist, no weight-5**.
5. Cross-check every MIXED tile's support and per-corner Pauli against the figure (the hard requirement above).

### B. Circuit view (cells: SE → acceptance → LER)

Run the real lattice-surgery time sequence (mirrors `routed_ZX_LS.ipynb` `build_mixed_circuit`):

1. `QECSystem` with two rotated patches (the arms) at the figure's offsets.
2. `initialize` both patches; run `rounds` of SE with the (pure-CSS) `RotatedSurfaceCodeExtractionBlock`.
3. `activate_coupler` for the bent seam; `initialize` the seam/coupler data qubits.
4. Run `rounds` of SE that **directly measures the MIXED checks** — an X-basis syndrome ancilla per mixed tile, `CNOT(ancilla→data)` on its X-side corners and `CZ(ancilla,data)` on its Z-side corners (ported from `unrotated/SE_block.py`, adapted to rotated geometry). No transversal-H.
5. `apply_data_readout`.

---

## Notebook structure (cell-by-cell, mirroring `routed_ZX_LS.ipynb`)

1. **Title** — `Rotated X̄₁ Z̄₂ Bent Joint Lattice Surgery Measurement`.
2. **Construct** — build the bent rotated layout, re-type, produce X / Z / MIXED checks; print the **figure-grounded enumeration table**.
3. **Four hard requirements** — `#data − rank == 1` and all checks commute; `X̄₁·Z̄₂` in the check span (joint measured); no single `X̄₁`/`Z̄₂` measured; MIXED bent checks present with **no Y (twist)**.
4. **Identify operators** — print `X̄₁` and `Z̄₂` supports and the MIXED domain-wall checks with their X/Z corner assignment.
5. **Visualize** — draw the rotated bent layout; color X/Z/MIXED tiles; label each mixed corner X/Z; **gold-highlight the readout chain** (the figure's green circles). Visual must match `rotated.png`.
6. **Circuit-level mixed SE** — `build_mixed_circuit(rounds)`; report qubit / detector / observable counts and number of active MIXED checks.
7. **Acceptance** — DEM valid; 200-shot noiseless determinism (all detectors 0, observable 0); detectors really measure **instantaneous mixed data checks** (X & Z on different data at the same tick) via `detecting_regions`; **joint-measurement signature** (init `|0…0⟩`: `Z̄₂` parity deterministic = commutes/preserved; `Z̄₁` parity randomized ≈0.5 = anticommutes) and the measured joint confirmed by `peek_observable_expectation`.
8. **Circuit diagram** — `circuit.diagram("detslice-with-ops-svg")`.
9. **LER vs p** — distance-parameterized noisy build + adaptive PyMatching (MWPM) decode; sweep `p` for `d ∈ {3,5,7}`; log-log plot, one curve per distance, 95% error bars (rule-of-three upper bounds where 0 errors).

---

## Verification & acceptance (what "done" means)

- **Figure fidelity:** every X / Z / MIXED check matches `rotated.png` in support and per-corner Pauli (independent multi-agent read, reconciled). The visualization is visually congruent with the figure (including the green readout chain).
- **Algebra:** the four hard requirements pass (joint `X̄₁·Z̄₂` measured, no singles, mixed checks present, no twist).
- **Circuit:** DEM-valid, noiseless-deterministic, measures instantaneous mixed data checks; `peek_observable_expectation` confirms the merge measures `X̄₁·Z̄₂` and not a single logical.
- **Scaling:** LER vs p shows distance suppression for `d ∈ {3,5,7}`.

---

## Risks / open points

- **Bent-corner mixed checks** are the crux: the exact support and X/Z split at the turn must match the figure. Mitigation: multi-agent figure read + re-typing cross-check + the four hard requirements as a backstop.
- **Orientation flip** (figure X̄ horizontal / Z̄₂ vertical vs library default): pinned by figure + `peek_observable_expectation`, never by the interaction-type label.
- **Rotated mixed SE schedule**: must produce commuting, hook-error-safe CNOT/CZ ordering on the rotated lattice. Mitigation: port and adapt the unrotated `_append_mixed_stabilizer_measurements` batching; verify via the acceptance cell.
- **Distance generalization**: the figure shows one illustrative distance; the generator must reproduce the same bent structure at `d = 3,5,7`. Mitigation: assert the generated small-d layout reproduces the figure before scaling.
- **No library files modified** unless the user approves; everything lives in the notebook. (If the rotated `SE_block` / coupler are later promoted into the library, that is a separate change.)

---

## Implementation phases

1. **Figure-grounded layout + enumeration** — transcribe data qubits and X/Z/MIXED checks from `rotated.png` (multi-agent read + reconcile); emit the enumeration table; assert it matches the figure. *No circuit yet.*
2. **Algebraic construction + four hard requirements** — re-typing generator, `X̄₁·Z̄₂` span check, no-twist check; cells 2–5 incl. the figure-matching visualization.
3. **Mixed-check SE + acceptance** — rotated mixed CNOT/CZ schedule; cells 6–8 (build, acceptance, detslice).
4. **LER vs p** — distance-parameterized noisy build + MWPM sweep + plot; cell 9.

Each phase verifies before the next; the figure cross-check (phase 1) and the four hard requirements (phase 2) gate everything downstream.
