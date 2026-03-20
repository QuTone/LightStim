# LightStim Engineering Chronicle

A record of significant design decisions, solved challenges, and architectural milestones.

---

## 2026-03 — Multi-Patch Lattice Surgery Coupler

**Problem**: Implementing N-patch lattice surgery (weight-N Pauli measurements) for distillation circuits required dynamic qubit management — couplers must be activated/deactivated mid-circuit, and coupler qubits reused across sequential measurements.

**Solution**:
- **Qubit reuse via `active_qubit_indices`** in `QECSystem` — when a coupler is deactivated and its qubits measured, those qubit indices become "dormant" and can be reassigned when a new coupler occupies the same coordinates. Prevents qubit count from growing unboundedly with sequential coupler cycles.
- **`UnrotatedMultiPatchCoupler`** with auto-detected path axis and fixed-width corridor geometry compatible with the 6-tick SE schedule.
- **Patch rotation** (`transpose_coords`) to ensure Z boundaries face the corridor for Z-product measurements. Without this, X boundaries face the corridor and X-type coupler stabilizers anti-commute with Z logicals, consuming them incorrectly.

**Key files**: `src/ir/qec_system.py` (activate/deactivate, `active_qubit_indices`), `src/qec_code/surface_code/unrotated/multi_patch_coupler.py`

---

## 2026-03 — State Injection as Reusable LogicalOpSet

**Problem**: State injection (|Y⟩ preparation via corner protocol) was a monolithic experiment script. Needed it as a composable operation for distillation circuits.

**Solution**:
- Refactored into `RotatedSurfaceCodeLogicalOpSet.state_injection()` and `logical_unencode()`, callable via `LogicalExecutor`.
- **Incremental `process_data_measurement`** in tracker — added Step 4 (partial tableau reduction) and Step 5 (state persistence) to support calling `apply_data_readout` multiple times within a single circuit (unencode + physical measurement).
- **Noiseless tag** on `apply_data_readout` and `apply_unitary_block` — instructions tagged `"noiseless"` are skipped by all noise rules (`FlipBeforeMeasurement`, `FlipAfterReset`, `DepolarizeAfterGate`).
- Post-selection modes: `full_postselection` (all syndromes), `full_qec` (no post-selection), `hybrid` (logical strip only).

**Key files**: `src/qec_code/surface_code/unrotated/operation.py`, `src/qec_code/surface_code/rotated/operation.py`, `src/ir/tracker.py`

---

## 2026-03 — Steane 7-to-1 |Y⟩ Distillation (Lattice Surgery)

**Problem**: First full distillation circuit in LightStim — 5 patches, 4 sequential ZZZZ measurements via lattice surgery, with mid-circuit measurement + reinjection of ancilla magic state.

**Solution**:
- Demonstrated the full pipeline: init → SE → coupler activate → SE → mid-circuit MX → deactivate → reinjection → repeat × 4 → final readout.
- **Observable post-selection + target LER** added to `SimulationPipeline` — `post_select_observable_indices` filters shots by observable values, `target_observable_indices` counts errors only on specified observables.
- Verified `P_out ≈ 7P_in³` distillation formula with circuit-level noise.

**Key files**: `eval/LS_distillation/LS_distillation_7_to_1.py`, `src/simulation/decoder_backend/pipeline.py`

---

## 2026-03 — Transversal-Gate 7-to-1 |Y⟩ Distillation (Zhou et al.)

**Problem**: Implement the "algorithmic fault tolerance" distillation protocol — 8 working patches (hypercube [[7,1,3]] Steane encoding) + 7 magic patches (Y injection), transversal CNOTs, Θ(1) SE rounds per gate.

**Challenges solved**:
1. **Global vs local patch references**: `system.add_patch()` returns a global patch (global `data_indices`), but `system.patches[name][0]` stores the local patch (local UIDs). `transversal_cnot` needs global patches; `fold_transversal_*` needs local patches (for `data_coords` → `qubit_coords` lookup). Solution: maintain both `gp` (global) and `lp` (local) dicts.

2. **state_injection with multi-patch SE**: Corner injection with `post_select_coords=None` auto-tags all syndromes for post-selection. For patch-growth scenarios (full QEC), pass `post_select_coords=set()`. Also: must use **global** patch for `state_injection` (needs global `data_indices` for system coord lookup).

3. **Circuit topology**: The H gate in the teleportation step should be on **working patches 1-7** (not magic patches). Equivalent optimization: apply noiseless S on W0 during the teleportation CNOT tick, then measure working patches in X and magic patches in Z — avoids an extra H+SE layer.

4. **Observable analysis**: 4 observables span multiple patches. Manual analysis showed they correspond to [[7,1,3]] X stabilizers + distilled output. Built `src/simulation/observable_analysis.py` with GF(2) Gaussian elimination to automatically separate target (distilled output on W0) from post-select (outer code stabilizers) observables. Generalizes to any distillation circuit.

5. **Batched parallel gates**: Multiple transversal CNOTs/S gates on disjoint patches batched into single `apply_unitary_block` calls (1 TICK instead of N TICKs).

**Key files**: `eval/TG_distillation/TG_distillation_7_to_1.py`, `src/simulation/observable_analysis.py`, `src/qec_code/surface_code/unrotated/operation.py`

---

## 2026-03 — SyndromeTracker Scalability (Open)

**Problem**: TG distillation with 15 simultaneously-active patches hits tracker RREF scalability wall. Build times: d=3 (12s), d=5 (10min), d=7 (2hr). Root cause: `process_mid_measurement` does full RREF on the combined stabilizer+logical tableau, which is O(n³) where n = total active qubits ≈ 15d².

**Contrast**: LS distillation (5 patches + couplers) never hit this because mid-circuit measurements reduce the tableau between each coupler cycle. TG distillation keeps all 15 patches active throughout — no mid-circuit measurement to shrink the tableau.

**When this arises**: Whenever many patches are simultaneously active without intermediate measurements. Transversal gate circuits are the primary case. Lattice surgery circuits naturally avoid it via mid-circuit measurements.

**Identified fix — Block-diagonal RREF**: Before inter-patch gates, the tableau is block-diagonal (patches independent). Per-block RREF is O(d⁶) per block × 15 blocks, vs O((15d²)³) = O(3375 × d⁶) for monolithic RREF — a **225x speedup** for 15 patches. After inter-patch gates, blocks merge but partial structure remains exploitable.

**Future optimizations**:
- Sparse RREF (rows have O(d) non-zeros in O(d²) columns)
- Incremental RREF (structured gate transformations → incremental decomposition updates)
- Stim-native detector computation (bypass tracker RREF entirely for standard circuits)

**Key files**: `src/ir/tracker.py` (`process_mid_measurement`), `src/utils/linear_algebra.py` (`solve_linear_decomposition`, `row_echelon`)
