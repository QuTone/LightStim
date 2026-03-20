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

**Implemented**: Block-diagonal RREF via Union-Find in `solve_linear_decomposition`. Auto-detects independent blocks and dispatches per-block RREF. Build time results: d=3 (12s→5s), d=5 (603s→167s). Transparent to all callers; all 31 tests pass.

**Key files**: `src/ir/tracker.py` (`process_mid_measurement`), `src/utils/linear_algebra.py` (`solve_linear_decomposition`, `row_echelon`)

---

## Open Problems — Tracker Extensions

### Rotated Surface Code Coupler + Distillation

Rotated surface code lattice surgery coupler has different construction from unrotated (different boundary geometry). Needed for rotated SC distillation circuits. Requires implementing `RotatedMultiPatchCoupler` following the unrotated pattern, then building LS/TG distillation experiments for rotated codes.

### Flag Qubit Support

Current tracker assumes a simple data-qubit + syndrome-qubit model. Flag qubits are ancilla qubits used to detect high-weight errors during syndrome extraction (e.g., in heavy-hex architectures or flag-based FTQEC). Supporting flags requires the tracker to handle additional measurement rounds with conditional error propagation paths.

### Syndrome Qubit Operations (Mid-SE Gate Insertion)

**Specific blocker**: `fold_transversal_s` on rotated surface code. The rotated SC implementation of S_L requires CZ gates applied to **syndrome qubits** mid-SE-round, at the point where the rotated code momentarily looks like an unrotated code. The current tracker assumes syndrome qubits are passive (reset → CNOT schedule → measure), and any gate on a syndrome qubit outside this pattern breaks the back-propagation logic.

**Possible approaches**:
1. Temporarily reclassify syndrome qubits as data qubits during the gate, then restore
2. Extend the tracker's SE model to allow "gate insertion points" between CNOT layers
3. Implement as a custom SE block that inlines the CZ gates within the extraction schedule

---

## Performance Roadmap

### C++/Native RREF Backend

The GF(2) Gaussian elimination (`row_echelon` in `linear_algebra.py`) operates on numpy bool arrays with Python-level loops. A C++ implementation with:
- Bitwise row operations (64 columns per uint64 word → 64x fewer operations)
- Cache-friendly memory layout (row-major, aligned)
- OpenMP parallelization for independent block processing

Could provide 100-1000x speedup over current Python+numpy. This would make d=7 TG distillation build times sub-minute and d=9+ feasible.

**Implementation options**:
- pybind11 extension module (cleanest integration)
- Cython with typed memoryviews (easier build, good performance)
- ctypes wrapper around standalone C library (simplest, no build dependency)

---

## Strategic Directions

### Direction 1: Protocol Library + Benchmark Suite

Expand LightStim into a comprehensive QEC protocol library with standardized benchmarks.

**Goals**:
- Every major QEC protocol (memory, CNOT, lattice surgery, transversal gates, distillation, color codes) as a reproducible benchmark with fixed parameters and standard output format
- Tutorial-style documentation: each protocol = 1 notebook (interactive) + 1 eval script (batch)
- pip-installable package (`pyproject.toml`) for low barrier to entry
- "Protocol gallery" for the QEC community — and as structured training data for AI-assisted QEC research

**Implemented protocols**:
- Memory experiment (rotated/unrotated/toric SC, BB code)
- Transversal CNOT (rotated/unrotated/toric SC)
- Lattice surgery CNOT (unrotated SC, 2-patch and N-patch)
- GHZ state preparation
- State injection + unencode (rotated/unrotated SC)
- Fold-transversal H, S, S† (unrotated SC)
- LS distillation 7-to-1 |Y⟩ (lattice surgery, Steane code)
- TG distillation 7-to-1 |Y⟩ (transversal gates, Zhou et al.)

**Pending protocols**:
- Rotated SC lattice surgery coupler + distillation
- Fold-transversal S on rotated SC (blocked by syndrome qubit ops)
- Color code experiments (6-6-6 hexagonal, triangular)
- 15-to-1 |T⟩ distillation factory
- Code switching / code enlarging

### Direction 2: Industrial-Grade Performance

Scale LightStim to handle large circuits (d=9+, 50+ patches) with acceptable build times.

**Priority roadmap**:
1. **C++ RREF backend** — single biggest bottleneck; 100-1000x via bitwise GF(2) ops
2. **Sparse tableau storage** — rows have O(d) non-zeros in O(d²) columns
3. **Circuit caching** — build once, reload for subsequent experiments (`--load-circuits`)
4. **Incremental RREF** — exploit structured gate transformations
5. **Packaging + CI** — `pyproject.toml`, GitHub Actions, automated testing + benchmarks
