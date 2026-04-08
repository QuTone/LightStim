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

---

## 2026-04 — Decoder Backend Empirical Findings (Benchmarking Campaign)

### Worker Parallelism Model

`SimulationPipeline` with `num_workers=N` (N > 1) spawns N independent `mp.Process` instances sharing three
`multiprocessing.Value` counters (`shots_counter`, `errors_counter`, `post_counter`) and a single `Lock`.
Each worker independently samples a batch, decodes it, then acquires the lock briefly to update the shared
counters and check the stopping condition. The lock is held for microseconds per batch; at `batch_size=10000`
it contributes negligibly to total runtime.

For the GPU backend (`nv-qldpc-decoder`), worker `wid` is assigned `gpu_id = wid` **unless**
`CUDA_VISIBLE_DEVICES` is already set in the environment — in which case the env var takes precedence and
all workers land on the same GPU. Implication: when running with a pre-set `CUDA_VISIBLE_DEVICES=6`, always
use `--num-workers 1`; multi-worker GPU parallelism requires leaving `CUDA_VISIBLE_DEVICES` unset so the
pipeline assigns worker 0 → GPU 0, worker 1 → GPU 1, etc.

### Memory Bandwidth as the CPU Parallelism Ceiling

Empirical result: 32 CPU workers (`bposd`) on a d=7 circuit (2268 detectors, ~48k error mechanisms) achieved
only ~8× speedup. The explanation is memory-bandwidth saturation, not lock contention or compute limits.

bposd's BP iterations are memory-bandwidth bound: each iteration sweeps the full `H` matrix
(`num_errors × num_detectors`) for message passing. The per-worker working set is dominated by:

| Circuit | Detectors | Error mechs | ~Working set / worker |
|---|---|---|---|
| TG d=3 | 180 | ~3k | ~1 MB |
| TG d=5 | 840 | ~12k | ~3 MB |
| TG d=7 | 2268 | ~48k | ~12 MB |
| BB [[144,12,12]] | ~10k+ | ~100k+ | ~50+ MB |

The machine (2× AMD EPYC 9534, 1.5 TB RAM) has ~256 MB L3 cache per socket. Once the aggregate working
set across all workers exceeds the L3, every BP iteration incurs DRAM latency, and additional workers only
increase contention on the memory bus. Estimated saturation thresholds:

- **d=5**: saturates at ~85 workers (rarely hit in practice)
- **d=7**: saturates at ~21 workers — explains the 32-worker/8× empirical observation
- **BB [[144,12,12]]**: saturates at ~5 workers

The H100 GPU bypasses this entirely: HBM3 bandwidth (~3.35 TB/s) is ~10× higher than host DRAM bandwidth,
and the entire working set fits in 80 GB VRAM. This is why `nv-qldpc-decoder` finishes in seconds for
circuits where CPU decoders hang for hours.

### Practical Decoder Selection Rules (from Benchmarking)

Established during the Bell teleportation + logical gate benchmark campaign (2026-04):

1. **PyMatching**: only for circuits with no hyperedges (memory, LS); never for fold-transversal H/S or TG.
2. **bposd CPU**: fold-transversal gates at d=3,5; TG at d=3,5. Beyond d=5 at low p, too slow and risks
   hangs at high error density.
3. **MWPF CPU**: TG at d=7 when p ≤ 2e-3. Hangs at p ≥ 5e-3 on d=7 (2268 detectors, dense syndrome —
   ~200 detector firings per shot, 72% hyperedges in DEM). Same hang pattern as bposd.
4. **nv-qldpc-decoder GPU**: required for d=7 fold-transversal H/S/CNOT_trans at any p; required for TG
   d=7 at p ≥ 5e-3. Use 1 worker, pin to a specific GPU via `CUDA_VISIBLE_DEVICES`.

Key finding: TG d=7 p=5e-3 has raw (undecoded) observable error rate ~49%, but after GPU decoding LER drops
to ~5%. CPU decoders fail not because the circuit is wrong but because one syndrome per ~80k shots is a
pathological matching instance that causes exponential blowup in MWPF/OSD.

See `docs/decoder_selection_guide.md` for the full reference table.

### Post-Paper TODO — Decoder Auto-Selection and Profiling

**Goal**: develop a workflow that automatically selects the optimal decoder backend and `num_workers` for a
given circuit, replacing the current manual trial-and-error process.

**Planned work**:
1. **Profiling harness**: for a given circuit + p, measure (a) per-shot decode time distribution,
   (b) DEM hyperedge fraction, (c) average syndrome weight — these three statistics determine which decoder
   regime applies.
2. **Memory working-set estimator**: given `num_detectors` and `num_errors`, estimate bposd working set and
   compute the theoretical CPU worker ceiling before memory-bandwidth saturation.
3. **Auto-selection logic**:
   - If `max_edge_weight ≤ 2`: use PyMatching
   - Else if `estimated_working_set × num_workers ≤ L3_cache`: use bposd CPU with computed optimal workers
   - Else: use `nv-qldpc-decoder` GPU (1 worker per GPU)
4. **Hang detection + fallback**: if a decode batch exceeds N× the median batch time, abort and switch to
   GPU. This is the root failure mode observed for MWPF/bposd at high p + large d.

**Priority**: post paper submission. Current workaround: `docs/decoder_selection_guide.md`.

---

## Post-Paper Roadmap

### Performance: Beyond RREF

C++ RREF (bitpacked uint64) accelerates `row_echelon` by 27-94x, but total circuit build only improves 2-3x due to Amdahl's law — RREF is only 30-40% of build time. Remaining hot paths (profiling needed):

1. **`check_commutativity`** — symplectic inner product, called per-measurement against full tableau. Currently numpy matmul. Could be bitpacked.
2. **`process_unitary_block`** — Stim tableau → symplectic matrix → numpy matmul for tableau conjugation. Large matrix multiply on every gate.
3. **`_get_back_propagated_pauli`** — Stim `Tableau.from_circuit(ignore_reset=True)` inversion. Stim-internal, hard to optimize externally.
4. **Python loop overhead** — `process_mid_measurement` iterates measurements one-by-one with per-iteration numpy calls.

**Strategy**: Profile with `cProfile` on a d=7 TG build to identify which of these dominates. Then targeted C++ acceleration of the top 1-2 bottlenecks. Full tracker C++ rewrite (2-4 weeks) only if incremental optimization plateaus.

### New Protocols to Implement

- ~~Color code 6-6-6 hexagonal memory~~ (DONE — 2026-03-12, space-multiplexed SE, MWPF decoder verified d=3→7)
- Rotated SC lattice surgery coupler + distillation
- 15-to-1 |T⟩ distillation factory (Zhou et al. ED Fig. 3)
- Fold-transversal S on rotated SC (requires syndrome qubit operation support)
- Color code transversal gates
- Code switching (e.g., surface code ↔ color code)

### Optional — Post-Paper Tasks (2026-03-26)

These are **stretch goals** before paper submission. If time permits, do them; otherwise defer to after paper.

1. **Rotated SC lattice surgery coupler** — rotated surface code版本的lattice surgery，目前只有unrotated版本
2. **Rotated SC fold-transversal S gate** — 需要 syndrome qubit 上的 operation 支持
3. **Color code Bell flagging SE circuit** — see implementation plan below
4. **5-qubit code distillation / Golay-based code (GBC) on color code** — 高级distillation协议
5. **Chromobius decoder integration for color code** — 需要 detector coordinate annotation

### Color Code Flag Qubit SE — Implementation Plan (2026-03-26)

**Difficulty**: Medium-low. Tracker core (`tracker.py`) does NOT need modification.

**Key insight**: Flag qubits are just syndrome qubits that, after back-propagation, only have Pauli support on other syndrome qubits (not data qubits). The tracker's existing decomposition naturally handles this:
- Flag measurement → back-propagated Pauli has zero data qubit support → empty decomposition → trivially deterministic → auto-generates DETECTOR with single measurement record
- Syndrome measurement → back-propagated Pauli on data qubits → normal stabilizer decomposition → normal DETECTOR

**Implementation steps**:
1. **SE block**: Write `ColorCodeFlagExtractionBlock`. Flag qubits registered as `role='syndrome'`. Add CX gates between syndrome-syndrome (flag-syndrome entangling). Geometry lookup relaxed from `if target in data_indices` to `if target in qubit_indices` for flag connections.
2. **Alternating X/Z rounds**: Color code flag circuits measure Z and X stabilizers in separate rounds. Combine both into a single REPEAT block so the tracker sees complete stabilizer coverage per combined round. Unmeasured stabilizer rows in a half-round simply don't get new measurement records — they're stale but valid (Pauli frame unchanged).
3. **Verify**: Run noiseless memory experiment, check all detectors are valid (zero events on 1000 samples). Then inject noise, extract DEM, decode with MWPF.

**What could go wrong**: The tracker's `process_first_round_detectors` / `process_repeated_round_detectors` may not handle partial stabilizer measurement gracefully (some rows have new records, others don't). If so, the fix is in how measurement records are diffed for REPEAT blocks, not in the core back-propagation/decomposition logic.

**Paper impact**: If completed before deadline, adds two lines to Figure 4 (memory experiment): Color code space-multiplexed (current) vs flag-qubit SE scheduling. Demonstrates modular SE block replacement on a non-trivial code family.

### Framework Maturity

- `pyproject.toml` + pip installable package
- Comprehensive test suite (target >80% coverage)
- CI/CD with GitHub Actions (test both Python fallback and C++ backend)
- API documentation (auto-generated from docstrings)
- Tutorial notebooks: one per protocol, suitable as both documentation and AI training data
