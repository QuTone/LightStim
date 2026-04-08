# Decoder Selection Guide

Based on empirical benchmarking experience across memory, logical gate, and Bell teleportation experiments.

## Quick Reference

| Circuit Type | d=3,5 | d=7, p≤2e-3 | d=7, p≥5e-3 |
|---|---|---|---|
| Memory / LS (no hyperedges) | `pymatching` | `pymatching` | `pymatching` |
| Fold-transversal H/S (hyperedges) | `bposd` CPU | `nv-qldpc-decoder` GPU | `nv-qldpc-decoder` GPU |
| Transversal CNOT (hyperedges) | `bposd` or GPU | `nv-qldpc-decoder` GPU | `nv-qldpc-decoder` GPU |
| TG Bell teleportation (hyperedges) | `bposd` CPU | `mwpf` CPU | `nv-qldpc-decoder` GPU |

---

## Why Decoder Choice Matters

The surface code DEM can contain **hyperedges** (error mechanisms touching >2 detectors). This happens in:
- Fold-transversal H and S gates (SWAP + CZ operations create weight-up-to-10 hyperedges)
- Transversal CNOT across patches
- TG Bell teleportation (correlated multi-patch syndrome)

PyMatching requires a standard graph (weight ≤ 2 edges). It either fails or silently decomposes hyperedges with degraded accuracy. **Never use PyMatching on circuits with hyperedges.**

---

## Decoder Profiles

### `pymatching` — CPU, Blossom algorithm
- **O(n^1.5)** per shot, never hangs
- **Works**: memory circuits, LS circuits (no/few hyperedges)
- **Fails**: circuits with hyperedges (fold-transversal gates, TG)
- **Ideal p range**: all p (fast even at high error density)
- Num workers: 8–16 for throughput

### `bposd` (stimbposd) — CPU, BP+OSD
- Handles hyperedges correctly
- **Works well**: d=3,5 for any circuit; d=7 when p ≤ 2e-3 and syndrome density is low
- **Hangs**: d=7 at p ≥ 5e-3 (dense syndrome → hard OSD instances, can stall for hours)
- **Performance**: ~10–100× slower than PyMatching for same shot count
- Num workers: 8–16

### `mwpf` — CPU, Min-Weight Perfect Matching with hyperedges
- Handles hyperedges
- **Works well**: d=7 at p ≤ 2e-3 for TG Bell teleportation
- **Hangs**: same as bposd — d=7 at p ≥ 5e-3 with dense syndrome (2268 detectors, ~200 firings/shot)
- Num workers: 8

### `nv-qldpc-decoder` — GPU, CUDA parallel BP+OSD
- Handles hyperedges; parallelism prevents individual-shot hangs
- **Works**: all circuit types, all d, all p — including cases that hang CPU decoders
- **Required for**: fold-transversal H/S at d=7; transversal CNOT at d=7; TG at d=7 p≥5e-3
- **Performance**: typically 10–100× faster than CPU bposd for d=7
- Num workers: 1 (GPU manages internal parallelism)
- CUDA_VISIBLE_DEVICES: pick an idle GPU

---

## Detailed Guidance by Circuit Type

### Memory Benchmark
- DEM has no hyperedges (standard syndrome graph)
- Use `pymatching` at all d and p
- No need for bposd or GPU

### LS Bell Teleportation (ZZ-LS, XX-LS)
- Standard graph DEM (lattice surgery measurements are local)
- Use `pymatching` at all d and p
- Fast, reliable, no hangs

### TG Bell Teleportation
The TG circuit has correlated multi-patch syndromes with hyperedges. At p=5e-3 d=7:
- DEM: 48,249 error mechanisms, **72% are hyperedges** (max weight=10)
- Raw (undecoded) observable error rate ≈ 49% — nearly random before decoding
- Syndrome density: ~200 detector firings per shot out of 2268

| Config | Recommendation | Reason |
|---|---|---|
| d=3,5, any p | `bposd` CPU, 8–16 workers | Small circuit (75–243 qubits), no hangs |
| d=7, p ≤ 2e-3 | `mwpf` CPU, 8 workers | Sparse syndromes, MWPF converges |
| d=7, p = 5e-3 | `nv-qldpc-decoder` GPU | CPU decoders hang; GPU finishes in ~137s |
| d=7, p = 1e-2 | `nv-qldpc-decoder` GPU | Same reason |

> **Note**: at d=7 p=5e-3, MWPF decoded the first 80k shots (in ~21s) but hung indefinitely on shot ~80,001. bposd showed the same hang pattern. This is a rare "hard instance" where the dense, hyperedge-heavy syndrome causes exponential blowup in matching. GPU decoder avoids this entirely.

### Fold-Transversal H and S Gates
The fold operation (transversal H + SWAP on mirror pairs, or S/S† + CZ) produces hyperedges with max weight up to ~8. PyMatching cannot be used.

| Config | Recommendation | Reason |
|---|---|---|
| d=3,5, any p | `bposd` CPU, 8–16 workers | Manageable circuit size |
| d=7, p ≤ 5e-4 | `nv-qldpc-decoder` GPU | CPU bposd takes 6–8 hours and only gets 30 errors |
| d=7, p ≥ 1e-3 | `nv-qldpc-decoder` GPU | GPU is 5–50× faster, no risk of hang |

### Transversal CNOT
Multi-observable (2 logical qubits) circuit with hyperedges.

| Config | Recommendation |
|---|---|
| d=3,5, p ≥ 1e-3 | `nv-qldpc-decoder` GPU (fast) or `bposd` CPU |
| d=5, p ≤ 5e-4 | `nv-qldpc-decoder` GPU (CPU bposd can take 3+ hours for 30 errors) |
| d=7, all p | `nv-qldpc-decoder` GPU |

---

## GPU Usage Protocol

Before launching any GPU experiment:

```bash
# 1. Check which GPUs are free
nvidia-smi

# 2. Use a free GPU (e.g. GPU 6)
CUDA_VISIBLE_DEVICES=6 python run_xxx.py --decoder nv-qldpc-decoder --num-workers 1

# 3. Use at most 1 GPU per experiment (GPU manages its own parallelism)
```

---

## num-workers Guidelines

| Decoder | Recommended num-workers |
|---|---|
| `pymatching` | 8–16 |
| `bposd` | 8–16 |
| `mwpf` | 8 |
| `nv-qldpc-decoder` | 1 |

For `nv-qldpc-decoder`, num-workers > 1 increases kernel launch overhead without meaningful throughput gain — use 1.

---

## Common Pitfalls

1. **CPU decoder hangs at high p + large d**: MWPF and bposd can stall indefinitely on a single shot with a complex syndrome. Switch to GPU.

2. **PyMatching on hyperedge circuits**: Reports 0 errors (silently wrong) or crashes. Always verify the circuit DEM before choosing PyMatching:
   ```python
   dem = noisy_circuit.detector_error_model(decompose_errors=False)
   max_weight = max(
       len([t for t in inst.targets_copy() if t.is_relative_detector_id()])
       for inst in dem.flattened() if inst.type == 'error'
   )
   # max_weight > 2 → hyperedges present → do NOT use PyMatching
   ```

3. **Using system Python instead of venv**: `cudaq_qec` (GPU decoder) is only installed in `venv/`. Always use `venv/bin/python` or `source venv/bin/activate`.
