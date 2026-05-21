# Steane 7-to-1 |Y⟩ Distillation — Simulation Results

**Protocol:** 4 sequential ZZZZ lattice surgery measurements on 5 unrotated surface code patches
**Post-select:** L0, L1, L3 (Steane syndrome checks); **Target:** L2 (distilled output)
**Decoder:** PyMatching (CPU), circuit-level depolarizing noise (uniform p)
**Layout:** `gap = 2*d + 2`, corridor width = `2*d + 1` columns
**Script:** `eval/LS_distillation/run_sweep.py`

## Results

| d | rounds | p | qubits | detectors | shots | kept | PS rate | errors | LER | time |
|---|--------|---|--------|-----------|-------|------|---------|--------|-----|------|
| 3 | 3 | 1e-3 | 272 | 1,732 | 1.6M | 328K | 20.50% | 16,357 | 4.99e-2 | 1.6s |
| 3 | 3 | 1e-4 | 272 | 1,732 | 1.8M | 1.5M | 81.23% | 1,775 | 1.21e-3 | 2.1s |
| 5 | 5 | 1e-3 | 768 | 8,552 | 1.9M | 234K | 12.67% | 1,336 | 5.70e-3 | 7.1s |
| 5 | 5 | 1e-4 | 768 | 8,552 | 10M | 5.8M | 57.80% | 28 | 4.84e-6 | 18.6s |
| 7 | 7 | 1e-3 | 1,520 | 24,044 | 4.8M | 600K | 12.51% | 297 | 4.95e-4 | 34.1s |
| 7 | 7 | 1e-4 | 1,520 | 24,044 | 10M | 3.7M | 36.79% | 1 | 2.72e-7 | 53.6s |

## Cross-Distance Comparison

### p = 1e-3

| d | LER | vs d=3 |
|---|-----|--------|
| 3 | 4.99e-2 | — |
| 5 | 5.70e-3 | 8.8x better |
| 7 | 4.95e-4 | 101x better |

### p = 1e-4

| d | LER | vs d=3 |
|---|-----|--------|
| 3 | 1.21e-3 | — |
| 5 | 4.84e-6 | 250x better |
| 7 | 2.72e-7 | 4,449x better |

## Analysis

### 1. Distance scaling is exponential

At p=1e-3, LER drops roughly 10x per +2 in d (5e-2 → 6e-3 → 5e-4). This is consistent with an effective distance scaling where errors require O(d/2) faults to cause a logical failure. The LER ratio d=3→d=7 is ~100x, close to (p_eff)^2 where p_eff ≈ 0.1.

At p=1e-4, the scaling is even steeper: 250x (d=3→5) and 4,400x (d=3→7). Lower physical error rates amplify the exponential suppression from larger d.

### 2. Post-selection rate

| d | PS rate (p=1e-3) | PS rate (p=1e-4) |
|---|-------------------|-------------------|
| 3 | 20.50% | 81.23% |
| 5 | 12.67% | 57.80% |
| 7 | 12.51% | 36.79% |

At p=1e-3, PS rate stabilizes around 12-20% — most shots trigger at least one Steane check observable. At p=1e-4, PS rate drops with d (81% → 37%) because larger circuits have more detectors that can fire spuriously, triggering the post-selection condition more often.

### 3. Distillation vs raw memory

From Experiment 0 (see `example_circuit/LS_distillation_result.md`): single-patch Z-memory LER ≈ 9.2e-4 at d=3, p=1e-3. The Z-memory LER scales as ~A * p^(d/2) for the surface code.

| d | p | Raw memory LER (est.) | Distilled LER | Regime |
|---|---|----------------------|---------------|--------|
| 3 | 1e-3 | ~9e-4 | 4.99e-2 | distillation overhead > benefit |
| 5 | 1e-3 | ~2e-5 | 5.70e-3 | transitioning |
| 7 | 1e-3 | ~5e-7 | 4.95e-4 | distillation-limited |
| 3 | 1e-4 | ~9e-5 | 1.21e-3 | distillation overhead > benefit |
| 5 | 1e-4 | ~2e-8 | 4.84e-6 | transitioning |
| 7 | 1e-4 | ~5e-12 | 2.72e-7 | distillation-limited |

At small d, the distilled LER is **worse** than raw memory — the 4 sequential lattice surgery operations introduce more error than the Steane code can correct. At larger d, raw memory error drops exponentially while distillation overhead grows only polynomially, so the output converges to the intrinsic distillation error floor.

This matches the expected behavior: distillation protocols are designed for the regime where physical errors are below threshold and code distance provides adequate protection of the logical operations themselves.

### 4. Effective distillation exponent

For the Steane [7,1,3] code, the theoretical distillation relation is LER_out ~ c * p_in^3 (distance-3 code corrects up to 1 error). In our circuit, p_in is the per-measurement-round logical error, not the raw physical error. The observed LER scaling between p=1e-3 and p=1e-4 at fixed d:

| d | LER(p=1e-3) | LER(p=1e-4) | ratio | effective exponent |
|---|-------------|-------------|-------|-------------------|
| 3 | 4.99e-2 | 1.21e-3 | 41x | ~1.6 |
| 5 | 5.70e-3 | 4.84e-6 | 1,178x | ~3.1 |
| 7 | 4.95e-4 | 2.72e-7 | 1,820x | ~3.3 |

At d=3 the exponent is only ~1.6 (sub-cubic), indicating the circuit is above the distillation threshold — errors are too frequent for the Steane code to correct effectively. At d=5 and d=7, the exponent reaches ~3, consistent with the theoretical cubic suppression of the [7,1,3] code.

### 5. Statistical limitations

d=7, p=1e-4 has only 1 error in 3.7M post-selected shots. The 95% Poisson confidence interval for the true LER spans roughly [7e-9, 1.5e-6]. A precise measurement would require >100M shots or >100 errors.

## Reproduction

```bash
python eval/LS_distillation/run_sweep.py
```

Detailed per-run data in `results.json` and `results.csv`.
