# PyMatching Decoder — Build the DEM with `decompose_errors=True`

**Date:** 2026-06-27
**Status:** Approved design, pending implementation
**Scope:** `PyMatchingDecoder` ([lightstim/simulation/decoder_backend/decoders/pymatching.py](../../../lightstim/simulation/decoder_backend/decoders/pymatching.py)) and the two DEM-construction sites in the simulation backend ([worker.py](../../../lightstim/simulation/decoder_backend/worker.py), [pipeline.py](../../../lightstim/simulation/decoder_backend/pipeline.py)). Other decoders unchanged.

## Problem

The MWPM / PyMatching decode path builds the detector error model (DEM) it hands to
`pymatching` with **`decompose_errors=False`**, which produces **incorrect logical
error rates far below threshold** and therefore wrong fitted effective distances.

Mechanism. The per-shot DEM is built as

```python
dem = circuit.detector_error_model(
    decompose_errors=getattr(decoder, "decompose_errors", False)
)
```

([worker.py:49-51](../../../lightstim/simulation/decoder_backend/worker.py)), but
`PyMatchingDecoder` defines **no `decompose_errors` attribute**
([pymatching.py](../../../lightstim/simulation/decoder_backend/decoders/pymatching.py)),
so `getattr(..., False)` evaluates to **False**. The resulting DEM is **not
graphlike** — e.g. the unrotated `d=3` Z-memory DEM contains 259 error mechanisms
that flip more than two detectors (up to degree 6). `pymatching.Matching.from_detector_error_model`
then graphlike-izes that DEM internally in a way that mishandles the hyperedges,
injecting a spurious low-power logical-error channel that dominates at low `p` and
**flattens the `LER`-vs-`p` slope**.

This contradicts the in-code comment at
[pipeline.py:131-132](../../../lightstim/simulation/decoder_backend/pipeline.py)
("pymatching now uses decompose_errors=True ... handles hyperedges correctly") — the
implementation has regressed from that stated intent.

### Evidence (verified)

Paired test: draw one set of shots, decode the **same** shots with a
`decompose_errors=False` matching and a `decompose_errors=True` matching, unrotated
surface code, deep-`p` window `{8e-5, 4e-5, 2e-5}`:

| case | `decompose=False` slope | `decompose=True` slope | ideal `(d+1)/2` |
|---|---|---|---|
| `d=3`, **X basis** | **1.20** ✗ | **2.00** ✓ | 2.0 |
| `d=3`, Z basis | 2.01 ✓ | 2.01 ✓ | 2.0 |
| `d=5`, Z basis | drifts to ≈2.3 ✗ | not yet cross-checked | 3.0 |

- The `d=3` **X-basis** case is decisive: the circuit distance is 3, so the LER
  exponent *must* be 2. `decompose=False` fits **1.20** — at `p=2e-5` it flags
  ~8× more logical errors than `decompose=True` **on the identical shots**.
  `decompose=True` restores **2.00**.
- `d=3` **Z basis** is (coincidentally) unaffected — both ≈2.0 — which is why the
  bug went unnoticed: the default Z-memory benchmarks look correct.
- Circuit-level distances are **full**: `len(circuit.shortest_graphlike_error())`
  returns 3 / 5 / 7 for `d = 3 / 5 / 7` (Z basis; X basis verified 3 / 5). So the
  true effective distances are the ideal `(d+1)/2 = 2 / 3 / 4`; the sub-ideal fitted
  slopes are decoder artifacts, not physics.

## Goal

Make the PyMatching path decode against a **graphlike (decomposed) DEM**, so MWPM
behaves correctly and the fitted effective-distance exponents return to the ideal
`(d+1)/2`.

## Non-Goals (out of scope)

- BP-OSD / external matrix decoders — they consume the *undecomposed* DEM via
  `dem_to_matrices` / `dem_to_check_matrices` by design and are unaffected.
- Noise model, SE schedule, circuit construction — untouched.
- Re-running / re-publishing the effective-distance study, and correcting any prior
  write-ups — downstream follow-up, not part of this code change.

## Root cause

`PyMatchingDecoder` carries no `decompose_errors` flag, so every DEM-build site that
reads `getattr(decoder, "decompose_errors", False)` falls back to `False`:

- multi-process worker — [worker.py:49-51](../../../lightstim/simulation/decoder_backend/worker.py)
- single-process — [pipeline.py:245-249](../../../lightstim/simulation/decoder_backend/pipeline.py)
  (`getattr(decoder_instance, "decompose_errors", False) or self.config.allow_gauge_detectors`)

## Design

Add a single class attribute to `PyMatchingDecoder`:

```python
class PyMatchingDecoder(sinter.Decoder):
    """MWPM decoder backed by pymatching."""
    decompose_errors = True   # MWPM requires a graphlike DEM; see 2026-06-27 spec

    def compile_decoder_for_dem(self, *, dem):
        return _CompiledPyMatching(pymatching.Matching.from_detector_error_model(dem))
```

Both DEM-build sites already honor the flag through `getattr`, so this one attribute
corrects **both** the multi-process and single-process paths with no other code
change. `compile_decoder_for_dem` needs no change — it now receives an
already-decomposed DEM. Refresh the stale comment at
[pipeline.py:131-132](../../../lightstim/simulation/decoder_backend/pipeline.py) so it
matches the actual behavior.

**Backward compatibility:** the registered name/aliases (`"pymatching"`, `"mwpm"`)
and the public decoder API are unchanged. Only the DEM that pymatching is built from
changes (decomposed instead of raw). Z-basis near-threshold benchmarks are essentially
unaffected; X-basis and deep-`p` results change (they were wrong before).

## Data flow

```
worker / pipeline
  └─ dem = circuit.detector_error_model(
             decompose_errors=getattr(decoder,"decompose_errors",False))  # now True
        # stim splits each hyperedge into graphlike (<=2-detector) components
  └─ PyMatchingDecoder.compile_decoder_for_dem(dem)
        └─ pymatching.Matching.from_detector_error_model(dem)   # clean matching graph
```

## Error handling

- `circuit.detector_error_model(decompose_errors=True)` can raise if an error
  mechanism cannot be decomposed into graphlike pieces. For CSS surface-code
  circuit-level noise the DEM **is** decomposable (verified: `d=3/5/7`, both bases,
  build cleanly), so the default path is safe.
- If a future circuit yields a genuinely non-decomposable DEM, **fail loudly** rather
  than silently dropping edges. Do **not** globally enable
  `ignore_decomposition_failures`: dropping undecomposable hyperedges removes
  low-weight logical paths and *underestimates* LER (the opposite failure mode). The
  existing `allow_gauge_detectors` path already gates that behavior on the
  single-process branch and should remain opt-in.

## Testing (TDD)

1. **Regression for the bug (load-bearing):** unrotated **X-basis `d=3`** memory,
   deep-`p` window — the fitted slope must be ≈ 2.0 (was ≈ 1.2). Equivalent unit
   form: on a fixed shot set the X-basis `d=3` decoded LER at `p=2e-5` drops by
   ~8× relative to the pre-fix decoder.
2. **Z-basis unchanged:** unrotated Z-basis `d=3` slope stays ≈ 2.0 (guards the case
   that already worked).
3. **DEM is graphlike:** assert the DEM that `PyMatchingDecoder` is built from has no
   error mechanism flipping >2 detectors (decomposition succeeded).
4. **Cross-decoder agreement:** on identical sampled shots the post-fix pymatching
   LER agrees (within statistics) with an independent graphlike decode at the deep
   points.
5. **`d=5`/`d=7` sanity:** with the fix, low-`p` slopes approach `(d+1)/2 = 3 / 4`
   from above and do **not** drift below.
6. **No exceptions:** `d=3/5/7`, both bases, build + decode without
   `DecompositionFailure`.

## Risks

- A circuit whose DEM is genuinely non-graphlike would now raise instead of silently
  mis-decoding. This is the intended "fail loud" behavior, but any such call sites
  must be identified; all current surface-code memory / SE circuits are decomposable
  (verified `d=3/5/7`).
- Decomposition adds a small per-build cost, negligible against sampling + decoding.
- **Downstream:** previously-collected MWPM results in the **X basis** or at **deep
  `p`** (and any biased-noise studies) are suspect and should be re-run after this
  lands. Z-basis near-threshold benchmarks are largely unaffected. The effective
  distance notebooks (`notebooks/Memory/unrotated_effective_distance_convergence*.ipynb`)
  should be re-run and any sub-`(d+1)/2` "effective distance" conclusions retracted —
  the circuit distances are full, so the true `d_e` is the ideal `(d+1)/2`.
