---
name: extend-new-decoder
description: >
  Integrate a new decoder into LightStim's decoder backend so it works
  with `SimulationPipeline` and `DecoderConfig`. Use this skill whenever
  the user asks to add a custom decoder, wrap a third-party decoder
  library (e.g. a neural decoder, a research-paper decoder), register a
  CPU or GPU decoder backend, build a decoder from scratch using a DEM,
  or extend the existing BPOSD/PyMatching/MWPF stack with a new variant.
user-invocable: true
---

# Add a New Decoder

LightStim's decoder backend is a registry of decoder classes. Register a class
under a name + backend (`cpu` / `gpu` / `fpga`), and
`DecoderConfig(name="…", backend="…", params={…})` finds it at runtime —
everywhere: pipeline, server, notebooks, benchmarks.

Adding a decoder is always three things:

1. **One new file** in `lightstim/simulation/decoder_backend/decoders/`.
2. **One soft-import line** in `decoders/__init__.py`.
3. **One smoke test** in `tests/test_simulation_backend_quality.py`.

A runnable version of everything below (with live benchmarks) is
`tutorials/how-to-add-new-decoders.ipynb`.

## Which pattern?

| Situation | Pattern |
|---|---|
| Upstream library ships a `sinter.Decoder` | **A** — register the class directly |
| Anything else | **C** — subclass `ExternalDecoder` (recommended) |
| You need to own the unpack → decode → pack loop yourself | **B** — build from the DEM matrices |

---

## The contract (duck-typed)

The pipeline never checks `isinstance(..., sinter.Decoder)`. A decoder is any
object with:

```python
class MyDecoder:                       # sinter.Decoder base optional
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel):
        return _MyCompiled(...)        # one-time prep per DEM

class _MyCompiled:                     # sinter.CompiledDecoder base optional
    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        # in:  (n_shots, ceil(n_detectors/8))   uint8, LSB-first
        # out: (n_shots, ceil(n_observables/8)) uint8, LSB-first
        ...
```

Subclassing the sinter bases is the repo convention and the documented API, but
it is not a runtime requirement — raw `sinter.collect` accepts duck-typed
classes too (verified on sinter 1.15/1.16). Matching this signature is why
sinter-native decoders plug in unchanged.

**Pattern C skips this contract entirely** — the `ExternalDecoder` facade
implements it for you; you only write methods on plain unpacked arrays.

---

## Pattern A — sinter-native: register directly

```python
# decoders/my_decoder.py
from ..registry import register_decoder

try:
    from mwpf import SinterMWPFDecoder
    register_decoder("mwpf", SinterMWPFDecoder)
except ImportError:
    pass
```

That's the entire built-in `mwpf.py`; `relay_bp.py` is the same shape. User
params flow straight through `DecoderConfig(params={…})` to the upstream class
— don't add a translation layer for a single library.

**Lazy variant** (`tesseract.py`): when the upstream package is a native
extension whose import can fail on a given host, don't import it at module
load. Register a thin `sinter.Decoder` subclass that imports inside
`__init__` and delegates `compile_decoder_for_dem` to the upstream object,
guarded by `importlib.util.find_spec` so a broken wheel breaks only
`DecoderConfig("tesseract")` and not the whole registry. This also covers the
case — as with Tesseract — where the upstream class merely duck-types the
sinter interface instead of subclassing it.

**Need renamed params or defaults?** Give a thin wrapper class an
`__init__(**params)` that merges defaults and renames keys before constructing
the inner decoder, and delegate `compile_decoder_for_dem` to it. `bposd.py`
does this to share one parameter vocabulary between its CPU (`stimbposd`) and
GPU (`cudaq_qec`) backends.

---

## Pattern C — `ExternalDecoder` facade (recommended for non-sinter)

Subclass, declare `output_type`, build in `setup`, implement **one** of
`decode_single` / `decode_batch`. The facade owns bit-packing, the
observable-matrix multiply, worker plumbing, and failure-flag routing.

```python
import numpy as np
from ..external import ExternalDecoder
from ..registry import register_decoder

class MyDecoder(ExternalDecoder):
    output_type = "correction"   # REQUIRED, no default — see below

    def setup(self, *, H, obs_matrix, priors, num_detectors,
              num_observables, dem):
        # Called once per DEM. self.params holds DecoderConfig(params=...).
        # H / obs_matrix are scipy CSR; .toarray() only for small DEMs.
        self._inner = my_lib.Decoder(H, priors, **self.params)

    def decode_single(self, syndrome):        # (n_dets,) uint8, unpacked
        correction, converged = self._inner.run(syndrome)
        return correction, converged          # flag: True/None ok, False failed

register_decoder("my-decoder", MyDecoder, backend="cpu")
```

Three knobs:

- **`output_type`** — `"correction"`: you return a correction over error
  mechanisms, length `H.shape[1]`, and LightStim computes the observable
  flips. `"observables"`: you return the logical flips directly, length
  `n_obs`. Forgetting it raises at compile time.
- **single vs batch** — implement whichever is natural
  (`decode_batch(syndromes) -> (preds, flags)`); LightStim bridges the other.
- **failure flags** — return `False` for shots the decoder failed on (e.g. BP
  non-convergence), `True`/`None` otherwise.
  `DecoderConfig(on_decode_failure=...)` decides what a `False` means:
  `"error"` (default, count as logical error), `"discard"` (herald it out of
  the denominator), `"ignore"` (trust the prediction).

What `setup` receives: the facade calls
`dem_to_matrices(dem, sparse=True, merge_duplicates=True)`, so `H` and
`obs_matrix` are CSR and duplicate-footprint mechanisms are already merged —
**`H.shape[1]` can be smaller than `dem.num_errors`** (gotcha 2).

Working examples: `decoders/ldpc_bp.py` (correction + real convergence
flags), and the `tests/test_simulation_backend_quality.py` externals.

---

## Pattern B — custom decoder from the DEM (full control)

Only when you must own the `CompiledDecoder` lifecycle (e.g. GPU memory
management). Reference implementation: `decoders/cudaqx.py`.

Get matrices from the shared helper — never re-derive the conversion:

```python
from ..dem_matrices import dem_to_matrices

H, obs_matrix, priors = dem_to_matrices(dem)              # dense uint8, C-order
H, obs_matrix, priors = dem_to_matrices(dem, sparse=True) # scipy CSR
```

Two behaviours it encodes that hand-rolled versions get wrong:

- **Parity, not assignment.** stim cancels a target listed an even number of
  times (`error(0.1) D0 D0 D1` flips only `D1`); the helper XORs. Writing
  `H[d, e] = 1` yourself silently builds a wrong matrix.
- **`merge_duplicates=True` (default) changes the column count.** Mechanisms
  with identical (detector, observable) footprints fuse into one column,
  priors combined with the XOR rule `p1(1-p2) + p2(1-p1)` — stim leaves such
  duplicates in z_only-style circuits and they measurably degrade BP. Pass
  `merge_duplicates=False` if you need DEM-column alignment.

Then write the compiled decoder: unpack syndromes
(`np.unpackbits(..., bitorder="little")[:, :n_dets]`), decode, compute
`(predictions @ obs_matrix.T) & 1` if your decoder returns corrections, and
`np.packbits(..., bitorder="little")` the result.

---

## Ship it

**Soft-import** in `decoders/__init__.py` so missing libraries never break the
registry:

```python
if importlib.util.find_spec("my_lib") is not None:
    try:
        from . import my_decoder  # noqa: F401 — registers my-decoder/cpu
    except ImportError as exc:
        _log.debug("my_decoder import failed: %s", exc)
```

**Smoke test** in `tests/test_simulation_backend_quality.py`:

```python
def test_my_decoder_registered_and_runs():
    importorskip_safe("my_lib")   # or pytest.importorskip
    assert "my-decoder" in list_decoders()
    stats = SimulationPipeline(
        decoder_config=DecoderConfig("my-decoder"),
        max_shots=200, max_errors=10_000, batch_size=100,
        num_workers=1, print_progress=False,
    ).run(_simple_observable_circuit(error_probability=0.1))
    assert stats.shots > 0
```

Run `venv/bin/python -m pytest tests/ -m "not slow" -q` before opening a PR,
and document any new `params` in `lightstim/simulation/README.md`.

---

## Gotchas

1. **Bit order is little-endian** (LSB-first) for syndromes in and predictions
   out. Big-endian gives the right shapes and wrong answers. (Pattern C
   handles this for you.)
2. **`H.shape[1]` is the mechanism count, not `dem.num_errors`** — duplicate
   merging can shrink it (a chen_p96 z_only DEM had ~36k duplicate columns).
   Sizing a `"correction"` output from the DEM gives wrong-length predictions.
3. **C-contiguous matrices for GPU/C extensions** — `dem_to_matrices` dense
   output is `order="C"`; keep it that way or the decode silently transposes.
4. **GPU decoders need `num_workers=1`** — `cudaq_qec` pre-allocates GPU
   memory per compile; more workers OOM or corrupt results.
5. **Failure flags need `SimulationPipeline`** — they ride an in-process side
   channel (`compiled.last_flags`), not the bit-packed return, so raw
   `sinter.collect` never sees them.

---

## Where things live

| File | What it is |
|---|---|
| `decoder_backend/registry.py` | `register_decoder()`, `get_decoder()`, `list_decoders()` |
| `decoder_backend/decoders/__init__.py` | soft-import dispatcher — add your import here |
| `decoder_backend/decoders/{mwpf,relay_bp}.py` | Pattern A references — direct registration |
| `decoder_backend/decoders/tesseract.py` | Pattern A lazy variant — deferred native import |
| `decoder_backend/decoders/bposd.py` | Pattern A + param translation |
| `decoder_backend/decoders/cudaqx.py` | Pattern B reference (DEM matrices + GPU) |
| `decoder_backend/external.py` | Pattern C — the `ExternalDecoder` facade |
| `decoder_backend/decoders/ldpc_bp.py` | Pattern C reference (flags, serial BP) |
| `decoder_backend/decoders/chain.py` | multi-level chain over registered decoders |
| `decoder_backend/dem_matrices.py` | `dem_to_matrices(dem, *, sparse, merge_duplicates)` |
| `decoder_backend/pcm.py` | older `dem_to_check_matrices` — merges priors by **summing**; prefer `dem_to_matrices` |
| `decoder_backend/config.py` | `DecoderConfig` (incl. `on_decode_failure`) |
| `tutorials/how-to-add-new-decoders.ipynb` | runnable walkthrough of all three patterns |
