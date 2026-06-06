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

LightStim's decoder backend is a small registry of `sinter.Decoder`
subclasses. Each decoder is registered under a name + backend (`cpu`,
`gpu`, or `fpga`), then `DecoderConfig(name="…", backend="…", params={})`
looks it up at runtime.

Adding a decoder is **always one new file** in
`lightstim/simulation/decoder_backend/decoders/`, plus **one line** in
`decoders/__init__.py` to soft-import it, plus **one smoke test** in
`tests/test_simulation_backend_quality.py`.

This skill walks through three real patterns in the repo, ordered by
complexity. Pick the one that matches your situation.

---

## The contract

Every decoder must subclass `sinter.Decoder` and implement
`compile_decoder_for_dem`. That method takes a `stim.DetectorErrorModel`
and returns a `sinter.CompiledDecoder`:

```python
import sinter
import stim

class MyDecoder(sinter.Decoder):
    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> sinter.CompiledDecoder:
        ...   # one-time prep per DEM
```

The compiled decoder must implement
`decode_shots_bit_packed(*, bit_packed_detection_event_data: np.ndarray)
-> np.ndarray` — bit-packed in, bit-packed out:

```python
class _MyCompiledDecoder(sinter.CompiledDecoder):
    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        # input  shape: (n_shots, ceil(n_detectors / 8)), uint8, LSB-first
        # output shape: (n_shots, ceil(n_observables / 8)), uint8, LSB-first
        ...
```

If your underlying library already implements `sinter.Decoder` and
`sinter.CompiledDecoder` correctly (e.g. `stimbposd.SinterDecoder_BPOSD`,
`mwpf.SinterMWPFDecoder`), **you don't need to write a CompiledDecoder
yourself** — see Pattern A.

---

## Pattern A: thin wrapper around an existing `sinter.Decoder`

**When to use**: the upstream library already implements `sinter.Decoder`.
You just want it registered under a LightStim name.

**Real example**: `lightstim/simulation/decoder_backend/decoders/mwpf.py`
(30 lines including imports). It's literally:

```python
from ..registry import register_decoder

try:
    from mwpf import SinterMWPFDecoder
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

if _AVAILABLE:
    register_decoder("mwpf", SinterMWPFDecoder, aliases=[])
```

That's it. No subclass needed.

---

## Pattern B: wrapper with parameter translation

**When to use**: the upstream library implements `sinter.Decoder`, but its
parameter names don't match LightStim's user-facing conventions, or you
want to set defaults.

**Real example**:
`lightstim/simulation/decoder_backend/decoders/bposd.py`. The user passes
`max_iterations`, `bp_method='min_sum'`, etc., but `stimbposd`'s API uses
`max_bp_iters` and `bp_method='minimum_sum'`. The wrapper translates:

```python
class BpOsdCpuDecoder(sinter.Decoder):
    def __init__(self, **params):
        translated = _unified_to_cpu({**_DEFAULTS, **params})
        self._inner = SinterDecoder_BPOSD(**translated)

    def compile_decoder_for_dem(self, *, dem):
        return self._inner.compile_decoder_for_dem(dem=dem)

if _BPOSD_AVAILABLE:
    register_decoder("bposd", BpOsdCpuDecoder, aliases=["bp_osd"], backend="cpu")
```

Key things to copy from this pattern:
1. `__init__(self, **params)` — accept arbitrary kwargs so users can pass
   anything through `DecoderConfig(params={...})`.
2. `_DEFAULTS` dict at module level — defines sensible defaults; user
   params override them via dict-merge.
3. A `_translate(params: dict) -> dict` function — rename keys, normalize
   case, drop irrelevant ones.
4. Hold the wrapped decoder in `self._inner` and delegate
   `compile_decoder_for_dem` to it.

---

## Pattern C: custom decoder from a DEM matrix

**When to use**: the upstream library is **not** sinter-compatible (most
research-paper decoders, GPU libraries, neural networks). You need to
parse the DEM into matrices yourself and write the decode loop.

**Real example**:
`lightstim/simulation/decoder_backend/decoders/cudaqx.py` — wraps NVIDIA's
`cudaq_qec` GPU decoder, which takes raw H matrices, not stim DEMs.

The pattern has three parts:

### 1. DEM → matrices

Convert a `stim.DetectorErrorModel` into the (H, observable matrix,
priors) triple your decoder needs. Steal this verbatim from `cudaqx.py`
(`_dem_to_matrices`); it handles flattened DEMs, detector targets,
observable targets, and decomposed errors.

```python
def _dem_to_matrices(dem: stim.DetectorErrorModel):
    """Returns (H, obs_matrix, priors) — see cudaqx.py for full impl."""
    n_dets = dem.num_detectors
    n_obs = dem.num_observables
    error_cols = []
    for inst in dem.flattened():
        if inst.type != "error": continue
        p = inst.args_copy()[0]
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():    dets.append(t.val)
            elif t.is_logical_observable_id(): obs.append(t.val)
        error_cols.append({"p": p, "dets": dets, "obs": obs})
    n_err = len(error_cols)
    H   = np.zeros((n_dets, n_err), dtype=np.uint8, order="C")
    obs = np.zeros((n_obs,  n_err), dtype=np.uint8, order="C")
    p   = np.zeros(n_err, dtype=np.float64)
    for e, col in enumerate(error_cols):
        p[e] = col["p"]
        for d in col["dets"]: H[d, e] = 1
        for o in col["obs"]:  obs[o, e] = 1
    return H, obs, p
```

### 2. Custom `CompiledDecoder`

Holds your already-prepared decoder instance + the observable matrix.
Implements the unpack → decode → pack loop:

```python
class _MyCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, inner, obs_matrix, n_detectors):
        self._inner = inner
        self._obs_matrix = obs_matrix
        self._n_dets = n_detectors

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        # 1. Unpack syndromes: bit_packed → (n_shots, n_dets) uint8
        syndromes = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little"
        )[:, : self._n_dets]
        # 2. Call your underlying decoder shot-by-shot or batched
        predictions = self._inner.decode_batch(syndromes)  # (n_shots, n_err) uint8
        # 3. Compute which observables flipped: obs_flip = predictions @ obs_matrix.T  mod 2
        obs_flips = (predictions @ self._obs_matrix.T) & 1
        # 4. Pack back: (n_shots, n_obs) → bit-packed uint8
        return np.packbits(obs_flips.astype(np.uint8), axis=1, bitorder="little")
```

### 3. Top-level `sinter.Decoder`

Calls `_dem_to_matrices`, constructs the inner decoder, returns the
`CompiledDecoder`:

```python
class MyDecoder(sinter.Decoder):
    def __init__(self, **params):
        self._params = params

    def compile_decoder_for_dem(self, *, dem):
        H, obs_matrix, priors = _dem_to_matrices(dem)
        inner = my_lib.Decoder(H=H, priors=priors, **self._params)
        return _MyCompiledDecoder(inner, obs_matrix, dem.num_detectors)
```

---

## Registration

After your decoder file is written, add one call at module bottom:

```python
register_decoder(
    name="my-decoder",          # canonical name; user types DecoderConfig(name="my-decoder")
    decoder_class=MyDecoder,
    aliases=["my_dec", "mwpm2"],  # optional; excluded from list_decoders()
    backend="cpu",              # or "gpu" or "fpga"
)
```

A single name can have multiple backends. E.g. `cudaqx.py` registers both
the GPU `nv-qldpc-decoder` name and a `bposd` `backend="gpu"` override, so
`DecoderConfig(name="bposd", backend="cpu")` hits the CPU implementation
and `backend="gpu"` hits the GPU one.

---

## Wire it into discovery

`lightstim/simulation/decoder_backend/decoders/__init__.py` does
**soft imports** so missing libraries don't crash the registry. Follow
the existing pattern — add your module behind an `importlib.util.find_spec`
guard:

```python
# In decoders/__init__.py, append:
if importlib.util.find_spec("my_lib") is not None:
    try:
        from . import my_decoder  # noqa: F401 — registers my-decoder/cpu
    except ImportError as exc:
        _log.debug("my_decoder import failed: %s", exc)
else:
    _log.debug("my_lib not installed; skipping my-decoder")
```

This is mandatory. If you `import` your module unconditionally and the
underlying library isn't installed, every LightStim user will hit an
`ImportError` at startup.

---

## Smoke test

Add to `tests/test_simulation_backend_quality.py`. The minimum bar: your
decoder must appear in `list_decoders()` and successfully decode the
trivial single-qubit observable circuit defined at the top of that file.

```python
def test_my_decoder_registered_and_runs():
    from lightstim.simulation.decoder_backend import list_decoders
    assert "my-decoder" in list_decoders()

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("my-decoder"),
        max_shots=100,
        max_errors=1,
        batch_size=50,
        num_workers=1,
        print_progress=False,
    )
    stats = pipeline.run(_simple_observable_circuit(error_probability=0.1))
    assert stats.shots > 0
```

If your decoder needs an optional dependency, gate the test:

```python
my_lib = pytest.importorskip("my_lib")
```

so CI doesn't fail when the library isn't installed.

---

## Where things live (quick reference)

| File | What it is |
|---|---|
| `lightstim/simulation/decoder_backend/registry.py` | The registry — `register_decoder()`, `get_decoder()`, `list_decoders()` |
| `lightstim/simulation/decoder_backend/decoders/__init__.py` | Soft-import dispatcher — add your import here |
| `lightstim/simulation/decoder_backend/decoders/pymatching.py` | **Pattern A reference** (30 lines, simplest) |
| `lightstim/simulation/decoder_backend/decoders/mwpf.py` | **Pattern A reference** (no subclass at all) |
| `lightstim/simulation/decoder_backend/decoders/bposd.py` | **Pattern B reference** — param translation |
| `lightstim/simulation/decoder_backend/decoders/cudaqx.py` | **Pattern C reference** — DEM matrix + custom CompiledDecoder + GPU |
| `lightstim/simulation/decoder_backend/pipeline.py` | Consumer of the registry; you shouldn't need to touch it |
| `lightstim/simulation/decoder_backend/config.py` | `DecoderConfig` dataclass |
| `tests/test_simulation_backend_quality.py` | Decoder smoke tests live here |

---

## Gotchas

### 1. Bit-packing is little-endian (LSB first)
`bit_packed_detection_event_data` and the predictions you return both
use `np.unpackbits(..., bitorder="little")` / `np.packbits(...,
bitorder="little")`. Use big-endian by accident and every observable
prediction will be wrong but the shape will be right — a silent
correctness bug.

### 2. C-contiguous matrices for GPU
Pattern C constructs `H` and `obs_matrix` with `order="C"`. Most C++/CUDA
extensions assume row-major; passing Fortran-ordered arrays silently
transposes the decode and produces nonsense predictions.

### 3. GPU decoders must use `num_workers=1`
`cudaq_qec` decoders pre-allocate GPU memory in `compile_decoder_for_dem`.
Running with `num_workers > 1` either OOMs or returns garbage. Document
this in your decoder's docstring and in the smoke test.

### 4. `sinter.collect` is bypassed for GPU
If your decoder needs GPU resources, `SimulationPipeline` already handles
this — see the custom decode loop in `pipeline.py` and `worker.py`. You
don't need to do anything special, but **do not** rely on
`sinter.collect` semantics in your `compile_decoder_for_dem`.

### 5. Soft-import discipline
Inside your decoder file, the `try: import upstream_lib` block must
catch `ImportError`. Use `pytest.importorskip` in tests. Otherwise a user
without your library gets a hard import failure when they call
`SimulationPipeline(...)`, not your decoder.

### 6. Post-selection: usually free
If your decoder just consumes syndromes and emits predictions,
post-selection (state injection, distillation) works automatically — the
pipeline filters shots before they reach you. If you need to inspect the
post-selection mask, see `lightstim/simulation/decoder_backend/post_select.py`.

---

## End-to-end checklist

Before opening a PR:

- [ ] One new file in `lightstim/simulation/decoder_backend/decoders/`.
- [ ] `register_decoder(name, cls, aliases=[...], backend="…")` at module bottom.
- [ ] Soft-import hook added to `decoders/__init__.py` behind a
      `importlib.util.find_spec` guard.
- [ ] Smoke test added to `tests/test_simulation_backend_quality.py` that
      verifies registration and decodes the trivial circuit.
- [ ] CI passes: `venv/bin/python -m pytest tests/ -m "not slow" -q`.
- [ ] If the decoder needs special hardware (GPU/FPGA), document the
      `num_workers` constraint in the docstring.
- [ ] If your decoder needs new parameters in `DecoderConfig.params`,
      document them in `lightstim/simulation/README.md` (the decoder
      backend reference).

---

## Working examples

The simplest possible decoder to extend from is **PyMatching** — read
`lightstim/simulation/decoder_backend/decoders/pymatching.py` end to end
(it's 56 lines including imports). For parameter handling, read
`bposd.py`. For an end-to-end custom decoder with DEM parsing and GPU
integration, read `cudaqx.py`.

Once your decoder is registered, it's usable from anywhere:

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(
        name="my-decoder",
        backend="cpu",
        params={"max_iterations": 500},
    ),
    max_shots=100_000,
    max_errors=200,
    num_workers=4,
)
stats = pipeline.run(noisy_circuit)
print(f"LER: {stats.logical_error_rate:.3e}")
```

No other code changes are required. The HTTP server, notebooks, and
benchmark scripts all go through `DecoderConfig`, so they pick up your
new decoder by name with no further plumbing.
