# Unified Decoder Backend Architecture

## 1. Goal and Pipeline Summary

**Input**: `stim.Circuit` from Experiment (with detectors, logical observables, noise injected)

**Pipeline (4 steps)**:

1. **Sampling** - `dem.compile_sampler().sample(batch_size)`
2. **Post-selection** - Discard samples where any detector tagged `["post-select"]` flips; record keep/discard counts and post-selection rate
3. **Decoding** - Pass surviving samples to Decoder; compare predictions with logical observables to compute LER
4. **Parallel execution** - Batch tasks across multiple workers (CPU/GPU) to maximize throughput

---

## 2. Post-Select Detector Identification

| Mechanism | Implementation |
|-----------|----------------|
| **Tag on DETECTOR** | Add `tag="post-select"` when appending DETECTOR instructions |
| **Fallback** | `PipelineConfig.post_select_detector_indices` for experiments that cannot add tags |

**Implementation**: `get_post_select_detector_indices(circuit)` iterates the circuit and returns detector indices with the post-select tag.

---

## 3. Decoder Abstraction

- **Decoder**: Implements `sinter.Decoder` (`compile_decoder_for_dem`, `decode_shots_bit_packed`)
- **DecoderConfig**: `(name, backend, params)` — e.g. `DecoderConfig("bposd", backend="gpu")`
- **Registry**: `name → { backend → decoder_class }` (backend-keyed)

**Decoder support**:

| Name | Backend | Package | Notes |
|------|---------|---------|-------|
| `"pymatching"` | `"cpu"` | `pymatching` | MWPM; alias `"mwpm"` |
| `"bposd"` | `"cpu"` | `stimbposd` | BP+OSD; alias `"bp_osd"` |
| `"bposd"` / `"nv-qldpc-decoder"` | `"gpu"` | `cudaq_qec` | GPU BP+OSD via NVIDIA cudaq_qec |
| `"mwpf"` | `"cpu"` | `mwpf` | — |
| `"relay-bp"` | `"cpu"` | `relay_bp` | Relay-BP; aliases `"relay_bp"`, `"relaybp"` |
| `"tesseract"` | `"cpu"` | `tesseract_decoder` | Beam-search MLE; lazy import |
| `"ldpc-bp"` | `"cpu"` | `ldpc` | Plain BP (no OSD), via `ldpc.BpDecoder`; aliases `"ldpc_bp"`, `"bp"` |
| `"mle-ilp"` | `"cpu"` | `scipy>=1.9` | Exact most-likely-error via ACG-ALP/RPC with MILP fallback; aliases `"mle"`, `"ilp"` |
| `"chain"` | `"cpu"` | *(none — composes other registered decoders)* | Multi-level escalation, e.g. BP → relay-BP → MLE; aliases `"decoder-chain"`, `"multi-level"` — see §11 |

Requesting a backend with no registration raises `ImportError` immediately (e.g. `backend="gpu"` without `cudaq_qec`).

---

## 4. Unified BP+OSD Parameters

Both CPU and GPU bposd backends accept the same parameter names:

| Unified param | CPU (`stimbposd`) | GPU (`cudaq_qec`) | Default |
|---|---|---|---|
| `max_iterations` | `max_bp_iters` | `max_iterations` | `1000` |
| `bp_method` | `'minimum_sum'`/`'product_sum'` | `1`/`0` (int) | `'min_sum'` |
| `ms_scaling_factor` | `ms_scaling_factor` | `scale_factor` | `0` |
| `osd_order` | `osd_order` | `osd_order` | `10` |
| `osd_method` | `'osd_cs'` etc | `3` (int) | `'osd_cs'` |
| `use_osd` | *(ignored; always on)* | `use_osd` | `True` |

### Exact MLE parameters

`DecoderConfig("mle-ilp")` is exact for supplied priors in `[0, 0.5]` by default and has
no solve deadline. Large detector error models can have long MILP tails; use a
positive `time_limit` together with an explicit `on_decode_failure` policy when
bounded throughput is more important than completing every shot.

| Parameter | Default | Meaning |
|---|---:|---|
| `time_limit` | `0.0` | Soft seconds-per-shot budget; zero is unlimited |
| `max_prior` | `0.5` | Optional cap applied after validation; `c<0.5` maps each positive `p` to `min(p, c)` |
| `max_cut_rounds` | `200` | Maximum forbidden-set cut rounds before MILP fallback |
| `max_rpc_rounds` | `8` | Redundant-parity-check rounds; zero disables RPC generation |
| `max_rpc_pivots` | `64` | Gaussian-elimination pivots per RPC round |
| `max_rpc_weight` | `256` | Skip denser derived checks |
| `max_rpc_memory_mb` | `512` | Packed RPC workspace cap; zero is unlimited |

Supplied priors must be in `[0, 0.5]`; values above `0.5` are rejected rather
than clamped. The default `max_prior=0.5` changes no accepted prior. A smaller
cap intentionally changes accepted priors above that cap. Zero-probability
mechanisms are fixed off, and probability-0.5 mechanisms have exactly zero
objective weight. A timed-out shot is flagged; `"error"`
counts it as wrong, `"discard"` removes it, and `"ignore"` trusts the partial
prediction.

---

## 5. Simulation Pipeline Architecture

**PipelineConfig** (dataclass):
- `max_shots`, `max_errors` — stopping conditions
- `batch_size` — shots per sampling batch (default 10 000)
- `num_workers` — parallel processes
- `decoder`: DecoderConfig
- `post_select_detector_indices`: Optional[List[int]] — if None, infer from circuit tags
- `post_select_observable_indices`: Optional[List[int]] — discard shots where any listed observable is wrong (pre-decode)
- `post_select_corrected_observable_indices`: Optional[List[int]] — discard shots where corrected observable is non-zero (post-decode)
- `target_observable_indices`: Optional[List[int]] — count errors only on these observables (None = all)
- `output_dir`, `output_filename`, `output_format` — optional CSV/JSON/Parquet output
- `progress_enabled`, `progress_output`, `progress_interval_sec`, `progress_min_delta_shots` — unified progress controls
- `progress_file_path` (+ rotating options) — optional file logging sink

**Output stats** (`SimulationStats`):
- `shots`, `post_selected_shots`, `post_selection_rate`, `errors`, `seconds`, `json_metadata`
- `logical_error_rate` — `errors / post_selected_shots`
- `ler_error_bar(z=1.96)` — half-width of a z-sigma Wilson confidence interval (95% CI by default)

---

## 6. Worker Model

- **CPU/GPU, with or without post-selection**: unified custom loop; each worker performs sample → post-select → decode
- **Single-process**: one process executes the full loop
- **Multi-process**: worker processes only update shared counters; main process aggregates and emits progress
- Progress output is unified across all paths (`shots kept errors LER elapsed ETA`) with dual-threshold throttling (time + shot delta)

---

## 7. Relation to sinter

- The backend no longer depends on `sinter.collect` for progress/runtime flow control
- Decoders still implement the `sinter.Decoder` interface (`compile_decoder_for_dem`, `decode_shots_bit_packed`)
- **Bit packing convention**: pipeline uses little-endian (`np.packbits(..., bitorder=\"little\")`)

---

## 8. Module Layout

```
simulation/
├── decoder_backend/
│   ├── __init__.py        # public exports: SimulationPipeline, DecoderConfig, SimulationStats, ExperimentTask, dem_to_check_matrices, ...
│   ├── config.py          # DecoderConfig, PipelineConfig, SimulationStats
│   ├── registry.py        # backend-keyed decoder registry
│   ├── decoders/
│   │   ├── __init__.py    # soft-imports all decoders; safe if package missing
│   │   ├── pymatching.py  # PyMatchingDecoder (CPU)
│   │   ├── bposd.py       # BpOsdCpuDecoder + unified param translation (CPU)
│   │   ├── cudaqx.py      # CudaQxDecoder + CudaQxCompiledDecoder (GPU)
│   │   ├── mwpf.py        # MWPF decoder (CPU)
│   │   ├── relay_bp.py    # Relay-BP decoder (CPU, sinter-native)
│   │   ├── tesseract.py   # Tesseract beam-search MLE (CPU, lazy import)
│   │   ├── ldpc_bp.py     # Plain BP decoder (CPU, ExternalDecoder facade)
│   │   ├── mle_ilp.py     # Exact most-likely-error decoder (CPU, SciPy MILP)
│   │   └── chain.py       # Multi-level decoder chain (CPU, composes other decoders)
│   ├── pipeline.py        # SimulationPipeline, ExperimentTask
│   ├── post_select.py     # apply_post_selection, get_post_select_detector_indices
│   ├── progress.py        # ProgressReporter, ProgressSnapshot
│   ├── pcm.py             # dem_to_check_matrices (DEM → sparse PCM + priors)
│   └── worker.py          # _decode_worker_cpu (multiprocessing)
```

---

## 9. Dependencies

- `stim` — circuit representation and sampling
- `sinter` — Decoder interface
- `pymatching` — MWPM decoder: `pip install pymatching`
- `stimbposd` — CPU BP+OSD: `pip install stimbposd`
- `mwpf` — MWPF decoder: `pip install mwpf frozendict frozenlist`
- `relay_bp` — Relay-BP decoder: `pip install "relay-bp[stim]"`
- `tesseract_decoder` — Tesseract beam-search MLE: `pip install tesseract-decoder` (a prebuilt wheel may not match every CPU; build from source if it fails to import)
- `ldpc` — Plain BP decoder: `pip install ldpc`
- `scipy>=1.9` — exact MLE (`scipy.optimize.milp`)
- `cudaq_qec` — GPU BP+OSD: `pip install cudaq_qec` (NVIDIA GPU required)

---

## 10. Usage

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig

# CPU PyMatching
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_shots=1_000_000,
    max_errors=100,
    num_workers=4,
    progress_output="print",
    progress_interval_sec=10.0,
)
stats = pipeline.run(circuit, json_metadata={"d": 3, "p": 0.001})

# Exact MLE: unlimited by default.
mle_pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("mle-ilp"),
    max_shots=2_000,
    num_workers=4,
)

# A practical bounded run. Timeout shots are explicitly discarded.
bounded_mle_pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(
        "mle-ilp",
        params={"time_limit": 2.0},
        on_decode_failure="discard",
    ),
)

# GPU BP+OSD (cudaq_qec nv-qldpc-decoder)
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("bposd", backend="gpu", params={
        "max_iterations": 1000,
        "osd_order": 10,
        "osd_method": "osd_cs",
    }),
    max_shots=1_000_000,
    max_errors=100,
    num_workers=1,
    print_progress=True,
)

# Batch mode
tasks = [ExperimentTask(circuit, json_metadata={"p": p}) for p in p_list]
df = pipeline.run_batch(tasks)
```

---

## 11. Multi-Level Decoder Chain

Hierarchical decoding — a fast decoder handles most shots, and only the ones
it fails on escalate to a slower/more powerful decoder — is exposed as the
`"chain"` decoder (`lightstim/simulation/decoder_backend/decoders/chain.py`).
Stage *k+1* re-decodes only the shots stage *k* flagged as failed (via the
same `last_flags` side channel `ExternalDecoder` subclasses already use);
shots no stage resolves surface through the chain's own `last_flags`, so
`DecoderConfig(on_decode_failure=...)` still governs the outcome for the
pipeline as a whole. Decoders that never emit failure flags (`pymatching`,
`relay-bp`, ...) resolve every shot handed to them, so they only make sense
as the last stage.

Two equivalent ways to configure it:

```python
# Explicit chain config — stages as names, dicts, or DecoderConfigs.
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("chain", params={"stages": [
        {"name": "ldpc-bp", "params": {"max_iter": 200}},
        {"name": "relay-bp", "params": {"num_sets": 300, "stop_nconv": 1}},
    ]}, on_decode_failure="discard"),
    ...
)

# Shorthand: hand SimulationPipeline a plain list of DecoderConfigs.
# The last config's on_decode_failure becomes the chain-level policy.
pipeline = SimulationPipeline(decoder_config=[
    DecoderConfig("ldpc-bp", params={"max_iter": 200}),
    DecoderConfig("relay-bp", params={"num_sets": 300}, on_decode_failure="discard"),
])
```

A stage's own `on_decode_failure` is ignored — inside the chain, "failure"
means "escalate to the next stage." Unknown stage names are validated eagerly
(in the parent process, before workers spawn), same as a plain `DecoderConfig`.
