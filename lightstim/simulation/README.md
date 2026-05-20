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
│   │   └── mwpf.py        # MWPF decoder (CPU)
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
