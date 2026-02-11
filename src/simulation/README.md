# Unified Decoder Backend Architecture

## 1. Goal and Pipeline Summary

**Input**: `stim.Circuit` from Experiment (with detectors, logical observables, noise injected)

**Pipeline (4 steps)**:

1. **Sampling** - `circuit.compile_detector_sampler().sample(batch_size)`
2. **Post-selection** - Discard samples where any detector tagged `["post-select"]` flips; record keep/discard counts and post-selection rate
3. **Decoding** - Pass surviving samples to Decoder; compare predictions with logical observables to compute LER
4. **Parallel execution** - Batch tasks across multiple workers (CPU/GPU) to maximize throughput

---

## 2. Post-Select Detector Identification

Use **detector tags** consistent with the existing framework (e.g. `TaggedIdling` with `"SE_start"` on TICK instructions in `src/noise/rules.py`).

| Mechanism | Implementation |
|-----------|----------------|
| **Tag on DETECTOR** | Add `tag="post-select"` when appending DETECTOR instructions. Stim circuit instructions support tags. |
| **Avoid 4th coord** | Do not use 4th coord for post-select: outdated in sinter, interferes with stim's 3D match graph visualization. |
| **Fallback** | `PipelineConfig.post_select_detector_indices` for experiments that cannot add tags (e.g. external circuits). |

**Implementation**: Extend `src/ir/tracker.py` to accept optional `post_select=True` when constructing DETECTOR. Utility `get_post_select_detector_indices(circuit)` iterates the circuit and returns detector indices with the post-select tag.

---

## 3. Decoder Abstraction

- **Decoder**: Implements `sinter.Decoder` (compile_decoder_for_dem, decode_shots_bit_packed)
- **DecoderConfig**: (name, backend, **decoder_params) - e.g. `'pymatching'`, `'bposd'`, `'nv-qldpc-decoder'`
- **DecoderRegistry**: name -> Decoder class/factory

**Initial decoder support**: pymatching (CPU), bposd (CPU), nv-qldpc-decoder (GPU)

---

## 4. Simulation Pipeline Architecture

**PipelineConfig** (dataclass):
- `max_shots`, `max_errors` - stopping conditions
- `batch_size` - shots per sampling batch (e.g. 10000)
- `num_workers` - parallel processes
- `decoder`: DecoderConfig
- `post_select_detector_indices`: Optional[List[int]] - if None, infer from circuit (DETECTOR with tag="post-select")
- `output_dir`: Optional[str] - e.g. `"./data/results"`
- `output_filename`: Optional[str] - e.g. `"ler_{timestamp}.csv"`
- `output_format`: Literal["csv", "json", "parquet"] - default "csv"

**Output stats** (per task):
- `shots`, `post_selected_shots`, `post_selection_rate`, `errors`, `logical_error_rate`, `seconds`, `json_metadata`

---

## 5. Output Format

When `output_dir` is set, results are saved to:
- **Path**: `{output_dir}/{output_filename}` (supports `{timestamp}` placeholder)
- **Formats**: CSV (default, pandas-compatible), JSON, Parquet
- **Columns**: stats fields + flattened json_metadata keys (d, p, rounds, etc.)

---

## 6. Worker Model

- **CPU**: multiprocessing workers; each: sample batch -> post-select -> decode
- **GPU**: Same loop, workers pinned to GPU via `rank % num_gpus`; `CUDA_VISIBLE_DEVICES` set per worker
- **No post-selection**: Delegate to `sinter.collect` for full compatibility

---

## 7. Relation to sinter

- **No post-selection**: Delegate to `sinter.collect`
- **With post-selection**: Custom pipeline; decoders still implement `sinter.Decoder` interface
- **sinter**: CPU multiprocessing only; GPU requires manual `CUDA_VISIBLE_DEVICES` per worker

---

## 8. Module Layout

```
simulation/
├── decoder_backend/
│   ├── __init__.py
│   ├── config.py          # DecoderConfig, PipelineConfig
│   ├── registry.py        # DecoderRegistry
│   ├── decoders/
│   │   ├── pymatching.py
│   │   └── ...
│   ├── pipeline.py        # SimulationPipeline
│   ├── post_select.py     # apply_post_selection, get_post_select_detector_indices
│   └── worker.py          # CPU/GPU worker functions
├── decoder.py             # BaseDecoder (legacy, migrate to registry)
├── simulator.py           # QECSimulator uses SimulationPipeline
└── gpu_worker.py
```

---

## 9. Dependencies

- `stim` - circuit representation and sampling
- `sinter` - sampling/decoding (bundled with stim or install separately)
- `pymatching` - for pymatching decoder (install for LER simulations: `pip install pymatching`)

## 10. Usage

```python
circuit = MemoryExperiment(...).build()
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching", backend="cpu"),
    max_shots=1_000_000,
    max_errors=100,
    num_workers=4,
    output_dir="data/results",
)
stats = pipeline.run(circuit, json_metadata={"d": 3, "p": 0.001})

# Batch mode
tasks = [ExperimentTask(circuit=exp.build(), json_metadata=meta) for ...]
df = pipeline.run_batch(tasks)
```
