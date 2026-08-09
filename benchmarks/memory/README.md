# Memory Experiment Benchmark

Run memory experiments across any combination of QEC codes, distances, and error rates.
Results are saved to CSV with automatic checkpointing (safe to interrupt and resume).

## Quick Start

```bash
# Surface code, 3 distances, 5 p-values — finishes in ~5 min on 8 CPU cores
venv/bin/python benchmarks/memory/run_memory.py \
    --codes rotated_sc \
    --distances 3 5 7 \
    --p-values 1e-3 3e-3 5e-3 1e-2 2e-2 \
    --decoder pymatching --num-workers 8

# Plot the result
venv/bin/python benchmarks/memory/plot_memory.py \
    benchmarks/memory/results/rotated_sc_pymatching.csv
```

## Supported Codes

| Code | `--codes` name | Requires `--distances` |
|------|---------------|----------------------|
| Rotated Surface Code | `rotated_sc` | yes |
| Unrotated Surface Code | `unrotated_sc` | yes |
| Toric Code | `toric` | yes |
| Color Code (6-6-6) | `color` | yes |
| XZZX Surface Code | `xzzx_sc` | yes |
| BB [[72,12,6]] | `bb_72_12_6` | no (d=6 fixed) |
| BB [[90,8,10]] | `bb_90_8_10` | no (d=10 fixed) |
| BB [[108,8,10]] | `bb_108_8_10` | no (d=10 fixed) |
| BB [[144,12,12]] | `bb_144_12_12` | no (d=12 fixed) |
| BB [[288,12,18]] | `bb_288_12_18` | no (d=18 fixed) |
| HGP / unrotated SC [[13,1,3]] | `hgp_13_1_3` | no (d=3 fixed) |
| HGP / toric [[18,2,3]] | `hgp_18_2_3` | no (d=3 fixed) |
| HGP [[225,9,4]] | `hgp_225_9_4` | no (d=4 fixed) |

> **Not yet supported**: 4D geometric codes (`FourDGeoCode`) use an L-matrix parameter
> interface incompatible with the `--distances` flag. See `notebooks/Memory/memory_4D_hadamard.ipynb`
> for an interactive example, and `lightstim/qec_code/four_d_geo_code/configs.py` for named configs.

## Supported Decoders

| `--decoder` | Backend | Best for |
|-------------|---------|---------|
| `pymatching` (default) | CPU | Surface / toric codes |
| `mwpf` | CPU | General QLDPC |
| `cpu_bposd` | CPU | QLDPC codes, no GPU |
| `gpu_bposd` | GPU (CUDA) | BB/HGP codes at scale |

## Color Code SE Circuits

`--codes color` defaults to `space_multiplexing`. Select one or more circuits
with `--color-se-circuits`:

| CLI name | Layout | Extraction block | Bases |
|----------|--------|------------------|-------|
| `space_multiplexing` | `superdense` | `ColorCodeSpaceMultiplexingBlock` | X/Y/Z |
| `bell_multiplexing` | `superdense` | `ColorCodeBellMultiplexingBlock` | X/Y/Z |
| `bell_flagging` | `superdense` | `ColorCodeBellFlaggingBlock` | X/Y/Z |
| `time_multiplexing` | `triangular` | `ColorCodeTimeMultiplexingBlock` | X/Y/Z |
| `middle_out` | `rectangle` | `ColorCodeMiddleOutBlock` | X/Z |

For example, this runs all five circuits without exceeding the repository's
48-CPU benchmark limit:

```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes color --distances 3 5 7 \
    --color-se-circuits \
        space_multiplexing bell_multiplexing bell_flagging \
        time_multiplexing middle_out \
    --basis Z \
    --p-values 1e-4 2e-4 4e-4 7e-4 1e-3 2e-3 4e-3 \
    --decoder mwpf --num-workers 48 \
    --max-shots 10000000 --max-errors 500 \
    --output benchmarks/memory/results/color_all_mwpf.csv
```

Use `--decoder gpu_bposd --num-workers 1` for the GPU backend. The runner
always uses exactly one worker for GPU decoding.

## Common Use Cases

### Surface code family comparison
```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes rotated_sc unrotated_sc toric \
    --distances 3 5 7 \
    --p-values 1e-3 2e-3 5e-3 1e-2 \
    --decoder pymatching --num-workers 8
```

### BB codes (CPU, smaller scale)
```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes bb_72_12_6 bb_144_12_12 \
    --p-values 1e-3 3e-3 5e-3 1e-2 \
    --decoder cpu_bposd --num-workers 4
```

### BB codes (GPU)
```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes bb_72_12_6 bb_108_8_10 bb_144_12_12 \
    --p-values 1e-3 2e-3 5e-3 1e-2 \
    --decoder gpu_bposd
```

### HGP product-coloration memory (GPU)

```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes hgp_13_1_3 hgp_18_2_3 hgp_225_9_4 \
    --p-values 1e-3 2e-3 3e-3 \
    --basis Z X \
    --decoder gpu_bposd
```

With `--rounds` omitted, the three instances use their declared distances
`3`, `3`, and `4`. The first two are structural reference anchors: their HGP
patches are exactly the distance-3 unrotated surface code and, up to a
coordinate transpose, the distance-3 toric code. The third is the qLDPC
memory benchmark instance.

The bundled `[[225,9,4]]` seed is a fixed, independently generated
`(3,4)`-biregular instance with the same parameters as the smallest code in
arXiv:2308.08648. It is not the authors' unpublished random matrix. Result
rows identify its SE circuit explicitly as `product_coloration`.

The benchmark CLI intentionally selects complete, reproducible instances by
name instead of parsing parity-check matrices inline. For an arbitrary pair
`H1, H2`, construct `HGPCode(H1, H2, d=...)` programmatically.

### Both Z and X basis
```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes rotated_sc --distances 3 5 \
    --p-values 1e-3 5e-3 1e-2 \
    --basis Z X --decoder pymatching
```

### Color Code X/Y/Z memory experiments

```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes color --distances 3 5 \
    --color-se-circuits space_multiplexing bell_multiplexing \
        bell_flagging time_multiplexing \
    --p-values 1e-3 2e-3 4e-3 \
    --basis X Y Z --decoder mwpf --num-workers 16
```

`middle_out` is intentionally omitted because its memory interface currently
supports X/Z, not Y.

### Phenomenological noise model
```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes rotated_sc --distances 3 5 7 \
    --p-values 1e-3 5e-3 1e-2 \
    --noise-model phenomenological --decoder pymatching
```

### High-accuracy run (for paper-quality data)
```bash
venv/bin/python benchmarks/memory/run_memory.py \
    --codes rotated_sc --distances 3 5 7 \
    --p-values 1e-3 2e-3 5e-3 7e-3 1e-2 1.5e-2 \
    --decoder pymatching \
    --max-shots 10000000 --max-errors 500 \
    --num-workers 32
```

## Plotting

```bash
# Single CSV
venv/bin/python benchmarks/memory/plot_memory.py \
    benchmarks/memory/results/rotated_sc_pymatching.csv

# Merge multiple CSVs (e.g. compare codes)
venv/bin/python benchmarks/memory/plot_memory.py \
    benchmarks/memory/results/rotated_sc_pymatching.csv \
    benchmarks/memory/results/bb_72_12_6_cpu_bposd.csv \
    --output results/comparison.png

# Filter and title
venv/bin/python benchmarks/memory/plot_memory.py results/*.csv \
    --codes color --se-circuits bell_flagging time_multiplexing \
    --basis Z --distances 3 5 7 \
    --title "Color Code"
```

## Output Format

Results are saved as CSV with one row per complete benchmark configuration:

```
code, distance, p, basis, rounds, se_circuit, noise_model, decoder_name,
layout, block_class, shots, errors, logical_error_rate, seconds,
n_data, n_total, k
```

Default output path: `benchmarks/memory/results/<codes>_<decoder>.csv`

The `results/` directory is git-ignored. Benchmark data stays local unless it
is deliberately published elsewhere.

## Checkpointing

The runner automatically skips tasks already present in the output CSV.
Safe to interrupt with Ctrl+C and resume — just re-run the same command.
The SE circuit is part of the checkpoint key, so different Color Code circuits
can safely share one CSV.
