# Native Bacon–Shor memory / MWPM baseline

This is the numerical companion to
[memory_bacon_shor.ipynb](../../../notebooks/Memory/memory_bacon_shor.ipynb).
It uses `BaconShorCode`, `BaconShorCodeExtractionBlock`, `MemoryExperiment`,
`NoiseConfig` and `SimulationPipeline(DecoderConfig("pymatching"))`.
All detectors and the protected observable come from LightStim's tracker.

Run from the repository root with the LightStim dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m benchmarks.memory.bacon_shor.run --shots 1000000
```

The default output is `results/native_mwpm/`. It contains full and selected
Stim circuits, undecomposed MWPM DEMs, a summary with counts, exact 95%
Clopper–Pearson intervals, seeds, library versions and source/circuit hashes,
plus the three notebook figures. `--output PATH` chooses another directory.
`--shots` must be a positive multiple of the 10,000-shot pipeline batch size.
The script always takes fixed shots; it does not stop early at an error target.

## Committed baseline (2026-09-07)

- Z memory; d=3,5,7,9; d complete X-then-Z extraction pairs per shot.
- Dedicated geometric SE: X left then right; Z negative-y then positive-y.
- Native `circuit_level` noise: `p_2q = p_meas = p_reset = 0.001`,
  `p_idle = p_1q = 0`. CNOT depolarization, reset flips and readout flips;
  **final data readout is noisy**. There is no idle noise.
- CPU PyMatching, one worker, seed 0, 10,000 shots per batch, 1M shots per d.
  No postselection. LER counts protected-observable failures per complete
  memory shot, not per round.
- Select original pure Z-record detectors *after* building the full XZ memory.
  Every physical instruction, noise channel, readout and observable is retained.
  This drops complementary syndrome information; it is a decoding baseline.
  `select_memory_detectors` is an experiment helper for M/MX CSS memories,
  not a general detector-inference API.

| d | Errors / shots | LER per shot |
| --- | --- | --- |
| 3 | 708 / 1,000,000 | 7.08e-4 |
| 5 | 201 / 1,000,000 | 2.01e-4 |
| 7 | 102 / 1,000,000 | 1.02e-4 |
| 9 | 63 / 1,000,000 | 6.30e-5 |

[precompute/summary.json](precompute/summary.json) records the complete metadata
and uncertainties. The notebook assets are in
[`notebooks/Memory/assets/bacon_shor/`](../../../notebooks/Memory/assets/bacon_shor/).
To update the committed baseline, rerun the command, review its output, then
copy `summary.json` to `precompute/` and the three figures to the notebook
asset directory. Keep the notebook table and this README consistent with the
new counts. Do not overwrite the historical review results.

For each d, the runner verifies zero ideal detector/observable flips,
preservation of the physical circuit, and graphlikeness of the complete
undecomposed selected DEM. The shortest error in that projection gives a
lower bound on full-circuit distance; an undetected logical-fault witness in
the full DEM with physical locations gives an upper bound. Both equal d for
these Z-memory circuits and this fault set. Finite-distance suppression here
does not establish an asymptotic threshold.

The earlier ideal-final-readout, generic/dedicated and CPU BP+OSD comparisons
remain in [playground/subsystem](../../../playground/subsystem/README.md).
