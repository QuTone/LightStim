# Native Bacon–Shor memory / MWPM baseline

This is the numerical companion to
[memory_bacon_shor.ipynb](../../../notebooks/Memory/memory_bacon_shor.ipynb).
It uses `BaconShorCode`, `BaconShorCodeExtractionBlock`, `MemoryExperiment`,
`NoiseConfig` and `SimulationPipeline(DecoderConfig("pymatching"))`.
All detectors and the protected observable come from LightStim's tracker.

## Reproduce through the shared runner

From the repository root, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python benchmarks/memory/run_memory.py --codes bacon_shor --distances 3 5 7 9 \
    --p-values 0.001 --p-idle 0 --p-1q 0 --basis Z --decoder pymatching \
    --max-shots 1000000 --max-errors 1000001 --num-workers 1 --batch-size 10000
```

The usual output is `benchmarks/memory/results/bacon_shor_pymatching.csv`.
Existing rows are checkpointed; choose a fresh `--output` for a new run.
To generate review figures and circuit-distance audits from that CSV:

```bash
python -m playground.subsystem.bacon_shor_memory.make_assets \
    --input benchmarks/memory/results/bacon_shor_pymatching.csv
```

The asset generator performs no decoding. It rebuilds circuits with the shared
`build_circuit` interface, checks them, reads counts from the CSV and writes
figures, Stim/DEM files and `summary.json` under `results/native_mwpm/` here.
`--output PATH` selects another asset directory. The CSV does not record seeds
or workers; preserve the actual command alongside any new run when archiving.

## Local reference baseline (2026-09-07)

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
  The shared runner applies this selection only for Bacon–Shor MWPM.

| d | Errors / shots | LER per shot |
| --- | --- | --- |
| 3 | 708 / 1,000,000 | 7.08e-4 |
| 5 | 201 / 1,000,000 | 2.01e-4 |
| 7 | 102 / 1,000,000 | 1.02e-4 |
| 9 | 63 / 1,000,000 | 6.30e-5 |

The local `results/reference_2026-09-07/summary.json` preserves the original
reference metadata and uncertainties. New summaries are generated under the
output directory chosen above. These JSON files and exported figures remain
in playground and are not committed.

The notebook renders its Stim diagram and MWPM plot directly when executed;
it does not load external assets. Review a new run before updating the counts
in this README, and preserve historical results separately.

For each d, the asset generator verifies zero ideal detector/observable flips,
preservation of the physical circuit, and graphlikeness of the complete
undecomposed selected DEM. The shortest error in that projection gives a
lower bound on full-circuit distance; an undetected logical-fault witness in
the full DEM with physical locations gives an upper bound. Both equal d for
these Z-memory circuits and this fault set. Finite-distance suppression here
does not establish an asymptotic threshold.

The earlier ideal-final-readout, generic/dedicated and CPU BP+OSD comparisons
remain in [playground/subsystem](../../../playground/subsystem/README.md).
