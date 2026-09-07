# Bacon-Shor dedicated / generic SE review

The native `BaconShorCodeExtractionBlock` fixes X interactions left→right and
Z interactions negative-y→positive-y in the patch frame. This review compares
it with `GenericCSSGaugeExtractionBlock` using the same code declaration and
LightStim-generated detectors. Both use two CNOT layers per basis.

Run from the repository root with the LightStim environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python playground/subsystem/bacon_shor_schedule/run.py
python playground/subsystem/bacon_shor_schedule/make_review.py
```

`--shots` sets fixed Z-memory MWPM shots per distance and schedule (default
1,000,000). `--shots 0` performs just the instrument, detector and distance
checks. `--output` selects the asset directory (default `results/2026-09-06`).

The audit uses d=3,5,7,9, X/Z memories, and d complete XZ pairs. It compares:

- all signed input/output/record flows of one SE round;
- affine detector row spaces and logical observables modulo detectors;
- circuit distance: an exact complete graphlike detector projection gives a
  lower bound, and a physical witness in the full DEM gives an upper bound.

Two Pauli fault sets are checked: reset and ancilla measurement flips p,
CNOT `DEPOLARIZE2(p)`, with either ideal or independently noisy final data
readout. p=0.001; neither model includes idle noise. The optional decoding
comparison uses ideal final readout and selects Z-record detectors before
calling LightStim's PyMatching decoder. LER is per complete memory shot of d
XZ pairs, for one Z-memory observable. Runs use independent seeds, no
postselection, and pointwise 95% Clopper-Pearson intervals.

Artifacts include `summary.json`, checkpointed `checks.json`, `mwpm.csv`, all
Stim circuits and undecomposed DEMs, minimum-fault witnesses and their physical
locations, batch seeds/counts, and each decoding job's first bit-packed batch.
Source and circuit SHA256 hashes identify the actual implementation.

The review notebook is
[`playground/subsystem/bacon_shor_se_review.ipynb`](../../../playground/subsystem/bacon_shor_se_review.ipynb).
Figures and native Stim diagrams are standalone files alongside the results.

Earlier `subsystem_crosscheck` and Foundation baseline assets used generic
extraction. Their generators now select that class explicitly to preserve
their schedules after the public Bacon-Shor default changed.

## CPU decoder and literature review

```bash
PYTHONPATH=. python playground/subsystem/bacon_shor_schedule/decoder_review.py --workers 4
PYTHONPATH=. python playground/subsystem/bacon_shor_schedule/audit_decoders.py
PYTHONPATH=. python playground/subsystem/bacon_shor_schedule/make_decoder_review.py
```

The checkpointed sweep covers d=3,5,7,9 and p=0.0005,0.001,0.002 on the
dedicated SE. MWPM and serial BP+OSD share the same Z-detector samples;
serial/parallel BP+OSD on the full XZ DEM are separate profiles at p=0.001.
Each job targets 100 failures, capped at 2M shots or 180 seconds. The final
report labels incomplete statistics instead of extrapolating them.
`--max-seconds` changes each job's cumulative time budget on resumption.

Outputs: [`results/2026-09-06-decoder-review/REPORT.md`](results/2026-09-06-decoder-review/REPORT.md),
CSV/JSON counts, paired decoder disagreements, PNG/SVG figures, first-batch
samples, and an executed
[`bacon_shor_decoder_review.ipynb`](../../../playground/subsystem/bacon_shor_decoder_review.ipynb).
The paired runner uses the LightStim decoder registry directly so that every
decoder prediction and shared input can be saved. Ordinary independent runs
can pass the selected circuit into `SimulationPipeline` instead.
