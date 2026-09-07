# Subsystem review notebooks

These research reviews have moved out of `notebooks/Memory/`. The regular
[Bacon–Shor memory demo](../../notebooks/Memory/memory_bacon_shor.ipynb) contains
the dedicated memory circuit diagram and a live CPU MWPM baseline.
The early SHYPS memory checks are archived locally in
`archive/memory_subsystem.ipynb`; a dedicated notebook will accompany its later
integration. See the [SHYPS implementation notes](../../lightstim/qec_code/shyps/README.md).

| Review | Purpose | Reproduction sources and local assets |
| --- | --- | --- |
| [subsystem_crosscheck.ipynb](subsystem_crosscheck.ipynb) | Original Bacon–Shor capacity/literature checks and SHYPS reference comparisons | [subsystem_crosscheck](subsystem_crosscheck/README.md) |
| [bacon_shor_se_review.ipynb](bacon_shor_se_review.ipynb) | Dedicated/generic signed flows, detector spaces, distance and decoding comparisons | [bacon_shor_schedule](bacon_shor_schedule/README.md) |
| [bacon_shor_decoder_review.ipynb](bacon_shor_decoder_review.ipynb) | Paired CPU BP+OSD / MWPM scaling and saved-batch replay | [bacon_shor_schedule](bacon_shor_schedule/README.md#cpu-decoder-and-literature-review) |

Saved numerical and PNG outputs remain visible without running the reviews.
Large inline Stim SVG outputs are cleared; executing their cells regenerates
the views. Rerunning these historical notebooks requires their local
`playground/subsystem/*/results/2026-09-06*` directories, restored from the existing
research archives or regenerated with the linked research scripts. Raw
samples, reference snapshots, literature downloads and run directories are
not committed. Generators now write notebooks to this playground directory.

The historical circuit benchmarks use ideal final data readout and no idle
noise. The new formal demo uses native `NoiseConfig`, including noisy final
data readout. Its diagram and decoding cells run without these historical
archives. Reference counts, circuit hashes and other generated assets stay
in the local playground results directories.

## Regular memory experiments

Bacon–Shor uses the shared `benchmarks/memory/run_memory.py --codes bacon_shor`
entry point and the shared `benchmarks/memory/results/` CSV directory.
[Notebook asset generation](bacon_shor_memory/README.md) consumes that CSV;
it does not implement another decoder runner. Review scripts below this
playground directory retain their historical experiment configurations.

Local compressed subsystem archives live in `playground/subsystem_artifacts/`
(the main workspace copy is under `Research/LightStim/playground/`). Old
archives preserve the original directory names and source hashes. Restore
`bacon_shor_schedule/` and `subsystem_crosscheck/` contents into this directory;
do not recreate their former `benchmarks/memory/` locations. Historical
reports/manifests remain immutable records of their original runs. The
saved-batch validator reverses import-path relocation before checking the
historical source hash.
