# Subsystem review notebooks

These research reviews have moved out of `notebooks/Memory/`. The regular
[Bacon–Shor memory demo](../../notebooks/Memory/memory_bacon_shor.ipynb) contains
the dedicated extraction schedule and a self-contained CPU MWPM baseline.
[SHYPS memory and visualization](../../notebooks/Memory/memory_subsystem.ipynb)
remain in the original subsystem demo.

| Review | Purpose | Reproduction sources and local assets |
| --- | --- | --- |
| [subsystem_crosscheck.ipynb](subsystem_crosscheck.ipynb) | Original Bacon–Shor capacity/literature checks and SHYPS reference comparisons | [subsystem_crosscheck](../../benchmarks/memory/subsystem_crosscheck/README.md) |
| [bacon_shor_se_review.ipynb](bacon_shor_se_review.ipynb) | Dedicated/generic signed flows, detector spaces, distance and decoding comparisons | [bacon_shor_schedule](../../benchmarks/memory/bacon_shor_schedule/README.md) |
| [bacon_shor_decoder_review.ipynb](bacon_shor_decoder_review.ipynb) | Paired CPU BP+OSD / MWPM scaling and saved-batch replay | [bacon_shor_schedule](../../benchmarks/memory/bacon_shor_schedule/README.md#cpu-decoder-and-literature-review) |

Saved numerical and PNG outputs remain visible without running the reviews.
Large inline Stim SVG outputs are cleared; executing their cells regenerates
the views. Rerunning these historical notebooks requires their local
`benchmarks/memory/*/results/2026-09-06*` directories, restored from the existing
research archives or regenerated with the linked benchmark scripts. Raw
samples, reference snapshots, literature downloads and run directories are
not committed. Generators now write notebooks to this playground directory.

The historical circuit benchmarks use ideal final data readout and no idle
noise. The new formal demo uses native `NoiseConfig`, including noisy final
data readout. Its separate committed summary and figures can be opened and
reproduced without these historical archives.
