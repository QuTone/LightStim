# IonQ Beam Search beta: short paper-point check

Tested on LightStim branch `Ming/ionq-beam-BP-beta`, based on `f271d33`.
Upstream revision: `084a475b05fb64308103317a1ce5a0c4b0be58aa`, locally compiled
without source changes. Native build used the existing Python 3.13.5 environment
and upstream's C++20 / `-O3` build settings.

## Experiment

- Authors' BB[[144,12,12]] circuits, X/Z memories, 12 syndrome rounds.
- Physical error probability: 0.003.
- Configuration: `beam8_230iters` (max_rounds=10, beam_width=8,
  initial_iters=30, iters_per_round=20, num_results=1).
- Fixed 2,000 shots per basis; seeds 20260908 and 20260909.
- Score every returned prediction, as upstream does; no discarded shots.

| Basis | Shots | Observable errors | Invalid corrections | Strict-policy errors | Native-adapter decoding time |
|---|---:|---:|---:|---:|---:|
| X | 2,000 | 4 | 4 | 4 | 18.84 s |
| Z | 2,000 | 6 | 4 | 6 | 22.97 s |

Paper-style LER per syndrome round:

`(4/2000 + 6/2000) / 12 = 4.17e-4`.

Conservative approximate 95% joint interval: **[1.60e-4, 1.09e-3]**, formed by
summing the independent bases' 97.5% Wilson intervals and dividing by 12.
The archived plot's orange p=0.003 marker is approximately **3.3e-4**. It lies
inside this interval. Thus this short run is consistent with the published
plot, but only ten observed logical errors are insufficient for a precise LER
reproduction claim. It does not establish the whole curve or low-p performance.
The archived plot does not provide raw counts or reference uncertainty.

For both circuits, H, observable matrix, priors, and column ordering matched
upstream exactly (H shape 936 x 8784). On 64 identical shots per basis,
LightStim and the original upstream wrapper had **zero prediction differences**.
The other 3,872 shots were decoded only through the LightStim adapter.

The strict LightStim policy and upstream observable-only policy happened to
produce the same error count here; this is not guaranteed for other data.
Times above include adapter decoding overhead, exclude sampling and reference
comparison, and are not a reproduction of the paper's hardware timings.

## Reproduce

Follow [installation instructions](../../docs/ionq_beam_search.md), then run:

```bash
PYTHONPATH=. python benchmarks/decoding/ionq_beam_search_paper.py \
    --upstream /path/to/BeamSearchDecoder \
    --shots 2000 --p 0.003 --seed 20260908
```

The local machine's tested checkout is `/tmp/lightstim-ionq-beam-upstream`.
It is temporary and is not part of this branch. The result JSON is generated
under `benchmarks/decoding/results/` and ignored by Git. This Markdown report
keeps the short-run results reviewable without bundling upstream artifacts.

Versions: numpy 2.2.6, scipy 1.17.1, stim 1.15.0, sinter 1.15.0,
ldpc 2.4.1, stimbposd 0.1.0. Stim reproducibility also depends on version,
machine SIMD width, and sampling call pattern, not just the seed.

Circuit SHA-256:

- X: `c22326f0170a657474525acec4eb97b6585bfa9ab512bd543ce6b0f18e7e2ddd`
- Z: `690b3b4450ea9cc20bd17d69a5b27379817c705eff7c04860eb0e072010e5e8f`

Sources: [paper Fig. 2](https://arxiv.org/html/2512.07057v1),
[archived plot](https://github.com/ionq-publications/BeamSearchDecoder/blob/084a475b05fb64308103317a1ce5a0c4b0be58aa/simulation_results/beam_144_12.pdf),
[upstream scoring](https://github.com/ionq-publications/BeamSearchDecoder/blob/084a475b05fb64308103317a1ce5a0c4b0be58aa/simulation_functions.py).
