# IonQ BP-guided beam search (beta)

`DecoderConfig("ionq-beam-search")` uses the original IonQ C++ implementation
through its Cython binding. It is distinct from the `tesseract` decoder.
LightStim contains only the adapter; it neither bundles nor builds upstream code,
and its default dependencies and package build remain unchanged.

## Install the optional upstream extension

Use the same Python environment for building upstream and running LightStim.
A C++20-capable compiler is required for this one-time manual build. The tested
upstream revision is `084a475b05fb64308103317a1ce5a0c4b0be58aa`.

```bash
# With the LightStim virtual environment activated:
python -m pip install numpy scipy ldpc Cython setuptools
# Choose an external checkout location (not inside the LightStim package).
git clone https://github.com/ionq-publications/BeamSearchDecoder.git /path/to/BeamSearchDecoder
cd /path/to/BeamSearchDecoder
git checkout 084a475b05fb64308103317a1ce5a0c4b0be58aa
cd decoder
python setup.py build_ext --inplace
export PYTHONPATH="/path/to/BeamSearchDecoder/decoder${PYTHONPATH:+:$PYTHONPATH}"
python -c "from beam_search_decoder import BeamSearchDecoder"
```

The `PYTHONPATH` setting must also be present in worker processes and notebook
kernels. Restart an existing kernel with that environment if necessary. On
Windows, set the equivalent environment variable with Windows path separators.
The upstream checkout-root import `decoder.beam_search_decoder` is supported too.
An unavailable or unloadable extension raises an installation hint only when
this decoder is selected; importing LightStim and using other decoders is safe.

Upstream source and artifacts retain their own CC BY-NC-SA 4.0 license:
[repository and license](https://github.com/ionq-publications/BeamSearchDecoder).
No upstream source or circuits are redistributed by this integration.

## Use

```python
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(
        "ionq-beam-search",
        params={
            "max_rounds": 10,
            "beam_width": 8,
            "num_results": 1,
            "initial_iters": 30,
            "iters_per_round": 20,
        },
        on_decode_failure="error",
    ),
    max_shots=2000,
    num_workers=2,
)
stats = pipeline.run(circuit)  # an existing Stim circuit
```

All five parameters are positive integers. These defaults are the paper's
`beam8_230iters`. `max_rounds` means **search branching rounds**, not syndrome
extraction rounds or a decoding window. No window decoding is implemented here.

| Paper configuration | max_rounds | beam_width | initial_iters | iters_per_round | num_results |
|---|---:|---:|---:|---:|---:|
| beam8_230iters | 10 | 8 | 30 | 20 | 1 |
| beam32_340iters | 10 | 32 | 40 | 30 | 1 |
| beam64_640iters | 20 | 64 | 40 | 30 | 1 |
| beam64_32res_640iters | 20 | 64 | 40 | 30 | 32 |

The adapter consumes undecomposed DEM hyperedges and uses LightStim's sparse
`dem_to_matrices(..., merge_duplicates=True)` conversion, including observable
footprints in the duplicate key. It returns corrections on that matrix's columns.
It validates the returned correction against the original syndrome, rather than
trusting the native `converge` flag of the last explored branch. Inputs and
outputs are copied across the native call. Empty DEMs and empty shot batches are
supported. Nonempty models require finite priors strictly between zero and one;
deterministic error mechanisms are rejected rather than silently clamped.

Failure handling is the existing LightStim policy:

- `error` (default): an invalid correction counts as a failure even if its
  observable prediction happens to agree with the sampled observable.
- `discard`: exclude invalid corrections and report the reduced acceptance.
- `ignore`: score the returned observable prediction regardless of validity.
  This matches the paper repository's Sinter wrapper and is the policy to use
  when comparing its LER. Do not compare a discard-conditioned LER to the paper.

The decoder also works as a stage in `DecoderConfig.chain(...)`. Ordinary
full-shot batches and existing process workers are reused; this adapter does
not promise a native batch API or the runtime measured on the paper's hardware.

## Short paper-point check

The reproducible script uses the authors' BB[[144,12,12]] X/Z memory circuits,
12 syndrome rounds, p=0.003, and `beam8_230iters`. It checks that LightStim and
upstream matrices/priors are identical for both circuits, compares 64 predictions
per basis on identical shots, and estimates LER with 2,000 fixed shots per basis.
It records invalid corrections and strict-policy errors separately.

```bash
python -m pip install stimbposd==0.1.0
# Run from LightStim's root; the script exposes the explicitly supplied checkout.
PYTHONPATH=. python benchmarks/decoding/ionq_beam_search_paper.py \
    --upstream /path/to/BeamSearchDecoder \
    --shots 2000 --p 0.003 --seed 20260908
```

Output defaults to `benchmarks/decoding/results/ionq_beam_search.json` and includes
upstream revision, library versions, circuit hashes, seeds, counts, timings,
reference agreement, and an approximate confidence interval. Increase `--shots`
for tighter uncertainty; this short check does not reproduce a full threshold
curve or the low-error-rate p=0.001 point.

The paper's plotted metric is `(X errors / X shots + Z errors / Z shots) / 12`,
not LightStim's per-shot LER and not a per-logical-qubit average. The script uses
that same approximation. It sums 97.5% Wilson bounds for the independent bases
and divides by 12 for a conservative approximate 95% joint interval. This is a
fixed-shot estimate, with no adaptive stopping on an error count.

The archived upstream plot has no accompanying numerical CSV; compare its
p=0.003 orange marker (roughly 3.3e-4 per round) as a plot-read approximation,
not an exact reference measurement with known uncertainty.

References: [paper, Fig. 2 and Table 1](https://arxiv.org/html/2512.07057v1),
[upstream simulation metric](https://github.com/ionq-publications/BeamSearchDecoder/blob/084a475b05fb64308103317a1ce5a0c4b0be58aa/simulation_functions.py),
[archived plot](https://github.com/ionq-publications/BeamSearchDecoder/blob/084a475b05fb64308103317a1ce5a0c4b0be58aa/simulation_results/beam_144_12.pdf).

The completed beta check and numerical counts are recorded in the
[short-run report](../benchmarks/decoding/ionq_beam_search_result.md).

## Tests

```bash
# Without the optional package: adapter tests pass, three native cases skip.
python -m pytest tests/test_ionq_beam_search_decoder.py -q
# With upstream/decoder on PYTHONPATH, native cases run as well.
```
