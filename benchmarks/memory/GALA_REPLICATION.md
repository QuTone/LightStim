# GALA `[[132,30,12]]` replication notes

## Outcome

The local decoder does **not yet quantitatively reproduce** the logical-error
curve or the five-million-shot zero-error result in arXiv:2608.07431. The
experiments below isolate the remaining discrepancy to the decoder tail rather
than the GALA code construction or ordinary converged BP decisions:

- no converged-wrong T1/T2 result appeared in the principal controlled samples;
- failures were Relay-BP non-convergences, correctly identified for escalation;
- the open-source HiGHS exact fallback did not solve those GALA tail instances
  on a useful timescale, while the paper used Gurobi.

## Reproduction setup

`run_gala_replication.py` now constructs the paper-like circuit with:

- greedy CSS coloration and sequential X/Z stabilizer extraction;
- no idle noise;
- two-qubit depolarization of strength `p` after each CNOT;
- preparation/readout depolarization represented by the equivalent basis-flip
  probability `2p/3`;
- no artificial noise after the ideal Hadamards that express direct X-basis
  preparation/readout in Stim;
- single-basis detector decoding;
- published Relay parameters (`num_sets=300`, `set_max_iter=60`, `gamma0=0.1`,
  `pre_iter=0`, `stop_nconv=1`, F32, seed 0);
- deterministic sampling and per-tier attempt counts.

The default circuit has 12 rounds, following the bundled notebook and the
earlier decoder paper's `W=d` window. A monolithic 32-round DEM is available as
an ablation, but is not equivalent to a 32-round experiment stitched from
sliding decoder windows.

## Representative results

All rates below use the paper's
`1 - (1 - p_fail) ** (1 / (k * rounds))` normalization and exact 95%
Clopper-Pearson intervals.

| Configuration | Result | Tier attempts | Per-logical-per-round LER |
|---|---:|---:|---:|
| `p=0.0015`, 12 rounds, package-default Relay alpha | 1/100 unresolved | `[100, 4]` | `2.79e-5` (`7.03e-7`–`1.56e-4`) |
| `p=0.0015`, 12 rounds, dynamic Relay alpha (`alpha=0`) | 0/100 | `[100, 4]` | point estimate 0; 95% upper `1.02e-4` |
| `p=0.0025`, 12 rounds, dynamic Relay alpha | 17/100 unresolved | `[100, 37]` | `5.17e-4` (`3.00e-4`–`8.29e-4`) |
| `p=0.0015`, monolithic 32 rounds, default Relay alpha | 4/100 unresolved | `[100, 14]` | `4.25e-5` (`1.15e-5`–`1.09e-4`) |

These small runs cannot measure the paper's lowest rates; they are diagnostics
of convergence and fallback behavior. In particular, zero errors in 100 shots
is only an upper bound, not a confirmation of a `~5e-8` point.

## Decoder investigations

### Relay convergence routing

The prior LightStim adapter used Relay-BP's Sinter wrapper, which returns a
prediction but does not expose convergence. Consequently, a chained MLE tier
was never invoked. The new adapter uses Relay's detailed result API, publishes
per-shot convergence flags and iteration counts, and the chain now escalates
only non-converged shots.

Exactly zero is a special Relay `alpha`: it enables the iteration-dependent
min-sum ramp. It was previously normalized to `None`, silently disabling the
ramp. The adapter now preserves zero. The paper reports the ramp scale but does
not report `alpha`, so the benchmark leaves it unset by default and exposes the
zero-ramp interpretation as a CLI ablation.

On seven fixed 32-round BP-hard syndromes at `p=0.0015`, 60 Relay ensembles
resolved 4/7 with the package default, 5/7 with a slower dynamic-alpha ramp,
and 5/7 with fixed `alpha=0.8`; no converged-wrong output appeared. With all
300 ensembles, the respective best result was 6/7, leaving one shot at the
18,000-iteration ceiling.

### T1 alternatives

The paper says its unpublished T1 supports memory BP, adaptive damping and
several schedules/rules, but does not give the selected circuit-level T1
parameters. On 50 identical `p=0.0025` shots:

- `ldpc` dynamic min-sum converged on 25/50;
- fixed min-sum scaling 0.625 converged on 35/50, with zero converged-wrong
  outputs, but Relay still left 8/50 unresolved;
- product-sum was stopped after exceeding five minutes;
- Relay's native memory-BP pre-pass was much faster; its conservative
  `gamma0=0.1` integrated cascade resolved 41/50 with zero silent errors;
- more aggressive memory tuning reached at most 42/50 and one setting
  (`gamma0=0.15`) introduced a converged-wrong logical result.

The runner exposes the integrated native approximation as
`--tiers native-relay`, but does not call it the paper default.

### Exact and approximate fallbacks

The current ACG-ALP + SciPy MILP exact decoder did not return on representative
12- or 32-round GALA residuals within several minutes; a nominal 60-second
HiGHS time limit overshot beyond 180 seconds. Native Highs 1.15 with a valid
BP-OSD MIP start respected its 60-second limit but retained a 64% optimality
gap. BP-OSD orders through 20 and aggressively sparsified Tesseract runs were
fast but returned the wrong logical class on the known hard residuals.
An additional unlimited full-cascade run reached the first Relay-unresolved
shot and was externally terminated with `SIGKILL` (exit 137) before HiGHS
proved optimality, most likely at the container memory limit. No incumbent was
accepted and no logical result was recorded for that shot.

This does not invalidate the MLE formulation: it shows that open-source HiGHS
is not a practical substitute for the paper's Gurobi tier on this detector
graph without a stronger decomposition/window implementation.

The implementation itself is independently checked for correctness. Its
binary log-likelihood objective and parity constraints match exhaustive search
for every reachable syndrome of a random 14-mechanism model; the LP shortcut
and full MILP agree on circuit-generated surface-code cases; dense BB-code
cases agree with a separately constructed pure MILP; and an end-to-end DEM
test checks the predicted logical observables against exhaustive enumeration.
Zero-probability mechanisms are fixed off, successful solver results are
rechecked for integrality and exact syndrome satisfaction, and solver timeout
or numerical failure is heralded rather than accepted as a correction.

The replication runner consequently defaults to the complete unlimited
`BP -> Relay-BP -> MLE` hierarchy. A positive `--mle-time-limit` is an explicit
diagnostic choice; omitting it allows rare fallback shots to run until HiGHS
proves optimality, subject to the machine's memory/resource limits.

## Commands

Paper-like diagnostic (default Relay interpretation):

```bash
MPLCONFIGDIR=/tmp/matplotlib venv/bin/python \
  benchmarks/memory/run_gala_replication.py \
  --p 0.0015 --shots 1000 --workers 4 --batch-size 5 --tiers relay \
  --output benchmarks/memory/results/gala_132_relay.json
```

Dynamic-alpha and native memory-BP ablations:

```bash
# Exactly zero activates Relay's iteration-dependent alpha ramp.
MPLCONFIGDIR=/tmp/matplotlib venv/bin/python \
  benchmarks/memory/run_gala_replication.py \
  --p 0.0015 --shots 1000 --workers 4 --batch-size 5 \
  --tiers relay --relay-alpha 0

MPLCONFIGDIR=/tmp/matplotlib venv/bin/python \
  benchmarks/memory/run_gala_replication.py \
  --p 0.0015 --shots 1000 --workers 4 --batch-size 5 \
  --tiers native-relay --relay-gamma0 0.1
```

## Reproducibility limitations in the papers

The GALA paper does not state `N_c`, the full T1 parameters, sliding-window
stitching details, or a pinned decoder/software revision. Its cited hierarchy
paper specifies 32 circuit rounds, `W=d=12` decoder windows, Relay's table of
parameters and Gurobi MLE, but likewise omits the selected T1 configuration.
Those omissions prevent an exact independent reconstruction from the papers
alone.
