# Exact MLE Core Benchmark

This benchmark compares LightStim's optimized ACG-ALP/RPC decoder with a
direct `scipy.optimize.milp` solve on identical, deterministic syndromes. It
checks parity and objective agreement before reporting timing, so a faster but
inexact result cannot be recorded as a speedup.

```bash
PYTHONPATH=. venv/bin/python benchmarks/mle/run_mle.py
```

The default presets cover a rotated surface-code DEM and the denser BB
`[[72,12,6]]` DEM. Use `--quick` for a two-shot smoke test, `--shots N` to
override preset counts, or `--output path.json` to archive the JSON report.
Timing depends on the host and SciPy's vendored HiGHS version; the report
includes Python, platform, NumPy, SciPy, and Stim versions for that reason.

This harness intentionally does not install or compare highspy, SCIP, or
CP-SAT. Those packages substantially increase environment complexity and are
not runtime dependencies of the decoder.
