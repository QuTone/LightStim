# Reproducing the Surface Code Handbook figures on this branch

This branch (`unrotated-xz-joint-surgery`) contains the notebook-based pipelines needed to
reproduce the numerical figures of *"Surface Code Handbook: From Theory to Practice"*.
Files ported from other branches: `benchmarks/memory/plot_memory{,_vs_distance}.py` and the
two effective-distance notebooks in `notebooks/Memory/`.  The standalone e9/biased sweep
pipelines (Fig 16/17 and 19/20) are not part of this branch — see the Surface-code-Handbook
repository linked below.

## Setup

Python ≥ 3.10, then:

```bash
pip install numpy "stim>=1.15.0" "sinter>=1.15.0" "PyMatching>=2.0.0" pandas scipy matplotlib jupyter stimbposd
pip install -e .          # or: export PYTHONPATH=$(pwd)
```

Fig 24's sweep uses the GPU BP+OSD decoder (`cudaq_qec`, NVIDIA GPU); a CPU fallback is
`DecoderConfig("bposd", backend="cpu")` (needs `stimbposd`).

**Notebook kernel:** execute notebooks with the kernel of the environment above, e.g.
`jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=<your-kernel> <nb>`.
Without the flag, Jupyter uses each notebook's stored kernelspec, which may resolve to a
different Python on your machine.

## Figure → command

| Figure | Command (from repo root) |
|---|---|
| Fig 9 | `PYTHONPATH=. python paper_artifact/memory/run_all.py --figure 4` then `PYTHONPATH=. python paper_artifact/memory/plot_fig4.py` (plot-only works from `precomputed/`) |
| Fig 18 | `jupyter nbconvert --to notebook --execute --inplace notebooks/Memory/unrotated_effective_distance_convergence_{X,Z}_basis.ipynb` (run from `notebooks/Memory/`) |
| Fig 22 | execute `notebooks/LogicalOps/logical_CNOT_LS_{rotated,unrotated}.ipynb` (from `notebooks/LogicalOps/`) |
| Fig 24 | execute `notebooks/LogicalOps/logical_CNOT_trans_{rotated,unrotated}.ipynb` (GPU) |
| Fig 26 | execute `notebooks/LogicalOps/logical_H_S_unrotated.ipynb` and `logical_H_rotated.ipynb` |

Full parameter tables, published data, and reference plots:
https://github.com/John-YuehanZhang/Surface-code-Handbook

## Scale warnings

- Fig 18 targets 500 errors down to p = 2×10⁻⁵ (~1.5 h per notebook at 60 workers).
- Fig 22/24/26 notebooks take hours at their stored settings.
- Reproduction is statistical (PID-dependent worker seeds), not bit-exact.

## Verification record (2026-07-04)

Every pipeline on this branch was executed end-to-end at reduced scale
(fewer p-points / lower shot caps; identical code paths):
Fig 9 (`--quick`, full), Fig 16/17 (18-cell sweep + both plots + fit),
Fig 18 (both notebooks), Fig 19/20 (one biased cell + fit/plot/summary),
Fig 22 (both notebooks), Fig 24 (both notebooks, `nv-qldpc-decoder` on an H100),
Fig 26 (both notebooks incl. the `bposd` sanity cell). All passed.
