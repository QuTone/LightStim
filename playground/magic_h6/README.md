# playground/magic_h6/

`H_six_roadmap_status.ipynb` — a branch-review audit for the `h6-msd` /
`feat/magic-h6-protocol` integration. It checks the maintainers' Magic-H6
roadmap (Steps 1–3) against the current branch and reproduces every claim:
patch conventions, `MemoryExperiment` baseline, `shortest_graphlike_error`
fault distance, Monte-Carlo `O(p^2)` slope fits, and the level-1 distillation
proxy.

It is an integration-status document, not a demo — it carries MC slope sweeps
and git branch/commit introspection, so it lives here rather than in
`notebooks/Memory/` (which is small per-code demos; see `memory_H_six.ipynb`
for that) or `benchmarks/`.
