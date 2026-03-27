# MWPF Decoder User Guide

## What is MWPF?

**Minimum-Weight Parity Factor** (MWPF) is a decoder that generalizes MWPM to hypergraphs. The algorithm (HyperBlossom) solves a primal-dual LP formulation on the full DEM, natively handling hyperedges without decomposition.

**Paper**: arXiv:2508.04969 (Yue Wu et al.)
**Package**: `pip install -U 'mwpf[stim]'`

## Relationship to Other Decoders

MWPF is a strict superset:

| Decoder | = MWPF with... |
|---------|----------------|
| **MWPM** (PyMatching) | No hyperedges (degree-2 edges only) |
| **Union-Find (UF)** | No primal-dual optimization (clusters grow until first valid solution) |
| **Hypergraph UF (HUF)** | UF on hypergraphs = `SolverSerialUnionFind` |

## decompose_errors

**Always use `decompose_errors=False` with MWPF.** This is already the default in LightStim's pipeline.

MWPF's entire advantage is native hyperedge handling. Using `decompose_errors=True` destroys hyperedge information and degrades MWPF to MWPM-equivalent performance.

## Decoder Types

Three solver classes, trading accuracy for speed:

| Class | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| `SolverSerialJointSingleHair` | Highest | Slowest | Default. Production benchmarks. |
| `SolverSerialSingleHair` | Medium | Medium | Compromise. |
| `SolverSerialUnionFind` | Lowest | Fastest | Real-time / speed-critical. |

Convenience aliases:
- `SinterMWPFDecoder` → `SolverSerialJointSingleHair`
- `SinterSingleHairDecoder` → `SolverSerialSingleHair`
- `SinterHUFDecoder` → `SolverSerialUnionFind`

## Key Parameter: cluster_node_limit (c)

**THE most important tuning knob.** Controls the maximum number of dual variables per cluster.

| `c` value | Behavior | When to use |
|-----------|----------|-------------|
| `0` | Unlimited (full HyperBlossom) | Highest accuracy, slowest. Code-capacity benchmarks. |
| `50` | Good tradeoff | **Surface code circuit-level** (default in the package) |
| `200` | More aggressive optimization | **Color codes, BB/qLDPC codes** — complex hypergraph structures need larger clusters |

**Rule of thumb**: Use `c=50` for surface/toric codes, `c=200` for color codes and qLDPC codes.

## Usage in LightStim

### Basic (surface code)
```python
from src.simulation.decoder_backend import DecoderConfig

decoder_config = DecoderConfig(
    name="mwpf",
    backend="cpu",
)
```
Uses defaults: `SolverSerialJointSingleHair`, `c=50`.

### Color code / qLDPC (larger clusters)
```python
decoder_config = DecoderConfig(
    name="mwpf",
    backend="cpu",
    params={"cluster_node_limit": 200},
)
```

### With BP pre-processing (for qLDPC codes)
```python
decoder_config = DecoderConfig(
    name="mwpf",
    backend="cpu",
    params={
        "cluster_node_limit": 50,
        "bp": True,
        "max_iter": 100,
        "bp_method": "ms",           # min-sum
        "ms_scaling_factor": 0.625,
        "bp_weight_mix_ratio": 1.0,
    },
)
```
Requires `ldpc` package. Runs BP first, then mixes BP weights with MWPF weights.

### Fast mode (HUF — Hypergraph Union-Find)
```python
decoder_config = DecoderConfig(
    name="mwpf",
    backend="cpu",
    params={
        "decoder_type": "SolverSerialUnionFind",
        "cluster_node_limit": 0,
    },
)
```

## Recommended Configurations per Code Family

| Code | `c` | BP? | Notes |
|------|-----|-----|-------|
| Rotated Surface Code | 50 | No | PyMatching is faster and equally accurate for surface codes (no hyperedges) |
| Unrotated Surface Code | 50 | No | Same as rotated |
| Toric Code | 50 | No | Same as surface |
| Color Code (6-6-6) | **200** | No | Hyperedges are critical — MWPF outperforms PyMatching significantly |
| BB Code (qLDPC) | **200** | Optional (`bp=True`) | Large `c` for complex hypergraph. BP pre-processing can help further. |
| 4D Geometric Code | 200 | Optional | Similar to BB codes |

## When to Use MWPF vs Other Decoders

| Scenario | Recommended Decoder | Why |
|----------|-------------------|-----|
| Surface code, fast | **PyMatching** | No hyperedges → MWPM is optimal and much faster |
| Color code | **MWPF (c=200)** | Hyperedges are essential for accuracy; MWPF handles them natively |
| BB / qLDPC code | **GPU BP+OSD** or **MWPF (c=200, bp=True)** | Both work; GPU BP+OSD is faster for large codes |
| Speed-critical (any code) | **MWPF HUF** or **PyMatching** | Fastest options |
| Highest accuracy (any code) | **MWPF (c=0)** | Full optimization, but can be slow |

## Performance Notes

- MWPF decode time scales roughly O(n) at low physical error rates
- At high error rates (near threshold), decode time increases significantly
- Color code d=5 circuit-level: ~15 shots/s/worker with `c=50`, expect slower with `c=200`
- Surface code d=7 circuit-level: ~1000+ shots/s/worker (but PyMatching is even faster)
- Multi-process: MWPF works correctly with `num_workers > 1` in LightStim's pipeline

## Sinter Version Warning

Use `mwpf >= 0.2.8`. Older sinter versions (< 1.15) had adaptor bugs that caused incorrect LER. Always import via `from mwpf import SinterMWPFDecoder` rather than using sinter's built-in `mw_parity_factor` decoder name.
