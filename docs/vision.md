# LightStim: Vision & Philosophy

## The Problem

Simulating quantum error correction experiments with [Stim](https://github.com/quantumlib/Stim) is powerful but labor-intensive. A researcher who wants to answer "what is the logical error rate of this distillation protocol at distance 5?" must:

1. Manually assign qubit indices and coordinates
2. Construct the circuit gate-by-gate (resets, CNOTs, measurements)
3. Hand-annotate every DETECTOR instruction with the correct measurement record offsets
4. Hand-annotate OBSERVABLE_INCLUDE instructions
5. Verify the detector error model is valid
6. Set up sampling, decoding, and post-selection logic
7. Analyze results

Steps 3-4 alone can take days for a complex multi-patch circuit, and a single off-by-one error in a measurement record offset produces silent, hard-to-diagnose failures.

## The Solution

LightStim eliminates the manual bookkeeping by providing **the right level of abstraction** for QEC experiments:

| Layer | Abstraction | What it automates |
|-------|-------------|-------------------|
| **Code Definition** | `QECPatch` | Qubit coordinates, stabilizer definitions, logical operators |
| **Multi-Patch System** | `QECSystem` + `LogicalCouplerProtocol` | Global index space, lattice surgery corridors, qubit lifecycle |
| **Circuit Generation** | `CircuitBuilder` + `SyndromeTracker` | Gate sequences, **automatic DETECTOR/OBSERVABLE generation** via Pauli tableau tracking |
| **Noise** | `NoiseInjector` + `NoiseRule` | Post-processing noise injection with composable rule patterns |
| **Simulation** | `SimulationPipeline` | Sampling → post-selection → decoding → error rate computation |

The key innovation is **automatic detector generation**: the `SyndromeTracker` maintains a symplectic Pauli tableau that evolves with every gate. When a syndrome qubit is measured, the tracker automatically decomposes the measurement into a product of prior measurements using GF(2) linear algebra (RREF). No manual annotation needed.

## Design Principles

### 1. Physics separated from geometry

`QECPatch` strictly separates:
- **Physics**: integer qubit indices, Pauli stabilizer strings, logical operators
- **Geometry**: float coordinates for visualization and layout

This means the same physical code can be rendered in different coordinate systems, rotated, transposed, or shifted — without changing the circuit semantics.

### 2. Define-by-run

Patches can be added to a `QECSystem` dynamically. The system auto-syncs the tracker and builder when patches appear or disappear. Couplers can be registered, activated, deactivated, and re-registered — enabling sequential lattice surgery operations that reuse qubit indices.

### 3. Clean circuit, then noise

Circuits are constructed noiseless. Noise is injected as a post-processing step via `NoiseInjector`, which walks the circuit and applies `NoiseRule`s (depolarizing after gates, flip before measurement, idle noise, etc.). This separates correctness concerns from noise modeling.

### 4. One pipeline for everything

`SimulationPipeline` handles the full simulation loop with a unified interface: sampling, optional post-selection (on detectors or observables), decoding (PyMatching, BP+OSD, MWPF), and error rate computation. It supports both single-process and multi-process execution, with adaptive stopping (stop early when enough errors are collected for statistical significance).

## LightStim + AI: A New Workflow

The abstractions above create an unexpected benefit: **the gap between a physicist's protocol description and executable simulation code becomes small enough for AI to bridge**.

### The pattern we've demonstrated

In building the Steane 7-to-1 |Y⟩ distillation experiment:

```
Human writes setup.md          AI generates experiment script
(protocol logic, layout,   →   (using LightStim API:
 measurement sequence)          patches, couplers, builder,
                                tracker, pipeline)
        ↓                              ↓
AI analyzes results             Pipeline runs simulation
(scaling, thresholds,       ←   (sampling, post-selection,
 comparison tables)              decoding, error rates)
```

The human provides:
- Which code to use (unrotated surface code, distance d)
- How patches are arranged (layout formula)
- What measurements to perform (4 sequential ZZZZ on specific subsets)
- What to post-select on (Steane syndrome checks)
- What to measure (output observable LER)

The AI handles:
- Translating the protocol into LightStim API calls
- Managing qubit indices, coupler registration/activation, tracker expansion
- Setting up noise injection and simulation pipeline
- Running parameter sweeps and analyzing results

### Why this works

Traditional QEC simulation frameworks operate at the wrong abstraction level for AI assistance:
- **Too low** (raw Stim): The AI would need to manage measurement record offsets, detector decompositions, and qubit index bookkeeping — too error-prone
- **Too high** (black-box simulators): The AI can't express novel protocols that aren't pre-built

LightStim hits the sweet spot: **high enough that the AI doesn't need to think about bookkeeping, low enough that it can express arbitrary protocols**. The `SyndromeTracker` is the key enabler — it eliminates the most error-prone part of circuit construction (detector annotation) entirely.

### What this enables

For any protocol that can be described as a sequence of:
- Patch creation/initialization
- Syndrome extraction rounds
- Coupler activation/deactivation (lattice surgery)
- Mid-circuit measurements
- Final readout

...an AI with access to LightStim's API can generate a working simulation script from a natural-language or structured protocol description.

This means:
- **Rapid prototyping**: Try a new distillation protocol ([[15,1,3]], [[8,3,2]]) by describing it, not coding it
- **Parameter sweeps**: "Run this at d=3,5,7,9 with p=1e-3 to 1e-5" becomes a one-line instruction
- **Reproducibility**: The setup.md serves as both documentation and specification
- **Accessibility**: Researchers who understand QEC theory but aren't Stim experts can run simulations

### The frontier

What's demonstrated today (distillation circuits, memory experiments, lattice surgery) is just the beginning. The same pattern extends to:

- **Fault-tolerant logical gate sequences** — compile a logical circuit into a sequence of lattice surgery operations, simulate end-to-end
- **Architecture comparison** — same logical algorithm on different codes (surface, color, BB), automated by AI
- **Noise model exploration** — "what if measurement errors are 10x worse than gate errors?" — change one line
- **Real-time experiment design** — analyze simulation results, identify bottlenecks, propose circuit modifications, iterate

The vision: **a physicist describes what they want to learn, and the system handles how to simulate it**.

## Getting Started

See `docs/user_guide.md` for API reference and `eval/` for complete experiment examples.
