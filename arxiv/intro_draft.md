# Intro Draft (Polished)

---

## §1 Introduction

Quantum computing is entering the fault-tolerant era, where quantum error correction (QEC) is
integrated to protect fragile quantum states from physical noise
[knill1996threshold, gottesman2013fault, preskill2018quantum].
In recent years, multiple hardware platforms---including superconducting circuits
[acharya2024quantumerrorcorrectionsurface], trapped ions
[pogorelov2025experimental, postler2024demonstration, reichardt2024demonstration],
and neutral-atom arrays [bluvstein2024logical]---have demonstrated quantum memory with
multi-round error correction, establishing that fault-tolerant operation is within reach.
Building on these successes, the community is increasingly turning toward the next frontier:
implementing logical operation primitives [bluvstein2024logical, postler2024demonstration,
reichardt2024demonstration] and composing them into larger-scale logical circuits for
practical quantum algorithms [gidney2021factor, litinski2019game, bluvstein2025architectural].

---

This progression from memory to computation brings an explosion of QEC protocols: diverse
code families (surface codes [fowler2012surface], color codes [bombin2006topological],
qLDPC codes [bravyi2024high]), multiple computing paradigms for realizing the same logical
operation (transversal gates [gottesman1999demonstrating], lattice surgery
[horsman2012surface, litinski2019game]), and composite logical circuits such as
teleportation and magic state distillation [litinski2019magic, gidney2024magic].
Evaluating the correctness and performance of these protocols---and conducting fair
comparisons across them---is critical for guiding both near-term hardware experiments and
long-term architectural decisions.

---

Currently, the gold standard for such evaluation is circuit-level noisy simulation using
Stim [gidney2021stim], with the *logical error rate* (LER) as the end-to-end metric
[dennis2002topological, fowler2012surface, acharya2024quantumerrorcorrectionsurface].
This simulation pipeline imposes a **dual burden** of compilation on the protocol
(Fig. 1).
First, the QEC protocol must be *translated* into physical operations---gates, measurements,
and resets---with errors inserted after each operation according to a specified noise model.
Second, a *Detector Error Model* (DEM) must be constructed: a description of which
physical errors affect which syndrome outcomes, which the decoder uses to infer and correct
errors.
The DEM is the necessary input to the decoder; without it, LER cannot be computed.

---

The DEM is built from two types of objects.
A **detector** is a set of measurement records in the physical circuit whose total parity is
deterministic in the absence of noise---it is a parity check that must evaluate to 0 in a
perfect execution.
A **logical observable** is a set of measurement records whose parity encodes the logical
eigenvalue of a specific logical qubit at the end of the circuit.
In current practice, both are manually annotated by the developer at the appropriate
locations in the Stim circuit, as shown in Fig. [X].
As we move from memory to logical operations and circuits, however, manual DEM construction
becomes infeasible due to three compounding challenges:

---

**Challenge 1: Diverse computing schemes.**
Different logical operation paradigms require fundamentally different DEM structure.
In lattice surgery, boundaries between code patches are dynamically *merged* and *split*,
causing detectors to span measurement records across patch boundaries in ways that depend
continuously on code geometry and protocol parameters---there is no fixed template to
hard-code.
In transversal gates, fault-tolerant decoding requires *correlated* detectors spanning
measurement records across multiple code blocks simultaneously [XXX].
A static, protocol-specific annotation is therefore not reusable across paradigms.

**Challenge 2: Explosion of state combinations.**
Unlike memory experiments that focus on a single code block in a fixed basis, logical
operations apply to multiple logical qubits with varying initial states and measurement
bases, each combination yielding a different logical observable structure.
The number of distinct configurations grows exponentially in the number of logical qubits,
making any hard-coded approach untenable.

**Challenge 3: Evolving logical information.**
As logical operations execute, the logical information evolves through transversal gates,
patch merges and splits during lattice surgery, and Pauli frame corrections from
feed-forward.
The DEM must track these transformations throughout the circuit, because the detectors and
logical observables at the end of the protocol depend on the entire history of operations.

---

These complexities make QEC protocol implementation a painstaking process.
Today, the DEM for each protocol is constructed manually: researchers derive by hand which
measurement parities form valid detectors and how logical observables propagate, then
hard-code this logic into protocol-specific scripts [gidney2021factor, gidney2023cleaner,
litinski2019game].
This is analogous to the era before compilers, where every program was written in assembly
for a specific target machine: each new protocol requires weeks of low-level bookkeeping
unrelated to the protocol's actual design, implementations are one-off scripts tied to
specific code geometries and parameter sets, and fair comparison across protocols is nearly
impossible when each lives in a separate codebase with different noise conventions.
The vast majority of publicly available implementations are accordingly restricted to
memory experiments---the simplest tier of QEC evaluation---leaving the more complex and
architecturally relevant protocols without standardized benchmarks.

---

In this work, we present **AutoDEM**, a framework that *fully automates* DEM compilation
across the full range of QEC protocols, from memory experiments to complex logical circuits.

The key insight is that every syndrome measurement in a stabilizer circuit can be associated
with a Pauli operator, and a set of measurement records forms a valid detector *if and only
if* the corresponding Pauli operators multiply to a product of stabilizers in the current
stabilizer group.
Tracking how the stabilizer group evolves through each Clifford operation---via standard
tableau mechanics---therefore determines the complete DEM automatically.
Because Clifford evolution is protocol-agnostic, the same algorithm handles memory
experiments, transversal gates, lattice surgery, state injection, and their arbitrary
compositions, without any protocol-specific annotation from the user.
The computation reduces entirely to classical linear algebra over GF(2).

Just as automatic differentiation in machine learning frees practitioners from manually
deriving gradients for each new model architecture [XXX], AutoDEM's stabilizer tracking
frees QEC researchers from manually deriving the DEM for each new protocol---once the
circuit is specified, the DEM follows automatically.

---

In summary, this paper makes the following contributions:

- **Automated DEM compilation.** We present a protocol-agnostic algorithm based on
  stabilizer evolution tracking that automatically derives all detectors and logical
  observables for general QEC protocols, eliminating the most labor-intensive and
  error-prone step in QEC simulation.

- **Unified evaluation framework.** AutoDEM provides a modular framework spanning code
  definition, circuit construction, configurable noise injection, and decoding, enabling
  systematic and fair comparison across codes, operations, and computing paradigms.

- **Comprehensive benchmark suite.** We demonstrate the broadest collection of
  automatically verified QEC experiments to date, from standard memory benchmarks to
  complex distillation circuits, all generated from a unified codebase.

- **New protocol enablement.** We show that AutoDEM's automation enables rapid prototyping
  of novel protocols (surface-PQRM lattice surgery) that would be prohibitively difficult
  to implement and verify manually.
