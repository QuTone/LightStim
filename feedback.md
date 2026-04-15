# Paper Reviewer Feedback — AutoDEM (ASPLOS)

Internal review. Fix these before submission. Organized by section, then prioritized.

---

## Section-by-Section Analysis

---

### § Abstract

**A1.** Grammar error: "sets AutoDEM as a systematic foundation" → **"set AutoDEM"** (subject is "capabilities", plural).

**A2.** "demonstrate a new cross-code lattice surgery design" → **"demonstrates"** (verb agreement).

**A3.** The abstract never quantifies the scale of the problem. A one-phrase anchor helps: e.g., "distillation circuits require hundreds of detectors spanning multiple code patches" gives non-experts a sense of why manual annotation is infeasible.

---

### § Introduction

**I1. [Stakes missing]** The three challenges are listed but their *consequences* are never stated. A reviewer needs to feel: "because of this bottleneck, the community currently cannot do X." Right now the cost of inaction is only implied.
> Fix: Add a paragraph after the three challenges — because no general DEM tool exists, Categories B/C/D have zero circuit-level evaluation in open-source tools; all architectural decisions rely on theory bounds or isolated hardcoded data points.

**I2. [Table 1 not activated]** The paper's most persuasive evidence (Categories B/C/D have no prior implementations) is mentioned in one sentence and never reinforced. Most readers skim; this needs to be explicitly stated in prose.

**I3. [AutoDiff analogy abandoned too quickly]** The analogy is introduced in one sentence. The parallel should be drawn explicitly: AutoDiff removed gradient derivation as the complexity ceiling on model architecture; AutoDEM removes DEM complexity as the ceiling on evaluable protocol complexity.

**I4. [Contribution list is a feature list, not an impact list]** All four bullets describe what was built, not what this enables. At least one should be an impact statement, e.g.: "For the first time, we show that gate-specific LER exceeds the memory baseline by up to 11.6×—a gap inaccessible from memory simulation alone, with direct implications for FTQC architecture design."

**I5. [Table 1 credibility risk]** Bell Teleportation row says "Not available" but FM flagged a potential prior arxiv implementation (2504.05611). If a reviewer knows this paper, the claim is undermined. Must investigate and either cite it or state explicitly how AutoDEM differs.

---

### § Background

**B1. [Fig. DEM-BG never referenced in text]** The figure exists but is never cited. Either reference it ("as illustrated in Fig. 2...") or remove it.

**B2. [Transition to Sec. 5 is abrupt]** Background explains what a DEM is, but doesn't bridge to *why it's hard to automate*. The reader has to make that logical leap themselves. One concluding sentence like: "Constructing this DEM automatically for arbitrary protocols requires tracking how logical information evolves throughout the circuit—the core challenge that motivates Sec. 5."

**B3. ["Deterministic parity" introduced casually]** The concept is foundational to the entire paper but stated informally. The reasoning — that noiseless stabilizer repetition must yield identical outcomes, giving a deterministic XOR — should be stated as a logical argument, not just as an observation about consecutive measurements.

**B4. [Update rules lack "why"]** Rule 1 (conjugation) and Rule 2 (measurement) are stated as facts without intuition. Readers without QEC background won't understand why conjugation is the right update. One phrase of physical intuition per rule helps accessibility.

**B5. [Decoder acronyms in Noisy Simulation paragraph]** PyMatching, BPOSD, MWPF are mentioned without any explanation. A systems reviewer won't know these. They're explained in Setup — but this creates a forward reference problem. Either briefly gloss them here or move the mention to Setup.

---

### § Sec. 5 — Pauli Tracker (Tech1)

**T1. [Complexity analysis absent]** The tracker operates on an n×m Pauli tableau with GF(2) RREF operations. What is the time complexity per measurement? Per SE round? For ASPLOS, this is a basic question about whether the approach scales. At minimum, state the complexity class and give one concrete example (e.g., "for the [[144,12,12]] BB code, the tableau has X rows and Y columns; each measurement requires O(?) operations").

**T2. [Protocol-agnosticism claim needs grounding]** Sec. 5.1 states "this process is entirely protocol-agnostic." This is a key claim. A skeptical reviewer will ask: under what conditions does this break? What assumptions does the tracker require (Clifford circuits? Stabilizer codes only? No mid-circuit state resets on data qubits?)? These boundaries should be stated.

**T3. [WriteBack framed as consequence, not design choice]** The locality constraint enforced by WriteBack is crucial for decoder performance, but it reads like a side effect: "without this refresh, a syndrome record from an early round could be referenced..." This is actually a deliberate design decision that distinguishes AutoDEM from purely algebraic approaches. Frame it as a design choice with explicit motivation.

**T4. [`buf` data structure underdefined in Algorithm 1]** The record buffer `buf[t]` is described in prose but its exact structure and role in `RREF_Solve` is unclear. A reader trying to implement the algorithm from the pseudocode would be confused about what `buf[i']` contains and how it flows into the detector emission.

**T5. [Data Readout section (5.3) is underspecified]** Sec. 5.3 is notably thinner than 5.2. Edge cases: what happens when readout Paulis don't span all remaining stabilizers? What if a logical row can't be fully decomposed into single-qubit readout Paulis? These cases arise in practice (e.g., partial readout in mid-circuit recycling) and the current description doesn't address them.

**T6. [Appendix pointer missing in Sec. 5]** The motivating examples were moved to Appendix. Sec. 5 should have a pointer at the start: "Readers may find Appendix A useful as a concrete worked example before reading this section."

---

### § Sec. 6 — Pipeline (Tech2)

**P1. [ActivateCoupler is a black box]** This is the most technically novel and complex of the five atomic operations — handling the discontinuity in stabilizer sets at lattice surgery boundaries is the hard part. The current description ("AutoDEM handles this automatically via the QEC System") is effectively "magic happens here." At minimum, describe what the tracker does when the coupler is activated: new stabilizers enter the tableau as anti-commuting rows, and the tracker transitions from Case A to Case B as the patch merge stabilizes.

**P2. ["Define-by-run" never formally defined]** The term appears in Sec. 6.1 but is not explained. What does it mean operationally? How does the system auto-sync Tracker and Builder when patches are added mid-circuit? This is presented as a feature but without enough explanation to evaluate.

**P3. [Noise injection mechanism underspecified]** "Automatically injects noise after the clean circuit is constructed" — what does "after" mean precisely? Is it a post-processing pass over circuit instructions? How does it distinguish which operations to inject what noise into for circuit-level vs. phenomenological models? This is crucial for reproducibility.

**P4. ["Fair comparison under identical noise conditions" needs support]** Different code families have different gate sets and qubit counts. How does AutoDEM ensure noise parameters are truly comparable across, say, surface code and BB code? This is a strong reproducibility claim that needs a sentence of justification.

**P5. [No mention of usability / API]** ASPLOS systems papers typically show that the system is *usable*. How many lines of code does it take to implement a new protocol? How does a user specify a QEC patch? A brief code snippet or LOC comparison (e.g., "implementing Bell teleportation required X lines vs. Y lines of manual Stim annotation") would dramatically strengthen the systems contribution claim.

---

### § Sec. 7 — Experiment Setup

**S1. ["Fragmented, code-specific scripts" claim needs evidence]** The claim that prior evaluation relied on "fragmented, code-specific scripts" is asserted without evidence. Table 1 is the evidence — draw the connection explicitly. Or: cite specific examples of how prior papers had to implement their own one-off evaluation scripts.

**S2. [Why PQRM supports transversal T is unexplained]** PQRM codes "natively support transversal non-Clifford gates" — but why? What mathematical structure enables this? One sentence of explanation prevents this from feeling like a black box.

**S3. [6-tick joint SE circuit for CrossLS deserves more explanation]** This is arguably a novel hardware-level design contribution. How was it derived? Was correctness verified formally (e.g., does it correctly measure the intended stabilizers)? One more sentence of justification would help.

**S4. [CrossLS vs. magic state cultivation: space overhead claim is unquantified]** "CrossLS is highly space-efficient" compared to cultivation — by how much? The LER comparison is there but the space overhead comparison is purely qualitative. Even an order-of-magnitude estimate would be useful.

**S5. [Decoder hyperparameter choices unexplained]** `bp_iteration=1000, osd_order=10`, `cluster=50` — why these values? Were they tuned? Is there a sensitivity analysis? Reviewers may ask whether different decoder settings would change the conclusions.

---

### § Sec. 8 — Evaluation

**E1. ["LER_rotated > LER_unrotated > LER_toric" explanation is imprecise]** The explanation given — "more physical qubits provide more syndrome information" — is not quite right. More qubits also means more error locations. The actual reason at fixed d is that the syndrome density (ratio of stabilizers to data qubits) and the code geometry differ. The explanation should be more precise.

**E2. ["Circuits exactly match but LER agrees within 15%" tension unresolved]** Sec. 8.1 claims "we verify that AutoDEM-generated circuits *exactly* match Stim's built-in circuit in detector count, logical observable annotations, and injected noise" — but then says "simulated LER agrees within 15%." If the circuits are identical, why isn't the LER match tighter? This apparent contradiction needs one sentence of explanation (e.g., different random seeds for Monte Carlo sampling, different shot counts, statistical variance).

**E3. [BB code "additional detectors" finding is underexplored]** This is an interesting and somewhat surprising result — AutoDEM discovers detectors that are absent in hand-annotated circuits due to linearly dependent stabilizers. But it's mentioned in one sentence. How many additional detectors? Do they actually improve decoding performance? This deserves a quantified result.

**E4. [S_TG overhead explanation lacks reference]** FM's TODO ("can you provide references that confirm these results?") is still outstanding. If the 11.6× overhead for S_TG is AutoDEM-original, say so. If it should match prior work, add the reference.

**E5. [State injection "linear with p, independent of d" stated without explanation]** The result is stated but the reason is obvious once you say it: injection error is proportional to p, and the injected state is either accepted or rejected — d rounds of SE don't suppress this because the injection happens before the QEC kicks in. Adding one explanatory sentence turns a result into an insight.

**E6. [Memory baseline comparison for logical circuits: depth-normalization?]** The overheads (3.0×, 9.4×, 11.5× above memory baseline) are at the same p and d — but the circuits have different numbers of SE rounds. A reviewer may ask: shouldn't you normalize by circuit depth (total number of SE rounds) to get a fair comparison? Needs one sentence acknowledging this or justifying the current comparison.

**E7. [CrossLS "2–3 orders of magnitude below PER" lacks anchor]** This claim needs a specific parameter point: "at d_surf=7, p=10^{-3}, the Z-LER for PQRM(1,2,4) is X, which is Y orders of magnitude below the PER."

**E8. [CrossLS post-selection mitigation unquantified]** "X-stabilizer post-selection partially mitigates this" — by how much? What is the post-selection rate, and what is the LER improvement? Even a rough number would make this concrete.

**E9. [Distillation section lacks concrete numbers]** Bell tele section has specific numbers everywhere. Distillation only says "output LER rises above 7p³." Add at least one quantification: "at p=5×10⁻⁴, d=5, the output LER exceeds the theoretical 7p³_in by X×, showing that circuit noise dominates at this operating point."

---

### § Conclusion

**C1. [No quantitative summary]** The conclusion is entirely qualitative. A good conclusion reminds the reader of the strongest numbers: what was the most surprising finding? "Gate-specific overhead exceeds memory by up to 11.6× for S_TG" or "CrossLS suppresses LER 2–3 orders below PER" — something concrete.

**C2. ["Long limited rigorous QEC evaluation" overclaims]** The intro never establishes "long" — it's a new enough problem that it hasn't been "long" limiting anything. Be more precise: "has limited evaluation beyond basic memory experiments."

**C3. [Hardware-calibrated noise: already partially done?]** XZ-biased noise is already supported. The "future work" description may be understating what's already implemented.

---

## CRITICAL (Must Fix Before Submission)

| ID | Issue | Location |
|----|-------|----------|
| **C-1** | Related Work section completely missing | `main.tex` |
| **C-2** | All `\todo{FM: ...}` comments must be removed | Figs 7, 8, 9; Table 1 |
| **C-3** | Table 1 Bell Tele "Not available" claim — verify vs. arxiv 2504.05611 | Table 1 |
| **C-4** | `fig: DEM-BG` not referenced in text | §Background |
| **C-5** | Abstract grammar: "sets" → "set", "demonstrate" → "demonstrates" | Abstract |

---

## MAJOR (Directly Affects Score)

### Impact & Framing
| ID | Issue | Location |
|----|-------|----------|
| **M-1** | Intro missing "stakes" — consequences of no AutoDEM never stated | §Intro |
| **M-2** | Table 1 not activated in prose — Category B/C/D gap needs explicit call-out | §Intro |
| **M-3** | AutoDiff analogy underdeveloped | §Intro |
| **M-4** | Contribution list is features, not impact | §Intro |

### Technical Rigor
| ID | Issue | Location |
|----|-------|----------|
| **M-5** | Framework performance/overhead completely uncharacterized (compile time vs. circuit size) | §Tech or §Eval |
| **M-6** | Complexity analysis of tracker algorithm absent | §Tech1 |
| **M-7** | "Circuits exactly match but LER agrees within 15%" tension unresolved | §Eval 8.1 |
| **M-8** | BB code "additional detectors" finding is a one-liner — needs quantification | §Eval 8.1 |
| **M-9** | ActivateCoupler is a black box — the hardest atomic op gets the least explanation | §Tech2 |
| **M-10** | "Protocol-agnostic" claim needs explicit scope boundaries | §Tech1 |

### Positioning
| ID | Issue | Location |
|----|-------|----------|
| **M-11** | Key Insight §6.1 antagonizes Litinski/Gidney — rephrase to not call their work "misleading" | §Eval 8.2 |
| **M-12** | CrossLS contribution framing ambiguous (new protocol vs. framework demo) | §Setup, §Eval |
| **M-13** | CrossLS vs. cultivation: space overhead claim unquantified | §Setup |

---

## MINOR (Polish, Lower Risk)

| ID | Issue | Location |
|----|-------|----------|
| **m-1** | Distillation section lacks concrete numbers (quantify "rises above 7p³") | §Eval 8.3 |
| **m-2** | State injection: "1.2× PS improvement" needs parameter specification (which d, p) | §Eval 8.2 |
| **m-3** | State injection result "linear with p, independent of d" needs one-sentence explanation | §Eval 8.2 |
| **m-4** | Memory baseline comparison for circuits: acknowledge depth-normalization question | §Eval 8.3 |
| **m-5** | CrossLS "2–3 orders below PER" needs specific anchor (which d, p) | §Eval 8.4 |
| **m-6** | CrossLS post-selection effect unquantified | §Eval 8.4 |
| **m-7** | S_TG overhead reference missing (FM TODO still open) | §Eval 8.2 |
| **m-8** | Background→Tech transition: add bridge sentence | §Background |
| **m-9** | Noise injection mechanism underspecified for reproducibility | §Tech2 |
| **m-10** | No API/usability demonstration (LOC comparison, code snippet) | §Tech2 |
| **m-11** | "Fragmented scripts" claim needs evidence or example citation | §Setup |
| **m-12** | Why PQRM supports transversal T: add one-sentence explanation | §Setup |
| **m-13** | Decoder hyperparameter choices unexplained | §Setup |
| **m-14** | Appendix pointer missing in Sec. 5 opening | §Tech1 |
| **m-15** | `buf` data structure in Algorithm 1 underdefined | §Tech1 |
| **m-16** | "LER_rotated > LER_unrotated > LER_toric" explanation imprecise | §Eval 8.1 |
| **m-17** | Conclusion has no quantitative summary | §Conclusion |
| **m-18** | "long limited" overclaims in Conclusion | §Conclusion |
| **m-19** | WriteBack framed as side effect, not design choice | §Tech1 |
| **m-20** | "Define-by-run" term never formally defined | §Tech2 |

---

## Master Priority Order

```
PHASE 1 — Pre-submission blockers (do first, regardless of anything else)
  C-1   Write Related Work
  C-2   Remove all \todo{FM:} comments + fix font sizes
  C-3   Verify Table 1 Bell Tele prior work claim
  C-5   Fix Abstract grammar

PHASE 2 — Impact/framing (highest ROI for reviewer perception)
  M-1   Add "stakes" paragraph to Intro
  M-2   Activate Table 1 in Intro prose
  M-3   Extend AutoDiff analogy
  M-4   Rewrite one contribution bullet as impact statement
  M-11  Rephrase Key Insight §6.1 (Litinski/Gidney)
  M-12  Clarify CrossLS framing

PHASE 3 — Technical rigor (expert reviewers will push here)
  M-5   Add framework overhead table (compile time)
  M-6   Add complexity analysis for tracker
  M-7   Resolve "exact match but 15% LER gap" in Eval 8.1
  M-8   Quantify BB code additional detectors
  M-9   Explain ActivateCoupler mechanism
  M-10  State protocol-agnosticism scope
  C-4   Reference fig: DEM-BG in text

PHASE 4 — Quantitative completeness (solidifies Eval)
  m-1   Add number to distillation section
  m-2   Specify d,p for post-selection 1.2x number
  m-5   Add anchor to CrossLS "2–3 orders" claim
  m-6   Quantify CrossLS post-selection effect
  m-7   Resolve S_TG reference (FM TODO)
  m-4   Acknowledge depth-normalization for circuit overhead comparison

PHASE 5 — Polish (do last, time permitting)
  m-3, m-8 through m-20
```
