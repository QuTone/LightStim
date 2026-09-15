# H-family core integration walkthrough

## Scope and provenance

This is the first memory milestone of the H6 integration: a general H-family
patch, default concurrent unflagged extraction, H6 compatibility, and
transversal H, an H6-only unflagged encoder, and the reviewed color-code
Clifford gate contribution. It adapts Maggie
Bao's assets from [PR #98](https://github.com/QuTone/LightStim/pull/98),
reviewed at commit `9cf7df700a85300e91434633e5cd43a04a66791b`, against base
`0cb663f9cf498e0b6b5a1a5f9125a1d79aaee2e1`.

The integration branch starts from that PR head, so all seven original
contributor commits remain ancestors. Maintainer changes narrow the final
diff and strengthen validation. Deferred files remain available at the pinned
PR commit for subsequent work.

| Original asset | First milestone | Reason / follow-up |
|---|---|---|
| `H_six/code_patch.py` | Generalize in `H_code/code_patch.py`; retain thin legacy imports | HCode(n) with even n >= 6; n=6 preserves the contributor's logical convention and coordinates |
| `HSixExtractionBlock` | Alias dedicated concurrent family extraction | n+2 CNOT layers, simultaneous X/Z pipeline, native detector generation |
| Memory notebook | Adapt the original PR as `memory_H_code.ipynb`; keep old-path pointer | Compact general-n configuration, X/Z memory, detector slices, and a fault check |
| H6 tests | Preserve compatibility checks and extend to the family | Algebra, full extraction tableau, coordinates/global indices, active checks, memory faults, and scaling |
| `prep_circuits.py` | Adopt Fig. 1(d) in `HSixLogicalOpSet.encode`; defer flagged preparation | Independent X/Y/Z input bases; Fig. 5's flagged \|00> preparation has a separate acceptance contract |
| `h_six_encoded_memory.py` | Defer | Flag operations bypass the tracker; encoded X path needs correction |
| `HSixLogicalXCheckBlock` | Defer | Bell readout represents joint logical parity plus a flag, not independent logical readouts |
| `H_six/operation.py` | Adopt H in `HCodeLogicalOpSet`; specialize `HSixLogicalOpSet` for encoding | H acts on all n-4 slots; encoder is H6-only; defer other gates pending phase/global-index review |
| Generic CSS Pauli gates | Defer the added API; restore shared IR to the PR base | Support-specific logical Pauli application and uniform all-data X/Z need distinct semantics |
| Color-code Clifford gates | Retain code-specific X/Z/H/S/S_DAG with larger-distance phase patterns | Preserve signed logical action and validate across registered layouts |
| `playground/magic_h6` roadmap notebook | Defer | Mixed branch state and unfinished protocols are not a reproducible core demo |

[Browse all original assets at the reviewed commit](https://github.com/maggie-bao202/LightStim/tree/9cf7df700a85300e91434633e5cd43a04a66791b).

## Suggested reading order

1. [Family README](../../lightstim/qec_code/H_code/README.md): code algebra and what
   the memory result actually means.
2. [Patch](../../lightstim/qec_code/H_code/code_patch.py): geometry, checks, and
   canonical logical pairs.
3. [Dedicated extraction](../../lightstim/qec_code/H_code/SE_block.py):
   concurrent X/Z checks in n+2 CNOT layers, using system-global indices.
4. [MemoryExperiment](../../lightstim/protocols/memory.py): initialization,
   extraction, readout, and noise injection through the existing builder.
5. [Executed notebook](../../notebooks/Memory/memory_H_code.ipynb): inspect the
   compact memory setup and detector time boundaries.
6. [Fault audit](../../tests/test_H_code_fault_distance.py): distinguish the
   distance proof under the selected noise model from the Monte Carlo check.

The key distinction is between three objects: the stabilizer code defines
which errors are detectable in principle; a physical extraction schedule can
spread faults; a protocol's acceptance rule determines which corrupted shots
survive. A correct patch is necessary, but does not certify every circuit built
from it.

## Adopted logical-operation contribution

`HCodeLogicalOpSet` adopts the physical H layer from Maggie's H6 operation set
and validates it for the general family. H exchanges each registered X/Z
logical pair and implements H on all n-4 slots together. `HSixLogicalOpSet`
inherits it and adds the unflagged Fig. 1(d) encoder for independently chosen
logical X/Y/Z eigenstates. Legacy operation imports remain supported.
Gates use global patch indices and the builder's
unitary-block path so tracker and noise semantics are retained.

The proposed shared CSS `transversal_x/z/pauli` API is deferred. Every CSS
logical X or Z has a physical Pauli representative on its registered support,
but uniform X or Z on every data qubit need not preserve a general CSS code.
For a multi-logical code, a uniform layer can also act on several logical
slots. The first asset leaves `lightstim/ir/operation.py` identical to the
original PR base, including the existing state-preparation semantics.

`ColorCodeLogicalOpSet` owns X, Z, H, S, S_DAG and inherits CNOT. Physical
X/Z on every data qubit commute with the even-weight checks and anticommute
with the odd-weight opposite-basis logical, giving the intended logical Pauli.
Its d=3
uniform phase-gate convention matches the PR: physical S_DAG implements
logical S for the registered weight-seven logicals. For larger distances,
weight-six checks need a nonuniform S/S_DAG pattern. The pattern is solved
from the registered supports: if r_q=1 denotes S_DAG, then each check c obeys
`sum(r_q on c) = weight(c)/2 mod 2`, while the logical support L obeys
`sum(r_q on L) = (weight(L)-1)/2 mod 2`. These enforce positive stabilizer
products and `X_L -> +Y_L`, fixing the logical S versus S_DAG sign. This is
the signed-support phase construction of [Kubica and Beverland, Section II.3](https://arxiv.org/html/1410.0069#S2.SS3),
implemented algebraically so it works across the registered layouts. The
physical pattern need not be the same geometric bipartition as the paper.

Use global patch views from `system.add_patch()`, including for shifted or
second patches. With an initialized builder and such a patch:

```python
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.qec_code.color_code import ColorCode, ColorCodeLogicalOpSet

executor = LogicalExecutor(builder)
executor.register_op_set(ColorCode, ColorCodeLogicalOpSet())
executor.apply_logical_operation("transversal_s", [patch])
executor.apply_logical_operation("transversal_x", [patch])
```

Tests cover signed X/Y/Z logical action, stabilizer-group preservation,
nonzero global indices, noiseless tagging, and automatic detector generation
with gates between extraction rounds. Color-code tests cover d=3,5,7,9 in
superdense, raw, triangular, and rectangle layouts, at full code boundaries.
In-cycle middle-out gates remain a separate spacetime protocol question.

## Why the deferred assets need separate validation

### Flag inventory at PR #98 head 9cf7df7

The remote PR head was checked again on September 15, 2026 and remains the
pinned commit above. The encoder and flag assets have distinct roles:

| Asset | Physical role | Current LightStim integration |
|---|---|---|
| `prep_circuits.get_dist_circ(data)` | Fig. 1(d) unflagged encoder with fixed \|++> inputs in the PR | Adopted in `HSixLogicalOpSet.encode` with two independently chosen X/Y/Z bases |
| `prep_circuits.get_ft_init_circ(data, flags)` | Fig. 5 flagged preparation of H6 logical \|00> using two flags | Returns a raw Stim circuit with flag resets/readout; requires a preparation block and declared acceptance |
| `encoded_memory_circuit(flag_verified=True)` | Experiment driver: Fig. 5 preparation, then generic coloration SE, then readout | Uses the builder for part of the circuit, but manually appends flag R/CX/M/DETECTOR instructions, bypassing the tracker |
| `HSixLogicalXCheckBlock` | Bell-pair Clifford proxy measuring joint logical XL0*XL1, plus a flag | Has a `.circuit` block and an SE-style call site; its logical-check/flag outcomes need distinct protocol semantics |

No independent flagged Pauli stabilizer-extraction block is included in the
PR: its `HSixExtractionBlock` is an alias for generic CSS coloration. The
flagged weight-four X-stabilizer measurements in Quantinuum's experimental
QASM are a separate source, documented in the [SE review](h_code_se_review.md#what-quantinuum-actually-used).

The encoded-memory helper is not a second encoding primitive. Its `flag_verified`
option chooses whether to flag the |00> preparation; its subsequent SE was
generic coloration at the PR head. The dedicated family schedule is a later
maintainer addition. The new `HSixLogicalOpSet.encode` is independently usable
with either extraction class; ordinary `MemoryExperiment` keeps its original
product-state initialization.

Independent Stim back-propagation of the PR's Bell-check unitary confirms
that its two measured Z operators pull back to `Z_a0 * X0 X1 X2 X3 X4 X5`
and `Z_a1`. With both ancillas initially in \|0>, these are the joint logical
`XL0 * XL1` check and the flag, respectively.

The encoder flags monitor data controls 0 and 2. Starting from six data
qubits in \|0>, its structure is:

```text
R  f0 f1
H  d0 d2
CX d0 f0 ; CX d2 f1
CX d0 d1 ; CX d2 d3
CX d0 d4 ; CX d2 d5
CX d0 d5 ; CX d2 d4
CX d0 f0 ; CX d2 f1
M  f0 f1                 # accept both flag outcomes 0
```

This is a state-preparation gadget, not the arbitrary-input Fig. 1(d) encoder
or a round measuring the four H-code stabilizers. Its state-specific fault
argument must not be extended to arbitrary logical basis pairs unchanged.

The original flag encoder directly appends reset, CNOT, measurement, and
DETECTOR instructions to the builder circuit. Its flags have no
`post-select` tags and share an ancilla coordinate. Refactoring needs complete
atomic blocks and a test of the simulation pipeline's actual acceptance mask.

The existing `zero_zero` encoder is intended for Z memory. Allowing it after
X initialization does not define the corresponding X-basis encoder. Each
supported preparation/readout combination needs an independent logical-state
and fault audit.

The Bell-pair Clifford check is sensitive to `XL0 * XL1`, with an additional
flag outcome. It cannot be documented as two independent logical-X
measurements. The desired state check must become an explicit acceptance
condition when incorporated into a protocol.

With the registered weight-three logical representatives, physical
`S^tensor6` sends logical X to minus logical Y and acts as logical S-dagger
on both slots. Naming this operation logical S loses the phase information.
Logical Pauli application also does not implement arbitrary-state logical
preparation. These are reasons to test gate semantics separately.

## Next milestones and acceptance gates

### Memory experiment API boundary

`MemoryExperiment.basis` chooses a uniform physical preparation/readout basis;
`data_basis_map` overrides it per physical data qubit. Neither parameter calls
an encoder or assigns independently encoded states to logical slots. The PR's
encoded-memory helper adds explicit preparation, optionally verified by flags,
before otherwise ordinary memory rounds.

The preferred future API is one `MemoryExperiment` with an optional preparation
stage and an explicit readout contract, rather than a parallel
`EncodedMemoryExperiment` class duplicating extraction, noise, and readout.
Preparation must remain a code-specific operation; logical bases do not in
general imply a compatible transversal physical readout map. This milestone
keeps the existing memory interface and provides a runnable H6 encoded-memory
composition in the family README. The generic API extension is follow-up work.

### Keep memory SE separate from H6 state verification

- Keep the memory default unflagged, with generic coloration as an explicit
  alternative. The Fig. 1(d) encoder is a code operation; an encoded-memory
  circuit is an experiment composing that operation, SE, and readout.
- Implement the H6 joint logical-H check and its Clifford proxy in a separate
  protocol experiment. Physical H^tensor6 realizes HL0*HL1; replacing every
  controlled-H with CX measures XL0*XL1, not one of the four stabilizers.
- Retain Fig. 5's flagged |00> preparation as a separate candidate when a
  protocol needs it. Do not infer fault-tolerant preparation for all nine
  logical basis pairs from the new unflagged encoder tests.
- No flagged four-check SE or general-family flag extension is planned in
  this milestone. A future X/Z flagged SE design would need a complete-round
  fault audit, even if its individual weight-four checks use a published
  primitive. Flags could be reused; four checks do not imply four dedicated
  flag qubits.
- For later flagged protocols, use complete atomic operations and tracker
  detectors; declare logical-check rejection, flags, and output verification
  separately, and audit the actual acceptance mask and residual data errors.

### Magic-H6 Level 1

- Pin the reference circuit and declare the Clifford proxy substitutions.
- State exactly what is accepted and what counts as an output error.
- Keep input-state infidelity and physical gate noise as separate parameters.
- Validate the intended suppression against the original physical noise
  assumptions before describing the proxy as reproducing the protocol.

The original paper discusses quadratic suppression against physical gate
failure rate and quartic suppression for self-concatenation. The PR README's
proposal to use only input-state infidelity for later scaling is a change of
validation model, and needs justification. [Paper, introduction and Section I](https://arxiv.org/html/2506.14688v1).

### Level 2 and true non-Clifford validation

After Level 1 is validated, compose the [[36,4,4]] construction and extend the
fault audit to the relevant low-weight combinations. A separate true
non-Clifford path should test the actual H states and controlled-H operations
against the declared proxy. Bare memory results do not establish either claim.

## Validation record

The earlier H-family milestone passed the targeted patch, compatibility,
fault-audit, and protocol tests. Its broader non-slow suite completed with
**855 passed and 1 skipped** after
excluding `tests/test_api.py`. The API tests previously stalled in this local
environment, including during the original PR review; they were not rerun in
this milestone, so the full CI command remains unverified. The broad run used:

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/ --ignore=tests/test_api.py -m "not slow" --timeout=60 --timeout-method=thread --maxfail=1 -q
```

The full ideal extraction tableau matches a serial reference, including
arbitrary logical inputs and ancilla states. The noisy memory audit covers
n=6,8,12,20,32,64, both bases, rounds=1,2,3, and both zero and nonzero idle
noise. No undecomposed single-fault logical-only signatures were found; every
case has an independent two-fault witness. This is a distance-two result for
these tested closed memory experiments, not open-output protocol certification.

All four slow scaling tests passed with p=0.002,0.004,0.008,0.016, 400,000 shots
per point, rounds=2, seeds 98 through 101, and no idle noise:

| Code | Z-memory slope | X-memory slope |
|---|---:|---:|
| H6 | 1.9947 | 1.9653 |
| H10 | 2.0662 | 2.0583 |

These finite-sample fits measure conditional block error after all-detector
postselection, including final readout. The stronger small-p evidence is
single-fault exclusion plus an independent two-fault witness. These results
replace the earlier generic-H6 schedule's slopes, because the default circuit
has changed. The generic CSS extraction remains available as an explicit
`extraction_block_class` override.

The simplified family notebook's five code cells were executed sequentially with IPython
in the LightStim virtual environment (Python 3.10.12, Stim 1.15.0), retaining
real detector SVG and check outputs. The in-process execution avoids the sandbox's
local-socket restriction on Jupyter kernels. The notebook records the method
in metadata and selects the `lightstim` kernel for normal interactive use.


### Extraction review and expanded audit

The [dedicated SE review](h_code_se_review.md) documents the exact comparison
with generic coloring, Quantinuum's flagged experimental QASM, and the locally
derived n+2 schedule. A 720-circuit audit scans every even n from 6 to 64:
the dedicated closed-memory circuit has distance two throughout; the current
generic ordering has distance two at n=6 and distance one at every tested
n>=8. Independent propagation tests also retain a generic H8 logical-fault
counterexample and the dedicated block's unflagged weight-two output residual.
Neither result is a family-wide open-output fault-tolerance proof.

The logical-operation review passed **73 gate tests** and **77 non-slow H-family
fault tests**. The broad non-slow run passed **929 tests, with 1 skipped**, still
excluding `tests/test_api.py`. Four additional distance-one CSS edge cases were
added after that broad run's collection and passed in the 73-test gate run.
The notebook's five code cells were re-executed successfully. The earlier H6/H10
scaling results remain applicable: this review did not change their SE circuit.

### First-asset consolidation (September 15, 2026)

The latest scope keeps dedicated extraction as the default, generic coloration
as an explicit override, H6 compatibility, family-wide transversal H, and the
code-specific color-code gates. The shared CSS Pauli extension has been
withdrawn; `lightstim/ir/operation.py` matches PR base `0cb663f` exactly.

The targeted non-slow run passed **461 tests**, with **4 slow scaling tests
deselected**: H-family patch/compatibility, SE fault audits, signed logical-H
action on every slot, global indices and tracker integration, color-code
X/Z/H/S/S_DAG across distances/layouts, and the packaged protocol tests. The
prior memory scaling results still apply because the extraction circuit and
noise model were unchanged. Flagged circuits were inspected at the unchanged
PR head; no flagged implementation was added to this first asset.

### Encoder / logical-check separation (September 15, 2026)

`HSixLogicalOpSet` now specializes the family operation set with the unflagged
Fig. 1(d) encoder. Both logical input bases are explicit, with X/Y/Z denoting
the +1 eigenstate. It accepts HSixCode and HCode(6), keeps the four CNOT layers
separate for noise injection, and preserves the legacy operation import.
It does not insert Fig. 5 preparation or a logical-parity check into memory SE.

The updated targeted run passed **492 tests**, with **4 slow scaling tests
deselected**. The 31 new encoder cases cover all nine basis pairs on both
constructors, signed logical expectations and all four code stabilizers,
nonzero global indices and other-patch isolation, tracker-generated encoded
memory, noiseless tags, preparation wrappers, and invalid-request rejection.
These establish the encoding and integration behavior, not fault-tolerant
preparation or noisy encoded-memory suppression. The ordinary memory circuit
and its prior fault-distance/scaling results remain unchanged.

### Push-readiness review (September 15, 2026)

The final non-slow library/protocol run passed **996 tests**, with **1 skipped**
and **8 slow tests deselected**. It excluded the eight API endpoint tests and
the unrelated, untracked surface-postselection experiment. All **8 API tests
also passed** when run outside the local sandbox, resolving the earlier
TestClient hang without a code change. Thus the two runs cover the non-slow
CI test scope of this contribution: **1004 passed and 1 skipped**.

All five H-family memory notebook code cells executed successfully. The
previous slow H6/H10 scaling evidence remains applicable because their
circuits/noise model are unchanged. The shared IR operation file is identical
to PR base `0cb663f`; the new logical operations live under their code families.
All seven contributor commits remain in the integration branch's ancestry.

The live main branch was checked at `eb53733`: its only change since the PR
base is `requirements-dev.txt` (PR #99), disjoint from this integration.
Remote CI on the proposed merge remains the final merge gate. Magic-H6
suppression, flagged preparation, and a generic memory preparation API are
follow-up work, rather than claims or acceptance criteria of this first asset.
