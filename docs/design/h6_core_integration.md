# H-family core integration walkthrough

## Scope and provenance

This is the first memory milestone of the H6 integration: a general H-family
patch, concurrent unflagged extraction, and H6 compatibility. It adapts Maggie
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
| Memory notebook | Extend as `memory_H_code.ipynb`; keep old-path pointer | H6/H8 Tanner graphs, concurrent schedule, size comparison, fault audit, and postselection |
| H6 tests | Preserve compatibility checks and extend to the family | Algebra, full extraction tableau, coordinates/global indices, active checks, memory faults, and scaling |
| `prep_circuits.py` | Defer | Useful reference encoders; need a separate preparation contract |
| `h_six_encoded_memory.py` | Defer | Flag operations bypass the tracker; encoded X path needs correction |
| `HSixLogicalXCheckBlock` | Defer | Bell readout represents joint logical parity plus a flag, not independent logical readouts |
| `H_six/operation.py` | Defer | Verify logical phases and multi-patch global indices before exposing gate APIs |
| Generic CSS and color-code gate changes | Restore base behavior | Broader API semantics belong in their own contribution |
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
   Tanner graph and detector time boundaries.
6. [Fault audit](../../tests/test_H_code_fault_distance.py): distinguish the
   distance proof under the selected noise model from the Monte Carlo check.

The key distinction is between three objects: the stabilizer code defines
which errors are detectable in principle; a physical extraction schedule can
spread faults; a protocol's acceptance rule determines which corrupted shots
survive. A correct patch is necessary, but does not certify every circuit built
from it.

## Why the deferred assets need separate validation

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

### H6-only flagged extraction and preparation

- Implement a separate H6 flag extraction block; general-family flags are
  outside the agreed scope. Keep protocol ancillas out of the HCode patch.
- Preserve the reference encoder's state and flag behavior.
- Use complete atomic operations; obtain detectors through the tracker.
- Distinguish preparation flags, syndrome rejection, and final verification.
- Verify coordinates, record offsets, tags, and the pipeline's acceptance mask.
- Audit all single faults, including accepted residual data errors, for each
  claimed output state. Compare flagged and unflagged circuits with the same
  noise model and acceptance rule.

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

## Validation record: concurrent H-family default

The targeted patch, compatibility, fault-audit, and protocol tests passed.
The broader non-slow suite completed with **855 passed and 1 skipped** after
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

The family notebook's eight code cells were executed sequentially with IPython
in the LightStim virtual environment (Python 3.10.12, Stim 1.15.0), retaining
real PNG, SVG, and sample outputs. The in-process execution avoids the sandbox's
local-socket restriction on Jupyter kernels. The notebook records the method
in metadata and selects the `lightstim` kernel for normal interactive use.
