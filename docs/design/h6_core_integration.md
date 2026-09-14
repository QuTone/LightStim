# H6 core integration walkthrough

## Scope and provenance

This is the first memory milestone of the H6 integration. It adapts Maggie
Bao's assets from [PR #98](https://github.com/QuTone/LightStim/pull/98),
reviewed at commit `9cf7df700a85300e91434633e5cd43a04a66791b`, against base
`0cb663f9cf498e0b6b5a1a5f9125a1d79aaee2e1`.

The integration branch starts from that PR head, so all seven original
contributor commits remain ancestors. Maintainer changes narrow the final
diff and strengthen validation. Deferred files remain available at the pinned
PR commit for subsequent work.

| Original asset | First milestone | Reason / follow-up |
|---|---|---|
| `H_six/code_patch.py` | Keep canonical patch and parity helpers | Correct k=2 algebra; remove protocol ancillas; fix accumulated shift metadata |
| `HSixExtractionBlock` | Keep generic CSS alias | Reuse existing scheduling and detector pipeline |
| Memory notebook | Keep and extend | Add a Tanner graph, acceptance semantics, and stronger distance reasoning |
| H6 tests | Adapt | Check registered logicals, code distance, placement, all single-fault signatures, and scaling |
| `prep_circuits.py` | Defer | Useful reference encoders; need a separate preparation contract |
| `h_six_encoded_memory.py` | Defer | Flag operations bypass the tracker; encoded X path needs correction |
| `HSixLogicalXCheckBlock` | Defer | Bell readout represents joint logical parity plus a flag, not independent logical readouts |
| `H_six/operation.py` | Defer | Verify logical phases and multi-patch global indices before exposing gate APIs |
| Generic CSS and color-code gate changes | Restore base behavior | Broader API semantics belong in their own contribution |
| `playground/magic_h6` roadmap notebook | Defer | Mixed branch state and unfinished protocols are not a reproducible core demo |

[Browse all original assets at the reviewed commit](https://github.com/maggie-bao202/LightStim/tree/9cf7df700a85300e91434633e5cd43a04a66791b).

## Suggested reading order

1. [H6 README](../../lightstim/qec_code/H_six/README.md): code algebra and what
   the memory result actually means.
2. [Patch](../../lightstim/qec_code/H_six/code_patch.py): geometry, checks, and
   canonical logical pairs.
3. [Generic extraction](../../lightstim/qec_code/generic_css/SE_block.py):
   four CNOT layers for each basis, using system-global indices.
4. [MemoryExperiment](../../lightstim/protocols/memory.py): initialization,
   extraction, readout, and noise injection through the existing builder.
5. [Executed notebook](../../notebooks/Memory/memory_H_six.ipynb): inspect the
   Tanner graph and detector time boundaries.
6. [Fault audit](../../tests/test_H_six_fault_distance.py): distinguish the
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

### Encoded and flagged preparation

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

## Validation record

For the scoped core implementation, the targeted patch, fault-audit, and
protocol regression tests passed (70 tests). The two slow scaling tests also
passed with 400,000 shots per point, rounds=2, and sampler seeds 98 through 101:

| Basis | Fitted slope | Rates |
|---|---:|---|
| Z | 2.0607 | 0.002, 0.004, 0.008, 0.016 |
| X | 1.9817 | 0.002, 0.004, 0.008, 0.016 |

These finite-sample fits measure conditional block error after all-detector
postselection. The stronger small-p evidence is exclusion of single-fault
logical signatures plus an independent two-fault witness. Counts and execution
details are visible in the tests and notebook.

The broader non-slow suite completed with **744 passed and 1 skipped** after
excluding `tests/test_api.py`. Including the API tests stalled in this local
environment, as it also did during the original PR review; the full CI command
is therefore still unverified. The completed broad run used:

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/ --ignore=tests/test_api.py -m "not slow" --timeout=60 --timeout-method=thread --maxfail=1 -q
```

The notebook's seven code cells were executed sequentially with IPython in the
LightStim virtual environment (Python 3.10.12, Stim 1.15.0), retaining real PNG,
SVG, and sample outputs. A normal Jupyter kernel launch was blocked by the local
sandbox's socket restriction. The notebook records its execution method in
metadata and selects the `lightstim` kernel for normal interactive use.
