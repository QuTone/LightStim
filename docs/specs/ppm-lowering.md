# PPM lowering: design note and capability table

Status: PR-B working document (general-ppm branch).

## Layering

```
PPMRequest ──lower_ppm()──► LoweringPlan ──apply_plan()──► QECSystem registration
   (pure, no mutation)          │                                │
                                │ certificate                    │ activate / merged
                                ▼                                ▼ rounds / split
                         LoweringCertificate          PPMOutcome (protocol output)
```

- `lower_ppm(specs, request, system=..., ...)` is PURE: it classifies the
  seam (live probe for cell-adjacent pairs), routes the explicit corridor,
  chooses the construction (plain/recoloured merge, stretched wall) and the
  merged-round schedule, and returns a declarative `LoweringPlan`.  It never
  touches the system, builder, or tracker.
- `apply_plan(system, plan, name)` is the single mutation: coupler
  registration.  Activation, merged rounds, split, and corridor readout
  remain with the caller (`SequentialPPMExperiment` is one such caller; a
  logical-level compiler is the intended other — see
  `test_kernel_api_composes_with_measurement_block_engine` for the
  driver-free consumption path).
- Routing and lowering are separate responsibilities.  The route is an
  explicit input everywhere; **there is no auto-router** and no API name
  suggests one.

## The algebraic certificate

Every accepted corridor layout carries `MultiPatchLayout.verify()`'s named
oracle items on `SubsetRoute.certificate` / `LoweringPlan.certificate`:

| item | meaning |
|---|---|
| `commute` | merged check set is mutually commuting |
| `joint` | the requested product IS in the measured span |
| `no_single` | no single target logical is exposed |
| `no_subjoint` | no proper subset product is exposed |
| `logical_count` | rank change is exactly one measured DOF |
| `no_twist` | no Y check appears in the construction |
| `no_weight1_logical` | no weight-1 undetected logical |
| `no_mpp`, `no_tick_collision`, `dem_valid` | circuit-level items (layouts without stretched checks) |

`certificate.measures_exactly_the_product` (= `joint & no_single &
no_subjoint`) is the **true weight-w instrument guarantee**: measuring
`M(P1*...*Pw)` reveals ONE bit; measuring `w-1` pairwise parities reveals
more logical information and is a different quantum instrument.  Wall
(stretched-stabilizer) plans currently carry `certificate=None`: their
validation lives in the rule-table laws and end-to-end distance tests — a
known gap.

## Protocol output vs evaluation observable

Each applied PPM banks a `PPMOutcome`: the joint's measurement-record
parity before the merge and after the split (`record_parity()` over the
record-carrying tracker rows; UNMEASURED-sentinel rows are excluded from
the reconstruction basis).  `None` before the merge of an anti-commuting
request is a **legitimately random protocol output**, not an error.

Whether a protocol output enters a deterministic, decodable quantity — an
`OBSERVABLE_INCLUDE`, a closure detector, a feed-forward dependency — is
the caller's EVALUATION choice, made from input states, the record
parities, and the final measurement bases.  The driver's final readout
emits the standing logical observables; the closure detectors it emits are
the certified correlations.  Tests: `tests/test_ppm_outcomes.py`
(commuting repeat reproducible shot-by-shot; anti-commuting corner-layout
pair is a per-shot coin while every certified correlation stays silent).

## Capability table

| axis | supported today | verified by |
|---|---|---|
| patch family | standard rotated rectangular patches (`PatchSpec` adapter; colour-swap knob) | whole suite |
| live geometry | cell-adjacent classification probes the LIVE system (`patch_view`); routed corridors consume orientation-stamped specs | `test_ppm_lowering` |
| distance | d=3 across the matrix; d=5 representative | `test_zz_pair_full_distance_d5` |
| Pauli letters | X, Z (homogeneous and mixed) | rule-row tests |
| Y | **rejected explicitly** (`UnsupportedPauliError`): needs a twist/Y-wall construction this stack does not implement | `test_y_letter_rejected_explicitly` |
| target weight | 2 (adjacent + routed), 3 (T corridor) | row tests, `test_three_target_one_step_t_corridor` |
| weight ≥ 4 | **rejected explicitly** (`BentLayoutError`): needs the snake / kf-wall attach machinery not carried by this variant | `test_weight4_straight_chain_is_an_explicit_gap` |
| route shapes | zero-cell (adjacent), straight, branched/T; bends force the diagonal schedule (machinery present, no dedicated matrix test yet) | corridor tests |
| sequential composition | commuting and anti-commuting consecutive PPMs; corridor reuse between steps | `test_ppm_outcomes`, driver tests |
| engine composition | measurement-block engine round over post-PPM tracker state (shared census) | `test_kernel_api_composes_with_measurement_block_engine` |
| deformed / non-rectangular patches | not supported: the corridor path reconstructs standard rectangles from specs; adjacent classification would reject a mismatched live view | — |

## Known reproducibility note

Circuit builds are process-nondeterministic without a pinned
`PYTHONHASHSEED` (set ordering reaches detector-coordinate assignment;
physics is unaffected).  Byte-level reproduction of a build requires
recording the hash seed alongside the commit.
