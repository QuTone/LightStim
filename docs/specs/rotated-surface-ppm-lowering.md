# Rotated-surface PPM lowering

Status: initial supported integration.

## Layering

The code-family backend lives under
`lightstim/qec_code/surface_code/rotated/ppm/`. The runnable experiment lives
in `lightstim/protocols/rotated_surface_ppm.py`. The backend never imports the
protocol layer.

```
RotatedSurfacePPMRequest ──lower_ppm()──► RotatedSurfacePPMPlan ──apply_ppm_plan()──► QECSystem registration
   (pure, no mutation)          │                                │
                                │ certificate                    │ activate / merged
                                ▼                                ▼ rounds / split
                         RotatedSurfacePPMCertificate          PPMOutcome (protocol output)
```

- `lower_ppm(placements, request, system=..., ...)` is PURE: it classifies the
  seam (live probe for cell-adjacent pairs), constructs the explicit corridor,
  chooses the construction (plain/recoloured merge, stretched wall) and the
  merged-round schedule, and returns a declarative `RotatedSurfacePPMPlan`.  It never
  touches the system, builder, or tracker.
- `apply_ppm_plan(system, plan, name)` is the single mutation: coupler
  registration. Couplers are define-by-run resources and must be registered
  after the baseline has established the logical patches, immediately before
  the corresponding PPM. Activation, merged rounds, split, and corridor
  readout remain with the caller (`RotatedSurfacePPMExperiment` is one such
  caller; a future logical compiler backend can be another; see
  `test_kernel_api_composes_with_measurement_block_engine` for the
  driver-free consumption path).
- Routing and lowering are separate responsibilities. The route is an
  explicit input everywhere; **there is no auto-router** and no API name
  suggests one.
- These request and plan types are deliberately named `RotatedSurface...`.
  They contain code-family details such as patch orientation, coarse-grid
  cells, seam construction, and extraction schedule, so they are not a
  backend-neutral logical PPM IR.

## The algebraic certificate

Every accepted corridor layout carries `MultiPatchLayout.verify()`'s named
oracle items on `SubsetRoute.certificate` / `RotatedSurfacePPMPlan.certificate`:

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
more logical information and is a different quantum instrument. Wall
(stretched-stabilizer) plans currently carry `certificate=None`: their
geometry is validated by the rule-table laws. The experiment driver does not
yet execute wall plans because their rounds require a mixed-measurement
lifecycle contract.

## Protocol output vs evaluation observable

Each applied PPM banks a `PPMOutcome`: the joint's measurement-record
parity before the merge and after the split (`record_parity()` over the
record-carrying tracker rows; UNMEASURED-sentinel rows are excluded from
the reconstruction basis).

Whether a protocol output enters a deterministic, decodable quantity — an
`OBSERVABLE_INCLUDE`, a closure detector, a feed-forward dependency — is
the caller's EVALUATION choice, made from input states, the record
parities, and the final measurement bases. The driver's final readout
emits the standing logical observables; the closure detectors it emits are
the certified correlations. The initial driver accepts commuting sequences
only. Anti-commuting sequences are rejected explicitly until the tracker has
a complete absorbed-relation retirement/liveness contract. Tests:
`tests/test_rotated_surface_ppm_outcomes.py`.

## Capability table

| axis | supported today | verified by |
|---|---|---|
| patch family | standard rotated rectangular patches (`RotatedSurfacePatchPlacement` adapter; colour-swap knob) | whole suite |
| live geometry | cell-adjacent classification probes the LIVE system (`patch_view`); routed corridors consume orientation-stamped placements | `test_rotated_surface_ppm_lowering` |
| distance | d=3 across the matrix; d=5 representative | `test_zz_pair_full_distance_d5` |
| Pauli letters | X and Z; adjacent mixed-Pauli rows are verified, while routed shape tests currently cover homogeneous Z products | rule-row and corridor tests |
| Y | **rejected explicitly** (`UnsupportedPauliError`): needs a twist/Y-wall construction this stack does not implement | `test_y_letter_rejected_explicitly` |
| target weight | verified for weight 2 (adjacent and routed) and one homogeneous-Z weight-3 T corridor | row tests, `test_three_target_certificate_pins_true_multibody_instrument` |
| weight 4+ | not generally supported; the tracked straight-chain weight-4 fixture is rejected because the required attach machinery is absent | `test_weight4_straight_chain_is_an_explicit_gap` |
| route shapes | homogeneous-Z zero-cell, straight (including a 3-cell route), bent/L (diagonal schedule), and branched/T | corridor tests, `test_longer_straight_route_full_distance`, `test_bent_route_forces_diagonal_and_full_distance` |
| adjacent seams | rows 1/4 execute as plain/recoloured merges; rows 2/3 lower to wall plans but experiment execution is rejected | rule-row tests |
| sequential composition | commuting consecutive PPMs and corridor reuse; anti-commuting sequences rejected explicitly | `test_rotated_surface_ppm_outcomes`, experiment tests |
| engine composition | measurement-block engine round over post-PPM tracker state (shared census) | `test_kernel_api_composes_with_measurement_block_engine` |
| deformed / non-rectangular patches | not supported: the corridor path reconstructs standard rectangles from placements; adjacent classification rejects a mismatched live view | — |
| reproducibility | circuit text is stable across process hash seeds for straight, bent/L, and branched/T routes | `test_rotated_surface_ppm_determinism` |

## Known interface gaps against the review's spec

Honestly listed, each a deliberate scope cut of this iteration:

- **wall execution and certificates**: wall plans carry `certificate=None`
  and are available to compiler consumers, but
  `RotatedSurfacePPMExperiment` rejects them until the tracker supports their
  mixed disposable/retained measurement lifecycle.
- **anti-commuting sequences**: the driver rejects them until absorbed
  relations can be retired or transformed soundly across later
  anti-commuting measurements.
- **paused/restored checks**: handled by coupler activation
  (`QECSystem.activate_coupler` pauses, `deactivate` restores), not
  exposed as a `RotatedSurfacePPMPlan` field.
- **no `post_state` field**: sequential composition reads the LIVE
  system's state after apply/split; the plan does not carry a
  self-contained state transition.
- **logical representatives**: an explicit live-probe input on the
  cell-adjacent path (`patch_view`); the corridor path derives standard
  representatives from the registered logical operators — which is also
  why deformed/non-rectangular patches are unsupported.
- **evaluation specification is implicit**: the initial/final state
  letters choose which correlations are deterministic; there is no
  first-class evaluation-spec object.  The protocol-output vs evaluation
distinction is carried by `PPMOutcome` plus the emitted
observables/closure detectors.

## Reproducibility

Circuit emission is deterministic across process hash seeds. Record XORs are
stored and emitted in canonical order, and subprocess regressions cover
straight, bent/L, and branched/T routes.
