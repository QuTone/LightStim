"""H6 magic-state distillation: the real "0-level distillation" protocol of
arXiv:2506.14688 ("Breaking even with magic", Quantinuum), built natively on
LightStim's own H-code family and simulation pipeline.

Two check variants are available, matching the paper's Fig. 1(e) and Fig.
1(f) respectively:

- ``variant="e"``: checks the symmetric pair |H+,H+>_L. Encodes |++>_L via
  the Fig. 1(d) unflagged encoder (``HSixLogicalOpSet.encode`` -- already
  native to ``lightstim.qec_code.H_code``), then verifies with the fused
  Bell-pair ancilla check of Quantinuum's ``Code614.py`` ``get_dist_circ``.
- ``variant="f"``: checks the asymmetric pair |H+,Y->_L that the paper's own
  benchmarking pipeline (Fig. 1(c)) actually consumes. Encodes |+>_L0|+i>_L1
  via the same Fig. 1(d) encoder, then applies a transversal logical Z1 (Z on
  data qubits 1,3,5 -- the shared support of X_L1/Z_L1) to flip the second
  slot from |+i>_L1 (Y+) to |-i>_L1 (Y-). The Bell-pair check couples aux1 to
  slot 1's support via CY instead of (e)'s CX (aux0 keeps CX to slot 0's
  still-X-proxied support) -- reusing (e)'s CX-only check unmodified measures
  an X_L1-type quantity with no definite value once slot 1 is Y-proxied,
  verified empirically to leave aux0's own detector randomly flipping ~50% of
  the time with zero injected noise. Slot 1's logical value is then extracted
  by a SEPARATE, dedicated flagged ancilla gadget -- the paper's Fig. 6
  circuit for measuring "Y2" non-destructively -- rather than by measuring
  slot 1's data qubits directly; see the note at that step below for why an
  earlier, more direct approach (mixing MX/MY into the same final measurement
  pass) doesn't work. The paper gives no gate-level diagram for any of this
  (Fig. 1(f)'s check is only described as "a nearly identical circuit" to
  (e); Appendix A is descriptive, not a circuit): the CX/CY check split and
  the Fig. 6 gadget's exact wiring are this module's own resolution,
  verified numerically (not a transcription of a circuit given in the
  paper) -- see the step 3 and step 3b notes below for exactly what was
  tried, what failed, and how each was confirmed correct.

In the real protocol both variants check non-Clifford |H+>_L magic states;
following the paper's own methodology for its scaling-law simulations, both
substitute a Clifford proxy state (reset -> encode, instead of the true
magic state) and study how the post-selected logical error rate scales with
the physical two-qubit gate error rate -- reproducing the paper's O(p^2)
scaling (Fig. 4) rather than measuring magic-state fidelity directly (which
needs an actual non-Clifford/PPVM simulator; see the separate
``H6-Distillation-DEQ`` repo, whose ``code614``/``h_check`` modules this
reimplements natively).

Neither variant implements the paper's X̄2 H̄1 H̄2 randomized twirl (Appendix
A) -- that twirl is applied before *consuming* a checked magic state via gate
teleportation (Fig. 1(c)), which this module does not build; both variants
stop at the checked state's own final logical readout.

Acceptance criterion: post-select on EVERY detector reading 0 (deterministic
in the noiseless case) -- both Bell-pair ancilla measurements and the H6
patch's own stabilizer/boundary detectors from its pre-check SE round(s),
all carrying the "post-select" detector tag, auto-discovered by
``SimulationPipeline``. This is deliberate: H6 is a distance-2 code, so it
can only DETECT a fault, never correct one -- feeding its own stabilizer
detectors to a real decoder instead of post-selecting them lets the decoder
"correct" based on a single round's own hook errors, which empirically
degrades the observed scaling from the paper's O(p^2) to O(p) (a decoder
mistake on every detected event, rather than a discard). The two Bell-pair
ancilla bits specifically are acceptance flags on whether the encoding+check
succeeded -- they are NOT independent logical-X measurements of the two
encoded qubits (a prior design review of this code family,
``docs/design/h6_core_integration.md`` on ``origin/h6-msd``, flags exactly
this distinction: the Bell-pair check is sensitive to a joint
X0_L*X1_L-type consistency condition plus a flag outcome, and must not be
documented as two independent logical readouts). The actual logical readout
-- used to compute the reported logical error rate among accepted shots --
is the final transversal MX measurement's OBSERVABLE_INCLUDE for X0_L and
X1_L, derived automatically by LightStim's own tracker (no manual rec[]
bookkeeping needed for that part).

Public API
----------
build_h6_distillation_circuit(rounds=1, variant="e") -> (circuit, circuit_info, system)
inject_noise(circuit, p, m=None) -> circuit
run_simulation(circuit, decoder_name=..., ...) -> stats
"""
from typing import Optional, Sequence, Tuple

import stim

from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.qec_code.H_code import (
    FlagAncillaPatch, HSixCode, HCodeExtractionBlock, HSixLogicalOpSet,
)

__all__ = [
    "build_h6_distillation_circuit",
    "inject_noise",
    "run_simulation",
]


class _BellCheckAncillaPatch(QECPatch):
    """Two bare ancilla qubits for the H6 Bell-pair logical-H check.

    Not a code: no stabilizers, no logical operators. Registered as its own
    ``QECSystem`` patch purely so the check ancillas get real coordinates
    like every other qubit in the circuit.
    """

    def _process_params(self):
        if self.params:
            raise ValueError(
                f"_BellCheckAncillaPatch takes no parameters; got {sorted(self.params)}."
            )

    def build(self):
        self.add_qubit(0.0, 0.0, role="syndrome")
        self.add_qubit(1.0, 0.0, role="syndrome")


class _Y2FlagAncillaPatch(QECPatch):
    """Two bare ancilla qubits for the Fig. 6 flagged, non-destructive Y_L1
    (paper's "Y2") measurement used by variant "f".

    Qubit 0 is the Y-parity syndrome ancilla (coupled to slot 1's support via
    CY, then read out in X); qubit 1 is the flag, entangled with the syndrome
    ancilla around its middle coupling gate to catch a fault on the syndrome
    ancilla itself spreading into a >weight-1 error on the data block, per
    the paper's own fault-tolerance claim for this circuit ("any hook error
    is a non-logical error that will be detected in subsequent error
    detection"). Not a code: no stabilizers, no logical operators.
    """

    def _process_params(self):
        if self.params:
            raise ValueError(
                f"_Y2FlagAncillaPatch takes no parameters; got {sorted(self.params)}."
            )

    def build(self):
        self.add_qubit(0.0, 0.0, role="syndrome")
        self.add_qubit(1.0, 0.0, role="syndrome")


def build_h6_distillation_circuit(
    *, rounds: int = 1, variant: str = "e", prep: str = "fig1d",
    anc_offset: Tuple[float, float] = (6.0, 4.0),
) -> Tuple[stim.Circuit, dict, QECSystem]:
    """Build the H6 0-level distillation gadget (noiseless).

    Steps:
      1. Encode the Clifford proxy state on a fresh ``HSixCode`` patch.
         ``prep="fig1d"`` (default) uses the family's own arbitrary-basis,
         unflagged Fig. 1(d) encoder (``HSixLogicalOpSet.encode``): |++>_L
         for ``variant="e"``, or |+>_L0|+i>_L1 followed by a transversal
         logical Z1 (Z on data qubits 1,3,5) to reach |+>_L0|-i>_L1 for
         ``variant="f"``. ``prep="fig5"`` (only supported for
         ``variant="e"``) instead uses the flag-verified Fig. 5 encoder
         (``encode(..., flagged=True)``) to prepare |00>_L, then a
         transversal Hadamard to reach |++>_L -- this is what the paper's
         own scaling-law simulations actually use, and it detects (rather
         than merely tolerates) a single fault during preparation itself.
      2. Run ``rounds`` ordinary syndrome-extraction rounds
         (``HCodeExtractionBlock``) before the check.
      3. Apply the fused Bell-pair ancilla check (the check half of
         Quantinuum's ``Code614.py`` ``get_dist_circ`` -- the encoder half is
         already exactly ``HSixLogicalOpSet.encode``'s own gate sequence, so
         it is not duplicated here). Each ancilla measurement gets its own
         DETECTOR, tagged "post-select". Identical for both variants: the
         paper describes (f)'s check as "a nearly identical circuit" to (e).
      4. Final transversal readout of the H6 patch, producing the code's own
         boundary-stabilizer detectors and logical observables automatically
         via the tracker: MX on both slots' support (X0_L/X1_L) for
         ``variant="e"``; MX on slot 0's support and MY on slot 1's support
         (X0_L/Y1_L) for ``variant="f"``, since X_L1 and Z_L1 share identical
         physical support {1,3,5} here, so per-qubit Y there reads out Y_L1
         directly.

    Args:
        rounds: Number of ordinary SE rounds run on the H6 patch before the
            Bell-pair check. Defaults to 1 -- REQUIRED for fault tolerance:
            with ``rounds=0`` LightStim's tracker has no prior round to
            anchor a boundary detector to, so the code's own two X-stabilizers
            (S^X_1=X0X1X2X3, S^X_2=X2X3X4X5) never get checked and a single
            physical fault can flip a logical observable undetected (measured
            empirically: ``rounds=0`` gives circuit fault distance 1;
            ``rounds>=1`` gives fault distance 2, matching the paper's
            protocol and ``H6-Distillation-DEQ``'s hand-built reference,
            which gets the same two stabilizer detectors "for free" from a
            single measurement pass since it isn't going through a tracker
            that requires an explicit prior round).
        variant: "e" for the paper's Fig. 1(e) symmetric |H+,H+>_L check
            (default), or "f" for the Fig. 1(f) asymmetric |H+,Y->_L check
            that the paper's actual Fig. 1(c) benchmarking pipeline consumes.
        prep: "fig1d" (default) for the unflagged, arbitrary-basis encoder,
            or "fig5" for the flag-verified |00>_L encoder plus a transversal
            Hadamard. "fig5" only supports ``variant="e"`` for now: reaching
            variant "f"'s asymmetric |+>_L0|-i>_L1 from a Fig. 5 |00>_L start
            would need an extra slot-1-only basis rotation (e.g. a
            transversal S/S_DAG restricted to qubits {1,3,5}, since slot 0's
            and slot 1's physical supports {0,2,4}/{1,3,5} are disjoint)
            that isn't implemented here.
        anc_offset: Placement of the 2 Bell-check ancilla qubits on the
            global coordinate canvas, clear of the H6 patch's own qubits.

    Returns:
        (circuit, circuit_info, system), matching the convention of
        ``lightstim/protocols/tg_distillation.py`` and ``ls_distillation.py``.
    """
    if variant not in ("e", "f"):
        raise ValueError(f"variant must be 'e' or 'f'; got {variant!r}.")
    if prep not in ("fig1d", "fig5"):
        raise ValueError(f"prep must be 'fig1d' or 'fig5'; got {prep!r}.")
    if prep == "fig5" and variant != "e":
        raise ValueError("prep='fig5' is only supported for variant='e'.")

    system = QECSystem()
    h6 = system.add_patch(HSixCode(), name="h6")
    anc = system.add_patch(_BellCheckAncillaPatch(), name="h6_check_anc", offset=anc_offset)
    y2_anc = None
    if variant == "f":
        y2_offset = (anc_offset[0] + 2.0, anc_offset[1])
        y2_anc = system.add_patch(_Y2FlagAncillaPatch(), name="h6_y2_anc", offset=y2_offset)
    prep_flag_anc = None
    if prep == "fig5":
        prep_flag_offset = (anc_offset[0], anc_offset[1] + 2.0)
        prep_flag_anc = system.add_patch(
            FlagAncillaPatch(), name="h6_prep_flag_anc", offset=prep_flag_offset
        )
    data = sorted(h6.data_indices)

    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    # 1. Encode the Clifford proxy state (stand-in for the two non-Clifford
    # |H+>_L magic states the real protocol checks).
    op_set = HSixLogicalOpSet()
    if prep == "fig5":
        # Fig. 5: flag-verified |00>_L, then transversal H to |++>_L -- the
        # paper's own route for its scaling-law simulations (variant "e" only).
        prep_flags = sorted(prep_flag_anc.syndrome_indices)
        op_set.encode(builder, h6, ("Z", "Z"), flagged=True, flag_qubits=prep_flags)
        builder.stabilizer_canonicalization()
        op_set.transversal_hadamard(builder, h6)
    elif variant == "e":
        op_set.encode(builder, h6, logical_bases=("X", "X"))  # |++>_L
    else:
        op_set.encode(builder, h6, logical_bases=("X", "Y"))  # |+>_L0 |+i>_L1
        # Flip slot 1 from Y+ to Y- via its logical Z (support {1,3,5}).
        flip = stim.Circuit()
        flip.append("Z", [data[1], data[3], data[5]])
        builder.apply_unitary_block(flip)
    # Required by encode()'s own contract: re-derive the tracker's
    # stabilizer/logical split from the code's canonical basis before any
    # further processing recognizes the encoded frame correctly.
    builder.stabilizer_canonicalization()

    # 2. Optional pre-check stabilization rounds. H6 is a distance-2 code:
    # it can only DETECT, never correct, a fault, so its own stabilizer
    # detectors must be post-selected (discard on any flag) rather than fed
    # to a decoder -- otherwise a real decoder will attempt "corrections"
    # from a single round's own hook errors and inject roughly O(p) decoder
    # mistakes, masking the protocol's real O(p^2) post-selected scaling
    # (confirmed empirically: leaving these undedicated to post-select gave
    # a fitted LER slope of ~1.1, not ~2). Tag them the same way
    # state_injection.py's "full_postselection" mode does.
    if rounds > 0:
        ps_coords = set()
        for stab in h6.stabilizers:
            syn_idx = stab.get("syn_idx")
            if syn_idx is None:
                continue
            xy = tuple(system.qubit_coords[syn_idx])
            ps_coords.add(xy + (0.0,))  # ordinary per-round SE detectors
            ps_coords.add(xy + (1.0,))  # the final boundary/closure detector
        builder.tracker.post_select_detector_coords |= ps_coords
        se = HCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    # 3. Fused Bell-pair logical-H check (Code614.py get_dist_circ, check half).
    # For "e" both slots are X-proxied, so aux1 couples to slot 1's support
    # {1,3,5} the same way aux0 couples to slot 0's support {0,2,4}: CX. For
    # "f" slot 1 is Y-proxied instead, so aux1's coupling must be CY there --
    # reusing CX (i.e. literally the same circuit as "e") leaves aux0's
    # detector maximally random in the noiseless case (verified empirically:
    # ~50% flip rate), since a CX-only check measures an X_L1-type quantity
    # that the Y-basis slot 1 has no definite value for. This CX/CY split is
    # this module's own resolution of the paper's "nearly identical circuit"
    # description for (f), which gives no gate-level diagram for it; verified
    # here by requiring both ancilla detectors deterministic in the noiseless
    # case, matching this module's own acceptance criterion.
    aux = sorted(anc.syndrome_indices)
    slot1_gate = "CX" if variant == "e" else "CY"

    check_circuit = stim.Circuit()
    check_circuit.append("H", [aux[0]])
    check_circuit.append("CX", [aux[0], aux[1]])
    check_circuit.append("CX", [aux[0], data[0]])
    check_circuit.append(slot1_gate, [aux[1], data[1]])
    check_circuit.append("CX", [aux[0], data[2]])
    check_circuit.append(slot1_gate, [aux[1], data[3]])
    check_circuit.append("CX", [aux[0], data[4]])
    check_circuit.append(slot1_gate, [aux[1], data[5]])
    check_circuit.append("CX", [aux[0], aux[1]])
    check_circuit.append("H", [aux[0]])
    builder.apply_unitary_block(check_circuit)

    # Measure the two check ancillas directly: this one-off acceptance flag
    # has no existing generic tracked-detector mechanism to reuse (it is not
    # a code stabilizer of either patch), so it is built by hand, exactly
    # like every other ancilla measurement in a circuit -- an M instruction
    # plus a DETECTOR referencing its rec[] -- with the tracker's own
    # measurement counter kept in sync so every later tracked call (the
    # final readout below) still resolves its own rec[] offsets correctly.
    #
    # The _replace_measured_ancillas_with_records call below is REQUIRED,
    # not optional bookkeeping, whenever slot1_gate is CY (variant "f"):
    # apply_unitary_block's forward-conjugation of the check circuit updates
    # every TRACKED row (including the code's own XA/XB stabilizer rows) as
    # it propagates through, and unlike CX, a controlled-Y entangles the
    # TARGET qubit's X-component with a Z-type factor on the control (CY
    # control=c,target=t: X_t -> Z_c X_t Y_t-ish, verified via
    # stim.Tableau on a 2-qubit CY -- "X1 fwd: +ZX"). So once aux[1] couples
    # via CY to data[1]/data[3] (both in XA's/XB's support), XA's and XB's
    # tracked rows pick up a phantom dependency on aux[1] -- a qubit that
    # IS about to be measured, but not through the tracker's own pipeline
    # (this block bypasses process_mid_measurement entirely). Left
    # unresolved, that phantom dependency makes XA/XB's rows inexpressible
    # in terms of the final MX readout alone, so process_data_measurement's
    # is_dependent check silently fails for them and their boundary/closure
    # detectors never get emitted -- this was the actual root cause of the
    # fault-distance-1 bug traced across several earlier attempts (mixed
    # MX/MY final readout, a transversal S_DAG conversion, and initially
    # this very check circuit): all of them, and the Fig. 6 Y2 gadget below
    # on its first attempt, hit this exact mechanism. Calling
    # _replace_measured_ancillas_with_records right after the raw M
    # instruction folds each measured ancilla's Pauli factor out of every
    # tracked row (stabilizers AND logicals), exactly like
    # process_mid_measurement would -- verified via shortest_graphlike_error
    # to restore fault distance 2 for variant "f" (see tests).
    if builder.circuit[-1].name != "TICK":
        builder.circuit.append("TICK")
    check_meas_base = builder.tracker.total_measurements
    builder.circuit.append("M", aux)
    builder.tracker.total_measurements += len(aux)
    builder.tracker._replace_measured_ancillas_with_records(
        measurement_qubit_indices=list(aux),
        measurement_bases=["X", "X"],
        measurement_base_idx=check_meas_base,
    )
    coord0 = tuple(system.qubit_coords[aux[0]]) + (0.0,)
    coord1 = tuple(system.qubit_coords[aux[1]]) + (0.0,)
    builder.circuit.append("DETECTOR", [stim.target_rec(-2)], list(coord0), tag="post-select")
    builder.circuit.append("DETECTOR", [stim.target_rec(-1)], list(coord1), tag="post-select")

    # 3b. variant "f" only: Fig. 6's flagged, non-destructive Y_L1 ("Y2")
    # measurement. This replaced an earlier, INCORRECT approach that read
    # slot 1's support directly via a destructive MY as part of the SAME
    # final measurement pass used for the code's own X-stabilizer closure
    # (XA=X0X1X2X3, XB=X2X3X4X5, both straddling BOTH logical slots) --
    # that mixed-basis final readout left the closure detectors for XA/XB
    # silently failing to form (verified via shortest_graphlike_error: fault
    # distance 1, a bare zero-detector weight-1 error flipping each logical
    # observable). The paper's own Fig. 6 circuit avoids this by keeping Y_L1
    # extraction on a SEPARATE, dedicated ancilla (independent of the QEDX
    # X-syndrome stage, per the paper's own text) -- the data qubits are
    # never destructively measured or basis-converted here, so they remain
    # available for an ordinary, uniform, variant-"e"-style final MX readout
    # afterward, which is what actually restores XA/XB's closure.
    #
    # Y_L1 = -(Y1 Y3 Y5) transversally (X_L1, Z_L1 share support {1,3,5}), so
    # a standard ancilla-parity-measurement circuit applies: prep the syn
    # ancilla in |+>, CY it onto each of data[1,3,5] (ancilla as control),
    # then read syn out in X. The flag ancilla brackets the MIDDLE coupling
    # gate with CX(flag, syn) so a fault on syn during that gate -- which
    # could otherwise spread into a weight-2+ error on the data block --
    # instead flips the flag, matching the paper's own fault-tolerance claim
    # for this circuit ("any hook error is a non-logical error that will be
    # detected in subsequent error detection"). Both ancilla outcomes are
    # acceptance flags (post-select on 0), not reported observables --
    # verified numerically (canonical_stabilizers()-based prototype) to be
    # deterministic in the noiseless case, same acceptance-flag role as the
    # Bell-check ancillas above.
    #
    # Same _replace_measured_ancillas_with_records requirement as the
    # Bell-check above, and for the identical reason: CY(syn, data[1]/[3])
    # entangles XA's/XB's tracked rows with a phantom dependency on `syn`
    # that must be folded out before those rows can close against the final
    # MX readout.
    if variant == "f":
        syn, flag = sorted(y2_anc.syndrome_indices)
        y2_circuit = stim.Circuit()
        y2_circuit.append("H", [syn])
        y2_circuit.append("CY", [syn, data[1]])
        y2_circuit.append("CX", [flag, syn])
        y2_circuit.append("CY", [syn, data[3]])
        y2_circuit.append("CX", [flag, syn])
        y2_circuit.append("CY", [syn, data[5]])
        y2_circuit.append("H", [syn])
        builder.apply_unitary_block(y2_circuit)

        if builder.circuit[-1].name != "TICK":
            builder.circuit.append("TICK")
        y2_meas_base = builder.tracker.total_measurements
        builder.circuit.append("M", [syn, flag])
        builder.tracker.total_measurements += 2
        builder.tracker._replace_measured_ancillas_with_records(
            measurement_qubit_indices=[syn, flag],
            measurement_bases=["Z", "Z"],
            measurement_base_idx=y2_meas_base,
        )
        syn_coord = tuple(system.qubit_coords[syn]) + (0.0,)
        flag_coord = tuple(system.qubit_coords[flag]) + (0.0,)
        builder.circuit.append("DETECTOR", [stim.target_rec(-2)], list(syn_coord), tag="post-select")
        builder.circuit.append("DETECTOR", [stim.target_rec(-1)], list(flag_coord), tag="post-select")

    # 4. Final logical readout: uniform MX for both variants (matching
    # variant "e" exactly restores XA/XB's automatic boundary/closure
    # detectors -- verified via shortest_graphlike_error, see tests: fault
    # distance 2 at rounds>=1 for BOTH variants). For variant "f", slot 1 was
    # already projected onto its Y_L1 eigenstate by step 3b's ancilla-based
    # measurement, which consumed that logical degree of freedom -- the
    # tracker correctly recognizes X_L1 is no longer well-defined (Y and X
    # don't commute) and emits only ONE observable for variant "f"
    # (X_L0, the genuine magic-state slot; circuit.num_observables == 1,
    # vs. 2 for variant "e"). Qubits 1,3,5 are still measured here (needed
    # for XA/XB closure), but their bits don't feed any observable.
    final_measurements = {q: "X" for q in h6.data_indices}
    builder.apply_data_readout(final_measurements)

    circuit = builder.circuit
    circuit_info = {
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "rounds": rounds,
        "variant": variant,
        "prep": prep,
    }
    return circuit, circuit_info, system


def inject_noise(circuit: stim.Circuit, *, p: float, m: Optional[float] = None) -> stim.Circuit:
    """Circuit-level depolarizing noise, matching Code614.py's own (p, m)
    parametrization: `p` for two-qubit gates/measurement, `m` for one-qubit
    idle-memory depolarization. Defaults `m` to `p` when not given.

    Matches tg_distillation.py's / ls_distillation.py's noise-injection
    convention (``NoiseInjector.from_circuit_level``).
    """
    all_qubits = list(range(circuit.num_qubits))
    cfg = NoiseConfig(p_1q=m if m is not None else p, p_2q=p, p_meas=p, p_reset=p)
    return NoiseInjector.from_circuit_level(cfg, all_qubits).inject_noise(circuit)


def run_simulation(
    circuit: stim.Circuit,
    *,
    decoder_name: str = "pymatching",
    max_shots: int = 1_000_000,
    max_errors: int = 100,
    batch_size: int = 10_000,
    num_workers: int = 4,
    target_observable_indices: Optional[Sequence[int]] = None,
    backend: str = "cpu",
    decoder_params: Optional[dict] = None,
    on_decode_failure: str = "error",
    print_progress: bool = False,
):
    """Run the noisy circuit through LightStim's standardized simulation
    pipeline. Post-selection is entirely automatic: ``SimulationPipeline``
    discovers every "post-select"-tagged detector on its own (see
    ``build_h6_distillation_circuit`` -- every detector in this circuit is
    tagged, so the decoder only ever sees an all-zero syndrome on surviving
    shots), so no post-select indices are passed here -- only which logical
    observable(s) to report the error rate against.

    Defaults to reporting joint failure across both encoded logical qubits
    (X0_L and X1_L); pass ``target_observable_indices=[0]`` to isolate one.
    """
    if target_observable_indices is None:
        target_observable_indices = list(range(circuit.num_observables))
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(
            decoder_name,
            backend=backend,
            params=decoder_params or {},
            on_decode_failure=on_decode_failure,
        ),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        target_observable_indices=list(target_observable_indices),
        print_progress=print_progress,
    )
    return pipeline.run(circuit)
