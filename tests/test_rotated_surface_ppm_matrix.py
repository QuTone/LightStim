"""Verification-matrix additions (review item 5).

- a larger-weight joint (weight 4) with the exactly-the-product
  certificate;
- a d=5 representative (slow-marked);
- composition with the measurement-block engine on post-PPM tracker
  state, driven through the compiler-facing kernel API (lower_ppm /
  apply_ppm_plan + builder primitives, no experiment driver).
"""
import contextlib
import io

import numpy as np
import pytest

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import (
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.rotated.bent_joint_se import (
    se_round_chunk,
)
from lightstim.qec_code.surface_code.rotated.ppm import (
    RotatedSurfacePatchPlacement,
    RotatedSurfacePPMLayoutError,
    RotatedSurfacePPMRequest,
    apply_ppm_plan,
    lower_ppm,
    origin_of,
)
from lightstim.protocols.rotated_surface_ppm import (
    RotatedSurfacePPMExperiment,
    RotatedSurfacePPMStep,
)

D = 3


def _spec(nm, a, b, o, d=D):
    return RotatedSurfacePatchPlacement(nm, origin_of(a, b, d, seam=True), d, o)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def test_weight4_straight_chain_is_an_explicit_gap():
    """KNOWN GAP fixture (review: leave a reproducible test, not a prose
    TODO): a weight-4 joint on a straight chain needs the snake / kf-wall
    attach machinery that this minimal explicit-route variant deliberately
    does not carry — alternating seam parity strands the far targets.  The
    physics layer rejects it loudly with the exact reason; nothing
    silently degrades.  Supported target weights today: 2 and 3 (T
    corridor); see the capability table."""
    px = [_spec("q1", 0, 0, "X_horizontal"), _spec("q2", 2, 0, "X_horizontal"),
          _spec("q3", 4, 0, "X_horizontal"), _spec("q4", 6, 0, "X_horizontal")]
    st = {q: "Z" for q in ("q1", "q2", "q3", "q4")}
    exp = RotatedSurfacePPMExperiment(
        px, [RotatedSurfacePPMStep([("q1", "Z"), ("q2", "Z"), ("q3", "Z"), ("q4", "Z")],
                     route=[(1, 0), (3, 0), (5, 0)])],
        initial_states=st, final_measure_states=st, rounds=D, rounds_init=1)
    with pytest.raises(RotatedSurfacePPMLayoutError, match="never reaches the corridor"):
        _build(exp)


def test_longer_straight_route_full_distance():
    # a 3-cell corridor between distant targets stays on the bent schedule
    from lightstim.noise.config import NoiseConfig
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 4, 0, "X_horizontal")]
    st = {"A": "Z", "B": "Z"}
    exp = RotatedSurfacePPMExperiment(
        px, [RotatedSurfacePPMStep([("A", "Z"), ("B", "Z")], route=[(1, 0), (2, 0), (3, 0)])],
        initial_states=st, final_measure_states=st, rounds=D, rounds_init=1,
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                 p_reset=1e-3, p_idle=1e-3))
    c = _build(exp)
    assert exp._sched[0] == 'bent'
    assert exp.plans[0].certificate.measures_exactly_the_product
    c.detector_error_model(decompose_errors=True)
    assert len(c.shortest_graphlike_error()) == D


def test_bent_route_forces_diagonal_and_full_distance():
    # any bend in the corridor puts the whole merged block on the diagonal
    # schedule; the certificate still pins the exact-product guarantee
    from lightstim.noise.config import NoiseConfig
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 1, "X_vertical")]
    st = {"A": "Z", "B": "Z"}
    exp = RotatedSurfacePPMExperiment(
        px, [RotatedSurfacePPMStep([("A", "Z"), ("B", "Z")], route=[(1, 0), (2, 0)])],
        initial_states=st, final_measure_states=st, rounds=D, rounds_init=1,
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                 p_reset=1e-3, p_idle=1e-3))
    c = _build(exp)
    assert exp._sched[0] == 'diagonal'
    assert exp.plans[0].certificate.measures_exactly_the_product
    c.detector_error_model(decompose_errors=True)
    assert len(c.shortest_graphlike_error()) == D


def test_three_target_certificate_pins_true_multibody_instrument():
    # the multi-body matrix item: the weight-3 T-corridor joint measures
    # the triple product and NOTHING finer (no pairwise parity leaks)
    px = [_spec("q1", 0, 0, "X_horizontal"), _spec("q2", 4, 0, "X_horizontal"),
          _spec("q3", 2, 1, "X_vertical")]
    st = {"q1": "Z", "q2": "Z", "q3": "Z"}
    exp = RotatedSurfacePPMExperiment(
        px, [RotatedSurfacePPMStep([("q1", "Z"), ("q2", "Z"), ("q3", "Z")],
                     route=[(1, 0), (2, 0), (3, 0)])],
        initial_states=st, final_measure_states=st, rounds=D, rounds_init=1)
    _build(exp)
    cert = exp.plans[0].certificate
    assert cert.ok
    assert cert.items['no_subjoint'] is True
    assert cert.items['no_single'] is True
    assert cert.measures_exactly_the_product


@pytest.mark.slow
def test_zz_pair_full_distance_d5():
    d = 5
    px = [_spec("A", 0, 0, "X_horizontal", d), _spec("B", 2, 0,
                                                     "X_horizontal", d)]
    st = {"A": "Z", "B": "Z"}
    from lightstim.noise.config import NoiseConfig
    exp = RotatedSurfacePPMExperiment(
        px, [RotatedSurfacePPMStep([("A", "Z"), ("B", "Z")], route=[(1, 0)])],
        initial_states=st, final_measure_states=st, rounds=d, rounds_init=1,
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                 p_reset=1e-3, p_idle=1e-3))
    c = _build(exp)
    c.detector_error_model(decompose_errors=True)
    assert len(c.shortest_graphlike_error()) == d


def test_kernel_api_composes_with_measurement_block_engine():
    """Compiler-consumer path: lower_ppm/apply_ppm_plan + builder primitives,
    no experiment driver — then ONE measurement-block engine round on the
    post-PPM tracker state (absorbed joint present), then readout.  The
    block path and the PPM tracker semantics must share one census."""
    specs = [_spec("A", 0, 0, "X_horizontal"),
             _spec("B", 2, 0, "X_horizontal")]
    system = QECSystem()
    for s in specs:
        p = RotatedSurfaceCode(distance=s.distance)
        if s.orientation == 'X_horizontal':
            p.transpose_coords()
        system.add_patch(p, name=s.name,
                         offset=(s.origin[0] - 1, s.origin[1] - 1))

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize(init_dict={q: 'Z' for q in system.data_indices},
                       n=system.num_qubits)
    orient = {s.name: s.orientation for s in specs}
    owner = system.index_to_owner_map
    domains = {tuple(system.qubit_coords[q]): orient[owner[q]]
               for q in system.data_indices if owner.get(q) in orient}
    builder.apply_syndrome_extraction(
        circuit_chunk=se_round_chunk(system, domains=domains), rounds=1)

    # Couplers are define-by-run resources: lower/register only after the
    # baseline has established the logical patches.
    plan = lower_ppm(specs,
                     RotatedSurfacePPMRequest(targets=(("A", "Z"), ("B", "Z")),
                                route=((1, 0),)),
                     system=system)
    apply_ppm_plan(system, plan, "ppm_0")

    builder.activate_coupler("ppm_0")
    coupler_patch = system.coupler_patches["ppm_0"]
    l2g = system.local_to_global_map["ppm_0"]
    corridor_init = {l2g[q]: plan.corridor_init_basis
                     for q in coupler_patch.data_indices}
    builder.initialize(init_dict=corridor_init, n=system.num_qubits)
    merged = dict(plan.route_result.layout.domains or {})
    for q in system.data_indices:
        nm = owner.get(q)
        if nm in orient:
            merged.setdefault(tuple(system.qubit_coords[q]), orient[nm])
    builder.apply_syndrome_extraction(
        circuit_chunk=se_round_chunk(system, domains=merged), rounds=D)
    builder.deactivate_coupler("ppm_0")
    builder.apply_data_readout(final_measurements=dict(corridor_init),
                               resolve_absorbed=False)

    # one measurement-block engine round on the post-PPM state
    block_chunk = se_round_chunk(system, domains=domains)
    builder.apply_syndrome_extraction(circuit_chunk=block_chunk, rounds=1,
                                      measurement_blocks=(block_chunk,))
    tracker.validate_logical_count(
        context="block round on post-PPM state")

    builder.apply_data_readout(
        final_measurements={q: 'Z' for q in system.data_indices})
    c = builder.circuit
    # same as the driver's ZZ flow: two logical DOFs, one consumed by the
    # joint (its value is the protocol output), one standing observable
    assert c.num_observables == 1
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
