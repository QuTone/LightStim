"""Sequential rotated-surface PPM experiment and lowering backend.

Routes are required on every RotatedSurfacePPMStep; automatic routing,
liveness, rotations, snake constructions, and Y births are not included.

Covered here (noiseless silence + deterministic observables + graphlike
distance d at p=1e-3):
  * explicit one-cell corridor Z⊗Z;
  * executable row-1/4 cell-adjacent merges;
  * pure lowering plus explicit experiment rejection for row-2/3 walls;
  * the Handbook Sec. 10.4 cap-slot regression (transposed row-3 E/W wall);
  * route-required enforcement at every layer.
"""
import contextlib
import io

import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.rotated.ppm import (
    RotatedSurfacePatchPlacement,
    RotatedSurfacePPMLayoutError,
    SeamRuleError,
    build_explicit_ppm_layout,
    lower_ppm,
    origin_of,
)
from lightstim.protocols.rotated_surface_ppm import (
    RotatedSurfacePPMExperiment,
    RotatedSurfacePPMStep,
    UnsupportedPPMExperimentError,
)

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return RotatedSurfacePatchPlacement(nm, origin_of(a, b, D, seam=True), D, o)


def _run(px, target, states, *, route=(), **kw):
    step_kw = kw.pop('step_kw', {})
    exp = RotatedSurfacePPMExperiment(
        px, [RotatedSurfacePPMStep(target, route=list(route), **step_kw)],
        initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    return exp, c


def _lower(px, target, states, *, route=(), **kw):
    step_kw = kw.pop('step_kw', {})
    step = RotatedSurfacePPMStep(target, route=list(route), **step_kw)
    exp = RotatedSurfacePPMExperiment(
        px, [step], initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, **kw)
    exp.system = QECSystem()
    exp._by_name = {s.name: s for s in exp.patches}
    for patch in exp.patches:
        exp._alloc_patch(patch.name)
    plan = lower_ppm(
        exp._specs(),
        exp._request(step),
        system=exp.system,
        conj_names=exp._conj(step),
        schedule_policy=exp.schedule,
    )
    return exp, plan


def _verify(exp, c, row=None):
    if row is not None:
        assert exp._rules[0].row == row
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), "detector fired at p=0"
    assert not obs.any(), "observable not deterministic at p=0"
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


# ── route is REQUIRED at every layer ─────────────────────────────────────────

def test_ppmstep_route_is_required():
    with pytest.raises(TypeError):
        RotatedSurfacePPMStep([("A", "Z"), ("B", "Z")])          # no route argument


def test_experiment_rejects_route_none():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    with pytest.raises(ValueError, match="route is required"):
        RotatedSurfacePPMExperiment(
            px, [RotatedSurfacePPMStep([("A", "Z"), ("B", "Z")], route=None)],
            initial_states={"A": "Z", "B": "Z"},
            final_measure_states={"A": "Z", "B": "Z"})


def test_build_explicit_ppm_layout_rejects_route_none():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    with pytest.raises(RotatedSurfacePPMLayoutError, match="no auto-router"):
        build_explicit_ppm_layout(
            px,
            [("A", "Z"), ("B", "Z")],
            route=None,
            seam=True,
        )


# ── explicit corridor ────────────────────────────────────────────────────────

def test_explicit_corridor_zz_full_distance():
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 2, 0, "X_horizontal")],
                  [("A", "Z"), ("B", "Z")], {"A": "Z", "B": "Z"},
                  route=[(1, 0)])
    assert c.num_observables == 1
    _verify(exp, c)


def test_joint_closure_detector_emitted():
    # A measurement-promoted joint has a legitimate long-range terminal
    # closure detector; support weight alone must not suppress it.
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 2, 0, "X_horizontal")],
                  [("A", "Z"), ("B", "Z")], {"A": "Z", "B": "Z"},
                  route=[(1, 0)])
    longrange = [inst for inst in c.flattened() if inst.name == "DETECTOR"
                 and len(inst.targets_copy()) > 2 * D + 2]
    assert longrange, "joint-closure long-range detector was not emitted"
    _verify(exp, c)


def test_three_target_one_step_t_corridor():
    # one step measures 3 patches through a 3-cell T corridor: two
    # independent pairwise products (obs=2), full distance
    exp, c = _run([_spec("q1", 0, 0, "X_horizontal"),
                   _spec("q2", 4, 0, "X_horizontal"),
                   _spec("q3", 2, 1, "X_vertical")],
                  [("q1", "Z"), ("q2", "Z"), ("q3", "Z")],
                  {"q1": "Z", "q2": "Z", "q3": "Z"},
                  route=[(1, 0), (2, 0), (3, 0)])
    assert c.num_observables == 2
    _verify(exp, c)


# ── the four rule-table rows on cell-adjacent pairs (route=[]) ──────────────

def test_row1_same_pauli_same_type_plain_merge():
    # both place(X_horizontal) = conjugate type; Z⊗Z through the E/W seam:
    # plain merge (seam line is DATA), bent schedule
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 1, 0, "X_horizontal")],
                  [("A", "Z"), ("B", "Z")], {"A": "Z", "B": "Z"})
    assert exp._sched[0] == 'bent'
    _verify(exp, c, row=1)


def test_row2_same_pauli_diff_type_wall():
    # A = place(X_horizontal), B = place(X_vertical) with swapped colours
    # (live X̄ horizontal, textbook positions); X⊗X through the N/S seam ->
    # uniform-domino wall on the diagonal schedule
    px = [_spec("A", 0, 0, "X_horizontal"),
          _spec("B", 0, 1, "X_vertical")]
    target = [("A", "X"), ("B", "X")]
    states = {"A": "X", "B": "X"}
    _, plan = _lower(px, target, states, colour_swapped={"B"})
    assert plan.rule.row == 2
    assert plan.kind == 'wall'
    assert plan.schedule == 'diagonal'
    with pytest.raises(UnsupportedPPMExperimentError, match="wall"):
        _run(px, target, states, colour_swapped={"B"})


def test_row3_mixed_diff_type_wall():
    # A minority -> conjugate type, Z̄ horizontal; B colour-swapped ->
    # textbook type, X̄ horizontal; Z⊗X through the N/S seam ->
    # mixed-domino wall, diagonal schedule
    px = [_spec("A", 0, 0, "X_horizontal"),
          _spec("B", 0, 1, "X_vertical")]
    target = [("A", "Z"), ("B", "X")]
    states = {"A": "Z", "B": "X"}
    _, plan = _lower(px, target, states, colour_swapped={"A", "B"})
    assert plan.rule.row == 3
    assert plan.kind == 'wall'
    assert plan.schedule == 'diagonal'
    with pytest.raises(UnsupportedPPMExperimentError, match="wall"):
        _run(px, target, states, colour_swapped={"A", "B"})


def test_row4_mixed_same_type_recoloured_merge():
    # both textbook positions, one recoloured; X⊗Z through the E/W seam ->
    # recoloured single-cell mixed checks, straight seam stays bent
    exp, c = _run([_spec("A", 0, 0, "X_vertical"),
                   _spec("B", 1, 0, "X_vertical")],
                  [("A", "X"), ("B", "Z")], {"A": "X", "B": "Z"},
                  colour_swapped={"B"})
    assert exp._sched[0] == 'bent'
    _verify(exp, c, row=4)


def test_row3_transposed_ew_wall_cap_slot():
    # Handbook Sec. 10.4 regression: X⊗Z through the E/W (vertical) seam —
    # ve/ho pair, same colour phase, no colour swap -> row-3 mixed stretched
    # wall.  The south end row already carries both patches' corner lobes
    # one gap away (no slot), so the cap must sit at the free NORTH end;
    # the old far-side-colour proxy put it south (three checks fighting one
    # corner -> 50% random detectors).
    px = [_spec("A", 0, 0, "X_vertical"),
          _spec("B", 1, 0, "X_horizontal")]
    target = [("A", "X"), ("B", "Z")]
    states = {"A": "Z", "B": "Z"}
    _, plan = _lower(px, target, states)
    assert plan.rule.row == 3
    assert plan.kind == 'wall'
    caps = [record for record in plan.wall.checks if len(record[1]) == 2]
    assert len(caps) == 1
    assert caps[0][0][1] == 0, \
        f"cap must sit at the free north end, got {caps[0][0]}"
    with pytest.raises(UnsupportedPPMExperimentError, match="wall"):
        _run(px, target, states)


def test_wall_step_rejects_bent_schedule():
    with pytest.raises(SeamRuleError, match="bent"):
        _run([_spec("A", 0, 0, "X_horizontal"),
              _spec("B", 0, 1, "X_vertical")],
             [("A", "X"), ("B", "X")], {"A": "X", "B": "X"},
             colour_swapped={"B"}, step_kw={'schedule': 'bent'})
