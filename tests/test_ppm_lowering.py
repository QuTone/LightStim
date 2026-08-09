"""The pure PPM lowering kernel: request in, plan + certificate out.

lower_ppm() must not mutate the system; apply_plan() is the single
registration step; the certificate carries the named algebraic oracle —
including the true-weight-w guarantee (no single logical, no proper
subset product leaks).  Y letters and missing routes are rejected
explicitly, never silently rewritten.
"""
import contextlib
import io

import pytest

from lightstim.protocols.ppm.spec import PatchSpec, PPMStep, origin_of
from lightstim.protocols.ppm.lowering import (
    PPMRequest, UnsupportedPauliError, lower_ppm)
from lightstim.protocols.ppm.coupler import route_and_build
from lightstim.protocols.ppm.seam_rules import SeamRuleError
from lightstim.protocols.ppm.sequential import SequentialPPMExperiment

D = 3


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _built_zz_experiment():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    exp = SequentialPPMExperiment(
        px, [PPMStep([("A", "Z"), ("B", "Z")], route=[(1, 0)])],
        initial_states={"A": "Z", "B": "Z"},
        final_measure_states={"A": "Z", "B": "Z"},
        rounds=D, rounds_init=1)
    with contextlib.redirect_stdout(io.StringIO()):
        exp.build()
    return exp


def test_lower_zz_pair_plan_fields():
    exp = _built_zz_experiment()
    plan = exp._plans[0]
    assert plan.kind == 'corridor'
    assert plan.bus == 'Z'
    assert plan.corridor_init_basis == 'X'
    assert plan.schedule == 'bent'
    assert plan.merged_checks, "plan must expose the merged check set"
    assert not plan.has_stretched_checks


def test_certificate_names_the_instrument_guarantees():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    r = route_and_build(px, [("A", "Z"), ("B", "Z")], seam=True,
                        route=[(1, 0)])
    assert r.ok and r.certificate is not None
    for item in ("commute", "joint", "no_single", "no_subjoint",
                 "logical_count"):
        assert r.certificate[item] is True, item
    exp = _built_zz_experiment()
    cert = exp._plans[0].certificate
    assert cert is not None and cert.ok
    # the true weight-w guarantee: the joint is measured and NO pairwise /
    # single logical information leaks (a different quantum instrument).
    assert cert.measures_exactly_the_product


def test_y_letter_rejected_explicitly():
    with pytest.raises(UnsupportedPauliError, match="different quantum"):
        PPMRequest(targets=(("A", "Y"), ("B", "Z")), route=())


def test_route_is_required_no_autorouter():
    with pytest.raises(ValueError, match="no auto-router"):
        PPMRequest(targets=(("A", "Z"), ("B", "Z")), route=None)


def test_adjacent_pair_requires_live_system():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 1, 0, "X_horizontal")]
    req = PPMRequest(targets=(("A", "Z"), ("B", "Z")), route=())
    with pytest.raises(ValueError, match="live probe"):
        lower_ppm(px, req, system=None)


def test_lower_ppm_is_pure_apply_is_the_mutation():
    exp = _built_zz_experiment()
    n_couplers = len(exp.system.coupler_patches)
    req = PPMRequest(targets=(("A", "Z"), ("B", "Z")), route=((1, 0),))
    plan = lower_ppm(exp._specs(), req, system=exp.system)
    assert len(exp.system.coupler_patches) == n_couplers, \
        "lower_ppm mutated the system"
    assert plan.route_result.ok


def test_wall_plan_and_bent_conflict():
    # Rule row 2 (same letter, different patch types; the
    # test_row2_same_pauli_diff_type_wall configuration): stretched-wall
    # construction, diagonal schedule mandatory.
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 0, 1, "X_vertical")]
    seq = [PPMStep([("A", "X"), ("B", "X")], route=[])]
    states = {"A": "X", "B": "X"}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, colour_swapped={"B"})
    with contextlib.redirect_stdout(io.StringIO()):
        exp.build()
    plan = exp._plans[0]
    assert plan.kind == 'wall'
    assert plan.schedule == 'diagonal'
    assert plan.wall is not None

    bad = SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, schedule='bent', colour_swapped={"B"})
    with pytest.raises(SeamRuleError, match="bent"):
        with contextlib.redirect_stdout(io.StringIO()):
            bad.build()
