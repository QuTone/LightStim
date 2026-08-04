"""Sequential PPM (explicit-route minimal driver) —
lightstim/protocols/ppm (spec / seam_rules / coupler / sequential).

These modules are copied from the author's CircLS repository
(https://github.com/John-YuehanZhang/CircLS @ 8802a5b) and intentionally
independent of it: route is REQUIRED on every PPMStep (no auto-router),
liveness / rotations / snake / Y births are not included.

Covered here (noiseless silence + deterministic observables + graphlike
distance d at p=1e-3):
  * explicit one-cell corridor Z⊗Z;
  * all four rule-table rows on cell-adjacent pairs (route=[]);
  * the Handbook Sec. 10.4 cap-slot regression (transposed row-3 E/W wall);
  * route-required enforcement at every layer;
  * independence from the circls package (subprocess import probe).
"""
import contextlib
import io
import subprocess
import sys

import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.ppm.spec import PatchSpec, PPMStep, origin_of
from lightstim.protocols.ppm.coupler import route_and_build, BentLayoutError
from lightstim.protocols.ppm.seam_rules import SeamRuleError
from lightstim.protocols.ppm.sequential import SequentialPPMExperiment

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _run(px, target, states, *, route=(), **kw):
    step_kw = kw.pop('step_kw', {})
    exp = SequentialPPMExperiment(
        px, [PPMStep(target, route=list(route), **step_kw)],
        initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    return exp, c


def _verify(exp, c, row=None, wall=None):
    if row is not None:
        assert exp._rules[0].row == row
    if wall is not None:
        assert (0 in exp._walls) == wall
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
        PPMStep([("A", "Z"), ("B", "Z")])          # no route argument


def test_experiment_rejects_route_none():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    with pytest.raises(ValueError, match="route is required"):
        SequentialPPMExperiment(
            px, [PPMStep([("A", "Z"), ("B", "Z")], route=None)],
            initial_states={"A": "Z", "B": "Z"},
            final_measure_states={"A": "Z", "B": "Z"})


def test_route_and_build_rejects_route_none():
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    with pytest.raises(BentLayoutError, match="no auto-router"):
        route_and_build(px, [("A", "Z"), ("B", "Z")], seam=True)


# ── explicit corridor ────────────────────────────────────────────────────────

def test_explicit_corridor_zz_full_distance():
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 2, 0, "X_horizontal")],
                  [("A", "Z"), ("B", "Z")], {"A": "Z", "B": "Z"},
                  route=[(1, 0)])
    assert c.num_observables == 1
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
    _verify(exp, c, row=1, wall=False)


def test_row2_same_pauli_diff_type_wall():
    # A = place(X_horizontal), B = place(X_vertical) with swapped colours
    # (live X̄ horizontal, textbook positions); X⊗X through the N/S seam ->
    # uniform-domino wall on the diagonal schedule
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 0, 1, "X_vertical")],
                  [("A", "X"), ("B", "X")], {"A": "X", "B": "X"},
                  colour_swapped={"B"})
    assert exp._sched[0] == 'diagonal'
    _verify(exp, c, row=2, wall=True)


def test_row3_mixed_diff_type_wall():
    # A minority -> conjugate type, Z̄ horizontal; B colour-swapped ->
    # textbook type, X̄ horizontal; Z⊗X through the N/S seam ->
    # mixed-domino wall, diagonal schedule
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 0, 1, "X_vertical")],
                  [("A", "Z"), ("B", "X")], {"A": "Z", "B": "X"},
                  colour_swapped={"A", "B"})
    assert exp._sched[0] == 'diagonal'
    _verify(exp, c, row=3, wall=True)


def test_row4_mixed_same_type_recoloured_merge():
    # both textbook positions, one recoloured; X⊗Z through the E/W seam ->
    # recoloured single-cell mixed checks, straight seam stays bent
    exp, c = _run([_spec("A", 0, 0, "X_vertical"),
                   _spec("B", 1, 0, "X_vertical")],
                  [("A", "X"), ("B", "Z")], {"A": "X", "B": "Z"},
                  colour_swapped={"B"})
    assert exp._sched[0] == 'bent'
    _verify(exp, c, row=4, wall=False)


def test_row3_transposed_ew_wall_cap_slot():
    # Handbook Sec. 10.4 regression: X⊗Z through the E/W (vertical) seam —
    # ve/ho pair, same colour phase, no colour swap -> row-3 mixed stretched
    # wall.  The south end row already carries both patches' corner lobes
    # one gap away (no slot), so the cap must sit at the free NORTH end;
    # the old far-side-colour proxy put it south (three checks fighting one
    # corner -> 50% random detectors).
    exp, c = _run([_spec("A", 0, 0, "X_vertical"),
                   _spec("B", 1, 0, "X_horizontal")],
                  [("A", "X"), ("B", "Z")], {"A": "Z", "B": "Z"})
    _verify(exp, c, row=3, wall=True)
    wall = [st for st in exp.system.stabilizers
            if st.get('patch_name') == 'ppm_0']
    caps = [st for st in wall if len(st['data_indices']) == 2]
    assert len(caps) == 1
    assert caps[0]['syn_coord'][1] == 0, \
        f"cap must sit at the free north end, got {caps[0]['syn_coord']}"


def test_wall_step_rejects_bent_schedule():
    with pytest.raises(SeamRuleError, match="bent"):
        _run([_spec("A", 0, 0, "X_horizontal"),
              _spec("B", 0, 1, "X_vertical")],
             [("A", "X"), ("B", "X")], {"A": "X", "B": "X"},
             colour_swapped={"B"}, step_kw={'schedule': 'bent'})


# ── independence from CircLS ─────────────────────────────────────────────────

def test_independent_of_circls():
    # in-process guard (cheap) …
    assert 'circls' not in sys.modules
    # … and the strong form: a fresh interpreter that imports the whole
    # protocol stack must never pull in circls
    code = ("import sys; "
            "import lightstim.protocols.ppm.sequential; "
            "import lightstim.protocols.ppm.coupler; "
            "import lightstim.protocols.ppm.seam_rules; "
            "import lightstim.protocols.ppm.spec; "
            "assert 'circls' not in sys.modules, 'circls leaked'; "
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'ok'
