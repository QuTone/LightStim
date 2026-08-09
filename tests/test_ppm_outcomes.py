"""Protocol output vs evaluation observable (review §3).

Every applied PPM exposes its physical measurement-record parity as a
PPMOutcome; whether that parity is deterministic is a property of the
request's relation to the tracked state, not a correctness question.
An anti-commuting outcome is a legitimate free coin (protocol output);
the deterministic quantities the circuit DOES certify (closure
detectors, standing observables) are the evaluation layer's choice.
"""
import contextlib
import io

import numpy as np

from lightstim.protocols.ppm.spec import PatchSpec, PPMStep, origin_of
from lightstim.protocols.ppm.sequential import SequentialPPMExperiment

D = 3


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def _bits(circuit, records, shots=400, seed=5):
    m = circuit.compile_sampler(seed=seed).sample(shots)
    out = np.zeros(shots, dtype=bool)
    for r in records:
        out ^= m[:, r]
    return out


def test_commuting_repeat_outcome_is_reproducible():
    # M(ZZ) twice: the second joint is pinned BEFORE its merge (pre-merge
    # records exist), and the two protocol outputs agree shot by shot — a
    # deterministic correlation between two protocol outputs.
    px = [_spec("A", 0, 0, "X_horizontal"), _spec("B", 2, 0, "X_horizontal")]
    seq = [PPMStep([("A", "Z"), ("B", "Z")], route=[(1, 0)]),
           PPMStep([("A", "Z"), ("B", "Z")], route=[(1, 0)])]
    st = {"A": "Z", "B": "Z"}
    exp = SequentialPPMExperiment(px, seq, initial_states=st,
                                  final_measure_states=st, rounds=D,
                                  rounds_init=1)
    c = _build(exp)
    o0, o1 = exp.ppm_outcomes[0], exp.ppm_outcomes[1]
    assert o0.records_post_split is not None
    assert o1.records_pre_merge is not None, \
        "a commuting repeat is pinned before its own merge"
    assert o1.records_post_split is not None
    b0 = _bits(c, o0.records_post_split)
    b1 = _bits(c, o1.records_post_split)
    assert (b0 == b1).all(), "repeated commuting PPM outcomes must agree"


def test_anticommuting_outcome_is_a_free_coin():
    # A sits on a corner: M(X̄A X̄B) through the E/W seam, then M(Z̄A Z̄C)
    # through the N/S seam — the joints anti-commute at A (X̄A vs Z̄A), and
    # each step is individually parallel-law-legal on its own seam (no
    # rotation needed).  The second outcome is a legitimately random
    # protocol output: no pre-merge parity, per-shot coin after the split
    # — while every correlation the circuit certifies (closures,
    # observables) stays silent at p=0.
    px = [_spec("A", 0, 0, "X_vertical"), _spec("B", 1, 0, "X_vertical"),
          _spec("C", 0, 1, "X_vertical")]
    seq = [PPMStep([("A", "X"), ("B", "X")], route=[]),
           PPMStep([("A", "Z"), ("C", "Z")], route=[])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"A": "X", "B": "X", "C": "Z"},
        final_measure_states={"A": "Z", "B": "X", "C": "Z"},
        rounds=D, rounds_init=1)
    c = _build(exp)
    o_0, o_1 = exp.ppm_outcomes[0], exp.ppm_outcomes[1]
    assert o_0.records_post_split is not None
    assert o_1.records_pre_merge is None, \
        "an anti-commuting joint must NOT be pinned before its merge"
    assert o_1.records_post_split is not None

    coin = _bits(c, o_1.records_post_split)
    assert coin.any() and not coin.all(), \
        "the anti-commuting outcome must be a per-shot coin"

    det = c.compile_detector_sampler(seed=0).sample(1024)
    assert not det.any(), \
        "every certified (evaluation-layer) correlation must stay silent"
