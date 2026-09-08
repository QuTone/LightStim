"""Statistical accounting for error-target paper reproduction."""
import numpy as np
import pytest
from scipy.integrate import quad

from benchmarks.decoding.ionq_beam_search_low_ler import confidence_sequence, summarize


@pytest.mark.parametrize('errors,shots', [(0,0),(0,100),(100,100),(1,100),(20,5000000)])
def test_confidence_sequence_contains_estimate_and_is_symmetric(errors, shots):
    low, high = confidence_sequence(errors, shots)
    assert 0 <= low <= high <= 1
    if shots:
        assert low <= errors/shots <= high
    flipped = confidence_sequence(shots-errors, shots)
    np.testing.assert_allclose([low, high], [1-flipped[1], 1-flipped[0]], atol=1e-12)


def test_confidence_sequence_inverts_beta_mixture_likelihood():
    # Integrate independently of the implementation's betaln expression.
    errors, shots, alpha = 3, 10, 0.025
    for p in confidence_sequence(errors, shots, alpha):
        mixture, _ = quad(
            lambda q: (q/p)**errors * ((1-q)/(1-p))**(shots-errors)
            / (np.pi*np.sqrt(q*(1-q))), 0, 1)
        assert mixture == pytest.approx(1/alpha, rel=1e-7)


def test_balanced_paper_metric_counts_errors_not_invalid_corrections():
    row = {'job_id': 0, 'bases': {}}
    for basis, errors, invalid in [('X',2,3), ('Z',4,5)]:
        row['bases'][basis] = dict(shots=10000, errors=errors,
            invalid_corrections=invalid, strict_policy_errors=invalid+errors,
            upstream_compared_shots=16, upstream_prediction_mismatches=0)
    result=summarize([row])
    assert result['logical_errors'] == 6
    assert result['paper_ler_per_round'] == pytest.approx((2/10000+4/10000)/12)
    assert result['bases']['X']['invalid_corrections'] == 3


def test_out_of_order_partial_checkpoint_has_no_anytime_interval():
    row = {'job_id': 1, 'bases': {b: dict(shots=100, errors=0,
        invalid_corrections=0, strict_policy_errors=0,
        upstream_compared_shots=16, upstream_prediction_mismatches=0) for b in 'XZ'}}
    partial = summarize([row])
    assert partial['complete_job_prefix'] is False
    assert partial['paper_ler_95_confidence_sequence'] is None
    complete = summarize([{'job_id': 0, 'bases': row['bases']}, row])
    assert complete['complete_job_prefix'] is True
    assert complete['paper_ler_95_confidence_sequence'] is not None
