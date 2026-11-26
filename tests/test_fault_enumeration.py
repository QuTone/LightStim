"""Probabilities, truncation, and the public experiment entry point."""
import math

import numpy as np
import pytest
import stim

from lightstim.analysis.faults import FaultEnumerator, QuantumLimits
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline


def test_distinct_location_weights_and_small_positive_tail():
    c = stim.Circuit("""
        R 0
        DEPOLARIZE1(0.15) 0
        X_ERROR(0.2) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    experiment = FaultEnumerator(c)
    report = experiment.enumerate(max_order=1)
    assert [r.configurations for r in report.per_order] == [1, 4]
    assert report.accepted_mass == pytest.approx(.97)
    assert report.loss_mass == pytest.approx(.1 * .8 + .2 * .85)
    assert report.omitted_probability == pytest.approx(.03)
    assert report.conditional_loss == pytest.approx(.25 / .97)
    full = experiment.enumerate(max_order=2)
    assert full.conditional_loss == pytest.approx(.1 + .2 - 2 * .1 * .2)
    assert full.full_loss_bounds == pytest.approx((.26, .26))
    assert [r.configurations for r in full.per_order] == [1, 4, 3]
    assert report.full_loss_bounds[0] <= full.conditional_loss <= report.full_loss_bounds[1]
    witness = full.witnesses[-1]
    assert experiment.evaluate(witness.faults) == witness.response
    with pytest.raises(ValueError, match="mutually exclusive"):
        experiment.evaluate(((0, 0), (0, 1)))
    with pytest.raises(ValueError, match="max_configurations"):
        experiment.enumerate(max_order=2, max_configurations=7)

    rare = FaultEnumerator(stim.Circuit("X_ERROR(1e-12) 0 1 2")).enumerate(max_order=2)
    assert math.isclose(rare.omitted_probability, 1e-36, rel_tol=1e-12)


def test_certain_fault_and_zero_acceptance():
    exp = FaultEnumerator("X_ERROR(1) 0\nM 0\nDETECTOR[post-select] rec[-1]")
    truncated = exp.enumerate(max_order=0)
    assert truncated.accepted_mass == 0
    assert truncated.omitted_probability == 1
    assert truncated.full_loss_bounds == (0, 1)
    full = exp.enumerate()
    assert full.omitted_probability == 0
    assert full.conditional_loss is None
    assert full.full_loss_bounds is None


def test_pipeline_and_born_conditioning_share_selection_with_signed_reference():
    text = """X 0
        X_ERROR(.3) 0
        X_ERROR(.25) 1
        M 0 1
        DETECTOR[post-select] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """
    circuit = stim.Circuit(text)
    sim = SimulationPipeline(num_workers=1, print_progress=False)
    cliff = sim.enumerate_faults(circuit)
    assert cliff.accepted_mass == pytest.approx(.75)
    assert cliff.conditional_loss == pytest.approx(.3)
    born = sim.enumerate_faults(text, backend="born",
                                decoder_model=circuit.detector_error_model())
    assert born.accepted_mass == pytest.approx(cliff.accepted_mass)
    assert born.loss_mass == pytest.approx(cliff.loss_mass)
    assert born.conditional_loss == pytest.approx(cliff.conditional_loss)
    override = SimulationPipeline(post_select_detector_indices=[], print_progress=False)
    assert override.enumerate_faults(circuit).accepted_mass == pytest.approx(1.)


def test_true_t_joint_faults_and_probe_loss_through_pipeline():
    # The pre-T X fixes |+>. A post-T X reverses the Y moment even when
    # the same Pauli also occurred before T. The joint loss is 1/2.
    physical = "H 0\nX_ERROR(.02) 0\nT 0\nX_ERROR(.02) 0\nEXP_VAL X0 Y0"
    def infidelity(bits, axes):
        return (1 - sum(axes) / math.sqrt(2)) / 2

    report = SimulationPipeline().enumerate_faults(physical, loss=infidelity)
    assert report.backend == "born"
    assert report.conditional_loss == pytest.approx(.01)
    assert report.per_order[2].loss_mass == pytest.approx(.02 ** 2 / 2)
    exp = FaultEnumerator(physical, loss=infidelity)
    assert exp.evaluate(((0, 0), (1, 0))).accepted_loss == pytest.approx(.5)
    assert not any(not w.faults for w in report.witnesses)
    with pytest.raises(ValueError, match="Born"):
        FaultEnumerator(physical, backend="clifford")
    with pytest.raises(ValueError, match="loss"):
        FaultEnumerator(physical)
    with pytest.raises(ValueError, match="branches"):
        FaultEnumerator("H 0\nT 0\nM 0", limits=QuantumLimits(max_branches=1)).enumerate()


def test_decoder_only_sees_observed_syndromes_and_failure_policy(monkeypatch):
    from lightstim.analysis.faults import _scoring
    seen = []
    class FlaggedDecoder:
        def compile_decoder_for_dem(self, *, dem):
            return self

        def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
            seen.extend(bit_packed_detection_event_data.tolist())
            self.last_flags = np.array([False])
            return np.array([[1]], dtype=np.uint8)

    monkeypatch.setattr(_scoring, "get_decoder", lambda *a, **kw: FlaggedDecoder())
    circuit = "X_ERROR(.25) 0\nM 0\nDETECTOR[post-select] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    def run(policy):
        return SimulationPipeline(
            DecoderConfig("flagged", on_decode_failure=policy),
            post_select_corrected_observable_indices=[0],
        ).enumerate_faults(circuit)

    error = run("error")
    assert error.accepted_mass == pytest.approx(.75)
    assert error.conditional_loss == 1.
    assert run("discard").accepted_mass == 0.
    assert run("ignore").accepted_mass == 0.
    assert seen == [[0], [0], [0]]


def test_multiple_targets_and_raw_postselection():
    circuit = stim.Circuit("""
        X_ERROR(.25) 0
        X_ERROR(.125) 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """)
    sim = SimulationPipeline(post_select_observable_indices=[0], target_observable_indices=[1])
    report = sim.enumerate_faults(circuit)
    assert report.accepted_mass == pytest.approx(.75)
    assert report.conditional_loss == pytest.approx(.125)


def test_model_reference_and_budget_errors_are_explicit():
    sim = SimulationPipeline()
    physical = "H 0\nT 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]"
    with pytest.raises(ValueError, match="decoder_model"):
        sim.enumerate_faults(physical)
    with pytest.raises(ValueError, match="counts"):
        sim.enumerate_faults(physical, decoder_model=stim.DetectorErrorModel("detector D0"))
    with pytest.raises(ValueError, match="indices"):
        FaultEnumerator(physical, post_select_detector_indices=[0])
    with pytest.raises(ValueError, match="reference"):
        FaultEnumerator(physical, observable_reference=[])
    with pytest.raises(ValueError, match="max_order"):
        FaultEnumerator(physical).enumerate(max_order=-1)
    with pytest.raises(ValueError, match="loss"):
        FaultEnumerator(physical, loss=lambda bits, probes: math.nan).enumerate()


def test_explicit_decoder_failure_policy_with_no_observables(monkeypatch):
    from lightstim.analysis.faults import _scoring

    class FailedDecoder:
        def compile_decoder_for_dem(self, *, dem):
            return self

        def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
            self.last_flags = np.array([False])
            return np.zeros((1, 0), dtype=np.uint8)

    monkeypatch.setattr(_scoring, "get_decoder", lambda *args, **kwargs: FailedDecoder())
    model = stim.DetectorErrorModel("detector D0")
    for phase in ("", "T 0\n"):
        circuit = phase + "M 0\nDETECTOR rec[-1]"
        run = SimulationPipeline(DecoderConfig("flagged", on_decode_failure="discard"))
        assert run.enumerate_faults(circuit, decoder_model=model).accepted_mass == 0
        run = SimulationPipeline(DecoderConfig("flagged", on_decode_failure="error"))
        assert run.enumerate_faults(circuit, decoder_model=model).conditional_loss == 1
