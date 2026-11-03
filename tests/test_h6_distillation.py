"""Correctness regressions for the native H6 magic-state distillation protocol.

See lightstim/protocols/h6_distillation.py for the protocol itself. These
tests pin down properties that were wrong during development and caught only
by actually running the circuit:

- rounds=0 (no pre-check SE round) is NOT fault tolerant (distance 1);
  rounds>=1 is required to reach the paper's distance-2 guarantee.
- every detector must carry the "post-select" tag, since H6 is a
  distance-2 code that can only detect, never correct, a fault -- feeding
  its own stabilizer detectors to a real decoder degrades the observed
  scaling from O(p^2) to O(p).
- variant "f"'s mixed-basis final readout used to silently drop H6's own
  XA/XB closure detectors (fault distance 1, not 2) until the dedicated
  Fig. 6 "Y2" flagged ancilla gadget was added to read Y_L1 non-destructively
  before the final MX readout; both variants are now checked for fault
  distance 2.
"""
import numpy as np
import pytest
import stim

from lightstim.protocols.h6_distillation import (
    build_h6_distillation_circuit,
    inject_noise,
    run_simulation,
)
from lightstim.utils.fault_audit import low_weight_fault_audit


def _flattened_detectors(circuit: stim.Circuit):
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            yield inst


@pytest.mark.parametrize("variant", ["e", "f"])
def test_noiseless_circuit_is_deterministic(variant):
    circuit, info, _ = build_h6_distillation_circuit(variant=variant)
    # variant "f" consumes slot 1's logical value in the Fig. 6 Y2 gadget, so
    # only X_L0 survives as an observable (X_L1 is no longer well-defined).
    assert info["num_observables"] == (2 if variant == "e" else 1)
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(2000, separate_observables=True)
    assert not det.any()
    assert not obs.any()


@pytest.mark.parametrize("variant", ["e", "f"])
def test_detector_error_model_builds_cleanly(variant):
    for rounds in (0, 1, 2):
        circuit, _, _ = build_h6_distillation_circuit(rounds=rounds, variant=variant)
        noisy = inject_noise(circuit, p=1e-3)
        noisy.detector_error_model()  # raises on gauge/inconsistent detectors


@pytest.mark.parametrize("variant", ["e", "f"])
def test_every_detector_is_post_selected(variant):
    circuit, _, _ = build_h6_distillation_circuit(rounds=1, variant=variant)
    tags = [inst.tag for inst in _flattened_detectors(circuit)]
    assert tags
    assert all(tag == "post-select" for tag in tags)


@pytest.mark.parametrize(
    "variant, expected_detectors, expected_observables", [("e", 8, 2), ("f", 10, 1)]
)
def test_circuit_shape(variant, expected_detectors, expected_observables):
    """Pins the detector/observable counts down: variant "f"'s dedicated
    Fig. 6 Y2 gadget adds 2 post-select detectors (the syndrome + flag
    ancilla) over variant "e", and collapses X_L1 out of the observable set
    since Y_L1 is measured directly instead."""
    circuit, info, _ = build_h6_distillation_circuit(rounds=1, variant=variant)
    assert info["num_detectors"] == expected_detectors
    assert info["num_observables"] == expected_observables


def test_zero_rounds_is_not_fault_tolerant():
    """Documents the known limitation: no pre-check SE round means the
    code's own X-stabilizers never get checked, so a single fault can flip
    a logical observable undetected."""
    circuit, _, _ = build_h6_distillation_circuit(rounds=0)
    noisy = inject_noise(circuit, p=1e-3)
    assert len(noisy.shortest_graphlike_error()) == 1


@pytest.mark.parametrize("variant", ["e", "f"])
@pytest.mark.parametrize("rounds", [1, 2])
def test_default_rounds_reach_fault_distance_two(rounds, variant):
    circuit, _, _ = build_h6_distillation_circuit(rounds=rounds, variant=variant)
    noisy = inject_noise(circuit, p=1e-3)
    assert len(noisy.shortest_graphlike_error()) == 2


def test_prep_fig5_requires_variant_e():
    with pytest.raises(ValueError, match="only supported for variant='e'"):
        build_h6_distillation_circuit(variant="f", prep="fig5")


def test_prep_fig5_is_deterministic_and_post_selected():
    circuit, info, _ = build_h6_distillation_circuit(rounds=1, prep="fig5")
    assert info["num_observables"] == 2
    # 2 extra post-select detectors over prep="fig1d" (the Fig. 5 flag pair).
    assert info["num_detectors"] == 10
    tags = [inst.tag for inst in _flattened_detectors(circuit)]
    assert tags and all(tag == "post-select" for tag in tags)
    dets, obs = circuit.compile_detector_sampler().sample(2000, separate_observables=True)
    assert not dets.any() and not obs.any()


@pytest.mark.parametrize("rounds", [1, 2])
def test_prep_fig5_reaches_fault_distance_two(rounds):
    circuit, _, _ = build_h6_distillation_circuit(rounds=rounds, prep="fig5")
    noisy = inject_noise(circuit, p=1e-3)
    assert len(noisy.shortest_graphlike_error()) == 2


def test_flagged_vs_unflagged_fault_audit():
    """Directly answers the design-review acceptance gate ("audit all single
    faults... compare flagged and unflagged circuits with the same noise
    model and acceptance rule", docs/design/h6_core_integration.md on
    origin/h6-msd): both prep paths reach fault distance 2, by different
    mechanisms -- prep="fig1d" via the post-hoc SE round + Bell-pair check,
    prep="fig5" via its own preparation-time flags -- certified exactly
    (every low-weight fault enumerated), not just via shortest_graphlike_error's
    single-number DEM check."""
    for prep in ("fig1d", "fig5"):
        circuit, _, _ = build_h6_distillation_circuit(rounds=1, variant="e", prep=prep)
        result = low_weight_fault_audit(circuit)
        assert result.has_circuit_fault_distance_two, f"prep={prep}: {result.summary()}"


@pytest.mark.slow
def test_reproduces_paper_quadratic_scaling():
    """Post-selected logical error rate should scale ~p^2 (arXiv:2506.14688
    Fig. 4; also matches the independently validated H6-Distillation-DEQ
    reference and the ~2.0-2.06 slopes recorded for this code family's own
    memory scaling in docs/design/h6_core_integration.md)."""
    ps = np.array([0.002, 0.004, 0.008, 0.016])
    lers = []
    for p in ps:
        circuit, _, _ = build_h6_distillation_circuit(rounds=1)
        noisy = inject_noise(circuit, p=float(p))
        stats = run_simulation(noisy, max_shots=300_000, max_errors=300, num_workers=4)
        assert stats.logical_error_rate > 0
        lers.append(stats.logical_error_rate)
    slope, _ = np.polyfit(np.log(ps), np.log(lers), 1)
    assert 1.5 < slope < 2.5
