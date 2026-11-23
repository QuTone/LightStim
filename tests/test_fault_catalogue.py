"""Circuit-level mechanism identities and independent Stim signature checks."""

import math
import numpy as np
import pytest
import stim

from lightstim.analysis.faults import FaultEnumerator
from lightstim.analysis.faults._catalogue import catalogue
from lightstim.analysis.faults._clifford import clifford_signatures


def test_physical_locations_retain_exclusive_and_invisible_alternatives():
    flat = stim.Circuit("""
        R 3 2 1 0
        REPEAT 2 {
            DEPOLARIZE2(1) 3 2 1 0
            TICK
        }
        M 0 1 2 3
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """).flattened()
    locations = catalogue(flat)
    assert [(f.instruction_index, f.target_start, f.qubits) for f in locations] == [
        (1, 0, (3, 2)), (1, 2, (1, 0)), (3, 0, (3, 2)), (3, 2, (1, 0)),
    ]
    for location, signatures in zip(locations, clifford_signatures(flat, locations)):
        assert location.probability == 1
        assert len(location.alternatives) == len(signatures) == 15
        for alternative, signature in zip(location.alternatives, signatures):
            assert alternative.probability == pytest.approx(1 / 15)
            by_qubit = dict(alternative.paulis)
            assert signature == (
                int(by_qubit.get(2) in ("X", "Y")) | int(by_qubit.get(3) in ("X", "Y")) << 2
            )


def test_signed_readout_fault_only_flips_record_and_drives_feedback():
    # The first result controls a correction. Re-measurement verifies that
    # corrupting a record never changes the qubit's physical postmeasurement state.
    flat = stim.Circuit("""
        R 0 1
        M(1) !0 0
        CX rec[-2] 1
        M 1 0
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-4] rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-3]
    """)
    locations = catalogue(flat)
    assert [f.target_start for f in locations] == [0, 1]
    assert all(f.alternatives[0].paulis == () for f in locations)
    signatures = clifford_signatures(flat, locations)
    assert signatures == ((3,), (4,))
    # With both faults forced, Stim reports the XOR of both masks.
    events, observables = flat.compile_detector_sampler().sample(32, separate_observables=True)
    assert np.all(events) and np.all(observables)


@pytest.mark.parametrize("name,targets", [("DEPOLARIZE1", [0]), ("DEPOLARIZE2", [0, 1])])
def test_depolarization_preserves_physical_probability_near_one(name, targets):
    p = math.nextafter(1., 0.)
    circuit = stim.Circuit()
    circuit.append(name, targets, p)
    (location,) = catalogue(circuit)
    assert location.probability == p
    circuit = stim.Circuit()
    circuit.append(name, targets, math.nextafter(0., 1.))
    with pytest.raises(ValueError, match="too small"):
        catalogue(circuit)


@pytest.mark.parametrize("operation,targets,starts", [
    ("MPP", "Z3*Z2 Z0 !Z1*Z0*Z3", [0, 3, 4]),
    ("MZZ", "3 1 !1 2 !0 3", [0, 2, 4]),
    ("MR", "3 !3 0", [0, 1, 2]),
])
def test_each_measurement_result_is_an_independent_location(operation, targets, starts):
    flat = stim.Circuit(f"""
        R 0 1 2 3
        {operation}(.25) {targets}
        DETECTOR rec[-3] rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """)
    locations = catalogue(flat)
    assert [f.target_start for f in locations] == starts
    assert [f.probability for f in locations] == [.25] * 3
    assert clifford_signatures(flat, locations) == ((1,), (1,), (4,))


def test_pauli_channel_sparse_probabilities_and_location_order():
    flat = stim.Circuit("""
        R 3 1
        PAULI_CHANNEL_2(0,0,0,0,0,.2,.3,0,0,0,0,0,0,0,.5) 3 1
        M 3 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    (location,) = catalogue(flat)
    assert [a.probability for a in location.alternatives] == [.2, .3, .5]
    assert [a.paulis for a in location.alternatives] == [
        ((3, "X"), (1, "Y")), ((3, "X"), (1, "Z")), ((3, "Z"), (1, "Z")),
    ]
    assert clifford_signatures(flat, (location,)) == ((3, 2, 0),)


@pytest.mark.parametrize("noise, expected_fraction", [
    ("DEPOLARIZE1(1e-90) 0", 2 / 3),
    ("DEPOLARIZE2(1e-90) 0 1", 8 / 15),
    ("PAULI_CHANNEL_1(0,1e-90,0) 0", 1),
    ("PAULI_CHANNEL_2(0,0,0,1e-90,0,0,0,0,0,0,0,0,0,0,0) 0 1", 1),
    ("X_ERROR(1e-90) 0", 1),
])
def test_tiny_positive_probabilities_retain_physical_signatures(noise, expected_fraction):
    audit = FaultEnumerator(f"{noise}\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    for i, alternative in enumerate(audit.locations[0].alternatives):
        flips_readout = dict(alternative.paulis).get(0) in ("X", "Y")
        assert audit.evaluate(((0, i),)).accepted_loss == int(flips_readout)
    assert math.isclose(audit.enumerate(max_order=1).loss_mass, expected_fraction * 1e-90)


def test_signature_probe_preserves_distinct_offsets_and_readout_rates():
    c = stim.Circuit("""
        DEPOLARIZE1(.1) 0
        DEPOLARIZE1(.2) 1 2
        DEPOLARIZE1(0) 0
        M(.3) 0
        M(1e-90) 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        OBSERVABLE_INCLUDE(1) rec[-2]
        OBSERVABLE_INCLUDE(2) rec[-1]
    """)
    locations = catalogue(c)
    assert clifford_signatures(c, locations) == (
        (1, 1, 0), (2, 2, 0), (4, 4, 0), (1,), (2,), (4,),
    )
    assert [(loc.instruction_index, loc.target_start) for loc in locations] == [
        (0, 0), (1, 0), (1, 1), (3, 0), (4, 0), (4, 1),
    ]


@pytest.mark.parametrize("declaration", ["DETECTOR", "OBSERVABLE_INCLUDE(0)"])
def test_random_baseline_rejected_even_for_empty_catalogue(declaration):
    flat = stim.Circuit(f"H 0\nM 0\n{declaration} rec[-1]")
    with pytest.raises(ValueError, match="deterministic"):
        clifford_signatures(flat, catalogue(flat))


@pytest.mark.parametrize("noise", [
    "E(.1) X0", "HERALDED_ERASE(.1) 0", "MPAD(.25) 0", "I_ERROR(.01) 0",
])
def test_unsupported_noise_explicitly_rejected(noise):
    with pytest.raises(ValueError, match="Unsupported noise instruction"):
        catalogue(stim.Circuit(noise))
