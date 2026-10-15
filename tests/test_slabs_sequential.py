"""Declared sequential PPM outputs must remain outside the detector space."""
from contextlib import redirect_stdout
from io import StringIO

import pytest
import stim

from lightstim.protocols.rotated_surface_ppm import (
    RotatedSurfacePPMExperiment, RotatedSurfacePPMStep,
)
from lightstim.qec_code.surface_code.rotated.ppm import (
    RotatedSurfacePatchPlacement, origin_of,
)
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _contains, _full_record_parities, _mask,
)


def _sequential_case(sequence, initials, rounds):
    experiment = RotatedSurfacePPMExperiment(
        [RotatedSurfacePatchPlacement(name, origin_of(x, 0, 3, seam=True), 3,
                                     "X_horizontal")
         for name, x in [("A", 0), ("B", 2), ("C", 4)]],
        [RotatedSurfacePPMStep([(name, "Z") for name in pair],
                               route=[(1 if pair == "AB" else 3, 0)])
         for pair in sequence],
        initial_states=dict(zip("ABC", initials)),
        final_measure_states=dict.fromkeys("ABC", "Z"),
        rounds=rounds, rounds_init=1, logical_tracking="slabs",
    )
    with redirect_stdout(StringIO()):
        circuit = experiment.build()
    return experiment, circuit


def _validate_outputs(experiment, circuit):
    physical, detectors, observables = _annotations(circuit)
    complete = _full_record_parities(physical)
    full_basis = _basis(complete)
    emitted = detectors + observables
    assert all(_contains(full_basis, p) for p in emitted)
    assert len(_basis(emitted)) == len(full_basis)
    assert physical.has_all_flows(
        [stim.Flow(measurements=list(_bits(p))) for p in emitted], unsigned=False)
    circuit.detector_error_model()

    # Read declared protocol outputs independently of D/OBS labels. Products
    # of individually random outcomes may still be deterministic targets.
    last_z, record = {}, 0
    for instruction in physical:
        if instruction.name in ("M", "MZ"):
            for offset, target in enumerate(instruction.targets_copy()):
                last_z[target.value] = record + offset
        record += instruction.num_measurements
    targets = [_mask(experiment.ppm_outcomes[i].records_post_split)
               for i in range(len(experiment.ppm_sequence))]
    for name in "ABC":
        operator = next(op for op in experiment.system.logical_ops
                        if op["patch_name"] == name and op["type"] == "Z")
        targets.append(_mask(last_z[q] for q in operator["pauli"]))
    deterministic = []
    for subset in range(1, 1 << len(targets)):
        parity = 0
        for index in _bits(subset):
            parity ^= targets[index]
        if _contains(full_basis, parity):
            deterministic.append(parity)
    expected = len(_basis(deterministic))
    d_rank = len(_basis(detectors))
    assert len(_basis(detectors + deterministic)) - d_rank == expected
    assert len(_basis(emitted)) - d_rank == expected
    assert len(_basis(emitted + deterministic)) - d_rank == expected
    assert circuit.num_observables == expected
    experiment.tracker.validate_logical_count(context="completed S/L/L_absorbed sequence")
    assert experiment.tracker.expected_num_logicals == 0
    return dict(detector_rank=d_rank, logical_rank=expected,
                full_parity_rank=len(full_basis), measurements=record,
                full_space=True, protected_output_space=True, signed_flows=True)


@pytest.mark.parametrize("sequence", [("AB", "BC"), ("BC", "AB"), ("BC", "AB", "BC")])
@pytest.mark.parametrize("initials", ["ZZZ", "ZXZ", "XXX"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_sequential_ppm_protects_all_deterministic_output_relations(sequence, initials, rounds):
    experiment, circuit = _sequential_case(sequence, initials, rounds)
    result = _validate_outputs(experiment, circuit)
    if initials == "ZZZ":
        assert result["logical_rank"] == len(sequence) + 3
