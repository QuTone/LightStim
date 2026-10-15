"""Three-patch PPM classification against an independent logical circuit.

The native experiment splits after every PPM. The oracle knows only three
logical qubits, their R/RX preparations, MPP ZZ outputs, and final M/MX
readouts. Physical annotations and tracker rows do not define expected ranks.
"""
from contextlib import redirect_stdout
from io import StringIO
from itertools import product

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


BASES = tuple("".join(bases) for bases in product("XZ", repeat=3))
SEQUENCES = (("AB", "BC"), ("BC", "AB"), ("AB", "BC", "AB"))


class _BoundaryExperiment(RotatedSurfacePPMExperiment):
    """Capture the real split-code boundaries for independent fault probes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundaries = []

    def _capture_boundary(self):
        physical, _, _ = _annotations(self.builder.circuit)
        operators = {}
        for operator in self.system.logical_ops:
            name, letter = operator["patch_name"], operator["type"]
            if name in "ABC" and letter in "XZ":
                operators[name, letter] = dict(operator["pauli"])
        assert len(operators) == 6
        self.boundaries.append((physical, operators))

    def _init_and_baseline(self, names):
        super()._init_and_baseline(names)
        self._capture_boundary()

    def _apply_ppm_step(self, i, step):
        super()._apply_ppm_step(i, step)
        self._capture_boundary()


def _logical_oracle(sequence, initials, finals):
    oracle = stim.Circuit()
    for qubit, basis in enumerate(initials):
        oracle.append("R" if basis == "Z" else "RX", [qubit])
    oracle.append("TICK")
    boundaries = [oracle.copy()]
    for pair in sequence:
        first, second = ("ABC".index(name) for name in pair)
        oracle.append("MPP", [stim.target_z(first), stim.target_combiner(),
                              stim.target_z(second)])
        oracle.append("TICK")
        boundaries.append(oracle.copy())
    for qubit, basis in enumerate(finals):
        oracle.append("M" if basis == "Z" else "MX", [qubit])
    relations = _full_record_parities(oracle)
    assert oracle.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in relations
    ], unsigned=False)
    return oracle, boundaries, relations


def _physical_case(sequence, initials, finals, rounds):
    experiment = _BoundaryExperiment(
        [RotatedSurfacePatchPlacement(name, origin_of(x, 0, 3, seam=True), 3,
                                     "X_horizontal")
         for name, x in (("A", 0), ("B", 2), ("C", 4))],
        [RotatedSurfacePPMStep([(name, "Z") for name in pair],
                               route=[(1 if pair == "AB" else 3, 0)])
         for pair in sequence],
        initial_states=dict(zip("ABC", initials)),
        final_measure_states=dict(zip("ABC", finals)),
        rounds=rounds, rounds_init=1, logical_tracking="slabs",
    )
    with redirect_stdout(StringIO()):
        circuit = experiment.build()
    return experiment, circuit


def _output_map(experiment, physical, finals):
    last, measured = {}, 0
    for instruction in physical:
        if instruction.name in ("M", "MZ", "MX"):
            basis = "X" if instruction.name == "MX" else "Z"
            for offset, target in enumerate(instruction.targets_copy()):
                last[target.value, basis] = measured + offset
        measured += instruction.num_measurements
    targets = []
    for index in range(len(experiment.ppm_sequence)):
        records = experiment.ppm_outcomes[index].records_post_split
        assert records is not None, f"PPM {index} lost its actual measured outcome"
        targets.append(_mask(records))
    for name, basis in zip("ABC", finals):
        operator = next(op for op in experiment.system.logical_ops
                        if op["patch_name"] == name and op["type"] == basis)
        assert all(letter == basis for letter in operator["pauli"].values())
        targets.append(_mask(last[qubit, basis] for qubit in operator["pauli"]))
    return targets


def _combine(rows, selector):
    result = 0
    for index in _bits(selector):
        result ^= rows[index]
    return result


def _sample_mask(circuit):
    return _mask(index for index, value in enumerate(circuit.reference_sample()) if value)


def validate_case(sequence, initials, finals, rounds, *, physical_factory=None,
                  boundary_indices=None, split_after_each_ppm=True):
    """Return evidence only after full-space, role, and logical-oracle checks."""
    context = dict(sequence=list(sequence), initials=initials, finals=finals,
                   rounds=rounds, split_after_each_ppm=split_after_each_ppm)
    oracle, logical_boundaries, logical_relations = _logical_oracle(
        sequence, initials, finals)
    if boundary_indices is not None:
        logical_boundaries = [logical_boundaries[index] for index in boundary_indices]
    experiment, circuit = (physical_factory or _physical_case)(sequence, initials, finals, rounds)
    physical, detectors, observables = _annotations(circuit)
    complete = _full_record_parities(physical)
    complete_basis = _basis(complete)
    emitted = detectors + observables
    outputs = _output_map(experiment, physical, finals)
    expected = len(logical_relations)

    # Z outputs have a direct physical map. Final X outputs additionally
    # contain split-corridor Pauli-frame records: raw final X parities alone
    # need not satisfy the ideal relation. Their full classification is
    # independently checked by the stage-response spaces below.
    ideal_basis = _basis(logical_relations)
    direct_outputs = _mask(range(len(sequence))) | _mask(
        len(sequence) + index for index, basis in enumerate(finals) if basis == "Z")
    expected_relations, direct_selectors = [], []
    for selector in range(1 << len(outputs)):
        if selector & ~direct_outputs:
            continue
        assert _contains(ideal_basis, selector) == _contains(
            complete_basis, _combine(outputs, selector)), (context, "output relation", selector)
        if selector and _contains(ideal_basis, selector):
            direct_selectors.append(selector)
            expected_relations.append(_combine(outputs, selector))
    direct_rank = len(_basis(expected_relations))
    assert direct_rank == len(_basis(direct_selectors)), (
        context, "distinct ideal output relations collapsed to one physical record parity")
    assert all(_contains(complete_basis, row) for row in emitted), context
    assert len(_basis(emitted)) == len(complete_basis), context
    assert len(complete_basis) == physical.count_determined_measurements(), context
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row)))
        for row in emitted + expected_relations
    ], unsigned=False), context
    circuit.detector_error_model()

    detector_rank = len(_basis(detectors))
    assert len(_basis(detectors + expected_relations)) - detector_rank == direct_rank, context
    assert len(_basis(emitted)) - detector_rank == expected, context
    assert len(_basis(emitted + expected_relations)) - detector_rank == expected, context
    assert circuit.num_observables == expected, context

    # Equal quotient dimensions alone permit D=d+h, O=h. Physical logical
    # insertions additionally require every emitted detector to be pure.
    physical_reference, logical_reference = _sample_mask(physical), _sample_mask(oracle)
    full_responses, observable_responses, ideal_responses = [], [], []
    assert len(experiment.boundaries) == len(logical_boundaries)
    for boundary, ((prefix, operators), logical_prefix) in enumerate(
            zip(experiment.boundaries, logical_boundaries)):
        assert physical[:len(prefix)] == prefix, context
        assert oracle[:len(logical_prefix)] == logical_prefix
        for (patch, letter), operator in operators.items():
            changed = prefix.copy()
            for qubit, factor in operator.items():
                changed.append(factor, [qubit])
            changed += physical[len(prefix):]
            delta = physical_reference ^ _sample_mask(changed)
            logical_changed = logical_prefix.copy()
            logical_changed.append(letter, ["ABC".index(patch)])
            logical_changed += oracle[len(logical_prefix):]
            ideal_delta = logical_reference ^ _sample_mask(logical_changed)

            for index, row in enumerate(detectors):
                assert not (row & delta).bit_count() % 2, (
                    context, "detector logical response", boundary, patch, letter,
                    index, list(_bits(row)))
            for selector, row in zip(direct_selectors, expected_relations):
                assert (selector & ideal_delta).bit_count() % 2 == (row & delta).bit_count() % 2, (
                    context, "direct Z-output response", boundary, patch, letter,
                    list(_bits(selector)), list(_bits(row)))
            full_responses.append(_mask(index for index, row in enumerate(complete)
                                        if (row & delta).bit_count() % 2))
            observable_responses.append(_mask(index for index, row in enumerate(observables)
                                              if (row & delta).bit_count() % 2))
            ideal_responses.append(_mask(index for index, row in enumerate(logical_relations)
                                         if (row & ideal_delta).bit_count() % 2))
    assert len(_basis(full_responses)) == expected, context
    assert len(_basis(observable_responses)) == expected, context
    # Compare complete response signatures over the same logical faults at
    # every protocol boundary, not just their dimensions. This handles the
    # physical split-frame records without asking tracker rows for a frame.
    def response_space(responses, count):
        return _basis(_mask(probe for probe, response in enumerate(responses)
                            if response & (1 << relation))
                      for relation in range(count))
    ideal_space = response_space(ideal_responses, expected)
    physical_space = response_space(full_responses, len(complete))
    assert len(ideal_space) == len(physical_space) == expected, context
    assert all(_contains(ideal_space, row) for row in physical_space.values()), (
        context, "ideal/physical stage-response space")
    experiment.tracker.validate_logical_count(context="three-patch oracle matrix")
    assert experiment.tracker.expected_num_logicals == 0, context
    if initials == finals == "ZZZ":
        assert expected == len(sequence) + 3
    return dict(context, measurements=physical.num_measurements,
                logical_rank=expected, detector_rank=detector_rank,
                full_parity_rank=len(complete_basis), logical_probes=len(full_responses),
                full_space=True, signed_flows=True, direct_z_output_space=True,
                pure_detector_space=True, ideal_stage_responses=True)


@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.parametrize("initials", BASES)
@pytest.mark.parametrize("finals", BASES)
@pytest.mark.parametrize("rounds", [1, 3])
def test_three_patch_ppm_matches_independent_logical_oracle(sequence, initials, finals, rounds):
    validate_case(sequence, initials, finals, rounds)
