"""Physical S/L/L_absorbed acceptance using the existing unrotated circuit blocks.

These tests do not classify tracker rows. Stim flows and independent physical
logical-Pauli insertions check the emitted detector/observable relation spaces.
"""
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedTwoPatchCoupler,
)

def _bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def _mask(indices):
    result = 0
    for index in indices:
        result ^= 1 << int(index)
    return result


def _basis(rows):
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return pivots


def _contains(pivots, row):
    while row:
        pivot = row.bit_length() - 1
        if pivot not in pivots:
            return False
        row ^= pivots[pivot]
    return True


def _annotations(circuit):
    physical = stim.Circuit()
    detectors, observables = [], {}
    measured = 0
    for instruction in circuit.flattened():
        if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            mask = _mask(measured + target.value for target in instruction.targets_copy())
            if instruction.name == "DETECTOR":
                detectors.append(mask)
            else:
                index = int(instruction.gate_args_copy()[0])
                observables[index] = observables.get(index, 0) ^ mask
        elif instruction.name != "QUBIT_COORDS":
            # Keep TICKs, so adjacent readout calls cannot merge across a cut.
            physical.append(instruction)
        measured += instruction.num_measurements
    return physical, detectors, list(observables.values())


def _full_record_parities(physical):
    """Eliminate all input/output Paulis from independent Stim flow rows."""
    pivots, parities = {}, []
    n = physical.num_qubits
    for flow in physical.flow_generators():
        endpoints = 0
        for end, pauli in enumerate((flow.input_copy(), flow.output_copy())):
            xs, zs = pauli.to_numpy()
            endpoints ^= _mask(2 * end * n + q for q in np.flatnonzero(xs))
            endpoints ^= _mask((2 * end + 1) * n + q for q in np.flatnonzero(zs))
        records = _mask(flow.measurements_copy())
        while endpoints:
            pivot = endpoints.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = endpoints, records
                break
            donor, donor_records = pivots[pivot]
            endpoints ^= donor
            records ^= donor_records
        if not endpoints:
            parities.append(records)
    return list(_basis(parities).values())


def _physical_case(initial, final, episodes, rounds, combined, *, explicit_deactivate=False, mode="slabs"):
    a = UnrotatedSurfaceCode(distance=3)
    a.rotate_coords(np.pi)
    a.reset_rotation_angle()
    system = QECSystem()
    system.add_patch(a, name="A")
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="B", offset=(0, 6))
    system.register_coupler(
        UnrotatedTwoPatchCoupler(), patch_names=["A", "B"],
        name="AB", interaction_type="ZZ",
    )
    tracker = SyndromeTracker(
        system.num_qubits, expected_num_logicals=2,
        **({"logical_tracking": mode} if mode != "legacy" else {}),
    )
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    owner = system.index_to_owner_map
    patch_data = {q for q in system.data_indices if owner[q] in ("A", "B")}
    corridor = {q for q in system.data_indices if owner[q] == "AB"}
    bases = dict(zip(("A", "B"), initial))
    builder.initialize({q: bases[owner[q]] for q in patch_data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)

    operators = []
    for operator in system.logical_ops:
        if operator["patch_name"] not in ("A", "B"):
            continue
        pauli = stim.PauliString(system.num_qubits)
        for qubit, factor in operator["pauli"].items():
            pauli[qubit] = factor
        operators.append(pauli)
    cuts = []

    def cut():
        prefix, _, _ = _annotations(builder.circuit)
        cuts.append(prefix.copy())

    cut()
    for episode in range(episodes):
        builder.activate_coupler("AB")
        builder.initialize({q: "X" for q in corridor}, n=system.num_qubits)
        builder.apply_syndrome_extraction(
            UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=rounds,
        )
        if combined and episode == episodes - 1:
            # Exercise the native combined readout: the real system still has
            # an active coupler when the corridor and patches are measured.
            if explicit_deactivate:
                builder.deactivate_coupler("AB")
            break
        builder.deactivate_coupler("AB")
        builder.apply_data_readout({q: "X" for q in corridor}, resolve_absorbed=False)
        cut()

    bases = dict(zip(("A", "B"), final))
    readout = {q: bases[owner[q]] for q in patch_data}
    if combined:
        readout.update({q: "X" for q in corridor})
    builder.apply_data_readout(readout)
    circuit = builder.to_stim_circuit()
    physical, _, _ = _annotations(circuit)
    if combined:
        indices = [
            index for index, instruction in enumerate(physical)
            if instruction.name == "MX"
            and {target.value for target in instruction.targets_copy()} == corridor
        ]
        assert indices
        cuts.append(physical[:indices[-1] + 1])
    for prefix in cuts:
        assert physical[:len(prefix)] == prefix
    return circuit, cuts, operators


def _validate_physical_relations(circuit, cuts, expected_logicals):
    """Compare annotations with the complete record kernel and stage responses.

    This uses Stim's physical flow generators, not any tracker role metadata.
    A detector is exactly a record parity with zero response to all admissible
    stage logical insertions; observables must span the remaining quotient.
    """
    physical, detectors, observables = _annotations(circuit)
    circuit.detector_error_model()

    full = _full_record_parities(physical)
    full_basis = _basis(full)
    assert all(_contains(full_basis, row) for row in detectors + observables)
    assert len(_basis(detectors + observables)) == len(full)
    assert len(full) == physical.count_determined_measurements()
    assert len(_basis(detectors + observables)) - len(_basis(detectors)) == expected_logicals

    reference = physical.reference_sample()

    def parity(row, sample):
        return sum(bool(sample[index]) for index in _bits(row)) % 2

    assert not any(parity(row, reference) for row in detectors + observables)
    full_responses, observable_responses = [], []
    flows = [stim.Flow(measurements=list(_bits(row))) for row in full]
    for prefix, operators in cuts:
        assert physical[:len(prefix)] == prefix
        for pauli in operators:
            modified = prefix.copy()
            for qubit in pauli.pauli_indices():
                modified.append("_XYZ"[pauli[qubit]], [qubit])
            modified += physical[len(prefix):]
            assert modified.has_all_flows(flows, unsigned=True)
            changed = modified.reference_sample()

            def response(row):
                return parity(row, reference) ^ parity(row, changed)

            assert not any(response(row) for row in detectors)
            full_responses.append(_mask(i for i, row in enumerate(full) if response(row)))
            observable_responses.append(
                _mask(i for i, row in enumerate(observables) if response(row))
            )
    assert len(_basis(full_responses)) == expected_logicals
    assert len(_basis(observable_responses)) == expected_logicals
    return {
        "measurements": physical.num_measurements,
        "qubits": physical.num_qubits,
        "detectors": len(detectors),
        "observable_labels": len(observables),
        "record_kernel_rank": len(full),
        "determined_measurements": physical.count_determined_measurements(),
        "detector_rank": len(_basis(detectors)),
        "annotation_rank": len(_basis(detectors + observables)),
        "observable_quotient_rank": len(full) - len(_basis(detectors)),
        "logical_response_rank": len(_basis(full_responses)),
        "observable_response_rank": len(_basis(observable_responses)),
        "logical_probes": len(full_responses),
        "detector_response_rank": 0,
        "valid_signed_relations": True,
        "complete_record_space": True,
        "detector_space_equals_stage_kernel": True,
        "noiseless_dem_valid": True,
    }

CASES = [
    ("00_split", "ZZ", "ZZ", 1, 1, False, 3),
    ("00_combined", "ZZ", "ZZ", 1, 1, True, 3),
    ("00_two_merges", "ZZ", "ZZ", 2, 1, False, 4),
    ("00_repeat_se", "ZZ", "ZZ", 1, 2, False, 3),
    ("0plus_teleport", "ZX", "XZ", 1, 1, False, 1),
    ("0plus_final_zz", "ZX", "ZZ", 1, 1, False, 2),
    ("plusplus_final_zz", "XX", "ZZ", 1, 1, False, 1),
]


@pytest.mark.parametrize("name,initial,final,episodes,rounds,combined,expected", CASES,
                         ids=[case[0] for case in CASES])
def test_slabs_physical_acceptance(name, initial, final, episodes, rounds, combined, expected):
    circuit, cuts, operators = _physical_case(initial, final, episodes, rounds, combined)
    _validate_physical_relations(circuit, [(cut, operators) for cut in cuts], expected)


def _native_case(initial, final, rounds, mode="slabs"):
    with redirect_stdout(StringIO()):
        experiment = TwoPatchLSExperiment(
            patch1_config={"distance": 3}, patch2_config={"distance": 3},
            offset=(0, 6), interaction_type="ZZ", rounds=rounds,
            initial_state_patch1=initial[0], initial_state_patch2=initial[1],
            measure_state_patch1=final[0], measure_state_patch2=final[1], noise_params=None,
            **({"logical_tracking": mode} if mode != "legacy" else {}),
        )
        return experiment.build()


@pytest.mark.parametrize("initial,final,expected", [
    ("ZZ", "ZZ", 3), ("ZX", "ZZ", 2), ("XX", "ZZ", 1),
    ("ZX", "XZ", 1), ("XX", "XZ", 0),
])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_slabs_existing_two_patch_experiment(initial, final, expected, rounds):
    circuit = _native_case(initial, final, rounds)
    physical, detectors, observables = _annotations(circuit)
    full = _full_record_parities(physical)
    assert len(_basis(detectors + observables)) == len(full)
    assert len(full) - len(_basis(detectors)) == expected
    circuit.detector_error_model()
