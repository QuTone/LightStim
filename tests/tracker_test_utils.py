"""Small Stim annotation and GF(2) helpers for tracker unit tests."""
import numpy as np
import stim


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
