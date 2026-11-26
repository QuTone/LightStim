"""Evaluate joint physical faults with their actual measurement instruments."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import stim

from ._catalogue import PAULI_NOISE_GATES, READOUT_GATES, FaultLocation
from ._phase import EXP_VAL_TAG, QuantumLimits, _evaluate_parsed


def _record_mask(op: stim.CircuitInstruction, measured: int) -> int:
    mask = 0
    for target in op.targets_copy():
        if not target.is_measurement_record_target:
            raise ValueError(f"Born enumeration requires record-backed {op.name} targets")
        index = measured + target.value
        if not 0 <= index < measured:
            raise ValueError(f"{op.name} references a measurement preceding the circuit")
        mask ^= 1 << index
    return mask


def _parities(records: int, masks: tuple[tuple[int, int], ...]) -> int:
    return sum(1 << i for i, mask in masks if (records & mask).bit_count() % 2)


class BornEvaluator:
    """Compile record parities once and resolve all Born branches of each fault set.

    Packed detectors use the supplied reference, packed logical observables use
    raw record parities, and EXP_VAL entries are conditional Pauli expectations.
    """
    def __init__(
        self,
        flat: stim.Circuit,
        locations: tuple[FaultLocation, ...],
        *,
        limits: QuantumLimits,
        detector_reference: tuple[bool, ...],
    ):
        self.flat = flat
        self.locations = locations
        self.limits = limits
        detectors = []
        observables = defaultdict(int)
        measured = 0
        for op in flat:
            if op.tag == EXP_VAL_TAG:
                continue
            if op.name == "DETECTOR":
                detectors.append(_record_mask(op, measured))
            elif op.name == "OBSERVABLE_INCLUDE":
                observable = int(op.gate_args_copy()[0])
                observables[observable] ^= _record_mask(op, measured)
            else:
                measured += op.num_measurements
        if len(detector_reference) != len(detectors):
            raise ValueError("detector_reference must contain one bit per detector")
        self._detectors = tuple(enumerate(detectors))
        self._observables = tuple(observables.items())
        self._reference = sum(int(bit) << i for i, bit in enumerate(detector_reference))

    def responses(
        self, chosen: tuple[tuple[int, int], ...]
    ) -> Iterator[tuple[float, int, int, tuple[float, ...]]]:
        paulis = defaultdict(list)
        flips = defaultdict(list)
        for location_index, alternative_index in chosen:
            location = self.locations[location_index]
            alternative = location.alternatives[alternative_index]
            if alternative.paulis:
                paulis[location.instruction_index].extend(alternative.paulis)
            else:
                flips[location.instruction_index].append(location.target_start)

        fixed = stim.Circuit()
        for index, op in enumerate(self.flat):
            if op.name in PAULI_NOISE_GATES:
                for qubit, axis in paulis.get(index, ()):
                    fixed.append(axis, [qubit])
            elif op.name in READOUT_GATES and op.tag != EXP_VAL_TAG:
                targets = op.targets_copy()
                for target_start in flips.get(index, ()):
                    # Invert only the *reported* eigenvalue. The eigenspaces,
                    # postmeasurement state and any MR reset are preserved.
                    targets[target_start] = stim.target_inv(targets[target_start])
                fixed.append(op.name, targets, tag=op.tag)
            else:
                fixed.append(op)

        for branch in _evaluate_parsed(fixed, self.limits):
            records = sum(1 << i for i, bit in enumerate(branch.records) if bit)
            syndrome = _parities(records, self._detectors) ^ self._reference
            yield branch.probability, syndrome, _parities(records, self._observables), branch.expectations
