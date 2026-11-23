"""Compile physical fault alternatives to deterministic Clifford flip signatures."""

import stim

from ._catalogue import PAULI_NOISE_GATES, READOUT_GATES, FaultLocation


def _signature_probe(flat: stim.Circuit) -> stim.Circuit:
    # A forced fault's signature depends only on its location and outcome.
    # Stim 1.15's explainer can drop depolarizing outcomes at small rates,
    # and mislabel sparse PAULI_CHANNEL_2 outcomes. Probe at ordinary rates
    # and keep the original physical catalogue for counts and weights.
    probe = stim.Circuit()
    for index, op in enumerate(flat):
        if op.name in PAULI_NOISE_GATES or op.name in READOUT_GATES:
            args = op.gate_args_copy()
            if any(args):
                probe_args = ([1 / 64] * 15 if op.name == "PAULI_CHANNEL_2"
                              else [1 / 64 if p else 0 for p in args])
                # Keep instruction offsets when adjacent channels acquire equal rates.
                probe.append(op.name, op.targets_copy(), probe_args, tag=str(index))
                continue
        probe.append(op)
    return probe


def clifford_signatures(
    flat: stim.Circuit, locations: tuple[FaultLocation, ...]
) -> tuple[tuple[int, ...], ...]:
    """Return the D-then-L bit mask for each location and alternative.

    Stim's explanations map individual physical mechanisms to detector and
    logical flips. Alternatives with no effect retain a zero mask.
    """
    try:
        # The explainer permits gauge detectors; XOR scoring requires that
        # every ideal detector and observable have a deterministic reference.
        flat.without_noise().detector_error_model(allow_gauge_detectors=False)
    except ValueError as exc:
        raise ValueError("Clifford enumeration requires deterministic detectors and observables") from exc
    if not locations:
        return ()

    indices = {
        (location.instruction_index, location.target_start, alternative.paulis): (i, j)
        for i, location in enumerate(locations)
        for j, alternative in enumerate(location.alternatives)
    }
    signatures = [[0] * len(location.alternatives) for location in locations]
    for effect in _signature_probe(flat).explain_detector_error_model_errors():
        mask = 0
        for term in effect.dem_error_terms:
            target = term.dem_target
            if target.is_logical_observable_id():
                mask ^= 1 << (flat.num_detectors + target.val)
            elif target.is_relative_detector_id():
                mask ^= 1 << target.val
            else:
                raise ValueError(f"Unexpected target in Clifford fault explanation: {target}")
        for source in effect.circuit_error_locations:
            start = source.instruction_targets.target_range_start
            instruction_index = source.stack_frames[0].instruction_offset
            paulis = tuple((item.gate_target.value, item.gate_target.pauli_type)
                           for item in source.flipped_pauli_product)
            key = (instruction_index, start, paulis)
            if key not in indices:
                if flat[instruction_index].name == "PAULI_CHANNEL_2":
                    continue  # zero-probability alternative in the original channel
                raise ValueError(f"Stim explained a fault absent from the physical catalogue: {key}")
            i, j = indices[key]
            signatures[i][j] = mask
    return tuple(tuple(row) for row in signatures)
