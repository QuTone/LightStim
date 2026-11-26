"""Independent physical fault locations in a flattened circuit."""

from dataclasses import dataclass
from itertools import product
from math import fsum

import stim


PAULI_NOISE_GATES = frozenset({
    "X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
})
READOUT_GATES = frozenset({
    "M", "MX", "MY", "MR", "MRX", "MRY", "MPP", "MXX", "MYY", "MZZ",
})
_TWO_QUBIT_PAULIS = tuple(a + b for a, b in product("IXYZ", repeat=2)
                          if (a, b) != ("I", "I"))


@dataclass(frozen=True)
class FaultAlternative:
    """One mutually exclusive outcome with its unconditional probability.

    A Pauli outcome lists its non-identity factors in circuit-target order.
    An empty Pauli tuple denotes a classical flip of a measurement result.
    """

    probability: float
    paulis: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FaultLocation:
    """One positive-rate stochastic channel on one target group.

    `instruction_index` indexes the flattened circuit and `target_start` is
    the index in that instruction's raw targets (including MPP combiners).
    """

    instruction_index: int
    target_start: int
    gate: str
    qubits: tuple[int, ...]
    alternatives: tuple[FaultAlternative, ...]
    # Keep the channel's supplied total. Summing rounded p/15 values can turn
    # a depolarizing probability just below one into an exactly certain fault.
    probability: float


def _channel_outcomes(name, args):
    if name.endswith("_ERROR"):
        return ((name[0], args[0]),)
    labels = _TWO_QUBIT_PAULIS if name.endswith("2") else "XYZ"
    if name.startswith("DEPOLARIZE"):
        probability = args[0] / len(labels)
        if probability == 0:
            raise ValueError(f"{name} alternative probability is too small for floating-point arithmetic")
        return tuple((label, probability) for label in labels)
    if fsum(args) > 1:
        raise ValueError(f"Alternative probabilities exceed one in {name}")
    return tuple((label, probability) for label, probability in zip(labels, args) if probability)


def catalogue(flat: stim.Circuit) -> tuple[FaultLocation, ...]:
    """Expand positive-rate physical channels, retaining invisible alternatives."""
    locations = []
    for index, op in enumerate(flat):
        if isinstance(op, stim.CircuitRepeatBlock):
            raise ValueError("catalogue expects a flattened circuit")
        args = op.gate_args_copy()
        if op.name in READOUT_GATES:
            if not any(args):
                continue
            start = 0
            for group in op.target_groups():
                qubits = tuple(t.value for t in group)
                locations.append(FaultLocation(index, start, op.name, qubits,
                                               (FaultAlternative(args[0], ()),), args[0]))
                # Stim strips the joining '*' targets when returning a group.
                start += 2 * len(group) - 1 if op.name == "MPP" else len(group)
        elif op.name in PAULI_NOISE_GATES:
            if not any(args):
                continue
            outcomes = _channel_outcomes(op.name, args)
            probability = fsum(args)
            for group_index, group in enumerate(op.target_groups()):
                qubits = tuple(t.value for t in group)
                alternatives = tuple(
                    FaultAlternative(probability, tuple((q, axis) for q, axis in zip(qubits, label)
                                                        if axis != "I"))
                    for label, probability in outcomes
                )
                locations.append(FaultLocation(index, group_index * len(group), op.name,
                                               qubits, alternatives, probability))
        elif stim.gate_data(op.name).is_noisy_gate or (op.name == "MPAD" and any(args)):
            raise ValueError(f"Unsupported noise instruction for physical fault enumeration: {op.name}")
    return tuple(locations)
