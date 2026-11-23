"""Bounded Born-branch propagation, adapted from Magic-State-Carving's
`magic_state_carving.evaluation.quantum` fixed-fault evaluator.

This module evaluates *joint* physical fault configurations.  The caller removes
stochastic noise and inserts the selected Pauli gates before calling it.  It
never combines final detector or logical signatures with XOR.  Sparse complex
amplitudes and exhaustive Born-rule measurement branches retain interference
through arbitrary sequences of the supported Clifford and T gates.  ``exact``
here means no Monte Carlo or fault-order approximation; arithmetic is floating
point, with a declared roundoff tolerance.

Supported instructions: I, X/Y/Z, H, S/S_DAG, SQRT_X/SQRT_X_DAG,
SQRT_Y/SQRT_Y_DAG, H_XY/H_YZ, CX/CY/CZ (including rec-controlled feedback),
SWAP, T/T_DAG, R/RX/RY, M/MX/MY, MR/MRX/MRY, MPP, MXX/MYY/MZZ, MPAD, EXP_VAL, REPEAT,
and Stim's detector/observable/coordinate/TICK annotations.  Stochastic channels,
sweep controls and other instructions are rejected, never silently ignored.
Reset measurements have hidden branches but do not append measurement records.

This is a small-circuit correctness backend, not an efficient general simulator.
Explicit branch, amplitude, instruction and work guards raise before returning
any partial result.  There is no dense allocation indexed by the largest qubit
number.  Protocol-specific backends can reuse an evaluated prefix via snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping, Sequence

import stim


T_TAG = "ENUM_T"
T_DAG_TAG = "ENUM_T_DAG"
EXP_VAL_TAG = "ENUM_EXP_VAL"
_PHYSICAL_NAMES = {T_TAG: "T", T_DAG_TAG: "T_DAG", EXP_VAL_TAG: "EXP_VAL"}
_ANNOTATIONS = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
_NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
          "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "E",
          "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"}


class QuantumEnumerationLimitError(ValueError):
    """Exact branch evaluation exceeded an explicit resource budget."""


@dataclass(frozen=True)
class QuantumLimits:
    max_branches: int = 4096
    max_amplitudes: int = 65536  # per pure-state branch
    max_total_amplitudes: int = 1_000_000  # all simultaneously live branches
    max_work: int = 20_000_000  # amplitude visits; a deterministic effort guard
    max_instructions: int = 10000
    roundoff_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("max_branches", "max_amplitudes", "max_total_amplitudes",
                     "max_work", "max_instructions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.roundoff_tolerance < 1e-6:
            raise ValueError("roundoff_tolerance must be in (0, 1e-6)")


@dataclass(frozen=True)
class QuantumBranch:
    """One resolved record trajectory, with its conditional pure state.

    Probability includes all branches leading to this snapshot. ``records``
    contains actual measurement bits (including target inversions), not detector
    events. Expectations are evaluated when each EXP_VAL instruction occurs.
    Sparse-state integer bits index ``qubit_ids`` rather than physical IDs.
    The full state is ``sparse_state`` tensored with ``product_factors`` at the
    indicated physical IDs; those factor positions are zero in sparse_state.
    Last-use measured qubits are factored out without discarding their state.
    Treat snapshots as immutable; the evaluator copies them before use.
    """
    records: tuple[bool, ...]
    probability: float
    expectations: tuple[float, ...]
    sparse_state: Mapping[int, complex] = field(repr=False)
    qubit_ids: tuple[int, ...] = ()
    product_factors: Mapping[int, tuple[complex, complex]] = field(default_factory=dict, repr=False)


@dataclass
class _Branch:
    records: tuple[bool, ...]
    probability: float
    expectations: tuple[float, ...]
    state: dict[int, complex]
    factors: dict[int, tuple[complex, complex]] = field(default_factory=dict)


class _Budget:
    def __init__(self, limits: QuantumLimits):
        self.limits = limits
        self.work = 0

    def charge(self, count: int) -> None:
        self.work += count
        if self.work > self.limits.max_work:
            raise QuantumEnumerationLimitError(
                f"Quantum branch work exceeds max_work={self.limits.max_work}; "
                "no partial probability sum was returned")

    def state(self, state: Mapping[int, complex]) -> None:
        if len(state) > self.limits.max_amplitudes:
            raise QuantumEnumerationLimitError(
                f"Pure-state support exceeds max_amplitudes={self.limits.max_amplitudes}")

    def branches(self, branches: Sequence[_Branch]) -> None:
        if len(branches) > self.limits.max_branches:
            raise QuantumEnumerationLimitError(
                f"Measurement branches exceed max_branches={self.limits.max_branches}")
        self.total_support(sum(len(b.state) for b in branches))
        for branch in branches:
            self.state(branch.state)

    def total_support(self, count: int) -> None:
        if count > self.limits.max_total_amplitudes:
            raise QuantumEnumerationLimitError(
                "Live sparse-state support exceeds "
                f"max_total_amplitudes={self.limits.max_total_amplitudes}")


def parse_quantum_circuit(
    text: str | stim.Circuit, *, max_instructions: int = 10000, max_targets: int = 20_000_000,
) -> stim.Circuit:
    """Bound repeat expansion and preserve extended physical gates while parsing.

    T/T_DAG use S[ENUM_T]/S[ENUM_T_DAG]; EXP_VAL uses MPP[ENUM_EXP_VAL].
    The phase evaluator restores their physical meanings before executing the
    instruction. The reserved tags are rejected in user input.
    Repeat expansion is bounded *before* flattening, including nested repeats.
    """
    if type(max_instructions) is not int or max_instructions < 1:
        raise ValueError("max_instructions must be a positive integer")
    if type(max_targets) is not int or max_targets < 1:
        raise ValueError("max_targets must be a positive integer")
    native = isinstance(text, stim.Circuit)
    if native:
        circuit = text
    elif isinstance(text, str):
        if re.search(r"\[(?:ENUM_T|ENUM_T_DAG|ENUM_EXP_VAL)\]", text):
            raise ValueError("Input uses a reserved quantum-enumeration parser tag")
        pattern = re.compile(
            r"(?m)(?P<prefix>^[ \t]*|(?<=[{}])[ \t]*)"
            r"(?P<name>T_DAG|T|EXP_VAL)(?:\[[^\]\n]*\])?(?=[ \t(]|$)")
        replacement = {"T": f"S[{T_TAG}]", "T_DAG": f"S[{T_DAG_TAG}]",
                       "EXP_VAL": f"MPP[{EXP_VAL_TAG}]"}
        encoded = pattern.sub(lambda m: m["prefix"] + replacement[m["name"]], text)
        circuit = stim.Circuit(encoded)
    else:
        raise TypeError("Expected physical circuit text or a stim.Circuit")

    def size(block: stim.Circuit) -> tuple[int, int]:
        operations = 0
        targets = 0
        for op in block:
            if native and op.tag in _PHYSICAL_NAMES:
                raise ValueError("Input uses a reserved quantum-enumeration parser tag")
            if isinstance(op, stim.CircuitRepeatBlock):
                body_ops, body_targets = size(op.body_copy())
                operations += op.repeat_count * max(1, body_ops)
                targets += op.repeat_count * body_targets
            else:
                target_count = len(op.targets_copy())
                if op.tag in _PHYSICAL_NAMES and target_count == 0:
                    raise ValueError(f"{_PHYSICAL_NAMES[op.tag]} requires at least one target")
                operations += 1
                targets += target_count
            if operations > max_instructions:
                raise QuantumEnumerationLimitError(
                    f"Expanded circuit exceeds max_instructions={max_instructions}")
            if targets > max_targets:
                raise QuantumEnumerationLimitError(
                    f"Expanded circuit exceeds max_targets={max_targets} (max_work)")
        return operations, targets

    size(circuit)
    return circuit.flattened()


def _single(state: dict[int, complex], q: int, matrix, budget: _Budget) -> dict[int, complex]:
    budget.charge(2 * len(state))
    mask = 1 << q
    result: dict[int, complex] = {}
    for basis, amplitude in state.items():
        bit = int(bool(basis & mask))
        for output in (0, 1):
            coefficient = matrix[output][bit]
            if coefficient:
                target = (basis | mask) if output else (basis & ~mask)
                result[target] = result.get(target, 0j) + coefficient * amplitude
    result = {k: v for k, v in result.items() if v != 0}
    budget.state(result)
    return result


def _pauli(state: dict[int, complex], terms: Sequence[tuple[int, str]],
           budget: _Budget, sign: int = 1) -> dict[int, complex]:
    budget.charge(len(state) * max(1, len(terms)))
    flip = sum(1 << q for q, p in terms if p in "XY")
    result: dict[int, complex] = {}
    for basis, amplitude in state.items():
        phase: complex = sign
        for q, pauli in terms:
            bit = bool(basis & (1 << q))
            if pauli == "Y":
                phase *= -1j if bit else 1j
            elif pauli == "Z" and bit:
                phase = -phase
        result[basis ^ flip] = phase * amplitude
    return result


def _expect(state: dict[int, complex], terms: Sequence[tuple[int, str]],
            budget: _Budget, sign: int = 1) -> float:
    acted = _pauli(state, terms, budget, sign)
    value = sum(state.get(k, 0j).conjugate() * v for k, v in acted.items())
    tolerance = budget.limits.roundoff_tolerance
    if abs(value.imag) > tolerance or abs(value.real) > 1 + tolerance:
        raise ValueError("Invalid Hermitian expectation from quantum branch state")
    return min(1., max(-1., float(value.real)))


def _measure(branch: _Branch, terms: Sequence[tuple[int, str]], budget: _Budget,
             *, sign: int = 1, record: bool = True) -> list[tuple[_Branch, bool]]:
    acted = _pauli(branch.state, terms, budget, sign)
    results = []
    keys = branch.state.keys() | acted.keys()
    budget.charge(2 * len(keys))
    for bit in (False, True):
        eigenvalue = -1 if bit else 1
        projected = {k: (branch.state.get(k, 0j) + eigenvalue * acted.get(k, 0j)) / 2
                     for k in keys}
        projected = {k: v for k, v in projected.items() if v != 0}
        probability = math.fsum(abs(v) ** 2 for v in projected.values())
        # Squared roundoff, not an importance/probability cutoff. Deterministic
        # projectors otherwise create spurious ~1e-32 branches after T gates.
        if probability <= budget.limits.roundoff_tolerance ** 2:
            continue
        scale = math.sqrt(probability)
        state = {k: v / scale for k, v in projected.items()}
        budget.state(state)
        results.append((_Branch(branch.records + ((bit,) if record else ()),
                                branch.probability * probability,
                                branch.expectations, state, dict(branch.factors)), bit))
    return results


def _products(targets: Sequence[stim.GateTarget], index: Mapping[int, int]):
    groups = []
    terms = []
    sign = 1
    joined = False
    for target in targets:
        if target.is_combiner:
            joined = True
            continue
        if terms and not joined:
            groups.append((terms, sign))
            terms, sign = [], 1
        pauli = "X" if target.is_x_target else "Y" if target.is_y_target else "Z" if target.is_z_target else None
        if pauli is None:
            raise ValueError("MPP/EXP_VAL requires Pauli-product targets")
        q = index[target.value]
        if any(old == q for old, _ in terms):
            raise ValueError("Repeated qubit in a Pauli product is unsupported")
        terms.append((q, pauli))
        if target.is_inverted_result_target:
            sign = -sign
        joined = False
    if terms:
        groups.append((terms, sign))
    return groups


def _matrices():
    h = 1 / math.sqrt(2)
    return {
        "I": ((1, 0), (0, 1)), "X": ((0, 1), (1, 0)),
        "Y": ((0, -1j), (1j, 0)), "Z": ((1, 0), (0, -1)),
        "H": ((h, h), (h, -h)), "S": ((1, 0), (0, 1j)),
        "S_DAG": ((1, 0), (0, -1j)),
        "T": ((1, 0), (0, complex(h, h))),
        "T_DAG": ((1, 0), (0, complex(h, -h))),
        "SQRT_X": (((1+1j)/2, (1-1j)/2), ((1-1j)/2, (1+1j)/2)),
        "SQRT_X_DAG": (((1-1j)/2, (1+1j)/2), ((1+1j)/2, (1-1j)/2)),
        "SQRT_Y": ((h, -h), (h, h)), "SQRT_Y_DAG": ((h, h), (-h, h)),
        "H_XY": ((0, complex(h, -h)), (complex(h, h), 0)),
        "H_YZ": ((h, -1j*h), (1j*h, -h)),
    }


_MATRICES = _matrices()


def evaluate_fixed_circuit(text: str | stim.Circuit, *, limits: QuantumLimits | None = None,
                           initial_branches: Sequence[QuantumBranch] | None = None
                           ) -> tuple[QuantumBranch, ...]:
    """Exhaust all measurement branches for one fixed joint-fault circuit.

    The returned probabilities sum to the input probability mass, normally one.
    No decoder or postselection is applied here. Those use ``records`` in the
    shared caller, so the simulator cannot leak the selected faults to a decoder.
    EXP_VAL probes do not collapse the quantum state. For an evaluated prefix,
    pass its complete returned branch tuple as ``initial_branches``.
    """
    limits = limits or QuantumLimits()
    parsed = parse_quantum_circuit(text, max_instructions=limits.max_instructions,
                                   max_targets=limits.max_work)
    return _evaluate_parsed(parsed, limits, initial_branches)


def _initial_state(parsed, initial_branches, budget):
    """Resume pure-state branches and remap sparse qubit IDs for this circuit."""
    initial = tuple(initial_branches) if initial_branches is not None else (
        QuantumBranch((), 1., (), {0: 1.}, ()),)
    if not initial:
        raise ValueError("initial_branches must not be empty")
    qubits = {q for branch in initial for q in branch.qubit_ids}
    last_use: dict[int, int] = {}
    for instruction_index, op in enumerate(parsed):
        for target in op.targets_copy():
            if target.is_qubit_target or target.is_x_target or target.is_y_target or target.is_z_target:
                if op.name != "MPAD":
                    qubits.add(target.value)
                    if op.name not in _ANNOTATIONS:
                        last_use[target.value] = instruction_index
    qubit_ids = tuple(sorted(qubits))
    index = {q: i for i, q in enumerate(qubit_ids)}
    branches = []
    total_support = 0
    for branch in initial:
        if not math.isfinite(branch.probability) or branch.probability <= 0:
            raise ValueError("Initial branch probability must be finite and positive")
        if len(set(branch.qubit_ids)) != len(branch.qubit_ids):
            raise ValueError("Initial branch qubit IDs must be unique")
        remapped = {}
        for basis, amplitude in branch.sparse_state.items():
            if basis < 0 or basis >= (1 << len(branch.qubit_ids)):
                raise ValueError("Initial sparse-state basis is outside qubit_ids")
            destination = sum(1 << index[q] for i, q in enumerate(branch.qubit_ids) if basis & (1 << i))
            remapped[destination] = complex(amplitude)
        norm = math.fsum(abs(v) ** 2 for v in remapped.values())
        if not math.isfinite(norm) or abs(norm - 1) > budget.limits.roundoff_tolerance:
            raise ValueError("Each initial pure state must have unit norm")
        factors = dict(branch.product_factors)
        for q, vector in tuple(factors.items()):
            if (q not in branch.qubit_ids or len(vector) != 2
                    or not math.isclose(math.fsum(abs(v)**2 for v in vector), 1.0,
                                        rel_tol=0.0, abs_tol=budget.limits.roundoff_tolerance)):
                raise ValueError("Initial product factor must be a normalized single-qubit state")
            mask = 1 << index[q]
            if any(basis & mask for basis in remapped):
                raise ValueError("A product-factor position must be zero in sparse_state")
            if q in last_use:
                # The prefix snapshot preserved this final measured state as a
                # factor. Materialize it only when the continuation uses it.
                budget.charge(2 * len(remapped))
                remapped = {basis | (mask if bit else 0): amplitude * coefficient
                            for basis, amplitude in remapped.items()
                            for bit, coefficient in enumerate(vector) if coefficient != 0}
                del factors[q]
                budget.state(remapped)
        branches.append(_Branch(tuple(branch.records), branch.probability,
                                tuple(branch.expectations), remapped, factors))
        total_support += len(remapped)
        budget.total_support(total_support)
    budget.branches(branches)
    return branches, qubit_ids, index, last_use


def _paired_gate(name, targets, branches, index, budget):
    """Apply a two-qubit gate or a Pauli controlled by a reported record."""
    for a, b in zip(targets[::2], targets[1::2]):
        for branch in branches:
            if a.is_measurement_record_target or b.is_measurement_record_target:
                if b.is_measurement_record_target:
                    if name != "CZ":
                        raise ValueError(f"Unsupported record target in {name}")
                    a, b = b, a
                if not b.is_qubit_target:
                    raise ValueError("Feedback requires one record and one qubit")
                if not -len(branch.records) <= a.value < 0:
                    raise ValueError("Feedback record index precedes available measurements")
                if branch.records[a.value]:
                    branch.state = _single(branch.state, index[b.value], _MATRICES[name[1]], budget)
                continue
            if not a.is_qubit_target or not b.is_qubit_target:
                raise ValueError("Only qubit and measurement-record controls are supported")
            qa, qb = index[a.value], index[b.value]
            ma, mb = 1 << qa, 1 << qb
            budget.charge(len(branch.state))
            result = {}
            for basis, amplitude in branch.state.items():
                aa, bb = bool(basis & ma), bool(basis & mb)
                target_basis, phase = basis, 1
                if name == "SWAP" and aa != bb:
                    target_basis ^= ma | mb
                elif name == "CX" and aa:
                    target_basis ^= mb
                elif name == "CY" and aa:
                    target_basis ^= mb
                    phase = -1j if bb else 1j
                elif name == "CZ" and aa and bb:
                    phase = -1
                result[target_basis] = phase * amplitude
            branch.state = result


def _factor_measured_qubit(child, q, physical_id, final_pauli, final_bit):
    """Separate a known last-use eigenstate, keeping both factors for continuation."""
    h = 1 / math.sqrt(2)
    if final_pauli == "Z":
        vector = (0j, 1+0j) if final_bit else (1+0j, 0j)
        choose = int(final_bit)
    else:
        second = (-h if final_bit else h) * (1j if final_pauli == "Y" else 1)
        vector, choose = (complex(h), complex(second)), 0
    mask = 1 << q
    child.state = {basis & ~mask: amplitude / vector[choose]
                   for basis, amplitude in child.state.items()
                   if bool(basis & mask) == bool(choose)}
    child.factors[physical_id] = vector


def _measure_qubits(name, targets, branches, index, budget, last_use, instruction_index):
    output_pauli = "X" if name.endswith("X") else "Y" if name.endswith("Y") else "Z"
    reset = name.startswith("R") or name.startswith("MR")
    record = name.startswith("M")
    # Unobserved reset implements trace-out followed by a fresh state.
    # Use Z-reset then a basis rotation, so RX/RY on virgin |0> do not
    # manufacture two identical hidden branches. Recorded MRX/MRY must
    # retain their actual X/Y measurement instruments.
    pauli = "Z" if reset and not record else output_pauli
    last_target = {target.value: i for i, target in enumerate(targets)}
    for target_index, target in enumerate(targets):
        q = index[target.value]
        expanded = []
        total_support = 0
        for branch in branches:
            sign = -1 if target.is_inverted_result_target else 1
            for child, bit in _measure(branch, [(q, pauli)], budget, sign=sign, record=record):
                physical_bit = bool(bit) ^ target.is_inverted_result_target
                if reset and physical_bit:
                    correction = "Z" if pauli == "X" else "X"
                    child.state = _single(child.state, q, _MATRICES[correction], budget)
                if reset and not record and output_pauli != "Z":
                    child.state = _single(child.state, q, _MATRICES["H"], budget)
                    if output_pauli == "Y":
                        child.state = _single(child.state, q, _MATRICES["S"], budget)
                if (last_use[target.value] == instruction_index
                        and last_target[target.value] == target_index):
                    final_bit = False if reset else physical_bit
                    _factor_measured_qubit(child, q, target.value, output_pauli, final_bit)
                expanded.append(child)
                total_support += len(child.state)
                if total_support > budget.limits.max_total_amplitudes or len(expanded) > budget.limits.max_branches:
                    budget.branches(expanded)
        branches = expanded
        budget.branches(branches)
    return branches


def _measure_products(name, targets, branches, index, budget):
    if name in {"MXX", "MYY", "MZZ"}:
        products = [
            ([(index[a.value], name[1]), (index[b.value], name[1])],
             -1 if a.is_inverted_result_target ^ b.is_inverted_result_target else 1)
            for a, b in zip(targets[::2], targets[1::2])
        ]
    else:
        products = _products(targets, index)
    for terms, sign in products:
        if name == "EXP_VAL":
            for branch in branches:
                branch.expectations += (_expect(branch.state, terms, budget, sign),)
        else:
            expanded = []
            total_support = 0
            for branch in branches:
                for child, _ in _measure(branch, terms, budget, sign=sign):
                    expanded.append(child)
                    total_support += len(child.state)
                    if (total_support > budget.limits.max_total_amplitudes
                            or len(expanded) > budget.limits.max_branches):
                        budget.branches(expanded)
            branches = expanded
            budget.branches(branches)
    return branches


def _evaluate_parsed(parsed: stim.Circuit, limits: QuantumLimits,
                     initial_branches: Sequence[QuantumBranch] | None = None) -> tuple[QuantumBranch, ...]:
    """Propagate a validated, flattened physical circuit with phase markers intact."""
    budget = _Budget(limits)
    if len(parsed) > limits.max_instructions:
        raise QuantumEnumerationLimitError(
            f"Fixed circuit exceeds max_instructions={limits.max_instructions}")
    branches, qubit_ids, index, last_use = _initial_state(parsed, initial_branches, budget)
    initial_mass = math.fsum(b.probability for b in branches)

    for instruction_index, op in enumerate(parsed):
        name = _PHYSICAL_NAMES.get(op.tag, op.name)
        targets = op.targets_copy()
        if name in _ANNOTATIONS:
            continue
        if name in _NOISE:
            raise ValueError(f"Fixed-fault quantum evaluator rejects stochastic instruction {name}; "
                             "remove random noise and insert the selected deterministic faults")
        args = op.gate_args_copy()
        if args and (name not in {"M", "MX", "MY", "MR", "MRX", "MRY", "MPP", "MPAD",
                                  "MXX", "MYY", "MZZ"} or any(args)):
            raise ValueError(f"Noisy/parameterized {name} is not supported in a fixed-fault circuit")
        if name in _MATRICES:
            for target in targets:
                if not target.is_qubit_target or target.is_inverted_result_target:
                    raise ValueError(f"{name} requires ordinary qubit targets")
                total_support = sum(len(b.state) for b in branches)
                for branch in branches:
                    total_support -= len(branch.state)
                    branch.state = _single(branch.state, index[target.value], _MATRICES[name], budget)
                    total_support += len(branch.state)
                    budget.total_support(total_support)
        elif name in {"CX", "CY", "CZ", "SWAP"}:
            _paired_gate(name, targets, branches, index, budget)
        elif name in {"R", "RX", "RY", "M", "MX", "MY", "MR", "MRX", "MRY"}:
            branches = _measure_qubits(
                name, targets, branches, index, budget, last_use, instruction_index)
        elif name in {"MPP", "MXX", "MYY", "MZZ", "EXP_VAL"}:
            branches = _measure_products(name, targets, branches, index, budget)
        elif name == "MPAD":
            values = tuple(bool(t.value) for t in targets)
            for branch in branches:
                branch.records += values
        elif name != "II":
            raise ValueError(f"Unsupported fixed-fault quantum instruction {name}")
        budget.branches(branches)
    total = math.fsum(b.probability for b in branches)
    if abs(total - initial_mass) > limits.roundoff_tolerance * max(1., initial_mass):
        raise ValueError(f"Quantum branch probability mass was not conserved: {total} != {initial_mass}")
    return tuple(QuantumBranch(b.records, b.probability, b.expectations,
                               dict(b.state), qubit_ids, dict(b.factors)) for b in branches)
