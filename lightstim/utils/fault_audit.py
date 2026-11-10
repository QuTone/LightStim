"""Low-weight fault audit for annotated ``stim`` circuits.

Given a circuit that already carries ``DETECTOR`` / ``OBSERVABLE_INCLUDE``
annotations and is deterministic under noiseless execution,
:func:`low_weight_fault_audit` inserts low-weight Pauli faults and records, for
each, whether it

* fires at least one detector  -> *detected*,
* fires no detector but flips at least one observable -> *undetected-logical*,
* fires no detector and flips no observable -> *undetected-harmless*.

Two fault classes are enumerated:

* **weight 1** -- every single-qubit Pauli (``X``, ``Y``, ``Z``) after every
  operation, on every qubit that operation touches;
* **weight 2 on two-qubit gates** (optional, on by default) -- every
  ``P_a ⊗ P_b`` with ``P`` in ``{X, Y, Z}`` after every two-qubit gate, on its
  operand pair. These are the correlated hook faults a ``DEPOLARIZE2`` channel
  would sample.

A circuit with **no undetected-logical fault at either weight** has circuit
fault distance ``>= 2``: under post-selection on the detectors, its conditional
logical-error rate is ``O(p^2)``. This is the correctness criterion the Magic-H6
roadmap uses for the flagged / post-selected circuits, checked exactly rather
than by a Monte-Carlo slope fit.

**Atomisation.** ``stim`` merges consecutive same-name gates into one
instruction (a whole transversal layer, or a whole ported encoder, can collapse
to a single ``CX``). A fault inserted *after* such a merged instruction is not
where a per-gate noise channel actually sits, and would spread through none of
the merged gate's siblings. So the audit first rewrites the circuit into
one-gate-per-instruction form (``TICK``-separated so ``stim`` cannot re-merge),
then probes between every physical gate. This matches how ``NoiseInjector``
places circuit-level noise and keeps the audit's verdict consistent with a
Monte-Carlo slope fit.

The probe is a probability-1 ``{X,Y,Z}_ERROR`` channel: ``stim`` references
detectors and observables to the *noiseless* run, so a deterministic prob-1
error reveals exactly the fault's syndrome and logical action. A weight-2 probe
is two such channels, one per operand.

The audit is code-agnostic -- it only needs an annotated circuit -- so it lives
in :mod:`lightstim.utils` rather than under any one code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import stim

_PAULI_TO_CHANNEL = {"X": "X_ERROR", "Y": "Y_ERROR", "Z": "Z_ERROR"}
_SKIP_INSTRUCTIONS = frozenset(
    {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"}
)
# Single-qubit Clifford gates whose merged layers are split for probing.
_ONE_QUBIT_GATES = frozenset(
    {
        "H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
        "SQRT_Y", "SQRT_Y_DAG", "H_XY", "H_YZ", "H_XZ", "C_XYZ", "C_ZYX", "I",
    }
)
# Two-qubit Clifford gates whose operand pair can take a correlated hook fault.
_TWO_QUBIT_GATES = frozenset(
    {
        "CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",
        "ZCX", "ZCY", "ZCZ", "SWAP", "ISWAP", "ISWAP_DAG", "CXSWAP", "SWAPCX",
        "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG",
        "SQRT_ZZ", "SQRT_ZZ_DAG",
    }
)


@dataclass(frozen=True)
class FaultRecord:
    """One inserted Pauli fault and the syndrome it produced."""

    instruction_index: int
    gate: str
    qubits: Tuple[int, ...]
    paulis: Tuple[str, ...]
    detectors_fired: Tuple[int, ...]
    observables_flipped: Tuple[int, ...]

    @property
    def weight(self) -> int:
        return len(self.qubits)

    # Back-compatible single-fault accessors.
    @property
    def qubit(self) -> int:
        return self.qubits[0]

    @property
    def pauli(self) -> str:
        return "⊗".join(self.paulis) if len(self.paulis) > 1 else self.paulis[0]

    @property
    def detected(self) -> bool:
        return bool(self.detectors_fired)

    @property
    def undetected_logical(self) -> bool:
        return not self.detectors_fired and bool(self.observables_flipped)


@dataclass
class FaultAuditResult:
    """Summary of :func:`low_weight_fault_audit` over a circuit."""

    num_locations: int
    audited_max_weight: int
    records: List[FaultRecord] = field(default_factory=list)

    @property
    def undetectable_logical_faults(self) -> List[FaultRecord]:
        return [r for r in self.records if r.undetected_logical]

    @property
    def undetected_harmless_faults(self) -> List[FaultRecord]:
        return [
            r for r in self.records
            if not r.detectors_fired and not r.observables_flipped
        ]

    @property
    def min_undetectable_logical_weight(self) -> Optional[int]:
        weights = [r.weight for r in self.undetectable_logical_faults]
        return min(weights) if weights else None

    @property
    def has_circuit_fault_distance_two(self) -> bool:
        """True iff no fault up to the audited weight flips a logical undetected.

        Requires the weight-2-on-two-qubit-gates sweep to have run
        (``audited_max_weight >= 2``); a weight-1-only audit cannot certify it.
        """
        return (
            self.audited_max_weight >= 2
            and self.min_undetectable_logical_weight is None
        )

    def summary(self) -> str:
        n = len(self.records)
        det = sum(r.detected for r in self.records)
        uh = len(self.undetected_harmless_faults)
        ul = self.undetectable_logical_faults
        by_w = {}
        for r in ul:
            by_w[r.weight] = by_w.get(r.weight, 0) + 1
        ul_str = ", ".join(f"{c} at weight {w}" for w, c in sorted(by_w.items())) or "0"
        # Every probe is a single fault *location* (a weight-2 probe is one
        # two-qubit gate taking a correlated Pauli), so any undetected-logical
        # hit means the circuit tolerates zero faults -> distance 1.
        if self.has_circuit_fault_distance_two:
            dist = ">= 2"
        elif self.audited_max_weight < 2 and not ul:
            dist = "1 or 2 (weight-2 sweep not run)"
        else:
            dist = "1"
        return (
            f"{n} fault probes (max weight {self.audited_max_weight}) over "
            f"{self.num_locations} atomic locations: {det} detected, {uh} "
            f"undetected-harmless, undetected-logical: {ul_str} "
            f"-> circuit fault distance {dist}"
        )


def atomise(circuit: stim.Circuit) -> stim.Circuit:
    """Rewrite ``circuit`` to one physical gate per instruction.

    Merged single- and two-qubit Clifford layers are split, with a ``TICK``
    after each piece so ``stim`` does not re-merge them on append. Resets,
    measurements, noise channels and annotations pass through untouched. The
    detector/observable behaviour is unchanged (``TICK`` is inert to sampling).
    """
    out = stim.Circuit()
    for inst in circuit.flattened():
        if not isinstance(inst, stim.CircuitInstruction):
            out.append(inst)
            continue
        targets = inst.targets_copy()
        args = inst.gate_args_copy()
        if inst.name in _ONE_QUBIT_GATES and len(targets) > 1:
            for t in targets:
                out.append(inst.name, [t], args)
                out.append("TICK")
        elif inst.name in _TWO_QUBIT_GATES and len(targets) > 2:
            for i in range(0, len(targets), 2):
                out.append(inst.name, targets[i:i + 2], args)
                out.append("TICK")
        else:
            out.append(inst)
    return out


def _flattened_instructions(circuit: stim.Circuit) -> List[stim.CircuitInstruction]:
    return [
        inst
        for inst in circuit.flattened()
        if isinstance(inst, stim.CircuitInstruction)
    ]


def _probe(base, pos, inserts):
    """Rebuild ``base`` with prob-1 error channels ``inserts`` after index ``pos``.

    ``inserts`` is a list of ``(channel_name, qubit)`` pairs. Returns
    ``(detectors_fired, observables_flipped)`` as sorted index tuples.
    """
    probe = stim.Circuit()
    for i, inst in enumerate(base):
        probe.append(inst)
        if i == pos:
            for channel, qubit in inserts:
                probe.append(channel, [qubit], 1.0)
    dets, obs = probe.compile_detector_sampler().sample(1, separate_observables=True)
    return (
        tuple(np.flatnonzero(dets[0]).tolist()),
        tuple(np.flatnonzero(obs[0]).tolist()),
    )


def low_weight_fault_audit(
    circuit: stim.Circuit, *, two_qubit_gate_faults: bool = True
) -> FaultAuditResult:
    """Enumerate low-weight Pauli faults in ``circuit`` (after :func:`atomise`).

    Args:
        circuit: A detector/observable-annotated circuit that samples all-zero
            detectors and observables under noiseless execution.
        two_qubit_gate_faults: Also enumerate the 9 ``P_a ⊗ P_b`` correlated
            faults after every two-qubit gate. On by default; the resulting
            ``has_circuit_fault_distance_two`` is only meaningful with it on.

    Returns:
        A :class:`FaultAuditResult`.
    """
    atomic = atomise(circuit)
    dets, obs = atomic.compile_detector_sampler().sample(
        64, separate_observables=True
    )
    if dets.any() or obs.any():
        raise ValueError(
            "circuit is not deterministic under noiseless execution "
            "(atomised sample had a firing detector/observable)."
        )

    base = _flattened_instructions(atomic)
    max_weight = 2 if two_qubit_gate_faults else 1
    result = FaultAuditResult(num_locations=0, audited_max_weight=max_weight)

    for pos, inst in enumerate(base):
        if inst.name in _SKIP_INSTRUCTIONS:
            continue
        targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if not targets:
            continue
        result.num_locations += 1

        for qubit in sorted(set(targets)):
            for pauli, channel in _PAULI_TO_CHANNEL.items():
                d, o = _probe(base, pos, [(channel, qubit)])
                result.records.append(
                    FaultRecord(pos, inst.name, (qubit,), (pauli,), d, o)
                )

        if two_qubit_gate_faults and inst.name in _TWO_QUBIT_GATES and len(targets) == 2:
            a, b = targets
            for pa, ca in _PAULI_TO_CHANNEL.items():
                for pb, cb in _PAULI_TO_CHANNEL.items():
                    d, o = _probe(base, pos, [(ca, a), (cb, b)])
                    result.records.append(
                        FaultRecord(pos, inst.name, (a, b), (pa, pb), d, o)
                    )
    return result


def single_fault_audit(circuit: stim.Circuit) -> FaultAuditResult:
    """Weight-1-only :func:`low_weight_fault_audit` (no two-qubit-gate faults)."""
    return low_weight_fault_audit(circuit, two_qubit_gate_faults=False)


__all__ = [
    "FaultRecord",
    "FaultAuditResult",
    "atomise",
    "low_weight_fault_audit",
    "single_fault_audit",
]
