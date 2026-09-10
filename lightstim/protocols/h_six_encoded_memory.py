"""Encoded-state memory experiment for the ``[[6, 2, 2]]`` H-code.

:func:`encoded_memory_circuit` prepares a codeword with a ported Magic-H6
encoder and then runs native syndrome extraction, so LightStim's
``SyndromeTracker`` generates the stabilizer detectors automatically for the
*encoded* circuit (not just the bare-qubit
:class:`~lightstim.protocols.memory.MemoryExperiment` baseline).

``encoder="zero_zero"``  -- the data-CX core of ``get_ft_init_circ``, prepares ``|00>_L``
``encoder=None``          -- no encoder (equivalent to ``MemoryExperiment``)

Every encoder gate is applied as its own unitary block so that
``NoiseInjector`` places one circuit-level channel per physical gate. (Passing
the whole encoder as a single block lets ``stim`` merge its six ``CX`` gates
into one instruction; the injected ``DEPOLARIZE2`` then lands after the *whole*
encoder instead of after each gate, which spuriously reports ``O(p)``.)

``flag_verified=True`` runs the Fig. 5 (arXiv:2506.14688) flag-verified
``|00>_L`` encoder: two ancillas are entangled with the encoder's control
qubits (0 and 2) before the data-CX core and disentangled + measured after, and
the caller **post-selects on the flags reading 0**. The flag ancillas are
reset/measured directly on the builder's circuit (bypassing the tracker, which
rejects a measurement block that follows a data-entangling unitary); the two
flag ``DETECTOR``s are appended explicitly, everything else -- SE rounds,
readout, observables -- is tracker-generated.

Under generic-CSS extraction the non-flagged ``|00>_L`` encoded Z-memory is
already circuit fault distance ``>= 2`` (``O(p^2)``); flag verification matters
for the ``|++>_L`` distillation path (``get_dist_circ`` + the Bell-pair
H-check), which is genuinely distance 1 -- that is roadmap stage 3, on the
``feat/magic-h6-protocol`` branch.

The bare-qubit memory experiment for any code is
:class:`~lightstim.protocols.memory.MemoryExperiment`; this module is the
[[6,2,2]]-specific *encoded* variant, alongside the other experiment drivers in
``lightstim.protocols``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.H_six import HSixCode, HSixExtractionBlock

# Data-CX core of get_ft_init_circ (flag ancillas / verification CX removed):
# prepares |00>_L. Controls 0 and 2 are the encoder "spine". Matches
# ``lightstim.qec_code.H_six.prep_circuits`` (the port of Code614.py).
_ZERO_ZERO_ENCODER_HEAD = (("H", (0, 2)),)
_ZERO_ZERO_ENCODER_CORE = (
    ("CX", (0, 1)), ("CX", (2, 3)),
    ("CX", (0, 4)), ("CX", (2, 5)),
    ("CX", (0, 5)), ("CX", (2, 4)),
)
_FLAG_CONTROLS = (0, 2)  # data labels the flag ancillas hook onto


def _apply_gate_by_gate(builder, gates, data):
    for gate, labels in gates:
        block = stim.Circuit()
        block.append(gate, [data[i] for i in labels])
        builder.apply_unitary_block(block)


def encoded_memory_circuit(
    *,
    basis: str = "Z",
    rounds: int = 2,
    encoder: Optional[str] = "zero_zero",
    flag_verified: bool = False,
) -> Tuple[stim.Circuit, dict]:
    """Builder-native encoded ``[[6, 2, 2]]`` memory experiment.

    Args:
        basis: Memory basis, ``"Z"`` or ``"X"`` (``"Z"`` recommended -- the
            ``zero_zero`` encoder prepares ``|00>_L``).
        rounds: Native syndrome-extraction rounds.
        encoder: ``"zero_zero"`` for a ``|00>_L`` codeword, or ``None`` for the
            bare-qubit baseline.
        flag_verified: Wrap the encoder in the Fig. 5 flag gadget (two extra
            ancillas + two flag ``DETECTOR``s to post-select on). Requires
            ``encoder="zero_zero"``.

    Returns:
        ``(circuit, info)`` -- a clean ``stim.Circuit`` (all-zero detectors and
        observables noiseless, ``num_observables == 2``) and an ``info`` dict.
        For ``flag_verified``, ``info["flag_detector_indices"]`` lists the two
        flag detectors (they are also post-selected as ordinary detectors).
    """
    basis = str(basis).upper()
    if basis not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X'; got {basis!r}")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if flag_verified and encoder != "zero_zero":
        raise ValueError("flag_verified requires encoder='zero_zero'")

    system = QECSystem()
    system.add_patch(
        HSixCode(h_check_ancillas=2 if flag_verified else 0), name="c622"
    )
    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    data = sorted(system.data_indices)
    builder.initialize({q: basis for q in data}, n=system.num_qubits)

    flag_detector_indices: list[int] = []
    if encoder is None:
        pass
    elif encoder == "zero_zero":
        _apply_gate_by_gate(builder, _ZERO_ZERO_ENCODER_HEAD, data)
        if flag_verified:
            patch = system.patches["c622"][0]
            l2g = system.local_to_global_map["c622"]
            flags = sorted(
                l2g[i]
                for i in (
                    patch.syndrome_indices
                    - patch.syndrome_indices_x
                    - patch.syndrome_indices_z
                )
            )
            if len(flags) != 2:
                raise RuntimeError("expected 2 bare flag ancillas")
            a0, a1 = flags
            c0, c1 = (data[_FLAG_CONTROLS[0]], data[_FLAG_CONTROLS[1]])
            builder.circuit.append("R", [a0, a1])
            builder.circuit.append("CX", [c0, a0, c1, a1])          # flag entangle
            _apply_gate_by_gate(builder, _ZERO_ZERO_ENCODER_CORE, data)
            builder.circuit.append("CX", [c0, a0, c1, a1])          # flag disentangle
            builder.circuit.append("M", [a0, a1])
            fx, fy = system.qubit_coords[a0]
            builder.circuit.append("DETECTOR", [stim.target_rec(-2)], [fx, fy, 0])
            builder.circuit.append("DETECTOR", [stim.target_rec(-1)], [fx, fy, 0])
            flag_detector_indices = [
                builder.circuit.num_detectors - 2,
                builder.circuit.num_detectors - 1,
            ]
        else:
            _apply_gate_by_gate(builder, _ZERO_ZERO_ENCODER_CORE, data)
    else:
        raise ValueError(f"unknown encoder {encoder!r}; use 'zero_zero' or None")

    builder.apply_syndrome_extraction(
        circuit_chunk=HSixExtractionBlock(system).circuit, rounds=rounds
    )
    builder.apply_data_readout(final_measurements={q: basis for q in data})

    circuit = builder.circuit
    info = {
        "code": "[[6,2,2]]",
        "basis": basis,
        "rounds": rounds,
        "encoder": encoder,
        "flag_verified": flag_verified,
        "flag_detector_indices": flag_detector_indices,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
    }
    return circuit, info


__all__ = ["encoded_memory_circuit"]
