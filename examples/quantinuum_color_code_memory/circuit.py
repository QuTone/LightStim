"""d=3 reconstruction of the memory circuit in arXiv:2610.24077v1.

Bell-flag preparation (Fig. 8(a)), followed by SWAP-superdense extraction
(Fig. 1(b)). This models the published gate schedule, not the Helios emulator.
"""
from __future__ import annotations

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.color_code import ColorCode, ColorCodeLogicalOpSet


# Paper slots TL, TR, R, BR, BL, L in LightStim's rotated patch coordinates.
PAPER_SLOT_TO_LOCAL = (0, 5, 4, 3, 2, 1)
ROUND_LABELS = ("cebfad", "bfadce", "adcebf")
NOISE_MODELS = ("circuit_level", "circuit_level_with_idling")


def _faces(system, labels):
    patch = system.patches["color"][0]
    mapping = system.local_to_global_map["color"]
    result = []
    for face in patch.faces:
        wires = {}
        for label, slot in zip(labels, PAPER_SLOT_TO_LOCAL):
            neighbor = face["data_neighbors"][slot]
            wires[label] = None if neighbor is None else mapping[neighbor[1]]
        wires["s"] = mapping[face["x_ancilla_idx"]]
        wires["t"] = mapping[face["z_ancilla_idx"]]
        result.append(wires)
    return result


def _layer(circuit, faces, pairs, gate="CX", *, tag="", tick=True):
    targets = [face[q] for face in faces for a, b in pairs
               if face[a] is not None and face[b] is not None for q in (a, b)]
    if len(targets) != len(set(targets)):
        raise ValueError(f"Gate collision in {gate}: {targets}")
    if targets:
        circuit.append(gate, targets, tag=tag)
    if tick:
        circuit.append("TICK")


def bell_flag_preparation(system):
    """Measure X checks using the data-H-conjugated Fig. 8(a) circuit.

    The final data H commutes with ancilla readout and precedes it to satisfy
    the atomic block's terminal-readout contract. Flags are decoded, not rejected.
    """
    faces = _faces(system, "dbfaec")
    data = sorted(system.data_indices)
    s, t = ([face[q] for face in faces] for q in ("s", "t"))
    c = stim.Circuit()
    c.append("H", data)
    c.append("RX", s + t)
    c.append("TICK", tag="SE_start")
    _layer(c, faces, [("s", "t")], "CZ")
    c.append("H", s)
    c.append("TICK")
    for a, b in (("a", "f"), ("b", "e"), ("c", "d")):
        _layer(c, faces, [(a, "s"), (b, "t")], "CZ")
    c.append("H", t)
    c.append("TICK")
    _layer(c, faces, [("s", "t")], "CZ")
    c.append("H", data)
    c.append("TICK")
    c.append("MX", s + t)
    return c


def swap_superdense_round(system, round_index):
    """Fig. 1(b), with three cyclic face schedules and virtual wire SWAPs."""
    faces = _faces(system, ROUND_LABELS[round_index % 3])
    s, t = ([face[q] for face in faces] for q in ("s", "t"))
    c = stim.Circuit()
    c.append("RX", s)
    c.append("R", t)
    c.append("TICK", tag="SE_start")
    _layer(c, faces, [("s", "t")])
    for a, b in (("a", "f"), ("b", "e")):
        _layer(c, faces, [(a, "s"), (b, "t")])
    # Software relabeling has no gate noise and consumes no separate moment.
    _layer(c, faces, [("c", "s"), ("d", "t")], "SWAP", tag="noiseless", tick=False)
    _layer(c, faces, [("c", "s"), ("d", "t")])
    for a, b in (("b", "e"), ("a", "f")):
        _layer(c, faces, [("s", a), ("t", b)])
    _layer(c, faces, [("s", "t")])
    c.append("MX", s)
    c.append("M", t)
    return c


def build_circuit(*, rounds=20, basis="Z", sign=1, noise=None,
                  noise_model="circuit_level"):
    """Prepare a signed logical Pauli state, store it, then rotate/read out Z.

    Returns an annotated Stim circuit. All annotations come from LightStim;
    feedback is a symbolic Pauli frame. There is no postselection or ideal
    terminal SE round. See README for the unresolved Y-boundary limitation.
    """
    if basis not in ("X", "Y", "Z") or sign not in (-1, 1):
        raise ValueError("basis must be X/Y/Z and sign must be +1 or -1")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
    system = QECSystem()
    patch = system.add_patch(ColorCode(distance=3, layout="superdense"), name="color")
    builder = CircuitBuilder(SyndromeTracker(system.num_qubits, system.num_logicals), system)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    builder.circuit.append("TICK")
    builder.apply_syndrome_extraction(bell_flag_preparation(system))
    # Preparation measures X checks only. Preserve the known, inherited Z
    # constraints as code stabilizers before entering the full CSS memory.
    builder.stabilizer_canonicalization()
    ops = ColorCodeLogicalOpSet()
    if sign < 0:
        ops.transversal_x(builder, patch)
    if basis in ("X", "Y"):
        ops.transversal_hadamard(builder, patch)
    if basis == "Y":
        ops.transversal_s(builder, patch)
    for r in range(rounds):
        builder.apply_syndrome_extraction(swap_superdense_round(system, r))
    if basis == "Y":
        ops.transversal_s_dag(builder, patch)
    if basis in ("X", "Y"):
        ops.transversal_hadamard(builder, patch)
    if sign < 0:
        ops.transversal_x(builder, patch)
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    if noise is not None:
        return builder.build_noisy_circuit(noise_params=noise, noise_model=noise_model)
    return builder.to_stim_circuit()


def noise_config(noise_model, p=0.001):
    """Fixed comparison: identical gate/SPAM rates, with or without moment idling."""
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
    return NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                       p_idle=p if noise_model == "circuit_level_with_idling" else 0)
