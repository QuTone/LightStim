"""
Export a stim.Circuit to the three JSON payloads consumed by the LightStim frontend:
  - DEM   : 3D detector-error-model graph (nodes = detectors, edges = error mechanisms)
  - Timeline : circuit operations grouped by TICK (gate timeline diagram)
  - DetSlice : per-tick Pauli support of each detector on data qubits (2D topology)
"""
from __future__ import annotations

import stim

# Gates that are noise channels, not physical operations
_NOISE_GATES = {
    "DEPOLARIZE1", "DEPOLARIZE2",
    "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
    "ELSE_CORRELATED_ERROR", "CORRELATED_ERROR",
    "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1",
}

# Gates with no physical meaning for visualization
_SKIP_GATES = {
    "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS", "TICK",
}

# Two-qubit gates (targets come in control/target pairs)
_TWO_QUBIT_GATES = {"CX", "CZ", "CY", "XCZ", "XCX", "YCZ", "SQRT_XX", "SQRT_ZZ", "ISWAP"}


def export_dem(
    circuit: stim.Circuit,
    *,
    source: str = "",
    distance: int | None = None,
    rounds: int | None = None,
    noise_model: str = "",
    physical_error_rate: float | None = None,
    decompose_errors: bool = False,
) -> dict:
    """Return the DEM JSON payload (matches front-end/schemas/dem_schema.json)."""
    flat = circuit.flattened()
    dem = flat.detector_error_model(decompose_errors=decompose_errors)
    det_coords = circuit.get_detector_coordinates()

    detectors = []
    for det_id, coords in det_coords.items():
        entry: dict = {"id": det_id, "coords": {}}
        if len(coords) >= 2:
            entry["coords"] = {"x": coords[0], "y": coords[1], "t": coords[2] if len(coords) > 2 else 0.0}
        detectors.append(entry)
    detectors.sort(key=lambda d: d["id"])

    num_observables = circuit.num_observables
    observables = [{"id": i} for i in range(num_observables)]

    error_mechanisms = []
    for inst in dem:
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        detector_ids = []
        observable_ids = []
        for t in inst.targets_copy():
            if t.is_separator():
                # ^ separator in decomposed mode: start a new mechanism
                error_mechanisms.append({
                    "probability": round(prob, 10),
                    "detector_ids": detector_ids,
                    "observable_ids": observable_ids,
                })
                detector_ids = []
                observable_ids = []
            elif t.is_relative_detector_id():
                detector_ids.append(t.val)
            elif t.is_logical_observable_id():
                observable_ids.append(t.val)
        error_mechanisms.append({
            "probability": round(prob, 10),
            "detector_ids": detector_ids,
            "observable_ids": observable_ids,
        })

    return {
        "metadata": {
            "source": source,
            "distance": distance,
            "rounds": rounds,
            "noise_model": noise_model,
            "physical_error_rate": physical_error_rate,
            "decompose_errors": decompose_errors,
        },
        "detectors": detectors,
        "observables": observables,
        "error_mechanisms": error_mechanisms,
    }


def export_timeline(
    circuit: stim.Circuit,
    *,
    source: str = "",
    distance: int | None = None,
    rounds: int | None = None,
) -> dict:
    """Return the Timeline JSON payload (matches front-end/schemas/circuit_timeline_schema.json)."""
    flat = circuit.flattened()
    qubit_coords_map = circuit.get_final_qubit_coordinates()
    det_coords = circuit.get_detector_coordinates()

    num_qubits = circuit.num_qubits
    qubits = []
    for qid in range(num_qubits):
        entry: dict = {"id": qid}
        if qid in qubit_coords_map:
            c = qubit_coords_map[qid]
            entry["coords"] = {"x": c[0], "y": c[1]}
        qubits.append(entry)

    # Group instructions into ticks
    ticks: list[dict] = []
    current_ops: list[dict] = []
    current_noise: list[dict] = []
    num_measurements = 0
    num_detectors = 0

    def _flush_tick():
        if current_ops or current_noise:
            ticks.append({"operations": list(current_ops), "noise": list(current_noise)})
            current_ops.clear()
            current_noise.clear()

    for inst in flat:
        name = inst.name
        if name == "TICK":
            _flush_tick()
            continue
        if name in _SKIP_GATES:
            if name == "DETECTOR":
                num_detectors += 1
            continue

        targets = [t.value for t in inst.targets_copy()]

        if name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
            num_measurements += len(targets)

        if name in _NOISE_GATES:
            if name in _TWO_QUBIT_GATES or name == "DEPOLARIZE2" or name == "PAULI_CHANNEL_2":
                pairs = [[targets[i], targets[i + 1]] for i in range(0, len(targets) - 1, 2)]
                current_noise.append({"gate": name, "qubit_pairs": pairs, "probability": inst.gate_args_copy()[0]})
            else:
                prob = inst.gate_args_copy()[0] if inst.gate_args_copy() else None
                entry: dict = {"gate": name, "qubits": targets}
                if prob is not None:
                    entry["probability"] = prob
                current_noise.append(entry)
        elif name in _TWO_QUBIT_GATES:
            pairs = [[targets[i], targets[i + 1]] for i in range(0, len(targets) - 1, 2)]
            current_ops.append({"gate": name, "qubit_pairs": pairs})
        else:
            current_ops.append({"gate": name, "qubits": targets})

    _flush_tick()

    detectors_list = [
        {"id": did, "coords": {"x": c[0], "y": c[1], "t": c[2] if len(c) > 2 else 0.0}}
        for did, c in det_coords.items()
    ]
    detectors_list.sort(key=lambda d: d["id"])

    return {
        "metadata": {
            "source": source,
            "distance": distance,
            "rounds": rounds,
            "num_qubits": num_qubits,
            "num_measurements": num_measurements,
            "num_ticks": len(ticks),
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
        },
        "qubits": qubits,
        "ticks": ticks,
        "detectors": detectors_list,
    }


def export_detslice(
    circuit: stim.Circuit,
    *,
    source: str = "",
    distance: int | None = None,
    rounds: int | None = None,
) -> dict:
    """Return the DetSlice JSON payload (matches front-end/schemas/detslice_schema.json).

    Uses stim.Circuit.detecting_regions() to get per-tick Pauli support of each
    detector on data qubits — no manual back-propagation needed.
    """
    qubit_coords_map = circuit.get_final_qubit_coordinates()
    det_coords = circuit.get_detector_coordinates()

    qubits = [
        {"id": qid, "coords": {"x": c[0], "y": c[1]}}
        for qid, c in qubit_coords_map.items()
        if len(c) >= 2
    ]
    qubits.sort(key=lambda q: q["id"])

    detector_coordinates = {
        str(did): {"x": c[0], "y": c[1], "t": c[2] if len(c) > 2 else 0.0}
        for did, c in det_coords.items()
    }

    # detecting_regions: {DemTarget -> {tick: stim.PauliString}}
    regions = circuit.detecting_regions()

    _pauli_map = {1: "X", 2: "Y", 3: "Z"}

    # Collect all ticks that have non-trivial detector support
    tick_to_detectors: dict[int, list[dict]] = {}
    for dem_target, tick_map in regions.items():
        if not dem_target.is_relative_detector_id():
            continue
        det_id = dem_target.val
        for tick, pauli_string in tick_map.items():
            pauli_support = []
            for qid, p_val in enumerate(pauli_string):
                if p_val != 0:
                    pauli_support.append({"qubit_id": qid, "pauli": _pauli_map[p_val]})
            if not pauli_support:
                continue
            tick_to_detectors.setdefault(tick, []).append({
                "detector_id": det_id,
                "pauli_support": pauli_support,
            })

    # Handle observables (L targets in detecting_regions)
    tick_to_observables: dict[int, list[dict]] = {}
    for dem_target, tick_map in regions.items():
        if not dem_target.is_logical_observable_id():
            continue
        obs_id = dem_target.val
        for tick, pauli_string in tick_map.items():
            pauli_support = []
            for qid, p_val in enumerate(pauli_string):
                if p_val != 0:
                    pauli_support.append({"qubit_id": qid, "pauli": _pauli_map[p_val]})
            if not pauli_support:
                continue
            tick_to_observables.setdefault(tick, []).append({
                "observable_id": obs_id,
                "pauli_support": pauli_support,
            })

    slices = []
    for tick in sorted(set(tick_to_detectors) | set(tick_to_observables)):
        entry: dict = {
            "tick": tick,
            "detectors": sorted(tick_to_detectors.get(tick, []), key=lambda d: d["detector_id"]),
        }
        obs = tick_to_observables.get(tick)
        if obs:
            entry["observable_support"] = sorted(obs, key=lambda o: o["observable_id"])
        slices.append(entry)

    num_ticks = sum(1 for inst in circuit.flattened() if inst.name == "TICK")

    return {
        "metadata": {
            "source": source,
            "distance": distance,
            "rounds": rounds,
            "num_ticks": num_ticks,
            "slices_with_data": len(slices),
        },
        "qubits": qubits,
        "detector_coordinates": detector_coordinates,
        "slices": slices,
    }


def export_all(
    circuit: stim.Circuit,
    *,
    source: str = "",
    distance: int | None = None,
    rounds: int | None = None,
    noise_model: str = "",
    physical_error_rate: float | None = None,
    decompose_errors: bool = False,
) -> dict:
    """Return all three payloads in one dict: {'dem': ..., 'timeline': ..., 'detslice': ...}."""
    kwargs = dict(source=source, distance=distance, rounds=rounds)
    return {
        "dem": export_dem(
            circuit,
            noise_model=noise_model,
            physical_error_rate=physical_error_rate,
            decompose_errors=decompose_errors,
            **kwargs,
        ),
        "timeline": export_timeline(circuit, **kwargs),
        "detslice": export_detslice(circuit, **kwargs),
    }
