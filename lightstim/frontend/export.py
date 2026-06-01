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
    """Return the Timeline JSON payload.

    RepeatBlock handling: the body is shown once with a `repeat_count` field on
    each tick entry; subsequent iterations are elided and counters are advanced.
    """
    qubit_coords_map = circuit.get_final_qubit_coordinates()
    det_coords = circuit.get_detector_coordinates()

    num_qubits = circuit.num_qubits
    qubits = []
    for qid in range(num_qubits):
        q_entry: dict = {"id": qid}
        if qid in qubit_coords_map:
            c = qubit_coords_map[qid]
            q_entry["coords"] = {"x": c[0], "y": c[1]}
        qubits.append(q_entry)

    _MEAS_NAMES = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"})

    ticks: list[dict] = []
    current_ops: list[dict] = []
    current_noise: list[dict] = []
    meas_idx = 0
    num_measurements = 0
    num_detectors = 0
    det_meas_records: dict[int, list[int]] = {}  # det_id → [absolute meas indices]
    obs_meas_records: dict[int, list[int]] = {}  # obs_id → [absolute meas indices]

    def _flush(repeat_count: int = 1) -> None:
        if current_ops or current_noise:
            t_entry: dict = {"operations": list(current_ops), "noise": list(current_noise)}
            if repeat_count > 1:
                t_entry["repeat_count"] = repeat_count
            ticks.append(t_entry)
            current_ops.clear()
            current_noise.clear()

    def _process(block: stim.Circuit, repeat_count: int = 1) -> None:
        nonlocal meas_idx, num_measurements, num_detectors

        for inst in block:
            # ── RepeatBlock: show body once, skip remaining iterations ──────────
            if isinstance(inst, stim.CircuitRepeatBlock):
                _flush(repeat_count)
                body = inst.body_copy()
                body_rc = repeat_count * inst.repeat_count
                _process(body, body_rc)
                # Flush trailing ops inside the body (e.g. measurement after the last TICK)
                # using the body's repeat_count, not the outer one.
                _flush(body_rc)
                extra = inst.repeat_count - 1
                if extra > 0:
                    flat_body = body.flattened()
                    body_meas = sum(
                        len(list(b.targets_copy()))
                        for b in flat_body if b.name in _MEAS_NAMES
                    )
                    body_dets = sum(1 for b in flat_body if b.name == "DETECTOR")
                    # Record measurement refs for extra repetitions (shifting indices by body_meas per rep)
                    first_det = num_detectors - body_dets
                    for rep in range(1, inst.repeat_count):
                        for off in range(body_dets):
                            src_refs = det_meas_records.get(first_det + off, [])
                            new_det_id = num_detectors + (rep - 1) * body_dets + off
                            det_meas_records[new_det_id] = [r + body_meas * rep for r in src_refs]
                    meas_idx += extra * body_meas
                    num_measurements = meas_idx
                    num_detectors += extra * body_dets
                continue

            name = inst.name

            if name == "TICK":
                _flush(repeat_count)
                continue

            # ── DETECTOR: record which measurements it references ─────────────
            if name == "DETECTOR":
                refs: list[int] = []
                for t in inst.targets_copy():
                    if t.is_measurement_record_target:
                        abs_idx = meas_idx + t.value  # t.value is negative (rec[-k])
                        if 0 <= abs_idx < meas_idx:
                            refs.append(abs_idx)
                det_meas_records[num_detectors] = refs
                num_detectors += 1
                continue

            # ── OBSERVABLE_INCLUDE: record which measurements an observable references ──
            if name == "OBSERVABLE_INCLUDE":
                try:
                    obs_id = int(inst.gate_args_copy()[0])
                except (IndexError, ValueError):
                    continue
                obs_refs: list[int] = []
                for t in inst.targets_copy():
                    if t.is_measurement_record_target:
                        abs_idx = meas_idx + t.value
                        if 0 <= abs_idx < meas_idx:
                            obs_refs.append(abs_idx)
                # Multiple OBSERVABLE_INCLUDE calls for the same id accumulate measurements
                if obs_id not in obs_meas_records:
                    obs_meas_records[obs_id] = []
                obs_meas_records[obs_id].extend(obs_refs)
                continue

            if name in _SKIP_GATES:
                continue

            targets = [t.value for t in inst.targets_copy()]

            # ── Measurements ─────────────────────────────────────────────────
            if name in _MEAS_NAMES:
                indices = list(range(meas_idx, meas_idx + len(targets)))
                meas_idx += len(targets)
                num_measurements = meas_idx
                current_ops.append({"gate": name, "qubits": targets, "measurement_indices": indices})
                continue

            # ── Noise / two-qubit / single-qubit ─────────────────────────────
            if name in _NOISE_GATES:
                if name in _TWO_QUBIT_GATES or name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
                    pairs = [[targets[i], targets[i + 1]] for i in range(0, len(targets) - 1, 2)]
                    current_noise.append({"gate": name, "qubit_pairs": pairs, "probability": inst.gate_args_copy()[0]})
                else:
                    prob = inst.gate_args_copy()[0] if inst.gate_args_copy() else None
                    n_entry: dict = {"gate": name, "qubits": targets}
                    if prob is not None:
                        n_entry["probability"] = prob
                    current_noise.append(n_entry)
            elif name in _TWO_QUBIT_GATES:
                pairs = [[targets[i], targets[i + 1]] for i in range(0, len(targets) - 1, 2)]
                current_ops.append({"gate": name, "qubit_pairs": pairs})
            else:
                current_ops.append({"gate": name, "qubits": targets})

    _process(circuit)
    _flush()

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
        "detector_measurement_records": {str(k): v for k, v in det_meas_records.items()},
        "observable_measurement_records": {str(k): v for k, v in obs_meas_records.items()},
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
        "circuit_str": str(circuit),
    }
