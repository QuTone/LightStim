"""Adopt an established active code as one registered patch, without gates."""

from __future__ import annotations

import copy

import numpy as np
import stim

from lightstim.ir.qec_patch import QECPatch
from lightstim.utils.linear_algebra import (
    check_commutativity,
    row_echelon,
    solve_linear_decomposition,
)


def _matrix(records, data, n):
    matrix = np.zeros((len(records), 2 * n), dtype=np.uint8)
    for row, record in enumerate(records):
        if "sign" in record or "phase" in record or "kf" in record:
            raise ValueError("combine_patches supports ordinary phase-free check records.")
        pauli = record.get("pauli", {})
        if not pauli or not set(pauli) <= data:
            raise ValueError("Output operators must have nonempty support on output data.")
        if set(record.get("data_indices", pauli)) != set(pauli):
            raise ValueError("Operator data_indices disagree with its Pauli support.")
        for qubit, factor in pauli.items():
            if factor not in ("X", "Y", "Z"):
                raise ValueError("Invalid Pauli factor in output patch.")
            matrix[row, qubit] = factor in ("X", "Y")
            matrix[row, n + qubit] = factor in ("Z", "Y")
    return matrix


def _rank(matrix):
    return int(row_echelon(matrix)[1]) if len(matrix) else 0


def _contains(basis, targets):
    if not len(targets):
        return True
    if not len(basis):
        return not np.any(targets)
    return bool(np.all(solve_linear_decomposition(
        basis=basis, targets=targets, reduce_weight=False,
    )[1]))


def _signature(record):
    return (
        record.get("type"), tuple(sorted(record["pauli"].items())),
        record.get("syn_idx"), record.get("syn_coord"),
    )


def _global_view(system, local, mapping, name):
    """Translate a template without registering or reallocating anything."""
    view = copy.deepcopy(local)
    view.qubit_coords = {mapping[q]: coord for q, coord in local.qubit_coords.items()}
    view.index_map = {coord: q for q, coord in view.qubit_coords.items()}
    view._rebuild_grid_map()
    view._local_to_global_map = dict(mapping)
    for attribute in (
        "data_indices", "syndrome_indices", "syndrome_indices_x", "syndrome_indices_z",
    ):
        setattr(view, attribute, {mapping[q] for q in getattr(local, attribute)})
    for attribute in ("stabilizers", "logical_ops", "gauges"):
        translated = [system._translate_record(record, mapping)
                      for record in getattr(local, attribute)]
        for record in translated:
            record["patch_name"] = name
        setattr(view, attribute, translated)
    return view


def _prepare_tracker(tracker, system, target_basis):
    """A classification change may use known rows, never invented constraints."""
    if tracker.num_qubits != system.num_qubits:
        raise ValueError("Tracker register does not match the system register.")
    if tracker.num_absorbed_dof() or tracker._gauge_logical_vectors:
        raise ValueError("combine_patches does not support pending absorbed or gauge logicals.")
    tracker._reject_pending_row_metadata("combine_patches")
    full = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    records = tracker.stabilizers.records + tracker.logicals.records
    known = [i for i, parity in enumerate(records) if all(r >= 0 for r in parity)]
    if not _contains(full[known], target_basis):
        raise ValueError(
            "Output checks are not established in the tracker's known state; "
            "run the physical preparation and syndrome extraction first."
        )

    known_stabs = [i for i, parity in enumerate(tracker.stabilizers.records)
                   if all(r >= 0 for r in parity)]
    if (_contains(tracker.stabilizers.matrix[known_stabs], target_basis)
            and tracker.logicals.count == tracker.expected_num_logicals):
        tracker.validate_logical_count(context="patch combination")
        return None

    # Some extraction paths leave an established code row in the logical
    # tableau. Reclassify only after proving every active check already has
    # a known record expression. Canonicalization cannot seed unknown rows.
    active = _matrix(system.active_stabilizers, system.data_indices, system.num_qubits)
    if not _contains(full[known], active):
        raise ValueError("Cannot reclassify while active code checks remain unestablished.")
    staged = copy.deepcopy(tracker)
    staged.stabilizer_canonicalization(system)
    staged.rebase_stabilizers_onto_code_basis(system)
    staged.validate_logical_count(context="patch combination")
    return staged


def combine_patches(builder, patch_names, output_patch: QECPatch, name: str) -> QECPatch:
    """Contract one code patch and active zero-logical couplers atomically.

    ``output_patch`` uses local indices and already-global coordinates. Its
    checks must equal the selected active checks, including syndrome sites.
    The physical preparation and measurement must already have happened.
    This initial contract preserves the protected logical count and rejects
    subsystem codes, disabled geometry, and external overlapping components.
    """
    system = builder.system
    names = list(patch_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("patch_names must be a nonempty list of distinct names.")
    if not isinstance(name, str) or not name:
        raise ValueError("The output patch name must be a nonempty string.")
    selected = set(names)
    if selected - system.patches.keys():
        raise ValueError(f"Unknown input patches: {sorted(selected - system.patches.keys())}.")
    if name in system.patches and name not in selected:
        raise ValueError(f"Output name {name!r} belongs to an unrelated patch.")
    ordinary = selected - system.coupler_patches.keys()
    couplers = selected & system.coupler_patches.keys()
    if len(ordinary) != 1 or not couplers:
        raise ValueError("combine_patches requires one code patch and active zero-logical couplers.")
    if not isinstance(output_patch, QECPatch):
        raise TypeError("output_patch must be a local QECPatch template.")
    if system.active_gauges or output_patch.gauges:
        raise ValueError("combine_patches does not support subsystem gauges.")
    for source in names:
        patch = system.patches[source][0]
        if patch.gauges:
            raise ValueError("combine_patches does not support subsystem gauges.")
        if source in couplers and (
            patch.num_logicals != 0 or source not in system.paused_stabilizer_indices
        ):
            raise ValueError("Every selected coupler must be active and have zero logicals.")
    source_patch = system.patches[next(iter(ordinary))][0]
    if output_patch.num_logicals != source_patch.num_logicals or output_patch.num_logicals < 1:
        raise ValueError("Patch combination must preserve the source protected logical count.")

    input_qubits = set().union(*(set(system.local_to_global_map[p].values()) for p in names))
    input_uids = set().union(*(system.patches[p][0]._registered_stabilizer_uids for p in names))
    active_uids = input_uids & system.active_stabilizer_indices
    if input_qubits & system.disabled_data_qubit_indices:
        raise ValueError("combine_patches does not support disabled data geometry.")
    if any(system.index_to_owner_map.get(q) not in selected for q in input_qubits):
        raise ValueError("Input qubits have ownership outside the selected components.")

    # The system does not store coupler endpoint names. Check their actual
    # registered resources, check supports, and restoration dependencies.
    for other, (patch, _) in system.patches.items():
        if other in selected:
            continue
        referenced = set(system.local_to_global_map[other].values())
        for uid in patch._registered_stabilizer_uids:
            record = system.stabilizers[uid]
            referenced.update(record["pauli"])
            if record.get("syn_idx") is not None:
                referenced.add(record["syn_idx"])
        conflicts = getattr(patch, "conflicting_stabilizer_coords", set())
        if (referenced & input_qubits
                or system.paused_stabilizer_indices.get(other, set()) & input_uids
                or conflicts & {system.qubit_coords[q] for q in input_qubits}):
            raise ValueError(f"Unselected component {other!r} references the combined region.")

    local = copy.deepcopy(output_patch)
    if (len(set(local.qubit_coords.values())) != len(local.qubit_coords)
            or local.index_map != {coord: q for q, coord in local.qubit_coords.items()}):
        raise ValueError("Output coordinate maps are inconsistent.")
    if set(local.qubit_coords.values()) != {system.qubit_coords[q] for q in input_qubits}:
        raise ValueError("Output geometry must contain exactly the input components' qubits.")
    mapping = {q: system.index_map[coord] for q, coord in local.qubit_coords.items()}
    if (not local.data_indices.isdisjoint(local.syndrome_indices)
            or local.data_indices | local.syndrome_indices != local.qubit_coords.keys()
            or not local.syndrome_indices_x <= local.syndrome_indices
            or not local.syndrome_indices_z <= local.syndrome_indices):
        raise ValueError("Output qubit roles are inconsistent.")
    view = _global_view(system, local, mapping, name)
    if view.data_indices != input_qubits & system.data_indices:
        raise ValueError("Output data roles differ from the current combined region.")
    for attribute in ("syndrome_indices", "syndrome_indices_x", "syndrome_indices_z"):
        if getattr(view, attribute) != input_qubits & getattr(system, attribute):
            raise ValueError("Output syndrome roles differ from the current combined region.")
    if not view.data_indices <= system.active_qubit_indices:
        raise ValueError("All combined data qubits must be initialized and live.")
    for record in view.stabilizers:
        syndrome = record.get("syn_idx")
        if syndrome not in view.syndrome_indices or record.get("syn_coord") != system.qubit_coords[syndrome]:
            raise ValueError("Output checks must reference their registered syndrome coordinates.")
    active_records = [system.stabilizers[u] for u in sorted(active_uids)]
    signatures = {_signature(record) for record in active_records}
    if (len(view.stabilizers) != len(signatures)
            or {_signature(record) for record in view.stabilizers} != signatures):
        raise ValueError("Output checks must match the current active checks and syndrome locations.")
    basis = _matrix(view.stabilizers, view.data_indices, system.num_qubits)
    if np.any(check_commutativity(basis, basis)):
        raise ValueError("Output stabilizers do not commute.")
    if len(view.data_indices) - _rank(basis) != local.num_logicals:
        raise ValueError("Output stabilizer rank disagrees with the declared logical count.")
    # Binary rank cannot detect a redundant generator product equal to -I.
    # QECPatch templates describe the positive-sign reference code; actual
    # measurement signs remain in the tracker's record expressions.
    paulis = []
    for record in view.stabilizers:
        pauli = stim.PauliString(system.num_qubits)
        for q, factor in record["pauli"].items():
            pauli[q] = factor
        paulis.append(pauli)
    try:
        stim.Tableau.from_stabilizers(paulis, allow_redundant=True, allow_underconstrained=True)
    except ValueError as exc:
        raise ValueError("Output template has inconsistent stabilizer signs.") from exc
    logicals = _matrix(view.logical_ops, view.data_indices, system.num_qubits)
    x = [i for i, record in enumerate(view.logical_ops) if record.get("type") == "X"]
    z = [i for i, record in enumerate(view.logical_ops) if record.get("type") == "Z"]
    k = local.num_logicals
    if len(x) != k or len(z) != k or len(logicals) != 2 * k:
        raise ValueError("Output must declare one X/Z logical pair per protected qubit.")
    if (np.any(check_commutativity(basis, logicals))
            or np.any(check_commutativity(logicals[x], logicals[x]))
            or np.any(check_commutativity(logicals[z], logicals[z]))
            or not np.array_equal(check_commutativity(logicals[x], logicals[z]), np.eye(k, dtype=np.uint8))):
        raise ValueError("Output logical operators do not form canonical X/Z pairs.")

    staged_tracker = _prepare_tracker(builder.tracker, system, basis)
    local._registered_stabilizer_uids = set(active_uids)
    local._registered_gauge_uids = set()
    view._registered_stabilizer_uids = set(active_uids)
    view._registered_gauge_uids = set()

    # Prepare every replacement before touching the live objects. Global
    # qubit IDs, check UIDs, active sets, and historical inactive records stay
    # allocated. Consumed restoration histories must never reactivate them.
    patches = {p: entry for p, entry in system.patches.items() if p not in selected}
    patches[name] = (local, (0, 0))
    maps = {p: entry for p, entry in system.local_to_global_map.items() if p not in selected}
    maps[name] = mapping
    owner = dict(system.index_to_owner_map)
    coord_owner = dict(system.coord_to_owner_map)
    for q in input_qubits:
        owner[q] = name
        coord_owner[system.qubit_coords[q]] = name
    stabilizers = list(system.stabilizers)
    for uid in active_uids:
        stabilizers[uid] = dict(stabilizers[uid], patch_name=name)
    logical_ops = [op for op in system.logical_ops if op.get("patch_name") not in selected]
    logical_ops.extend(copy.deepcopy(view.logical_ops))
    updates = {
        "patches": patches, "local_to_global_map": maps,
        "index_to_owner_map": owner, "coord_to_owner_map": coord_owner,
        "stabilizers": stabilizers, "logical_ops": logical_ops,
        "coupler_patches": {p: patch for p, patch in system.coupler_patches.items() if p not in couplers},
        "paused_stabilizer_indices": {p: uids for p, uids in system.paused_stabilizer_indices.items() if p not in couplers},
    }
    system.__dict__.update(updates)
    if staged_tracker is not None:
        builder.tracker.__dict__.update(staged_tracker.__dict__)
    return view
