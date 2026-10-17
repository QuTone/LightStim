"""Corner-anchored rotated-code contraction by destructive data measurement."""

from __future__ import annotations

import copy

import numpy as np

from lightstim.ir.patch_combination import _global_view, _prepare_tracker, _signature
from lightstim.qec_code._operation_utils import require_global_patch
from lightstim.utils.tableau_utils import stabilizers_to_symplectic

from .code_patch import RotatedSurfaceCode
from .enlargement import _check_signature, corner_initialization_basis
from .SE_block import RotatedSurfaceCodeExtractionBlock


def _prepare_target(builder, patch, target_distance, direction, rounds):
    if not isinstance(patch, RotatedSurfaceCode):
        raise TypeError("shrink requires a RotatedSurfaceCode patch")
    require_global_patch(builder, patch)
    if type(target_distance) is not int or target_distance < 3 or target_distance % 2 == 0:
        raise ValueError("target_distance must be an odd integer at least 3")
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer (optional target SE rounds)")
    if direction != "-x,-y":
        raise ValueError("shrink currently supports only direction='-x,-y'")
    if patch.distance_x != patch.distance_z or patch.distance_x < 3 or patch.distance_x % 2 == 0:
        raise ValueError("shrink requires a square source of odd distance at least 3")
    if target_distance >= patch.distance_x:
        raise ValueError("target_distance must be smaller than the current distance")
    if patch.is_transposed or not np.isclose(patch.rotation_angle % (2 * np.pi), 0):
        raise ValueError("shrink supports translated, unrotated/untransposed layouts only")
    system = builder.system
    if not builder.if_detector:
        raise ValueError("shrink requires the tracker-enabled builder (if_detector=True)")
    if system._builder is not builder or system._tracker is not builder.tracker:
        raise ValueError("Register this builder and tracker with QECSystem before shrinking")
    if system.disabled_data_qubit_indices & patch.data_indices:
        raise ValueError("shrink does not yet support defects")
    if system.active_gauges or patch.gauges:
        raise ValueError("shrink does not support subsystem gauges")
    if not patch.data_indices <= system.active_qubit_indices:
        raise ValueError("Source data must be initialized and active before shrinking")
    uids = set(patch._registered_stabilizer_uids)
    if not uids <= system.active_stabilizer_indices:
        raise ValueError("Source checks must all be active before shrinking")
    name, = [name for name, (local, _) in system.patches.items()
             if local._registered_stabilizer_uids == uids
             and set(system.local_to_global_map[name].values()) == set(patch.qubit_coords)]
    source_qubits = set(patch.qubit_coords)
    # An inactive coupler can still hold a boundary-restoration dependency.
    for other, (local, _) in system.patches.items():
        if other == name:
            continue
        referenced = set(system.local_to_global_map[other].values())
        for uid in local._registered_stabilizer_uids:
            check = system.stabilizers[uid]
            referenced.update(check["pauli"])
            if check.get("syn_idx") is not None:
                referenced.add(check["syn_idx"])
        if (referenced & source_qubits
                or system.paused_stabilizer_indices.get(other, set()) & uids
                or getattr(local, "conflicting_stabilizer_coords", set()) & set(patch.index_map)):
            raise ValueError(f"Registered component {other!r} depends on the source patch")

    origin = (min(x for x, _ in patch.data_coords), min(y for _, y in patch.data_coords))
    reference = RotatedSurfaceCode(distance=patch.distance_x)
    reference.shift_coords(origin[0] - 1, origin[1] - 1)
    if (set(reference.qubit_coords.values()) != set(patch.qubit_coords.values())
            or {_check_signature(reference, s) for s in reference.stabilizers}
            != {_check_signature(patch, s) for s in patch.stabilizers}):
        raise ValueError("Source geometry/checks are not the canonical rotated layout")
    target = RotatedSurfaceCode(distance=target_distance)
    target.shift_coords(origin[0] - 1, origin[1] - 1)
    return name, target, origin


def _adopt_target(builder, patch, local, name):
    """Replace the live definition; keep physical IDs and measurement history."""
    system = builder.system
    mapping = {q: system.index_map[xy] for q, xy in local.qubit_coords.items()}
    view = _global_view(system, local, mapping, name)
    released = set(patch.qubit_coords) - set(view.qubit_coords)
    old_uids = set(patch._registered_stabilizer_uids)
    target_uids = set()
    for check in view.stabilizers:
        key = (check["type"], tuple(sorted(check["pauli"].items())))
        uid = system._stabilizer_signatures.get(key)
        if uid is None or _signature(system.stabilizers[uid]) != _signature(check):
            uid = len(system.stabilizers)
            system.stabilizers.append(copy.deepcopy(check))
        else:
            system.stabilizers[uid] = dict(system.stabilizers[uid], patch_name=name)
        system._stabilizer_signatures[key] = uid
        target_uids.add(uid)
    system.active_stabilizer_indices.difference_update(old_uids)
    system.active_stabilizer_indices.update(target_uids)
    for uid in old_uids - target_uids:
        # Keep historical records allocated, but remove retired checks from
        # live deduplication. A future patch on this support needs a fresh
        # record carrying its own ownership and extraction metadata.
        record = system.stabilizers[uid]
        key = (record["type"], tuple(sorted(record["pauli"].items())))
        if system._stabilizer_signatures.get(key) == uid:
            del system._stabilizer_signatures[key]
        system.stabilizers[uid] = dict(system.stabilizers[uid], patch_name=None)
    for record in (local, view):
        record._registered_stabilizer_uids = set(target_uids)
        record._registered_gauge_uids = set()
    system.patches[name] = (local, (0, 0))
    system.local_to_global_map[name] = mapping
    system.logical_ops = [op for op in system.logical_ops if op.get("patch_name") != name]
    system.logical_ops.extend(copy.deepcopy(view.logical_ops))
    builder._retired_data_qubit_indices.update(patch.data_indices - view.data_indices)
    for attr in ("data_indices", "syndrome_indices", "syndrome_indices_x", "syndrome_indices_z",
                 "active_qubit_indices"):
        getattr(system, attr).difference_update(released)
    for q in released:
        system.index_to_owner_map.pop(q, None)
        system.coord_to_owner_map.pop(system.qubit_coords[q], None)
    # Keep index_map, qubit_coords and next_index: old circuit operations still
    # refer to these IDs, and later enlargement can reuse dormant coordinates.
    return view


def shrink(builder, patch, *, target_distance, direction="-x,-y", rounds=0):
    """Retain the minimum-coordinate corner and return its smaller global view.

    Removed data are measured in X on/below the relative diagonal and Z above.
    The known logical Pauli byproduct stays in the tracker's measurement-record
    parities; no physical correction or postselection is added. ``rounds=0``
    performs measurements only. Positive values append native target-code SE.
    """
    name, target, origin = _prepare_target(builder, patch, target_distance, direction, rounds)
    staged = copy.deepcopy(builder)
    system, tracker = staged.system, staged.tracker
    system.register_builder(staged)
    system.register_tracker(tracker)
    source_basis = stabilizers_to_symplectic(system, patch.stabilizers, tracker.num_qubits)
    classified = _prepare_tracker(tracker, system, source_basis)
    if classified is not None:
        tracker.__dict__.update(classified.__dict__)

    retained = {system.index_map[xy] for xy in target.data_coords}
    measurements = {q: corner_initialization_basis(*patch.qubit_coords[q], origin)
                    for q in sorted(patch.data_indices - retained)}
    first_record = tracker.total_measurements
    before_budget = tracker.expected_num_logicals
    before_observables = staged.circuit.num_observables
    staged.apply_data_readout(measurements, resolve_absorbed=False)
    ordered = ([q for q, basis in measurements.items() if basis == "X"]
               + [q for q, basis in measurements.items() if basis == "Z"])
    tracker.meas_rec_to_idx_map.update({first_record + i: q for i, q in enumerate(ordered)})
    if staged.circuit.num_observables != before_observables:
        raise RuntimeError("Shrinking unexpectedly measured a protected logical degree of freedom")
    columns = ordered + [q + tracker.num_qubits for q in ordered]
    if any(tableau.matrix[:, columns].any() for tableau in (tracker.stabilizers, tracker.logicals)):
        raise RuntimeError("Shrinking left tracked operators on removed data")
    output = _adopt_target(staged, patch, target, name)
    # All target checks must follow from real old syndromes and skirt outcomes.
    # Rebase carries those record parities; it cannot invent new constraints.
    tracker.rebase_stabilizers_onto_code_basis(system)
    # V2 reports the protected rank relative to the current code declaration.
    # Immediately after skirt readout, shortened target checks need not lie in
    # the surviving old-code check space. Compare ranks only after adopting
    # the target declaration and establishing its checks from actual records.
    if tracker.expected_num_logicals != before_budget:
        raise RuntimeError("Shrinking changed the protected logical rank after target-code rebase")
    staged._z_only_syn_qubit_indices = None
    staged._z_only_no_detector_mask = None
    staged._z_only_n_meas_per_round = None
    if rounds:
        extraction_system = copy.copy(system)
        extraction_system.active_stabilizer_indices = set(output._registered_stabilizer_uids)
        extraction_system.active_gauge_indices = set()
        extraction = RotatedSurfaceCodeExtractionBlock(extraction_system, scheduling="perpendicular")
        staged.apply_syndrome_extraction(extraction.circuit, rounds=rounds)
        # The legacy shortcut closes every active check against one shared
        # last-round ordering. Target-only SE cannot refresh spectator
        # records, so keep the cache invalidated in a multi-patch system.
        if (builder._z_only_syn_qubit_indices is not None
                and extraction_system.active_stabilizer_indices == system.active_stabilizer_indices):
            measured = [t.value for t in extraction.circuit[-1].targets_copy()]
            x_syndromes = set(extraction_system.active_syndrome_indices_x)
            staged._z_only_syn_qubit_indices = measured
            staged._z_only_no_detector_mask = np.array([q in x_syndromes for q in measured], dtype=bool)
            staged._z_only_n_meas_per_round = len(measured)

    original_system, original_tracker = builder.system, builder.tracker
    original_system.__dict__.clear()
    original_system.__dict__.update(system.__dict__)
    original_tracker.__dict__.clear()
    original_tracker.__dict__.update(tracker.__dict__)
    builder.__dict__.clear()
    builder.__dict__.update(staged.__dict__)
    builder.system, builder.tracker = original_system, original_tracker
    original_system.register_builder(builder)
    original_system.register_tracker(original_tracker)
    return output
