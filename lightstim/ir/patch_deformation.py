"""Internal registration update after code-specific destructive contraction.

Callers own the physical measurement recipe, dependency checks, and atomic
staging. This helper only retires the old definition and registers a smaller
one on retained physical sites. It does not synthesize or validate an arbitrary
code deformation; the caller must rebase the tracker onto established checks.
"""

from __future__ import annotations

import copy

from .patch_combination import _global_view, _signature


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
