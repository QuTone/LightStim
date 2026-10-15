"""Permanent, corner-anchored growth of a square rotated surface code.

The temporary coupler owns only added resources and changed checks.  After
physical preparation/SE, ``combine_patches`` adopts the target definition.
This module does not infer arbitrary code-deformation paths.
"""

from __future__ import annotations

import copy

import numpy as np

from lightstim.ir.coupler import LogicalCouplerProtocol
from lightstim.ir.patch_combination import _prepare_tracker
from lightstim.qec_code._operation_utils import require_global_patch
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic

from .code_patch import RotatedSurfaceCode
from .SE_block import RotatedSurfaceCodeExtractionBlock


def corner_initialization_basis(x: float, y: float, origin: tuple[float, float]) -> str:
    """The corner-injection X/Z split, without resetting an injection site."""
    ox, oy = origin
    return "X" if y - oy >= x - ox else "Z"


def _check_signature(patch, record):
    return tuple(sorted((patch.qubit_coords[q], axis)
                        for q, axis in record["pauli"].items()))


class _EnlargementCoupler(LogicalCouplerProtocol):
    """A zero-logical resource delta; not a separately prepared code."""

    EXPECTED_PATCH_COUNT = 1

    def _build_coupler_geometry(self, coupler_patch, patches, **params):
        source, = patches
        target = params["target_patch"]
        old_coords = set(source.index_map)
        for q, xy in sorted(target.qubit_coords.items()):
            if xy in old_coords:
                continue
            if q in target.data_indices:
                role = "data"
            elif q in target.syndrome_indices_x:
                role = "syndrome_x"
            else:
                role = "syndrome_z"
            coupler_patch.add_qubit(*xy, role=role)

        old_checks = {s["syn_coord"]: _check_signature(source, s)
                      for s in source.stabilizers}
        for check in target.stabilizers:
            coord = check["syn_coord"]
            signature = _check_signature(target, check)
            if old_checks.get(coord) == signature:
                continue
            if coord in old_checks:
                coupler_patch.conflicting_stabilizer_coords.add(coord)
            # Coordinate references may refer to retained qubits which are
            # owned by the source. Never allocate those a second time.
            coupler_patch.stabilizers.append({
                "pauli": dict(signature),
                "type": check["type"],
                "syn_coord": coord,
            })


def _prepare_target(builder, patch, target_distance, direction, rounds):
    """Validate without mutating either the circuit or its registered state."""
    if not isinstance(patch, RotatedSurfaceCode):
        raise TypeError("enlarge requires a RotatedSurfaceCode patch")
    require_global_patch(builder, patch)
    if type(target_distance) is not int or target_distance < 3 or target_distance % 2 == 0:
        raise ValueError("target_distance must be an odd integer at least 3")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer (total target SE rounds)")
    if direction != "+x,+y":
        raise ValueError("enlarge currently supports only direction='+x,+y'")
    if patch.distance_x != patch.distance_z:
        raise ValueError("enlarge currently requires a square source patch")
    if patch.distance_x < 3 or patch.distance_x % 2 == 0:
        raise ValueError("enlarge requires an odd source distance at least 3")
    if target_distance <= patch.distance_x:
        raise ValueError("target_distance must be greater than the current distance")
    if patch.is_transposed or not np.isclose(patch.rotation_angle % (2 * np.pi), 0):
        raise ValueError("enlarge currently supports translated, unrotated/untransposed layouts only")
    system = builder.system
    if system.disabled_data_qubit_indices & patch.data_indices:
        raise ValueError("enlarge does not yet support a patch with defects")
    if not builder.if_detector:
        raise ValueError("enlarge requires the tracker-enabled builder (if_detector=True)")
    if system._builder is not builder or system._tracker is not builder.tracker:
        raise ValueError("Register this builder and tracker with QECSystem before enlargement")
    if not patch.data_indices <= system.active_qubit_indices:
        raise ValueError("The source data must be initialized and active before enlargement")
    source_uids = set(patch._registered_stabilizer_uids)
    if not source_uids <= system.active_stabilizer_indices:
        raise ValueError("The source checks must all be active before enlargement")
    source_name, = [name for name, (p, _) in system.patches.items()
                    if getattr(p, "_registered_stabilizer_uids", None) == source_uids
                    and {system.local_to_global_map[name][q] for q in p.data_indices}
                    == patch.data_indices]

    # Actual geometry is authoritative. In particular, do not depend on the
    # historical ``shift`` bookkeeping in a caller-created patch.
    ox = min(patch.qubit_coords[q][0] for q in patch.data_indices) - 1
    oy = min(patch.qubit_coords[q][1] for q in patch.data_indices) - 1
    reference = RotatedSurfaceCode(distance=patch.distance_x)
    reference.shift_coords(ox, oy)
    if (set(reference.qubit_coords.values()) != set(patch.qubit_coords.values())
            or {_check_signature(reference, s) for s in reference.stabilizers}
            != {_check_signature(patch, s) for s in patch.stabilizers}):
        raise ValueError("Source geometry/checks are not the supported canonical rotated layout")
    target = RotatedSurfaceCode(distance=target_distance)
    target.shift_coords(ox, oy)
    target_coords = set(target.index_map)
    tx0, tx1, ty0, ty1 = target._get_bounds()
    for name, (other, _) in system.patches.items():
        if name == source_name:
            continue
        other_coords = set(other.index_map)
        if not other_coords:
            continue
        x0, x1, y0, y1 = other._get_bounds()
        footprint_overlap = max(tx0, x0) < min(tx1, x1) and max(ty0, y0) < min(ty1, y1)
        if target_coords & other_coords or footprint_overlap:
            raise ValueError(f"Enlarged patch overlaps registered patch {name!r}")

    # A declaration alone is not evidence that the input code was prepared.
    # Check the known state span, including logical-state rows whose current
    # representatives may contain stabilizer factors.
    tracker = builder.tracker
    full = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    records = tracker.stabilizers.records + tracker.logicals.records
    known = [i for i, rec in enumerate(records) if all(r >= 0 for r in rec)]
    canonical = stabilizers_to_symplectic(system, patch.stabilizers, tracker.num_qubits)
    if len(canonical):
        _, dependent, _ = solve_linear_decomposition(full[known], canonical, reduce_weight=False)
        if not dependent.all():
            raise ValueError("The source code checks must be established before enlargement")
    return source_name, target


def enlarge(builder, patch, *, target_distance, direction="+x,+y", rounds=1):
    """Grow a square patch toward +x,+y and return its new global patch view.

    ``rounds`` counts all target-distance SE rounds, including the growth
    projection. No old data are reset and no postselection policy is added.
    The original patch name and physical IDs survive; its old view retires.
    """
    source_name, target = _prepare_target(builder, patch, target_distance, direction, rounds)
    # Build against a private snapshot. A failed extraction or adoption must
    # not leave an allocated coupler or half-grown patch in the caller.
    staged = copy.deepcopy(builder)
    system = staged.system
    tracker = staged.tracker
    system.register_builder(staged)
    system.register_tracker(tracker)
    # A coherent encoder can establish the source checks before any SE has
    # classified its logical rows. Reuse adoption's known-span proof before
    # reclassifying staged state; this inserts no gates or new constraints.
    source_basis = stabilizers_to_symplectic(system, patch.stabilizers, tracker.num_qubits)
    classified_tracker = _prepare_tracker(tracker, system, source_basis)
    if classified_tracker is not None:
        tracker.__dict__.update(classified_tracker.__dict__)
    coupler_name = source_name + "__enlargement"
    suffix = 1
    while coupler_name in system.patches:
        coupler_name = f"{source_name}__enlargement_{suffix}"
        suffix += 1
    system.register_coupler(_EnlargementCoupler(), [source_name], name=coupler_name,
                            target_patch=target)
    coupler = system.coupler_patches[coupler_name]
    mapping = system.local_to_global_map[coupler_name]
    origin = (min(target.data_coords)[0], min(y for _, y in target.data_coords))
    init = {mapping[q]: corner_initialization_basis(*coupler.qubit_coords[q], origin)
            for q in sorted(coupler.data_indices)}
    if len(staged.circuit) and staged.circuit[-1].name != "TICK":
        staged.circuit.append("TICK")
    staged.initialize(init, n=system.num_qubits)
    staged.activate_coupler(coupler_name)

    # Extraction must touch only this patch. The real system remains intact
    # for tracker classification, including any entangled spectator patches.
    extraction_system = copy.copy(system)
    selected = (set(patch._registered_stabilizer_uids)
                | set(coupler._registered_stabilizer_uids))
    extraction_system.active_stabilizer_indices = system.active_stabilizer_indices & selected
    extraction_system.active_gauge_indices = set()
    extraction = RotatedSurfaceCodeExtractionBlock(extraction_system, scheduling="perpendicular")
    staged.apply_syndrome_extraction(extraction.circuit, rounds=rounds)
    if staged._z_only_syn_qubit_indices is not None:
        # The optional legacy Z-only final readout stores the last syndrome
        # order and count. Source-distance offsets cannot describe this new
        # round. Refresh from the actual native M instruction, while retaining
        # full X/Z detector generation during growth itself.
        measured = [target.value for target in extraction.circuit[-1].targets_copy()]
        x_syndromes = set(extraction_system.active_syndrome_indices_x)
        staged._z_only_syn_qubit_indices = measured
        staged._z_only_no_detector_mask = np.array(
            [q in x_syndromes for q in measured], dtype=bool,
        )
        staged._z_only_n_meas_per_round = len(measured)
    output = staged.combine_patches(
        patch_names=[source_name, coupler_name], output_patch=target, name=source_name,
    )

    # Preserve object identities held by QECExperiment and its LogicalExecutor.
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
