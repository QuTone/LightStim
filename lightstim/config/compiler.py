"""Compile the executable surface-code subset of the v1 experiment contract."""

from __future__ import annotations

from collections.abc import Mapping
import math

import stim

from lightstim.config.loader import COMPILE_FIELDS, ExperimentConfig, ResolvedConfig, resolve_compile_section
from lightstim.ir.builder import CircuitBuilder, DataReadoutStage
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler,
)


_SURFACE_CODES = {
    "rotated_sc": (RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock),
    "unrotated_sc": (UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock),
}


def _positive_integer(value, name):
    # A JSON number written as 3.0 has integer semantics in the contract.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _build_system(layout):
    system = QECSystem()
    code_ids = {patch["code"] for patch in layout["patches"]}
    if len(code_ids) != 1 or not code_ids <= _SURFACE_CODES.keys():
        raise ValueError(f"Expected a homogeneous rotated_sc or unrotated_sc system, got {sorted(code_ids)}")
    code_id = code_ids.pop()
    patch_class, block_class = _SURFACE_CODES[code_id]
    for declaration in layout["patches"]:
        name = declaration["id"]
        if declaration["code_family"] != "surface" or declaration.get("syndrome_extraction", "default") != "default":
            raise ValueError(f"Patch {name!r} requires an unsupported family or extraction schedule")
        params = declaration["params"]
        if set(params) not in ({"distance"}, {"distance_x", "distance_z"}):
            raise ValueError(f"Patch {name!r} requires distance or both distance_x and distance_z")
        distances = {key: _positive_integer(d, f"Patch {name!r} {key}") for key, d in params.items()}
        patch = patch_class(**distances)
        patch.apply_canvas_transform(declaration.get("transform", ()))
        offset = declaration["offset"]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in offset):
            raise ValueError(f"Patch {name!r} requires finite numeric offsets")
        overlapping = any(
            patch.snap_coord((x + offset[0], y + offset[1])) in system.index_map
            for x, y in patch.index_map
        )
        if overlapping:
            raise ValueError(f"Patch {name!r} overlaps another code patch")
        system.add_patch(patch, name=name, offset=tuple(offset))

    for coupler in layout.get("couplers", []):
        between = coupler["between"]
        if (code_id != "unrotated_sc" or len(between) != 2 or len(set(between)) != 2
                or coupler["interaction"] not in ("XX", "ZZ")):
            raise ValueError("Only two-patch unrotated XX and ZZ couplers are supported")
        if coupler.get("params"):
            raise ValueError(f"Unsupported coupler parameters on {coupler['id']!r}")
        system.register_coupler(UnrotatedTwoPatchCoupler(), patch_names=between,
                                name=coupler["id"], interaction_type=coupler["interaction"])
    return system, block_class


def _coupler_qubits(system, name):
    indices = system.local_to_global_map[name]
    return tuple(indices[q] for q in system.coupler_patches[name].data_indices)


def _data_bases(system, targets, basis):
    if isinstance(basis, str):
        bases = dict.fromkeys(targets, basis)
    elif isinstance(basis, dict) and set(basis) == set(targets):
        bases = basis
    else:
        raise ValueError("basis must be a single basis or a map covering exactly the targets")
    if any(value not in ("X", "Y", "Z") for value in bases.values()):
        raise ValueError("Each target basis must be X, Y or Z")

    # Match builder measurement-record order. The protocol traverses code
    # data in system order, then appends coupler data in patch-local order.
    data = {}
    for q in system.data_indices:
        owner = system.index_to_owner_map[q]
        if owner in bases and owner not in system.coupler_patches:
            data[q] = bases[owner]
    for name in targets:
        if name in system.coupler_patches:
            data.update(dict.fromkeys(_coupler_qubits(system, name), bases[name]))
    return data


def _apply_operation(builder, block_class, operation, *, terminal):
    system = builder.system
    name = operation["op"]
    targets = operation["targets"]
    active = [p for p in system.patches
              if p not in system.coupler_patches or p in system.paused_stabilizer_indices]
    if targets == ["@active"]:
        targets = active
    if len(set(targets)) != len(targets) or not set(targets) <= system.patches.keys():
        raise ValueError(f"Invalid or duplicate targets {targets!r}")

    params = operation.get("params", {})
    if name in {"initialize", "data_readout", "mid_data_readout"}:
        if params.keys() != {"basis"}:
            raise ValueError(f"{name} expects the basis parameter")
        data = _data_bases(system, targets, params["basis"])
        if name == "initialize":
            builder.initialize(init_dict=data, n=system.num_qubits)
        elif name == "mid_data_readout":
            builder.apply_mid_data_readout(measurements=data)
        else:
            closing = [p for p in targets if p in system.paused_stabilizer_indices] if terminal else []
            if closing:
                # Track the final LS boundaries in one physical measurement layer.
                stages = [DataReadoutStage(_coupler_qubits(system, p), (p,)) for p in closing]
                coupler_data = {q for stage in stages for q in stage.qubits}
                stages.append(DataReadoutStage(tuple(q for q in data if q not in coupler_data)))
                builder.apply_staged_data_readout(data, stages)
            else:
                builder.apply_data_readout(final_measurements=data)
    elif name == "syndrome_extraction":
        if params.keys() != {"rounds"}:
            raise ValueError("syndrome_extraction expects the rounds parameter")
        rounds = _positive_integer(params["rounds"], "rounds")
        if set(targets) != set(active):
            raise ValueError("Syndrome extraction must target every active patch and coupler")
        block = block_class(system)
        builder.apply_syndrome_extraction(
            circuit_chunk=block.circuit, rounds=rounds,
            measurement_blocks=getattr(block, "measurement_blocks", None),
        )
    elif name in {"activate_coupler", "deactivate_coupler"}:
        if params or len(targets) != 1 or targets[0] not in system.coupler_patches:
            raise ValueError(f"{name} requires a single coupler target and accepts no parameters")
        coupler = targets[0]
        if name == "activate_coupler" and coupler not in system.paused_stabilizer_indices:
            builder.activate_coupler(coupler)
        elif name == "deactivate_coupler" and coupler in system.paused_stabilizer_indices:
            builder.deactivate_coupler(coupler)
        else:
            raise ValueError(f"Invalid coupler state for {name} on {coupler!r}")
    else:
        raise ValueError(f"Operation {name!r} is not supported by the v1 compiler")


def compile(spec: ExperimentConfig | ResolvedConfig | Mapping) -> tuple[stim.Circuit, QECSystem]:
    """Build one noisy circuit and return its final system layout.

    A complete config or a mapping containing only parameters, system,
    operations and noise uses the default parameters. Pass a point from
    expand_sweep to compile other coordinates. The
    implementation entrypoint is metadata and is never imported or executed.
    """
    if isinstance(spec, ExperimentConfig):
        spec = spec.resolve()
    elif isinstance(spec, Mapping) and set(spec) == set(COMPILE_FIELDS):
        spec = resolve_compile_section(spec)
    elif not isinstance(spec, ResolvedConfig):
        spec = ExperimentConfig.from_dict(spec).resolve()

    system, block_class = _build_system(spec["system"])
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system)
    builder.write_coordinates()
    for i, operation in enumerate(spec["operations"]):
        try:
            _apply_operation(builder, block_class, operation, terminal=i == len(spec["operations"]) - 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"operations[{i}] ({operation['op']}): {exc}") from exc
    noise = spec["noise"]
    return builder.build_noisy_circuit(noise_params=NoiseConfig.from_spec(noise), noise_model=noise["model"]), system
