"""Load the packaged v1 contract and resolve one experiment at a time."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from itertools import product
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


SCHEMA_VERSION = "lightstim-config/v1"
COMPILE_FIELDS = ("parameters", "system", "operations", "noise")


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(files("lightstim.config").joinpath("schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _compile_validator() -> Draft202012Validator:
    validator = _validator()
    properties = validator.schema["properties"]
    return validator.evolve(schema={
        "type": "object", "additionalProperties": False, "required": list(COMPILE_FIELDS),
        "properties": {key: properties[key] for key in COMPILE_FIELDS},
    })


def _require_finite_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Invalid non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, f"{path}[{index}]")


def _validate(spec: Any, section: str | None = None) -> None:
    _require_finite_numbers(spec)
    validator = _validator()
    if section == "compile":
        validator = _compile_validator()
    elif section:
        validator = validator.evolve(schema={"$ref": f"#/$defs/{section}"})
    try:
        validator.validate(spec)
    except ValidationError as exc:
        raise ValueError(f"Invalid {section or 'experiment config'} at {exc.json_path}: {exc.message}") from exc


def _validate_names(spec: dict) -> None:
    system = spec["system"]
    patches = [patch["id"] for patch in system["patches"]]
    couplers = system.get("couplers", [])
    names = patches + [coupler["id"] for coupler in couplers]
    if len(set(names)) != len(names):
        raise ValueError("Patch and coupler ids must be unique")
    for coupler in couplers:
        unknown = set(coupler["between"]).difference(patches)
        if unknown:
            raise ValueError(f"Coupler {coupler['id']!r} refers to unknown patches: {sorted(unknown)}")
        if len(coupler["interaction"]) != len(coupler["between"]):
            raise ValueError(f"Coupler {coupler['id']!r} needs one Pauli per patch in 'between'")
    for index, op in enumerate(spec["operations"]):
        unknown = set(op["targets"]).difference(names, {"@active"})
        if unknown:
            raise ValueError(f"operations[{index}] refers to unknown targets: {sorted(unknown)}")


def _sweep_groups(spec: dict) -> list[dict]:
    sweep = spec.get("sweep", {})
    groups = [{axis: values} for axis, values in sweep.get("axes", {}).items()]
    groups.extend(sweep.get("zip", []))
    seen: set[str] = set()
    for group in groups:
        unknown = group.keys() - spec["parameters"].keys()
        if unknown:
            raise ValueError(f"Sweep uses undeclared parameters: {sorted(unknown)}")
        repeated = seen.intersection(group)
        if repeated:
            raise ValueError(f"Sweep axes occur more than once: {sorted(repeated)}")
        if len({len(values) for values in group.values()}) != 1:
            raise ValueError(f"Zipped sweep axes must have equal lengths: {list(group)}")
        seen.update(group)
    return groups


class _ConfigMapping(Mapping[str, Any]):
    spec: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.spec[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.spec)

    def __len__(self) -> int:
        return len(self.spec)


@dataclass
class ResolvedConfig(_ConfigMapping):
    """One independent run with literal parameters and its sweep coordinates."""

    spec: dict[str, Any]
    coordinates: dict[str, Any]


@dataclass
class ExperimentConfig(_ConfigMapping):
    """Validated experiment template. Only compile fields interpret parameter references."""

    spec: dict[str, Any]

    def __post_init__(self) -> None:
        self.spec = deepcopy(dict(self.spec))
        # The Zoo's two preview bundles predate the v1 version identifier.
        if self.spec.get("schema_version") == "lightstim-config/v1-draft":
            self.spec["schema_version"] = SCHEMA_VERSION
        _validate(self.spec)
        # JSON Schema integers include 8.0. Sampling and indexing APIs need
        # Python ints. Normalize after validation so fractional inputs fail.
        sim = self.spec["simulation"]
        for name in ("max_shots", "max_errors", "num_workers", "batch_size", "seed"):
            if name in sim:
                sim[name] = int(sim[name])
        if sim.get("target_observable_indices") is not None:
            sim["target_observable_indices"] = list(map(int, sim["target_observable_indices"]))
        selection = sim.get("postselection", {})
        for name in ("detector_indices", "observable_indices"):
            if name in selection:
                selection[name] = list(map(int, selection[name]))
        _validate_names(self.spec)
        _sweep_groups(self.spec)
        self.resolve()

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> ExperimentConfig:
        return cls(dict(spec))

    def resolve(self, overrides: Mapping[str, Any] | None = None) -> ResolvedConfig:
        """Substitute scalar overrides before following aliases and affine refs."""
        coordinates = dict(overrides or {})
        point = {key: deepcopy(value) for key, value in self.spec.items()
                 if key not in COMPILE_FIELDS and key != "sweep"}
        point.update(_resolve_compile_fields(self.spec, coordinates))
        return ResolvedConfig(point, coordinates)


def _resolve_compile_fields(spec: Mapping[str, Any], overrides: dict) -> dict:
    parameters = spec["parameters"] | overrides
    unknown = overrides.keys() - spec["parameters"].keys()
    if unknown:
        raise ValueError(f"Overrides use undeclared parameters: {sorted(unknown)}")
    for value in overrides.values():
        _validate(value, "scalar")
    resolved: dict[str, Any] = {}

    def parameter(key: str, path: tuple[str, ...]) -> Any:
        if key in resolved:
            return resolved[key]
        if key not in parameters:
            raise ValueError(f"Unknown parameter {key!r}")
        if key in path:
            raise ValueError(f"Parameter alias cycle: {' -> '.join((*path, key))}")
        resolved[key] = substitute(parameters[key], (*path, key))
        return resolved[key]

    def substitute(value: Any, path: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [substitute(item, path) for item in value]
        if isinstance(value, dict):
            if "$param" not in value:
                return {key: substitute(item, path) for key, item in value.items()}
            key = value["$param"]
            result = parameter(key, path)
            if "times" in value or "plus" in value:
                if type(result) not in (int, float):
                    raise ValueError(f"Affine reference to {key!r} requires a numeric value")
                try:
                    result = result * value.get("times", 1) + value.get("plus", 0)
                except OverflowError as exc:
                    raise ValueError(f"Affine reference to {key!r} exceeds the numeric range") from exc
            return result
        return value

    for key in parameters:
        parameter(key, ())
    fields = {"parameters": resolved}
    for key in ("system", "operations", "noise"):
        fields[key] = substitute(spec[key])
    # Parameter declarations permit any scalar. Its use can have tighter
    # constraints, such as a numeric offset or a physical error probability.
    _validate(fields, "compile")
    return fields


def resolve_compile_section(spec: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict:
    """Validate and resolve exactly the four compile fields of the v1 contract."""
    fields = dict(spec)
    _validate(fields, "compile")
    _validate_names(fields)
    return _resolve_compile_fields(fields, dict(overrides or {}))


def _invalid_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON number {token!r}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a v1 or Zoo v1-draft JSON file and validate the default point."""
    with Path(path).open(encoding="utf-8") as source:
        spec = json.load(source, parse_constant=_invalid_constant)
    if not isinstance(spec, dict):
        raise ValueError("An experiment config must be a JSON object")
    return ExperimentConfig.from_dict(spec)


def expand_sweep(config: ExperimentConfig) -> Iterator[ResolvedConfig]:
    """Yield the Cartesian product of axes and zip groups, or one default run."""
    groups = _sweep_groups(config.spec)
    dimensions = [
        [dict(zip(group, values)) for values in zip(*group.values())]
        for group in groups
    ]
    for choices in product(*dimensions):
        yield config.resolve({key: value for choice in choices for key, value in choice.items()})
