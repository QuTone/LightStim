"""Explore the geometry and operators of a QECPatch or QECSystem in standalone HTML.

This viewer is independent of circuit generation. It preserves physical qubit
IDs, coordinates, Pauli factors and signs; support polygons are not gate edges.
It does not infer code distance, stabilizer independence, or an SE schedule.
The Pauli color convention is X red, Z blue, and Y green.

Views are limited to 500 registered qubits, including auxiliary qubits. A
standalone patch must supply coordinates for its entire support. For couplers
and multi-patch geometry, pass the QECSystem that owns the global qubit IDs.

Example::

    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode

    export_patch_html(RotatedSurfaceCode(distance=3), "code.html")

Overlays may overlap. Each has a name, an iterable of existing qubit IDs and an
optional six-digit hex color. They are annotations, not additional operators.
They are supplied explicitly by the caller and are empty by default. The viewer
does not infer regions.

CLI::

    python -m lightstim.frontend.patch_html \\
        --factory lightstim.qec_code.surface_code.rotated:RotatedSurfaceCode \\
        --kwargs '{"distance": 3}' --output code.html
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from importlib.resources import files
import importlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any

import stim

from lightstim.ir.coupler import LogicalCouplerPatch
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem


_COLORS = ("#188578", "#a76b20", "#7553ab", "#4775a5")
_MAX_QUBITS = 500


@dataclass(frozen=True)
class PatchVisualization:
    """A standalone HTML snapshot, displayable in a trusted notebook.

    Leave this object as the cell's last expression, or pass it to ``display``.
    Each notebook display uses its own iframe so viewer scripts, element IDs
    and styles remain isolated. No IPython dependency or server is required.
    ``html`` and ``str(view)`` expose the complete standalone document.
    """

    html: str

    def __repr__(self) -> str:
        return "<PatchVisualization: interactive HTML; use .save_html(path) to export>"

    def __str__(self) -> str:
        return self.html

    def save_html(self, output: str | Path) -> Path:
        """Save this snapshot, creating parent folders, and return its path."""
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html, encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        # Only scripts are enabled; the embedded document cannot access the
        # notebook's DOM. Attribute escaping keeps the entire document in srcdoc.
        return (
            '<iframe title="LightStim QEC structure" sandbox="allow-scripts" '
            'style="width:100%;height:850px;border:0" '
            f'srcdoc="{escape(self.html, quote=True)}"></iframe>'
        )


def _qubit_id(value: Any) -> int:
    # numpy integers are valid IDs; floats and strings would silently remap IDs.
    import operator

    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"Qubit ID must be an integer, got {value!r}.") from exc
    if isinstance(value, bool) or result < 0:
        raise ValueError(f"Qubit ID must be a nonnegative integer, got {value!r}.")
    return result


def _operator(record: Mapping, category: str, index: int, known: set[int],
              names: Counter) -> dict:
    pauli = record.get("pauli")
    phase = complex(record.get("sign", 1))
    if "phase" in record:
        raise ValueError("Use a signed stim.PauliString or sign=+1/-1; phase metadata is ambiguous.")
    if isinstance(pauli, stim.PauliString):
        phase *= pauli.sign
        support = {q: "_XYZ"[pauli[q]] for q in pauli.pauli_indices()}
    elif isinstance(pauli, Mapping):
        support = {}
        for raw_q, factor in pauli.items():
            q = _qubit_id(raw_q)
            if factor not in ("I", "_", "X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli factor {factor!r} on qubit {q}.")
            if factor in ("X", "Y", "Z"):
                support[q] = factor
    else:
        raise ValueError("Operator 'pauli' must be a Pauli dictionary or stim.PauliString.")
    if phase not in (1, -1):
        raise ValueError("Displayed stabilizers, gauges and logical observables must have real sign +1 or -1.")
    missing = set(support) - known
    if missing:
        raise ValueError(f"Operator references qubits without coordinates: {sorted(missing)}.")
    factors = set(support.values())
    basis = next(iter(factors)) if len(factors) == 1 else ("mixed" if factors else "I")
    declared_type = str(record.get("type") or "")
    names[(category, declared_type or basis)] += 1
    number = names[(category, declared_type or basis)]
    prefix = {"stabilizer": "S", "logical": "L", "gauge": "G"}[category]
    default_name = f"{prefix}_{declared_type or basis}{number}"
    name = str(record.get("name") or record.get("label") or default_name)
    sign = 1 if phase == 1 else -1
    pairs = [[q, support[q]] for q in sorted(support)]
    syndrome = record.get("syn_idx")
    if syndrome is not None:
        syndrome = _qubit_id(syndrome)
        if syndrome not in known:
            raise ValueError(f"Syndrome qubit {syndrome} has no coordinates.")
    return {
        "id": f"{category}-{index}", "name": name, "category": category,
        "basis": basis, "declared_type": declared_type, "sign": sign,
        "support": pairs, "weight": len(pairs), "syndrome_qubit": syndrome,
        "pauli_string": ("+" if sign == 1 else "−") + " ".join(
            f"{factor}{q}" for q, factor in pairs
        ) if pairs else ("+I" if sign == 1 else "−I"),
    }


def _check_size(count: int, subject: str = "Patch") -> None:
    if count > _MAX_QUBITS:
        raise ValueError(
            f"{subject} visualization supports at most {_MAX_QUBITS} registered qubits "
            f"(including auxiliary qubits); got {count}."
        )


def _qubit_role(source: QECPatch | QECSystem, q: int) -> str:
    if q in source.data_indices:
        return "data"
    if q in source.syndrome_indices_x:
        return "syndrome X"
    if q in source.syndrome_indices_z:
        return "syndrome Z"
    if q in source.syndrome_indices:
        return "syndrome"
    return "other"


def _qubit_record(q: int, coord: Sequence[float], role: str) -> dict:
    q = _qubit_id(q)
    if len(coord) != 2 or not all(math.isfinite(float(v)) for v in coord):
        raise ValueError(f"Qubit {q} requires two finite coordinates.")
    return {"id": q, "x": float(coord[0]), "y": float(coord[1]), "role": role}


def _annotations(overlays: Sequence[Mapping] | None, known: set[int]) -> list[dict]:
    annotations = []
    for i, overlay in enumerate(overlays or []):
        color = overlay.get("color", _COLORS[i % len(_COLORS)])
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Overlay color must be a six-digit hex color, e.g. #188578.")
        members = sorted({_qubit_id(q) for q in overlay["qubits"]})
        if set(members) - known:
            raise ValueError(f"Overlay references unknown qubits: {sorted(set(members) - known)}.")
        annotations.append({"name": str(overlay["name"]), "qubits": members, "color": color})
    return annotations


def _configured_parameters(patch: QECPatch) -> dict:
    """Include only explicit displayable inputs to the patch constructor."""
    parameters = {}
    for key in ("distance", "distance_x", "distance_z", "boundary_conditions"):
        value = patch.params.get(key)
        if isinstance(value, Integral) and not isinstance(value, bool):
            parameters[key] = int(value)
        elif isinstance(value, str):
            parameters[key] = value
    return parameters


def patch_view_data(patch: QECPatch, *, title: str | None = None,
                    overlays: Sequence[Mapping] | None = None) -> dict:
    """Return the JSON-compatible viewer snapshot without modifying ``patch``.

    Operator records accept QECPatch Pauli dictionaries (optionally ``sign``
    +/-1) or signed Stim PauliStrings. ``type`` is a label, never a replacement
    for the actual Pauli factors. Gauges retain their separate category.
    Unregistered support and non-Hermitian operator phases are rejected.
    ``name`` and ``label`` are optional display metadata. Unnamed records get
    generated labels such as ``S_X1``, ``S_Z1`` or ``L_X1``. The exporter reads
    only the patch object and supplied overlays, not the patch's source files.
    Self-contained patches with at most 500 registered qubits are supported.
    Coupler patches require QECSystem context and raise ``ValueError``.
    """
    if not isinstance(patch, QECPatch):
        raise TypeError("Expected a QECPatch instance.")
    # Bound work before copying operators or building an HTML document. Count
    # every displayed qubit, including syndrome/flag qubits, not only data.
    _check_size(len(patch.qubit_coords))
    # Couplers can refer to qubits owned by other patches, even after system
    # registration. Resolving those references needs a system-level viewer;
    # inventing local IDs here would misrepresent the physical support.
    if isinstance(patch, LogicalCouplerPatch):
        raise ValueError(
            "Patch visualization supports self-contained patches only; coupler "
            "patches require surrounding QECSystem geometry to resolve cross-patch support."
        )
    qubits = [
        _qubit_record(q, coord, _qubit_role(patch, q))
        for q, coord in sorted(patch.qubit_coords.items())
    ]
    known = {q["id"] for q in qubits}
    names = Counter()
    operators = []
    for category, records in (("stabilizer", patch.stabilizers),
                              ("logical", patch.logical_ops),
                              ("gauge", getattr(patch, "gauges", []))):
        operators.extend(_operator(record, category, i, known, names)
                         for i, record in enumerate(records))
    return {
        "schema": "lightstim.qec-patch-view.v1",
        "title": str(title or type(patch).__name__), "patch_class": type(patch).__name__,
        "num_logicals": int(patch.num_logicals), "qubits": qubits,
        "operators": operators, "overlays": _annotations(overlays, known),
        "parameters": _configured_parameters(patch),
    }


def system_view_data(system: QECSystem, *, title: str | None = None,
                     overlays: Sequence[Mapping] | None = None) -> dict:
    """Describe current patch geometry, cross-patch checks and logical operators.

    Coordinates, Pauli factors, syndrome IDs and overlay IDs use the system's
    global numbering. Shared sites and deduplicated checks appear once. A check
    carries every declaring patch and its current activation state. Stabilizer
    support includes the system's defect projection. Retired patch definitions
    and sites retained only for circuit history are omitted. The logical count
    is the sum declared by those registered patches.
    """
    if not isinstance(system, QECSystem):
        raise TypeError("Expected a QECSystem instance.")

    patches = []
    site_roles = defaultdict(dict)
    check_owners = defaultdict(list)
    for name, (patch, offset) in system.patches.items():
        # Stored patches have shifted coordinates and local IDs. The registry
        # includes reuse of a dormant site by more than one patch.
        registration = system.local_to_global_map[name]
        for local_id, global_id in registration.items():
            site_roles[global_id][name] = _qubit_role(patch, local_id)
        descriptor = {
            "id": name, "kind": "coupler" if isinstance(patch, LogicalCouplerPatch) else "patch",
            "patch_class": type(patch).__name__, "qubits": sorted(registration.values()),
            "offset": [float(value) for value in offset],
        }
        parameters = _configured_parameters(patch)
        if parameters:
            descriptor["parameters"] = parameters
        patches.append(descriptor)
        for category, uids in (("stabilizer", patch._registered_stabilizer_uids),
                               ("gauge", patch._registered_gauge_uids)):
            for uid in uids:
                check_owners[category, uid].append(name)

    known = set(site_roles)
    _check_size(len(known), "System")
    qubits = []
    for q in sorted(known):
        roles = site_roles[q]
        if q not in system.qubit_coords:
            raise ValueError(
                f"Registered qubit {q} in {', '.join(roles)} has no global system coordinates."
            )
        owner = system.index_to_owner_map.get(q)
        role = roles[owner] if owner in roles else _qubit_role(system, q)
        qubit = _qubit_record(q, system.qubit_coords[q], role)
        qubit["owner"] = owner
        if len(set(roles.values())) > 1:
            qubit["roles_by_patch"] = dict(roles)
        if q in system.disabled_data_qubit_indices:
            qubit["disabled"] = True
        qubits.append(qubit)

    names = defaultdict(Counter)
    operators = []
    for category, records in (("stabilizer", system.stabilizers),
                              ("logical", system.logical_ops),
                              ("gauge", system.gauges)):
        for uid, record in enumerate(records):
            primary = record.get("patch_name")
            if category == "logical":
                if primary not in system.patches:
                    continue
                owners = [primary]
            else:
                owners = check_owners.get((category, uid))
                if not owners:
                    continue
                if primary not in owners:
                    primary = owners[0]
            if category == "stabilizer":
                record = system.effective_stabilizer(uid)
            op = _operator(record, category, uid, known, names[primary])
            op.update(id=f"{primary}:{op['id']}", patch=primary, patches=list(owners))
            if category == "stabilizer":
                op["active"] = uid in system.active_stabilizer_indices
            elif category == "gauge":
                op["active"] = uid in system.active_gauge_indices
            operators.append(op)

    return {
        "schema": "lightstim.qec-system-view.v1",
        "title": str(title or type(system).__name__), "system_class": type(system).__name__,
        "num_logicals": sum(int(p.num_logicals) for p, _ in system.patches.values()),
        "patches": patches, "qubits": qubits, "operators": operators,
        "overlays": _annotations(overlays, known),
    }


def patch_html(patch: QECPatch | QECSystem, *, title: str | None = None,
               overlays: Sequence[Mapping] | None = None) -> str:
    """Render the registered geometry as a self-contained HTML document."""
    if isinstance(patch, QECSystem):
        data = system_view_data(patch, title=title, overlays=overlays)
    else:
        data = patch_view_data(patch, title=title, overlays=overlays)

    assets = files("lightstim.frontend")
    template = assets.joinpath("_patch_viewer.html").read_text(encoding="utf-8")
    # Escape the raw JSON script context. A single substitution pass also keeps
    # literal template markers inside user-provided names and titles intact.
    parts = {
        "__TITLE__": escape(data["title"]),
        "__PAYLOAD__": json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c"),
        "__VIEWER_JS__": assets.joinpath("_patch_viewer.js").read_text(encoding="utf-8"),
    }
    return re.sub(r"__TITLE__|__PAYLOAD__|__VIEWER_JS__", lambda match: parts[match[0]], template)


def export_patch_html(patch: QECPatch | QECSystem, output: str | Path, *, title: str | None = None,
                      overlays: Sequence[Mapping] | None = None) -> Path:
    """Write the standalone viewer, creating parent folders, and return its path."""
    return PatchVisualization(patch_html(patch, title=title, overlays=overlays)).save_html(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factory", required=True, help="Importable module:callable returning a QECPatch or QECSystem.")
    parser.add_argument("--kwargs", default="{}", help="JSON object passed to the factory.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()
    try:
        module_name, factory_name = args.factory.split(":", 1)
        kwargs = json.loads(args.kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--kwargs must be a JSON object.")
        factory = getattr(importlib.import_module(module_name), factory_name)
        path = export_patch_html(factory(**kwargs), args.output, title=args.title)
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == "__main__":
    main()
