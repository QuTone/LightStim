"""Build a DEQ CODE block from a LightStim CSS QECPatch."""

from __future__ import annotations

from typing import Mapping

from lightstim.ir.qec_patch import QECPatch

from . import text


class DeqExportError(ValueError):
    """Raised when a LightStim object cannot be exported to DEQ."""


def is_css(patch: QECPatch) -> bool:
    """True if every stabilizer on ``patch`` is uniformly X or Z.

    DEQ's ``CODE { STABILIZER ... }`` block only supports same-basis Pauli
    products per check; LightStim subsystem/gauge codes can declare
    ``type == "Mixed"`` stabilizers, which this exporter does not support.
    """
    return all(stabilizer["type"] in ("X", "Z") for stabilizer in patch.stabilizers)


def local_index_map(patch: QECPatch) -> dict[int, int]:
    """Map each of the patch's global data-qubit indices to a local 0..n-1 index.

    This is the canonical local numbering used for the CODE's port qubits and
    must be reused verbatim by every gadget body built against this patch.
    """
    return {global_index: local_index for local_index, global_index in enumerate(sorted(patch.data_indices))}


def code_block(patch: QECPatch, name: str, *, distance: int | None = None) -> text.CodeBlock:
    """Render ``patch``'s stabilizers and logical operators as a DEQ CODE block."""
    if not is_css(patch):
        raise DeqExportError(
            f"{name}: DEQ CODE export only supports CSS (X/Z-only) stabilizers; "
            "found a non-CSS ('Mixed'-type) stabilizer. Subsystem/gauge codes "
            "are not supported by lightstim.deq."
        )

    local = local_index_map(patch)

    def to_product(pauli: Mapping[int, str]) -> text.PauliProduct:
        return text.PauliProduct.from_dict({local[qubit]: p for qubit, p in pauli.items()})

    logical_x = next(op["pauli"] for op in patch.logical_ops if op["type"] == "X")
    logical_z = next(op["pauli"] for op in patch.logical_ops if op["type"] == "Z")

    return text.CodeBlock(
        name=name,
        n=len(patch.data_indices),
        k=patch.num_logicals,
        d=distance,
        logical_x=to_product(logical_x),
        logical_z=to_product(logical_z),
        stabilizers=[to_product(stabilizer["pauli"]) for stabilizer in patch.stabilizers],
    )
