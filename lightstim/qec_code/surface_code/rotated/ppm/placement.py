"""Patch placement helpers for rotated-surface-code PPM lowering.

Adapted from John Yuehan Zhang's CircLS repository at commit ``8802a5b``.
This module describes physical rotated-surface-code placement only. Protocol
sequence objects live in :mod:`lightstim.protocols.rotated_surface_ppm`.
"""

from dataclasses import dataclass

from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode

_FLIP = {"X": "Z", "Z": "X"}


class RotatedSurfacePPMLayoutError(ValueError):
    """Raised when the requested patch placement cannot host the joint
    measurement (e.g. the two patches are misaligned, or their parity/seam
    endpoints are incompatible).  The message states the concrete geometric
    reason — it is NOT a generator hard-coding limit."""


@dataclass(frozen=True)
class RotatedSurfacePatchPlacement:
    """A logical patch placed on the coarse routing grid.

    origin:           bus-facing corner data coord, in the library (1,1)-corner convention.
    distance:         odd code distance.
    orientation:      "X_horizontal" (X̄ runs along a row) | "X_vertical" (X̄ runs up a column).

    Which logical each patch contributes to the joint ``M(∏ᵢ P̄ᵢ)`` is given
    separately by the PPM ``targets`` list (e.g.
    ``[("Q1", "Z"), ("Q2", "Z")]``), not by the patch itself.
    """
    name: str
    origin: tuple
    distance: int
    orientation: str


def place_patch(spec):
    """Build one rotated patch placed/oriented per ``spec``; return coord-keyed dicts.

    Returns dict(data=set[(col,row)], checks=list[checkdict], x_support, z_support).
    Reuses RotatedSurfaceCode: build at the (1,1) corner, transpose for X_horizontal,
    then shift so the corner lands at ``origin``.
    """
    if spec.orientation not in ("X_horizontal", "X_vertical"):
        raise ValueError(f"orientation must be X_horizontal|X_vertical, got {spec.orientation!r}")
    code = RotatedSurfaceCode(distance=spec.distance)
    if spec.orientation == "X_horizontal":
        code.transpose_coords()                       # default X̄ vertical -> horizontal
    code.shift_coords(spec.origin[0] - 1, spec.origin[1] - 1)
    qc = code.qubit_coords
    data = {tuple(qc[i]) for i in code.data_indices}
    checks = []
    for s in code.stabilizers:
        pauli = {tuple(qc[q]): P for q, P in s["pauli"].items()}
        checks.append({"syn": tuple(s["syn_coord"]), "type": s["type"],
                       "pauli": pauli, "corners": sorted(pauli)})
    x_support = next(sorted(tuple(qc[q]) for q in lo["pauli"])
                     for lo in code.logical_ops if lo["type"] == "X")
    z_support = next(sorted(tuple(qc[q]) for q in lo["pauli"])
                     for lo in code.logical_ops if lo["type"] == "Z")
    return dict(data=data, checks=checks, x_support=x_support, z_support=z_support)


def conjugate_patch_records(code):
    """Typeswap a built patch IN PLACE: every stabilizer/logical record swaps
    X<->Z (type and per-qubit Paulis) and the syndrome-role index sets swap.

    The result is the CONJUGATE-CONVENTION patch: textbook-shaped on the same
    qubits with the checkerboard colours exchanged — exactly the structure the
    seam-column painting assigns to a minority patch.  A minority patch
    registered this way matches the routed merged check set natively (rule-0
    lock), so the protocol contains no transversal-H morph anywhere."""
    for rec in code.stabilizers:
        rec["type"] = _FLIP.get(rec["type"], rec["type"])
        rec["pauli"] = {q: _FLIP.get(P, P) for q, P in rec["pauli"].items()}
    for rec in code.logical_ops:
        rec["type"] = _FLIP.get(rec["type"], rec["type"])
        rec["pauli"] = {q: _FLIP.get(P, P) for q, P in rec["pauli"].items()}
    xs = getattr(code, "syndrome_indices_x", None)
    zs = getattr(code, "syndrome_indices_z", None)
    if xs is not None and zs is not None:
        code.syndrome_indices_x, code.syndrome_indices_z = zs, xs
    return code


# -----------------------------------------------------------------------------
# The coarse routing grid (seam-column design, pitch 2d+2)
# -----------------------------------------------------------------------------

def _pitch(d, seam=False):
    """Coarse-grid pitch: ``2d`` for the abutting design, ``2d+2`` for the SEAM-COLUMN
    design (standard-LS style: one freshly-initialized data column/row between every
    pair of edge-adjacent occupied cells)."""
    return 2 * d + 2 if seam else 2 * d


def cell(a, b, d, seam=False):
    """The ``d×d`` data-qubit footprint of coarse cell ``(a, b)`` (rotated frame)."""
    P = _pitch(d, seam)
    x0, y0 = 1 + P * a, 1 + P * b
    return {(x, y) for x in range(x0, x0 + 2 * d, 2) for y in range(y0, y0 + 2 * d, 2)}


def origin_of(a, b, d, seam=False):
    """The data-qubit **origin** (bus-facing ``(1,1)``-corner) of coarse cell ``(a, b)`` —
    i.e. the :attr:`RotatedSurfacePatchPlacement.origin` that places a distance-``d`` patch on that cell.
    Inverse of :func:`cell_index`.  ``seam=True`` uses the seam-column grid (pitch ``2d+2``)."""
    P = _pitch(d, seam)
    return (1 + P * a, 1 + P * b)


def cell_index(origin, d, seam=False):
    """The coarse cell ``(a, b)`` a distance-``d`` patch placed at ``origin`` occupies.  Inverse of
    :func:`origin_of`.  Raises ``ValueError`` if ``origin`` is off the pitch grid (patches are
    placed only on that grid; use :func:`origin_of` to land on it)."""
    P = _pitch(d, seam)
    ox, oy = origin
    if (ox - 1) % P or (oy - 1) % P:
        raise ValueError(f"origin {origin} is not on the pitch-{P} coarse grid for d={d}")
    return ((ox - 1) // P, (oy - 1) // P)
