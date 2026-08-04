"""The XZXZ <-> XZZX Y-transition round, on LightStim global qubit indices.

Port of Craig Gidney's ``make_y_transition_round_nesw_xzxz_to_xzzx``
(``code/src/midout/circuits/steps/_measure_y_transition_round.py``, zenodo
record 7487893, *Inplace Access to the Surface Code Y Basis*).

What the round does
-------------------
One native round of gates that converts an ordinary rotated surface-code patch
(:class:`~lightstim.qec_code.surface_code.rotated.code_patch.RotatedSurfaceCode`,
== Gidney's ``make_xtop_qubit_patch``) into the degenerate XXZZ patch
(:func:`~lightstim.qec_code.surface_code.rotated.y_boundary_patch.make_degenerate_y_boundary_patch`,
== his ``make_ztop_yboundary_patch``), consuming every check of the former,
preparing every check of the latter, and measuring ``Y_L`` on the way through.
That is ``direction='measure'``.

``direction='init'`` is the time reverse of the same round: it consumes the
degenerate patch's checks, prepares the qubit patch's checks, and *prepares*
``Y_L``.  It is what Gidney's published ``basis='Y'`` memory circuit uses
(``qubit_to_boundary_round.inverted()`` in ``circuits/_y_memory_circuit.py``).

Output
------
A bare :class:`stim.Circuit` in **global** qubit indices, TICK-separated, with
no ``QUBIT_COORDS`` / ``DETECTOR`` / ``OBSERVABLE_INCLUDE`` annotations — a
chunk for a caller that owns detector bookkeeping.  Gates are emitted
**natively** (``R RX RY CX XCY H SQRT_X M MX MY``); nothing is compiled down to
``CZ``, and the ``MX`` / ``MY`` are left as they are rather than being
rewritten as ``H`` + ``M``.

Coordinate bridge
-----------------
Gidney data qubit ``g = gx + gy*1j``  ->  LightStim ``(2*gx + 1, 2*gy + 1)``.
Half-integer ``g`` (his measure qubits) therefore land on even LightStim
coordinates.  The origin ``g = 0`` is taken to be the lowest data coordinate of
the qubit patch, so a shifted placement works unchanged.

Faithfulness notes
------------------
* Gate order, target order and the pairing/orientation of every two-qubit gate
  reproduce ``gen.Builder`` exactly: single-qubit targets are sorted by
  ``complex_key = (real is not an integer, real, imag)``, two-qubit pairs by
  the same key applied to (first, second), and control/target orientation is
  whatever ``toward(..., sign)`` produced.
* ``direction='init'`` reproduces ``gen.Chunk.inverted()``
  (``gen/_flow_verifier.py::FlowStabilizerVerifier.invert``): instructions are
  emitted back to front, ``R/RX/RY <-> M/MX/MY``, targets reversed, and
  two-qubit gates keep their name with the *order of the pairs* reversed while
  each pair stays intact.
* **Deviation worth naming**: Gidney's ``REV_DICT`` maps ``SQRT_X -> SQRT_X``,
  not ``SQRT_X -> SQRT_X_DAG``.  The two differ by an ``X`` on the ``d-1``
  diagonal ancillas, so this is a Pauli-frame choice, not an error — and it is
  the choice baked into the published golden circuit, which is
  detector-deterministic as generated.  This port keeps his mapping so the
  ``init`` chunk is byte-identical to his.
* His inverter turns ``M`` into ``R`` only for measurements it has proved can
  be destructive.  For this round every measurement qualifies (checked against
  the golden circuit for d=3 and d=5), so the swap here is unconditional.
"""
from typing import AbstractSet, Dict, Iterable, List, Set, Tuple, Union

import stim

from lightstim.ir.qec_patch import QECPatch

__all__ = ["make_y_transition_chunk", "time_reverse_chunk"]

# --- Gidney _patches.py, verbatim -------------------------------------------
DIRS = [(0.5 + 0.5j) * 1j ** d for d in range(4)]
DR, DL, UL, UR = DIRS


# --- Gidney _measure_y_transition_round.py, verbatim -------------------------

def _m_basis(m: complex):
    if m.real % 1 == 0:
        return None
    is_x = int(m.real + m.imag) & 1 == 0
    return 'X' if is_x else 'Z'


def _split_dl_md_ur(ps: AbstractSet[complex]
                    ) -> Tuple[Set[complex], Set[complex], Set[complex]]:
    dl = set()
    ur = set()
    md = set()
    for m in ps:
        dst = ur if m.real > m.imag + 1 else md if m.real == m.imag or m.real == m.imag + 1 else dl
        dst.add(m)
    return dl, md, ur


# --- gen.Builder ordering ----------------------------------------------------

def _complex_key(c: complex):
    """Gidney ``gen/_util.py::complex_key``."""
    return (c.real != int(c.real), c.real, c.imag)


def _sorted_complex(values: Iterable[complex]) -> List[complex]:
    return sorted(values, key=_complex_key)


class _Emitter:
    """The slice of ``gen.Builder`` this port needs, writing global indices."""

    def __init__(self, q2i: Dict[complex, int]):
        self.q2i = q2i
        self.circuit = stim.Circuit()

    def gate(self, name: str, qubits: Iterable[complex]) -> None:
        qs = _sorted_complex(set(qubits))
        if not qs:
            return
        self.circuit.append(name, [self.q2i[q] for q in qs])

    def gate2(self, name: str, pairs: Iterable[Tuple[complex, complex]]) -> None:
        ps = sorted(pairs, key=lambda pair: (_complex_key(pair[0]),
                                             _complex_key(pair[1])))
        if not ps:
            return
        self.circuit.append(name, [self.q2i[q] for pair in ps for q in pair])

    def tick(self) -> None:
        self.circuit.append('TICK')

    def measure(self, qubits: Iterable[complex], *, basis: str) -> None:
        qs = _sorted_complex(set(qubits))
        if not qs:
            return
        self.circuit.append(f'M{basis}', [self.q2i[q] for q in qs])


# --- time reversal (gen/_flow_verifier.py) -----------------------------------

_FLIP_REV_SET = {"CX", "CY", "CZ", "XCY"}
_REV_DICT = {
    "I": "I", "X": "X", "Y": "Y", "Z": "Z",
    "C_XYZ": "C_ZYX", "C_ZYX": "C_XYZ",
    "H": "H", "H_XY": "H_XY", "H_XZ": "H_XZ", "H_YZ": "H_YZ",
    "S": "S", "S_DAG": "S",
    "SQRT_X": "SQRT_X", "SQRT_X_DAG": "SQRT_X",
    "SQRT_Y": "SQRT_Y", "SQRT_Y_DAG": "SQRT_Y",
    "SWAP": "SWAP", "XCX": "XCX", "ISWAP": "ISWAP", "ISWAP_DAG": "ISWAP",
}


def time_reverse_chunk(circuit: stim.Circuit) -> stim.Circuit:
    """Time-reverse a reset/unitary/measure chunk, Gidney's ``invert`` rules.

    Instructions are emitted back to front; ``R/RX/RY`` become ``M/MX/MY`` and
    vice versa; targets are reversed; two-qubit gates in ``_FLIP_REV_SET`` keep
    their name and orientation with only the order of their pairs reversed.
    """
    out = stim.Circuit()
    for inst in list(circuit)[::-1]:
        if isinstance(inst, stim.CircuitRepeatBlock):
            raise NotImplementedError("REPEAT blocks cannot be time-reversed here")
        name = inst.name
        args = inst.gate_args_copy()
        targets = inst.targets_copy()
        if name == 'TICK':
            out.append('TICK')
        elif name in _FLIP_REV_SET:
            new_targets = [
                targets[k + i]
                for k in range(0, len(targets), 2)[::-1]
                for i in range(2)
            ]
            out.append(name, new_targets, args)
        elif name in _REV_DICT:
            out.append(_REV_DICT[name], targets[::-1], args)
        elif name in ("R", "RX", "RY"):
            out.append(name.replace("R", "M"), targets[::-1], args)
        elif name in ("M", "MX", "MY"):
            out.append(name.replace("M", "R"), targets[::-1], args)
        else:
            raise NotImplementedError(f"cannot time-reverse {name!r}")
    return out


# --- LightStim glue ----------------------------------------------------------

PatchRef = Union[str, QECPatch]


def _resolve(system, patch: PatchRef) -> QECPatch:
    if isinstance(patch, str):
        if patch not in system.patches:
            raise ValueError(f"no patch named {patch!r} in the system; "
                             f"have {sorted(system.patches)}")
        return system.patches[patch][0]
    return patch


def _coords(patch: QECPatch) -> Set[Tuple[float, float]]:
    return {tuple(c) for c in patch.qubit_coords.values()}


def make_y_transition_chunk(system,
                            patch_qubit: PatchRef,
                            patch_degenerate: PatchRef,
                            direction: str = 'init') -> stim.Circuit:
    """Build one Y-transition round as a bare :class:`stim.Circuit`.

    Args:
        system: the :class:`~lightstim.ir.qec_system.QECSystem` both patches are
            registered on; supplies the coordinate -> global index map.
        patch_qubit: the ordinary ``RotatedSurfaceCode`` patch — its name in the
            system, or the patch object itself (already in global coordinates).
        patch_degenerate: the degenerate XXZZ patch, same footprint, same forms.
        direction: ``'measure'`` for Gidney's round as published (qubit patch ->
            degenerate patch, ``Y_L`` measured out); ``'init'`` for its time
            reverse (degenerate patch -> qubit patch, ``Y_L`` prepared).

    Returns:
        A TICK-separated chunk on global qubit indices, native gates only.
    """
    if direction not in ('init', 'measure'):
        raise ValueError(f"direction must be 'init' or 'measure', got {direction!r}")

    qubit_patch = _resolve(system, patch_qubit)
    degenerate_patch = _resolve(system, patch_degenerate)

    # --- origin and distance, read off the qubit patch ----------------------
    data_coords = sorted(tuple(qubit_patch.qubit_coords[i])
                         for i in qubit_patch.data_indices)
    if not data_coords:
        raise ValueError("the qubit patch has no data qubits")
    x0 = min(c[0] for c in data_coords)
    y0 = min(c[1] for c in data_coords)
    span_x = max(c[0] for c in data_coords) - x0
    span_y = max(c[1] for c in data_coords) - y0
    if span_x != span_y or span_x % 2:
        raise ValueError(f"the qubit patch is not a square odd-distance patch "
                         f"(x span {span_x}, y span {span_y})")
    distance = int(span_x // 2) + 1
    if len(data_coords) != distance * distance:
        raise ValueError(f"expected {distance ** 2} data qubits in the qubit "
                         f"patch, found {len(data_coords)}")

    def to_g(coord) -> complex:
        return complex((coord[0] - x0) / 2.0, (coord[1] - y0) / 2.0)

    def to_l(g: complex) -> Tuple[float, float]:
        return QECPatch.snap_coord((x0 + 2.0 * g.real, y0 + 2.0 * g.imag))

    # --- `used` = union of both patches' qubits, in Gidney coordinates ------
    used = {to_g(c) for c in _coords(qubit_patch) | _coords(degenerate_patch)}

    if to_g((x0, y0)) not in used:
        raise ValueError("the corner data qubit is missing from both patches")
    if 0 in {to_g(c) for c in _coords(degenerate_patch)}:
        raise ValueError("the degenerate patch still owns the corner data qubit "
                         "(g = 0); it must be excluded")

    q2i: Dict[complex, int] = {}
    for g in used:
        coord = to_l(g)
        if coord not in system.index_map:
            raise ValueError(f"coordinate {coord} (Gidney {g}) is not registered "
                             f"on the system")
        q2i[g] = system.index_map[coord]

    # ==========================================================================
    # Gidney's round, transcribed
    # ==========================================================================
    xs = {q for q in used if _m_basis(q) == 'X'}
    zs = {q for q in used if _m_basis(q) == 'Z'}
    top_row = {q for q in used if q.imag == -0.5}
    right_col = {q for q in used if q.real == distance - 0.5}

    def toward(qs: AbstractSet[complex], delta: complex, sign: int
               ) -> Set[Tuple[complex, complex]]:
        result = set()
        for q in qs:
            if q + delta in used:
                result.add((q, q + delta)[::sign])
        return result

    xs_dl, xs_md, xs_ur = _split_dl_md_ur(xs)
    zs_dl, zs_md, zs_ur = _split_dl_md_ur(zs)

    out = _Emitter(q2i)
    out.gate("RX", (xs - right_col) | top_row)
    out.gate("R", (zs - top_row) | right_col)
    out.tick()
    out.gate2('CX', toward(xs - right_col, DL, +1))
    out.gate2('CX', toward(zs - top_row, DL, -1))
    out.tick()
    out.gate2('CX', toward(xs - right_col, DR, +1))
    out.gate2('CX', toward(zs - top_row, UL, -1))
    out.tick()
    out.gate2('CX', toward(xs_ur | xs_md, UL, -1))
    out.gate2('CX', toward(zs_ur, DR, +1))
    out.gate2('XCY', toward(zs_md, DR, +1))
    out.gate2('CX', toward(xs_dl, UL, +1))
    out.gate2('CX', toward(zs_dl, DR, -1))
    out.tick()
    out.gate2('CX', toward(xs_ur, DL, -1))
    out.gate2('CX', toward(zs_ur, DL, +1))
    out.gate2('CX', toward(xs_dl, UR, +1))
    out.gate2('CX', toward(zs_dl, UR, -1))
    out.tick()
    out.gate2('XCY', toward(xs_md - top_row, DL, -1))
    out.tick()
    out.gate('H', [q for q in used if q.real > q.imag])
    out.gate('SQRT_X', [q for q in used if q.real == q.imag and q.real % 1 == 0.5])
    out.tick()
    xms = (xs - top_row) | right_col
    out.measure(xms, basis='X')
    out.measure({0}, basis='Y')
    out.measure((zs - right_col) | top_row, basis='Z')

    if direction == 'measure':
        return out.circuit
    return time_reverse_chunk(out.circuit)
