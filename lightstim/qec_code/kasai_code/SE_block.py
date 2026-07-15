"""Chen-style transversal-block syndrome extraction for Kasai codes.

Implements the syndrome-extraction schedule of Zhao et al.,
"Towards Ultra-High-Rate Quantum Error Correction with Reconfigurable Atom
Arrays" (arXiv:2604.16209), Sec. 2 / Fig. 2: each check row measures its
stabilizers through a sequence of transversal CNOT layers between the ancilla
block and one data block per step, ordered by the code's affine permutation
maps. The block-circulant structure lets all J active check rows run in
parallel — at X-step t, ancilla row s addresses data block (s + t) mod L/2
under the SAME permutation f_t, so the layers below are conflict-free by
construction. CNOT depth is L per basis (12 for the standard instances),
matching the generic coloration fallback.

The schedule is only hardware-efficient when the code satisfies the paper's
co-design condition: every within-run transition APM (f_{t+1} o f_t^{-1},
g_{t+1} o g_t^{-1}, and their Z-phase inverses) must commute with a common
reference APM whose orbits all have equal length — the orbit structure defines
the (P/len) x len atom layout, and commutation makes every transition a simple
column shift plus row permutation (Fig. 2b). ``find_commuting_layout_reference``
implements that search; the block enforces it at construction unless
``enforce_layout_condition=False``.
"""

from math import gcd
from typing import Any, List, Optional, Sequence, Tuple

import stim

from .code_patch import (
    Affine,
    affine_commutes,
    apply_affine,
    apply_affine_inverse,
    invert_affine,
)


def _compose(f: Affine, g: Affine, P: int) -> Affine:
    """Affine composition ``x -> f(g(x))`` modulo P."""
    (a1, b1), (a2, b2) = f, g
    return (a1 * a2 % P, (a1 * b2 + b1) % P)


def _uniform_orbit_length(a: int, b: int, P: int) -> Optional[int]:
    """Orbit length of the APM ``x -> a x + b`` if all orbits are equal, else None."""
    seen = [False] * P
    length: Optional[int] = None
    for x0 in range(P):
        if seen[x0]:
            continue
        ln, x = 0, x0
        while not seen[x]:
            seen[x] = True
            ln += 1
            x = (a * x + b) % P
        if length is None:
            length = ln
        elif ln != length:
            return None
    return length


def transition_apms(f: Sequence[Affine], g: Sequence[Affine], P: int) -> List[Affine]:
    """Within-run transition APMs of the Chen schedule.

    The ancilla orderings are [f_0..f_{m-1}] then [g_0..g_{m-1}] during the
    X phase and their inverses (g first, then f) during the Z phase; the
    F<->G and phase crossings are separate block-reordering steps (Fig. 2a)
    and are excluded from the commutation condition.
    """
    finv = [invert_affine(t, P) for t in f]
    ginv = [invert_affine(t, P) for t in g]
    runs = [list(f), list(g), ginv, finv]
    return [
        _compose(run[k + 1], invert_affine(run[k], P), P)
        for run in runs
        for k in range(len(run) - 1)
    ]


def find_commuting_layout_reference(
    f: Sequence[Affine],
    g: Sequence[Affine],
    P: int,
    min_orbit_length: int = 2,
) -> Optional[Tuple[Affine, int]]:
    """Best reference APM for the Chen layout condition, or ``None``.

    Searches every non-identity APM with uniform orbit length at least
    ``min_orbit_length`` that commutes with all within-run transition APMs,
    and returns the one with the longest orbits as ``((a, b), orbit_len)``.
    E.g. chen_p96 yields orbit length 32 with 3 orbits — the paper's
    "reference APM with 3 length-32 orbits".
    """
    trans = transition_apms(f, g, P)
    best: Optional[Tuple[Affine, int]] = None
    for a in range(1, P):
        if gcd(a, P) != 1:
            continue
        for b in range(P):
            if (a, b) == (1, 0):
                continue
            if not all(affine_commutes(t, (a, b), P) for t in trans):
                continue
            ln = _uniform_orbit_length(a, b, P)
            if ln is not None and ln >= min_orbit_length and (
                best is None or ln > best[1]
            ):
                best = ((a, b), ln)
    return best


class KasaiChenExtractionBlock:
    """Transversal-block syndrome extraction (arXiv:2604.16209) for Kasai codes.

    Schedule (m = L/2 data blocks per side, s = active block row, x = shift):
        X phase, step t < m:   ancilla (s, x) -> data block (s+t) % m, qubit f_t(x)
        X phase, step m + t:   ancilla (s, x) -> data block m + (s+t) % m, qubit g_t(x)
        Z phase, step t < m:   data block (s-t) % m, qubit g_t^{-1}(x) -> ancilla (s, x)
        Z phase, step m + t:   data block m + (s-t) % m, qubit f_t^{-1}(x) -> ancilla (s, x)

    Raises ``ValueError`` when the code violates the co-design (layout)
    condition, unless ``enforce_layout_condition=False``. The default
    ``min_orbit_length`` of ``P // (2 J)`` accepts every instance published in
    arXiv:2604.16209v2 (orbit length >= 32) and rejects codes without a usable
    rectangular layout (e.g. kasai_p768 caps at orbit length 16).
    """

    def __init__(self, system: Any, enforce_layout_condition: bool = True,
                 min_orbit_length: Optional[int] = None):
        self.system = system
        self.circuit = stim.Circuit()
        self._find_patch()

        code = self._patch
        if min_orbit_length is None:
            min_orbit_length = max(2, code.P // (2 * code.J))
        self.layout_reference = find_commuting_layout_reference(
            code.f, code.g, code.P, min_orbit_length=min_orbit_length,
        )
        if self.layout_reference is None and enforce_layout_condition:
            relaxed = find_commuting_layout_reference(code.f, code.g, code.P)
            achieved = relaxed[1] if relaxed else 0
            raise ValueError(
                "This Kasai code does not satisfy the arXiv:2604.16209 "
                "co-design condition: no reference APM with uniform orbit "
                f"length >= {min_orbit_length} commutes with all transition "
                f"APMs (best achievable orbit length: {achieved}). The "
                "transversal schedule would not map to column-shift/row-"
                "permutation atom moves. Pass enforce_layout_condition=False "
                "to build the circuit anyway, or use the coloration block."
            )

        self.depth_x = code.L
        self.depth_z = code.L
        self.cnot_depth = self.depth_x + self.depth_z
        self._build_circuit()

    def _find_patch(self):
        for _name, (patch, _) in self.system.patches.items():
            if all(hasattr(patch, attr) for attr in ("P", "J", "L", "f", "g")):
                self._patch = patch
                return
        raise ValueError("No KasaiCode patch found in the system.")

    # ── circuit construction ────────────────────────────────────────────────

    def _ancilla_slot(self, syn_coord, basis: str) -> Tuple[int, int]:
        """Recover (active block row s, shift x) from an ancilla coordinate."""
        code = self._patch
        sx, sy = code.shift
        x = int(round(syn_coord[0] - sx))
        row_slot = int(round(syn_coord[1] - sy)) - code.L
        if basis == "Z":
            row_slot -= code.J
        if not (0 <= row_slot < code.J and 0 <= x < code.P):
            raise ValueError(
                f"Ancilla coordinate {syn_coord} does not match the Kasai "
                "layout; was the patch geometry customised?"
            )
        return code.active_rows[row_slot], x

    def _data_index(self, block: int, pos: int) -> int:
        code = self._patch
        sx, sy = code.shift
        key = code.get_grid_key((pos + sx, block + sy))
        return self.system.grid_map[key]

    def _build_circuit(self):
        code = self._patch
        m = code.m
        active_syn_indices = sorted(self.system.active_syndrome_indices)
        active_x_syn = sorted(self.system.active_syndrome_indices_x)

        x_slots = [(stab["syn_idx"], *self._ancilla_slot(stab["syn_coord"], "X"))
                   for stab in self.system.active_stabilizers_x]
        z_slots = [(stab["syn_idx"], *self._ancilla_slot(stab["syn_coord"], "Z"))
                   for stab in self.system.active_stabilizers_z]

        self.circuit.append("R", active_syn_indices)
        self.circuit.append("TICK", tag="SE_start")
        if active_x_syn:
            self.circuit.append("H", active_x_syn)
        self.circuit.append("TICK")

        # X phase: ancilla is control (syndrome -> data)
        for t in range(2 * m):
            targets: List[int] = []
            for syn_idx, s, x in x_slots:
                if t < m:
                    block = (s + t) % m
                    pos = apply_affine(code.f[t], x, code.P)
                else:
                    block = m + (s + t - m) % m
                    pos = apply_affine(code.g[t - m], x, code.P)
                targets.extend([syn_idx, self._data_index(block, pos)])
            self.circuit.append("CNOT", targets)
            self.circuit.append("TICK")

        # Z phase: data is control (data -> syndrome)
        for t in range(2 * m):
            targets = []
            for syn_idx, s, x in z_slots:
                if t < m:
                    block = (s - t) % m
                    pos = apply_affine_inverse(code.g[t], x, code.P)
                else:
                    block = m + (s - (t - m)) % m
                    pos = apply_affine_inverse(code.f[t - m], x, code.P)
                targets.extend([self._data_index(block, pos), syn_idx])
            self.circuit.append("CNOT", targets)
            self.circuit.append("TICK")

        if active_x_syn:
            self.circuit.append("H", active_x_syn)
        self.circuit.append("TICK")
        self.circuit.append("M", active_syn_indices)
