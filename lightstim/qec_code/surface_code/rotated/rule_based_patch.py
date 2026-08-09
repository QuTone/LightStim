"""Rule-based rectangular rotated-surface-code patch.

Builds a ``d_z × d_x`` patch by APPLYING THE CONSTRUCTION RULES rather than emitting a
hardcoded stabilizer layout, so the boundary pattern comes out the way Litinski draws it
(*A Game of Surface Codes*, Fig 45 step 5) rather than the way the plain
:class:`RotatedSurfaceCode` builder happens to lay it out.

The rules are the documented ones (Handbook §10.4 / Fig 33-34, mirrored from
``multi_patch_coupler``'s deterministic construction):

1. **Bulk forced** — every complete weight-4 plaquette of the footprint is taken.
2. **Corners first** — exposed corner data qubits (≤2 orthogonal neighbours) are covered
   first, in lexicographic order, each by the first incident boundary tile that passes the
   vetoes.  *This is what anchors the alternating pattern of every boundary chain*, and it
   is the step the plain builder skips — it is why the rule-based patch has two same-type
   boundary lobes meeting at a corner (they share one data qubit, and sharing exactly one
   qubit commutes precisely when the two are the same type).
3. **Alternating-spacing sweep** — the remaining weight-2 candidates are swept in
   lexicographic order and taken unless a veto applies: one lattice spacing from an
   already-selected stabilizer *along the same edge*; anticommutes with something already
   selected; or linearly dependent on the current selection.

**Corner anchoring is for UNCOVERED corners only.**  A corner data qubit already inside
some selected check (bulk, a forced boundary check, or an earlier corner pass) needs no
anchor of its own; eagerly giving every corner its own tile is what mis-placed the
same-type corner pair against Litinski Fig 45 step 5 (it landed on the mirror corner)
and smuggled a redundant lobe next to a forced corner check against Kishony-Fowler
Fig 5b.  With the coverage check in place, the boundary pattern of BOTH papers' grown
patches — Litinski Fig 45 step 5 (d=3, grow down, 2d-1) and Kishony-Fowler Fig 5b
(d=5, grow right, 2d+1, conjugate orientation) — emerges from bulk + ``forced`` + the
plain lexicographic sweep alone, including the same-type corner pair at the far corner
of the growth.

**Destination anchoring.**  The grown bar's far side carries a mid-edge topological
corner (a boundary-type change along one edge), and its position is a protocol input,
not a geometric consequence: both papers pin it to the boundary of the DESTINATION
sub-patch the later shrink step will leave behind (visible at d=5, where the free edge
has slack; at d=3 the vetoes fix it).  The caller expresses it through ``forced``: add
the destination patch's far-side boundary checks alongside the kept live-patch checks.
"""
from typing import Dict, List, Tuple

from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode


def _sym_commute(a: Dict[Tuple[int, int], str], b: Dict[Tuple[int, int], str]) -> bool:
    """Do two coordinate-keyed Pauli operators commute?  (X/Z-only supports.)"""
    anti = sum(1 for q, p in a.items() if q in b and b[q] != p)
    return anti % 2 == 0


def _independent(sel: List[Dict], cand: Dict, coords: List[Tuple[int, int]]) -> bool:
    """Is ``cand`` outside the GF(2) span of ``sel``?"""
    n = len(coords)
    idx = {q: i for i, q in enumerate(coords)}

    def pack(op):
        v = 0
        for q, p in op.items():
            v |= 1 << (idx[q] + (0 if p == 'X' else n))
        return v

    rows: List[int] = []
    for op in sel:
        v = pack(op)
        for r in rows:
            v = min(v, v ^ r)
        if v:
            rows.append(v)
            rows.sort(reverse=True)
    v = pack(cand)
    for r in rows:
        v = min(v, v ^ r)
    return v != 0


class RuleBasedRectPatch(RotatedSurfaceCode):
    """A ``distance_x`` × ``distance_z`` patch whose boundary is chosen by the rules above.

    Same interface as :class:`RotatedSurfaceCode` (``distance_x``/``distance_z``/``shift``),
    so it drops straight into :meth:`QECSystem.grow_patch`.
    """

    def build(self):
        d_x, d_z = self.distance_x, self.distance_z
        data = [(2 * c + 1, 2 * r + 1) for c in range(d_z) for r in range(d_x)]
        dset = set(data)
        # ``forced``: checks that MUST be kept, given as {(syn_coord): (type, [supports])}.
        # When growing a live patch these are the existing patch's checks, so the grown
        # code contains them verbatim and their detector history survives the growth.
        forced = dict(self.params.get('forced') or {})

        # --- plaquette candidates: every centre with >= 2 data corners ------------
        cxs = range(0, 2 * d_z + 1, 2)
        cys = range(0, 2 * d_x + 1, 2)
        phase = self.params.get('phase', 0)

        def corners(cx, cy):
            return sorted(q for q in ((cx - 1, cy - 1), (cx - 1, cy + 1),
                                      (cx + 1, cy - 1), (cx + 1, cy + 1)) if q in dset)

        def ptype(cx, cy):
            return 'X' if (((cx + cy) // 2) + phase) % 2 else 'Z'

        bulk, cands = [], []
        for cx in cxs:
            for cy in cys:
                sup = corners(cx, cy)
                if len(sup) == 4:
                    bulk.append(((cx, cy), ptype(cx, cy), sup))
                elif len(sup) == 2:
                    cands.append(((cx, cy), ptype(cx, cy), sup))
        cands.sort()

        selected = list(bulk)                       # rule 1: bulk is forced
        # rule 1b: the caller's pre-existing checks are forced too (anchoring), unless the
        # enlarged bulk already contains them at that centre.
        have = {c for c, _, _ in selected}
        for sc, (t, sup) in sorted(forced.items()):
            sup = sorted(tuple(q) for q in sup)
            if sc not in have and all(q in dset for q in sup):
                selected.append((tuple(sc), t, sup))
                have.add(tuple(sc))

        def op_of(entry):
            _, t, sup = entry
            return {q: t for q in sup}

        def passes(entry):
            (cx, cy), t, sup = entry
            op = op_of(entry)
            for other in selected:
                if not _sym_commute(op, op_of(other)):
                    return False
                # spacing rule: one lattice spacing away ALONG THE SAME EDGE
                (ox, oy), _, osup = other
                if len(osup) == 2 and len(set(osup) & set(sup)) == 1 \
                        and (ox == cx or oy == cy):
                    return False
            return _independent([op_of(s) for s in selected], op, data)

        # --- rule 2: corners first ------------------------------------------------
        def n_orth(q):
            return sum(1 for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                       if (q[0] + dx, q[1] + dy) in dset)

        for cq in sorted(q for q in data if n_orth(q) <= 2):
            # anchor only UNCOVERED corners (see module docstring): a corner already
            # inside a selected check — bulk, forced, or an earlier corner pass — gets
            # no eager tile of its own.  Eager per-corner selection is what mis-placed
            # the same-type corner pair against Litinski Fig 45 step 5 and smuggled a
            # redundant lobe next to a forced corner check against Kishony-Fowler
            # Fig 5b; with this check, the sweep below reproduces both papers.
            if any(cq in sup for _, _, sup in selected):
                continue
            for entry in cands:
                if entry in selected or cq not in entry[2]:
                    continue
                if passes(entry):
                    selected.append(entry)
                    break

        # --- rule 3: alternating-spacing sweep ------------------------------------
        for entry in cands:
            if entry not in selected and passes(entry):
                selected.append(entry)

        # --- register qubits ------------------------------------------------------
        for q in data:
            self.add_qubit(*q, role='data')
        for (cx, cy), t, _ in selected:
            self.add_qubit(cx, cy, role='syndrome_x' if t == 'X' else 'syndrome_z')

        # --- stabilizers ----------------------------------------------------------
        for (cx, cy), t, sup in selected:
            self.create_stim_stabilizer({q: t for q in sup}, (cx, cy), t)

        # --- logical operators: the shortest surviving X / Z chains ---------------
        ops = [op_of(s) for s in selected]

        def logical(kind, chains, partner=None):
            """A chain that commutes with every check, is not a stabilizer, and — when a
            ``partner`` logical is given — ANTICOMMUTES with it (the two must cross, else
            they would not be a conjugate pair)."""
            for sup in chains:
                op = {q: kind for q in sup}
                if not all(_sym_commute(op, o) for o in ops):
                    continue
                if not _independent(ops, op, data):
                    continue
                if partner is not None and _sym_commute(op, partner):
                    continue
                return op
            return None

        cols = [[(2 * c + 1, 2 * r + 1) for r in range(d_x)] for c in range(d_z)]
        rows = [[(2 * c + 1, 2 * r + 1) for c in range(d_z)] for r in range(d_x)]
        # shortest chains first, so the stored representatives are minimum-weight ones
        chains = sorted(cols + rows, key=len)
        x_op = logical('X', chains)
        if x_op is None:
            raise ValueError(f"RuleBasedRectPatch({d_x}x{d_z}): no surviving X logical chain")
        z_op = logical('Z', chains, partner=x_op)
        if z_op is None:
            raise ValueError(
                f"RuleBasedRectPatch({d_x}x{d_z}): no Z logical chain anticommuting with X̄ — "
                f"the rule-selected boundary over-constrains this footprint")
        self.create_stim_logical(x_op, 'X')
        self.create_stim_logical(z_op, 'Z')

        self.num_logicals = 1
        if self.shift != (0, 0):
            self.shift_coords(*self.shift)
