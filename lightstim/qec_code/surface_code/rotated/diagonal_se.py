"""Diagonal syndrome-extraction schedule (Kishony & Fowler, *Surface code
off-the-hook: diagonal syndrome-extraction scheduling*).

One globally uniform gate ORDER for every patch shape and orientation —
Z-plaquette corners coupled NW→SE→NE→SW, X-plaquette SW→NE→SE→NW (y-up frame) —
so every ancilla hook lands on a DIAGONAL data pair and never aligns with a
horizontal or vertical logical path. This supports the bent and branched
rotated-surface PPM layouts whose logical representatives do not remain on one
axis; the verified layouts reach full distance at d = 3 and 5.

Slot assignment: the FIXED compact-7 table of K&F Fig 4 — no dynamic search.

* Plain plaquettes (and boundary lobes, restricted to present corners) read
  ``FIXED_SLOTS`` at their corner deltas: Z at slots 1-4, X at slots 4-7 over
  the 7-slot coupling window.  Round depth: R | H | 7 | H | M = 11 ticks.
* Single-cell mixed checks (corridor/recoloured-column stabilizers; not in
  the paper) read TWO fixed variant tables (``_M_SLOTS``) keyed by the
  (west,south)-most foot's Pauli — the single-cell twin of the wall's
  '+'/'-' alternation, derived offline by constraint solving and selected
  by the distance ladder (see the table's comment).
* K&F relay (stretched) checks use the Fig 4(c) program verbatim via
  ``_kf_events``; adjacent dominoes alternate between the two Fig 4 variants
  (offset 0 / +2, i.e. the paper's slot-1 and slot-3 anchored circuits).

The K&F Sec. II constraint (two checks sharing two anticommuting corners keep
a consistent relative gate order) and (slot, qubit) collision-freedom are now
VALIDATED, not searched for: ``_validate_fixed`` raises with the offending
coordinates if a layout violates the table, instead of packing around it.
"""
import stim


_NE, _NW, _SE, _SW = (1, 1), (-1, 1), (1, -1), (-1, -1)

#: K&F diagonal gate orders (y-up corner deltas, data − syndrome)
DIAGONAL_ORDERS = {'Z': (_NW, _SE, _NE, _SW), 'X': (_SW, _NE, _SE, _NW)}

#: K&F Fig 4 fixed slots (1-indexed over the 7-slot window): Z gates at 1-4,
#: X gates at 4-7.  These ARE the numbers printed on the paper's plaquettes;
#: ``_slot_maps`` subtracts 1 to get 0-indexed circuit slots.
FIXED_SLOTS = {
    'Z': {_NW: 1, _NE: 3, _SW: 4, _SE: 2},
    'X': {_NW: 7, _NE: 5, _SW: 4, _SE: 6},
}


#: K&F relay-template program (Fig 39 / interface plaquette): per-check slot
#: offsets for the two-aux + shared-relay realization of a wide mixed check.
#: A = flag aux (RX -> MZ, couples the Z feet via CZ), B = syndrome aux
#: (RZ -> MX, couples the X feet via CX), S = shared relay (RZ -> MZ).
#: p0 CX(A,S); p1 CZ(A,fz1)+CX(S,B); p2 CZ(A,fz2)+CX(B,fx1);
#: p3 CX(S,A)+CX(B,fx2); p4 CX(B,S).  This IS the Fig 4(c) circuit: offset 0
#: reproduces the paper's slot-1 anchored variant (feet at slots 2-4,
#: 1-indexed) and offset +2 the slot-3 anchored one (feet at 4-6) — the two
#: stretched-stabilizer schedules that alternate along the interface in
#: Fig 4(a,b).  Adjacent dominoes share two anticommuting feet, so a uniform
#: offset would violate the K&F Sec. II consistency constraint; the builders
#: therefore alternate '+'/'-' along the wall.
_KF_OFFSET = {'+': 0, '-': 2}

#: Single-cell mixed (corridor/recoloured-column) checks: TWO fixed variant
#: tables keyed by the Pauli of the check's (west-most, then south-most)
#: foot — the single-cell twin of the wall's '+'/'-' alternation.  A
#: recoloured column's checks alternate their foot Paulis row by row and
#: adjacent checks share two anticommuting feet with each other AND with
#: the neighbouring bulk plaquettes; one per-own-Pauli table cannot order
#: all those pairs consistently (K&F Sec. II).  These tables were derived
#: OFFLINE by constraint solving over five verified corridor layouts
#: (straight/stacked row-4, an obstacle-detour bent corridor, an L
#: corridor) and selected by the distance ladder: all scenarios reach
#: p=0-clean full graphlike distance under the diagonal schedule.
#: (Entries never exercised by a real check pattern carry the solver's
#: canonical values.)  Keys: (corner, foot Pauli) -> 1-indexed slot.
_M_SLOTS = (
    # variant 0: the check's (west,south)-most foot is X-support
    {(_NW, 'X'): 4, (_NW, 'Z'): 1, (_NE, 'X'): 1, (_NE, 'Z'): 5,
     (_SW, 'X'): 3, (_SW, 'Z'): 1, (_SE, 'X'): 4, (_SE, 'Z'): 2},
    # variant 1: it is Z-support
    {(_NW, 'X'): 7, (_NW, 'Z'): 4, (_NE, 'X'): 3, (_NE, 'Z'): 1,
     (_SW, 'X'): 1, (_SW, 'Z'): 5, (_SE, 'X'): 6, (_SE, 'Z'): 4},
)


def _kf_events(ch):
    """The (slot, gate, pair) events of one K&F relay check.  ``ch`` needs
    'offset', flag/shared/syn coords 'A'/'S'/'B', and per-side feet lists
    'fa' (A's side) / 'fb' (B's side), each a list of (subslot, coord, gate)
    with subslot in {0, 1} (the foot's POSITIONAL slot within the side's
    pair — like the paper's boundary lobes, a weight-1 side keeps the slot
    its corner position dictates, it does not slide to the first one) and
    gate 'CX' (X-support foot, aux is control) or 'CZ' (Z-support).
    The aux split is GEOMETRIC (each aux couples the feet on its own row),
    not Pauli-based: back-propagation gives MX(B) = the full check and keeps
    MZ(A)/MZ(S) deterministic for ANY foot-Pauli pattern, so the same
    template serves mixed dominoes (2+2 split) and uniform same-type
    dominoes (all-X or all-Z)."""
    o = ch['offset']
    A, S, B = ch['A'], ch['S'], ch['B']

    def foot(aux, coord, gate):
        return (gate, (aux, coord) if gate == 'CX' else (coord, aux))

    ev = [(o + 0, 'CX', (A, S)), (o + 1, 'CX', (S, B))]
    for sub, coord, gate in ch['fa']:
        ev.append((o + 1 + sub,) + foot(A, coord, gate))
    for sub, coord, gate in ch['fb']:
        ev.append((o + 2 + sub,) + foot(B, coord, gate))
    ev.extend([(o + 3, 'CX', (S, A)), (o + 4, 'CX', (B, S))])
    return ev


def _validate_fixed(placements):
    """Assert the fixed table produced a legal schedule — raise, don't search.

    ``placements``: list of ``(label, pauli_at, {coord: slot})`` covering every
    check's data couplings AND the K&F relay apparatus gates.  Checks the two
    invariants the old backtracking packer used to guarantee:

    * no (slot, qubit) cell is used twice;
    * K&F Sec. II — two checks sharing >= 2 anticommuting corners keep a
      consistent relative gate order on them (apparatus cells carry no Pauli
      and are exempt: ``pauli_at`` simply has no entry for them).
    """
    occupied = {}
    for label, _, cm in placements:
        for q, t in cm.items():
            if not 0 <= t < 7:
                raise RuntimeError(
                    f"fixed compact-7 schedule: {label} places qubit {q} at "
                    f"slot {t + 1}, outside the 7-slot window")
            other = occupied.setdefault((t, q), label)
            if other != label:
                raise RuntimeError(
                    f"fixed compact-7 schedule: slot {t + 1} on qubit {q} is "
                    f"used by both {other} and {label} — this layout violates "
                    f"the K&F Fig 4 table (fix the wall spec, do not repack)")
    by_corner = {}
    for k, (_, pauli_at, cm) in enumerate(placements):
        for q in cm:
            if q in pauli_at:
                by_corner.setdefault(q, []).append(k)
    pairs = {}
    for q, ks in by_corner.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                if placements[a][1][q] != placements[b][1][q]:
                    pairs.setdefault((a, b), []).append(q)
    for (a, b), qs in pairs.items():
        if len(qs) < 2:
            continue
        la, _, ca = placements[a]
        lb, _, cb = placements[b]
        rels = {ca[q] < cb[q] for q in qs}
        if len(rels) > 1:
            raise RuntimeError(
                f"fixed compact-7 schedule: K&F Sec. II consistency violated "
                f"between {la} and {lb} on shared anticommuting corners {qs}")


class DiagonalSurfaceCodeExtractionBlock:
    """Drop-in alternative to :class:`RotatedSurfaceCodeExtractionBlock` using the
    diagonal schedule.  Same contract: ``self.circuit`` is one noiseless SE round,
    last instruction is the syndrome measurement."""

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self.coupling_slots = None    # set by _build_circuit: always 7 (K&F Fig 4)
        self._build_circuit()

    def _gather(self):
        """Gather ALL active checks — pure X/Z plaquettes AND single-cell mixed
        checks (type 'MIXED'/'M', e.g. corridor stabilizers).  Corners are
        classified by the SIGN of their offset from the syn coord and then read
        the fixed table: a plain check its own type's table, a mixed check each
        FOOT's own Pauli's table (gates stay per-foot: CX for X-support / CZ
        for Z-support).  A check whose feet do not sit on four distinct
        diagonal corners cannot be expressed in the table — raise, don't guess."""
        system = self.system

        def sgn(v):
            return (v > 0) - (v < 0)

        checks, kf_checks = [], []
        for uid in sorted(system.active_stabilizer_indices):
            st = system.stabilizers[uid]
            if st.get('kf') is not None:
                kf_checks.append(self._gather_kf(st))
                continue
            t = st.get('type')
            typ = t if t in ('X', 'Z') else 'M'
            sc = st['syn_coord']
            deltas = {}
            pauli_of = {}
            for di in st['data_indices']:
                dc = system.qubit_coords[di]
                dxy = (sgn(dc[0] - sc[0]), sgn(dc[1] - sc[1]))
                if dxy in deltas:
                    raise RuntimeError(
                        f"fixed compact-7 schedule: check at {tuple(sc)} has "
                        f"two data qubits ({tuple(system.qubit_coords[deltas[dxy]])} "
                        f"and {tuple(dc)}) on the same {dxy} corner direction — "
                        f"a wide check must carry 'kf' relay metadata, the "
                        f"plain table cannot express it")
                deltas[dxy] = di
                pauli_of[di] = (st['pauli'][di] if typ == 'M' else typ)
            pauli_at = {tuple(system.qubit_coords[di]): p
                        for di, p in pauli_of.items()}
            checks.append({'type': typ, 'syn': tuple(sc), 'syn_idx': st['syn_idx'],
                           'deltas': deltas, 'pauli_of': pauli_of,
                           'pauli_at': pauli_at})
        return checks, kf_checks

    def _gather_kf(self, st):
        """One K&F relay check: resolve the apparatus coords (flag A, shared S,
        syn B = ``syn_coord``) and split the feet GEOMETRICALLY — each foot
        belongs to the aux one diagonal step away (its own gap side).  Both
        Fig 4 orientations are supported and told apart by the apparatus
        axis: A and B in the same COLUMN = vertical domino (feet above and
        below the aux band), same ROW = horizontal domino (feet left and
        right).  Each foot gets the POSITIONAL subslot of the Fig 4 table —
        vertical: Z side east first, X side west first (Fig 4b); horizontal:
        both sides north first on the '-' variant, south first on '+'
        (Fig 4a/c) — and carries its own gate: CX for X support, CZ for Z
        support."""
        system = self.system

        def sgn(v):
            return (v > 0) - (v < 0)

        kf = st['kf']
        B = tuple(st['syn_coord'])
        A = tuple(kf['flag'])
        S = tuple(kf['shared'])
        if A[0] == B[0] and A[1] != B[1]:
            horizontal = False
        elif A[1] == B[1] and A[0] != B[0]:
            horizontal = True
        else:
            raise RuntimeError(
                f"K&F relay check at {B}: apparatus A={A} and B={B} share "
                f"neither a row nor a column — malformed wall spec")
        axis = 0 if horizontal else 1     # feet split along this coordinate
        feet = {A: {}, B: {}}      # aux -> {delta-from-aux: (coord, pauli)}
        for di in st['data_indices']:
            dc = tuple(system.qubit_coords[di])
            p = st['pauli'][di]
            ax = A if abs(dc[axis] - A[axis]) == 1 else B
            if dc in (A, S, B) or abs(dc[0] - ax[0]) != 1 or abs(dc[1] - ax[1]) != 1:
                raise RuntimeError(
                    f"K&F relay check at {B}: foot {dc} is not one diagonal "
                    f"step from its aux {ax} (apparatus A={A}, S={S}, B={B}) — "
                    f"malformed wall spec")
            dxy = (sgn(dc[0] - ax[0]), sgn(dc[1] - ax[1]))
            if dxy in feet[ax]:
                raise RuntimeError(
                    f"K&F relay check at {B}: feet {feet[ax][dxy][0]} and {dc} "
                    f"occupy the same {dxy} corner of aux {ax} — malformed "
                    f"wall spec")
            feet[ax][dxy] = (dc, p)

        offset = _KF_OFFSET[kf['orient']]

        def side(ax):
            # K&F Fig 4 foot subslots for the stretched stabilizers.  The
            # subslot is POSITIONAL (which end of the side's short edge), so
            # a weight-1 side keeps the slot its corner dictates — exactly
            # like the paper's boundary lobes.
            # * vertical dominoes (Fig 4b): an X-support side couples its
            #   WEST foot first (time runs west -> east on the red halves),
            #   a Z-support side its EAST foot first.  Sides are
            #   Pauli-uniform in every wall family we build; a hypothetical
            #   mixed side gets the X rule.
            # * horizontal dominoes (Fig 4a/c): the order is keyed by the
            #   VARIANT, not the Pauli — the slot-3-anchored '-' circuit
            #   couples both sides north first (Fig 4c: tl=4,bl=5 / tr=5,
            #   br=6), the slot-1-anchored '+' couples both south first.
            out = []
            for coord, p in feet[ax].values():
                if horizontal:
                    north = coord[1] > ax[1]
                    north_first = (offset == 2)
                    sub = (0 if north else 1) if north_first else (1 if north else 0)
                else:
                    paulis = {q for _, q in feet[ax].values()}
                    east_first = (paulis == {'Z'})
                    east = coord[0] > ax[0]
                    sub = (0 if east else 1) if east_first else (1 if east else 0)
                out.append((sub, coord, 'CX' if p == 'X' else 'CZ'))
            return sorted(out)

        return {'A': A, 'S': S, 'B': B, 'fa': side(A), 'fb': side(B),
                'offset': offset,
                'syn_idx': st['syn_idx'],
                'flag_idx': system.index_map[A],
                'shared_idx': system.index_map[S],
                'pauli_at': {tuple(system.qubit_coords[di]): st['pauli'][di]
                             for di in st['data_indices']}}

    def _slot_maps(self, checks, kf_checks=()):
        """Fixed compact-7 assignment (K&F Fig 4) — a table lookup, no search.

        Plain X/Z checks (any corner subset) read ``FIXED_SLOTS[type]``;
        single-cell mixed checks read each foot's OWN Pauli's table at that
        foot's corner; K&F relay checks are already fully determined by
        ``_kf_events``.  Everything is then validated once.
        """
        system = self.system
        slot_maps = []
        for ch in checks:
            tm = {}
            # single-cell mixed checks read the _M_SLOTS variant table keyed
            # by the (west,south)-most foot's Pauli (see the table's comment)
            if ch['type'] == 'M':
                feet_coords = {tuple(system.qubit_coords[di]): di
                               for di in ch['deltas'].values()}
                first = feet_coords[min(feet_coords)]
                m_tab = _M_SLOTS[0 if ch['pauli_of'][first] == 'X' else 1]
            for dxy, di in ch['deltas'].items():
                pauli = ch['pauli_of'][di] if ch['type'] == 'M' else ch['type']
                if pauli not in FIXED_SLOTS:
                    raise RuntimeError(
                        f"fixed compact-7 schedule: check at {ch['syn']} has "
                        f"{pauli!r} support on qubit "
                        f"{tuple(system.qubit_coords[di])} — the K&F table "
                        f"only covers X and Z")
                if dxy not in FIXED_SLOTS[pauli]:
                    raise RuntimeError(
                        f"fixed compact-7 schedule: check at {ch['syn']} has a "
                        f"non-diagonal corner delta {dxy} (qubit "
                        f"{tuple(system.qubit_coords[di])}) — the K&F table "
                        f"only covers diagonal corners")
                if ch['type'] == 'M':
                    tm[di] = m_tab[(dxy, pauli)] - 1
                else:
                    tm[di] = FIXED_SLOTS[pauli][dxy] - 1
            slot_maps.append(tm)

        placements = []
        for ch, tm in zip(checks, slot_maps):
            placements.append(
                (f"{ch['type']}-check@{ch['syn']}", ch['pauli_at'],
                 {tuple(system.qubit_coords[di]): t for di, t in tm.items()}))
            # the check's own ancilla is busy at every one of its coupling
            # slots: register those cells too, so a collision against a K&F
            # apparatus qubit (or anything else) on the ancilla coord raises
            for t in tm.values():
                placements.append(
                    (f"{ch['type']}-check@{ch['syn']}:anc", {}, {ch['syn']: t}))
        for ch in kf_checks:
            feet_cm = {}
            for slot, gate, (u, v) in _kf_events(ch):
                for q in (u, v):
                    if q in (ch['A'], ch['S'], ch['B']):
                        # apparatus qubits recur across slots: one placement
                        # per event, carrying no Pauli (collision check only)
                        placements.append(
                            (f"kf-apparatus@{ch['B']}:t{slot}", {}, {q: slot}))
                    else:
                        feet_cm[q] = slot
            placements.append(
                (f"kf-check@{ch['B']}", ch['pauli_at'], feet_cm))
        _validate_fixed(placements)

        return 7, slot_maps

    def _build_circuit(self):
        system = self.system
        checks, kf_checks = self._gather()
        T, slot_maps = self._slot_maps(checks, kf_checks)
        self.coupling_slots = T

        idx = system.index_map
        # K&F relay apparatus: A = flag (RX -> MZ: H after reset only),
        # B = syndrome (RZ -> MX: H before measure only), S = relay (RZ -> MZ:
        # no H).  Plain X/M checks keep the H sandwich on both sides.
        kf_gates = {}                 # slot -> list of ('CX'|'CZ', u_idx, v_idx)
        for ch in kf_checks:
            for slot, gate, (u, v) in _kf_events(ch):
                kf_gates.setdefault(slot, []).append((gate, idx[u], idx[v]))

        plain_h = [ch['syn_idx'] for ch in checks if ch['type'] in ('X', 'M')]
        h_pre = sorted(plain_h + [ch['flag_idx'] for ch in kf_checks])
        h_post = sorted(plain_h + [ch['syn_idx'] for ch in kf_checks])
        syn = sorted(
            [ch['syn_idx'] for ch in checks]
            + [i for ch in kf_checks
               for i in (ch['syn_idx'], ch['flag_idx'], ch['shared_idx'])])

        self.circuit.append("R", syn)
        self.circuit.append("TICK", tag="SE_start")
        if h_pre:
            self.circuit.append("H", h_pre)
        self.circuit.append("TICK")
        for t in range(T):
            cx, cz = [], []
            for gate, u, v in kf_gates.get(t, ()):
                (cx if gate == 'CX' else cz).extend([u, v])
            for ch, smap in zip(checks, slot_maps):
                si = ch['syn_idx']
                for di, slot in smap.items():
                    if slot != t:
                        continue
                    if ch['type'] == 'X':
                        cx.extend([si, di])       # X plaquette: CNOT(anc -> data)
                    elif ch['type'] == 'Z':
                        cx.extend([di, si])       # Z plaquette: CNOT(data -> anc)
                    elif ch['pauli_of'][di] == 'X':
                        cx.extend([si, di])       # mixed, X-support: CNOT(anc -> data)
                    else:
                        cz.extend([di, si])       # mixed, Z-support: CZ(data, anc)
            if cx:
                self.circuit.append("CNOT", cx)
            if cz:
                self.circuit.append("CZ", cz)
            self.circuit.append("TICK")
        if h_post:
            self.circuit.append("H", h_post)
        self.circuit.append("TICK")
        self.circuit.append("M", syn)
