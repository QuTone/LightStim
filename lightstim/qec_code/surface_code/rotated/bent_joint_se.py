import itertools

import stim


# -----------------------------------------------------------------------------
# Rotated bent (XZ) joint lattice-surgery measurement — circuit builder
# -----------------------------------------------------------------------------
#
# The bent joint-measurement layout is transcribed from `rotated.png` as an
# explicit list of data-qubit coordinates plus checks (pure X, pure Z, and MIXED
# domain-wall checks carrying an explicit per-data Pauli). This module turns that
# layout into a circuit-level syndrome-extraction circuit so the notebook does
# NOT define the syndrome-extraction circuit inline — it lives here in
# `lightstim/qec_code/`, next to the other surface-code extraction blocks.
#
# Pure X/Z checks couple their corners in a hook-benign order (see BENIGN_TABLES).
# MIXED checks are measured with a single X-basis ancilla — X terms couple via
# CNOT(ancilla->data) and Z terms via CZ(ancilla, data) — scheduled in the SAME
# direction-by-tick slots the pure checks use, so the mixed checks inherit the
# pure checks' hook-error-benign (logical-perpendicular) orientation.


def _sgn(a):
    return (a > 0) - (a < 0)


# -----------------------------------------------------------------------------
# Hook-benign scheduling
# -----------------------------------------------------------------------------
#
# Each check couples its corners in a benign order so its hook error lands
# PERPENDICULAR to the same-type logical it could shorten.  Key points:
#
# * one benign direction-table pair PER ORIENTATION DOMAIN (X_horizontal is the
#   exact transpose = X/Z swap of X_vertical);
# * when two orientation domains meet, T=5 slots with a per-check FLOATING IDLE
#   SLOT: gate ORDER (hence hook geometry) never changes, only absolute slot
#   positions shift so seam-band checks dodge each other.  T=4 is provably
#   unsatisfiable for two domains; away from a seam every check keeps the
#   default trailing idle, so a uniform layout degenerates to plain T=4;
# * mixed (domain-wall) checks are interleaved into the SAME coupling window
#   (X corners via CNOT, Z corners via CZ), inheriting the pure checks' hook
#   geometry.

_NE, _NW, _SE, _SW = (1, 1), (-1, 1), (1, -1), (-1, -1)

#: Benign gate-order table per orientation domain: X and Z checks each couple
#: their corners in an order whose hook (the last-coupled corner pair) lies
#: PERPENDICULAR to the same-type logical it could shorten (X hook ⊥ X̄, Z hook
#: ⊥ Z̄).  ``X_horizontal`` is the exact transpose (X/Z swap) of ``X_vertical``.
#: Full circuit distance on single-patch memory in both bases; the routed
#: pipeline (``RoutedMultiPatchLSExperiment``) reaches full joint distance with it.
BENIGN_TABLES = {
    'X_vertical':   {'X': (_NE, _NW, _SE, _SW), 'Z': (_NE, _SE, _NW, _SW)},
    'X_horizontal': {'X': (_NE, _SE, _NW, _SW), 'Z': (_NE, _NW, _SE, _SW)},
}

_DEFAULT_ORIENTATION = 'X_horizontal'


def _check_orientation(check, domains, default=_DEFAULT_ORIENTATION):
    """Majority vote of the check's corner cells over the orientation-domain map."""
    votes = {}
    for q in check['pauli']:
        o = domains.get(q, default)
        votes[o] = votes.get(o, 0) + 1
    return max(sorted(votes), key=lambda o: votes[o])


def assign_hook_benign_schedule(checks, domains=None, default=_DEFAULT_ORIENTATION,
                                max_nodes=500_000, include_mixed=False):
    """Schedule the checks of one SE round with benign gate orders.

    Returns ``(T, tickmaps)`` where ``tickmaps[k] = {corner_coord: tick}`` for
    every scheduled check index ``k``.  By default only the pure (X/Z) checks
    are scheduled (mixed 'M' checks then run in their own batch blocks);
    ``include_mixed=True`` schedules the mixed checks INSIDE the same T-slot
    coupling window (fully interleaved round): a mixed check follows its
    domain's X table as a single corner-order permutation (per-check slot
    injectivity by construction), its X corners couple via CNOT and Z corners
    via CZ at their assigned slots.

    ``T`` is 4 for a single orientation domain and 5 when two domains meet; in
    the latter case each check carries one idle slot whose position is chosen
    by a deterministic greedy sweep (with bounded backtracking) so that no
    data qubit is touched twice in a tick and every pair of checks sharing
    two ANTICOMMUTING corners keeps a consistent relative gate order (for
    pure X/Z pairs this is the classic rule; it generalises to mixed-pure and
    mixed-mixed pairs, whose shared corners carry opposite Paulis along a
    domain wall).  The gate ORDER of each check is always exactly its
    domain's benign table — the idle slot only shifts absolute positions, so
    hook geometry is untouched.
    """
    domains = domains or {}
    tables = BENIGN_TABLES
    kinds = ('X', 'Z', 'M') if include_mixed else ('X', 'Z')
    sched = [k for k, ch in enumerate(checks) if ch['type'] in kinds]
    orients = {k: _check_orientation(checks[k], domains, default) for k in sched}
    has_mixed = any(checks[k]['type'] == 'M' for k in sched)
    # mixed checks reuse the X table, so wall-adjacent qubits need the floating
    # idle slot's freedom to dodge same-slot collisions -> T=5 whenever present
    T = 4 if len(set(orients.values())) <= 1 and not has_mixed else 5

    # direction index within the table, per corner (slot base position).
    # Corners are classified by the SIGN of their offset from the syn coord, so
    # WIDE checks (Fig 39-style stretched supports, e.g. the mixed-wall dominoes
    # whose feet sit rows away from the gap-centre ancilla) inherit the same
    # corner-order tables as ordinary (±1, ±1) plaquettes.
    def _sgn(v):
        return (v > 0) - (v < 0)

    base_pos = {}
    for k in sched:
        ch = checks[k]
        # mixed checks take the domain's X table as ONE corner permutation
        table = tables[orients[k]][ch['type'] if ch['type'] in ('X', 'Z') else 'X']
        sx, sy = ch['syn']
        pos_of = {d_: i for i, d_ in enumerate(table)}
        base_pos[k] = {}
        for q in ch['pauli']:
            dxy = (_sgn(q[0] - sx), _sgn(q[1] - sy))
            if dxy in pos_of:
                base_pos[k][q] = pos_of[dxy]

    # pairs sharing >=2 ANTICOMMUTING corners: relative gate order must agree
    # (pure X vs pure Z pairs always anticommute on shared corners; mixed
    # checks anticommute with neighbours exactly where the wall alternates)
    by_corner = {}
    for k in sched:
        for q in checks[k]['pauli']:
            by_corner.setdefault(q, []).append(k)
    xz_pairs = {}
    for q, ks in by_corner.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = sorted((ks[i], ks[j]))
                if checks[a]['pauli'][q] != checks[b]['pauli'][q]:
                    xz_pairs.setdefault((a, b), []).append(q)
    xz_pairs = {kk: qs for kk, qs in xz_pairs.items() if len(qs) >= 2}
    partners = {}
    for (a, b), qs in xz_pairs.items():
        partners.setdefault(a, []).append((b, qs))
        partners.setdefault(b, []).append((a, qs))

    def tickmap_of(k, idle):
        slots = [t for t in range(T) if t != idle]
        return {q: slots[base_pos[k][q]] for q in base_pos[k]}

    idle_options = [None] if T == 4 else [T - 1, 3, 2, 1, 0]
    order = sorted(sched, key=lambda k: checks[k]['syn'])
    assign, occupied = {}, {}
    nodes = [0]

    def consistent(k, tm):
        for q, t in tm.items():
            if (t, q) in occupied:
                return False
        for other, qs in partners.get(k, ()):
            if other in assign:
                to = assign[other]
                rel = None
                for q in qs:
                    if q in tm and q in to:
                        r = tm[q] < to[q]
                        if rel is None:
                            rel = r
                        elif rel != r:
                            return False
        return True

    def place(i):
        if i == len(order):
            return True
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise RuntimeError(
                "hook-benign scheduling exceeded its search budget; "
                "this layout's orientation-domain seams could not be resolved")
        k = order[i]
        for idle in idle_options:
            tm = tickmap_of(k, idle)
            if consistent(k, tm):
                assign[k] = tm
                for q, t in tm.items():
                    occupied[(t, q)] = k
                if place(i + 1):
                    return True
                for q, t in tm.items():
                    del occupied[(t, q)]
                del assign[k]
        return False

    if not place(0):
        raise RuntimeError(
            "hook-benign scheduling is unsatisfiable for this layout "
            f"(T={T}); no idle-slot assignment resolves the seam band")
    return T, assign


class RotatedBentJointMeasurement:
    """Circuit builder for the rotated bent (XZ) joint lattice-surgery measurement.

    Args:
        data:      list of data-qubit ``(col, row)`` coordinates (rotated frame).
        checks:    list of stabilizer dicts ``{'syn', 'type', 'pauli', 'corners'}``
                   where ``type`` is ``'X'`` / ``'Z'`` / ``'M'`` and ``pauli`` maps
                   each corner ``(col, row)`` to ``'X'`` / ``'Z'``.
        x_logical: data-qubit coords supporting the measured logical X̄ (read out
                   from the final X-basis data measurement as observable 0).

    The qubit index layout is: data qubits ``0 .. nq-1`` (in ``data`` order),
    then one ancilla per check ``nq + k`` (in ``checks`` order ``k``).
    """

    def __init__(self, data, checks, x_logical, domains=None):
        self.data = list(data)
        self.checks = list(checks)
        self.x_logical = list(x_logical)
        self.domains = dict(domains) if domains else {}
        self.nq = len(self.data)
        self.nst = len(self.checks)
        self.di = {q: i for i, q in enumerate(self.data)}
        self.aid = {k: self.nq + k for k in range(self.nst)}
        self.Xk = [k for k, c in enumerate(self.checks) if c['type'] == 'X']
        self.Zk = [k for k, c in enumerate(self.checks) if c['type'] == 'Z']
        self.Mk = [k for k, c in enumerate(self.checks) if c['type'] == 'M']
        self._hb_plan = None    # cached (T, tickmaps) schedule

    def _hb_schedule(self):
        if self._hb_plan is None:
            self._hb_plan = assign_hook_benign_schedule(
                self.checks, self.domains, include_mixed=True)
        return self._hb_plan

    def _se_round_hook_benign(self, c, p):
        """One FULLY INTERLEAVED SE round under the hook-benign schedule: every
        check — pure X, pure Z and mixed — couples inside the SAME T-slot
        window (mixed X corners via CNOT, Z corners via CZ; CNOT and CZ share a
        tick on disjoint qubits).  Round depth: R + H + T + H + M."""
        di, AID, CHECKS = self.di, self.aid, self.checks
        T, tickmaps = self._hb_schedule()
        anc = [AID[k] for k in range(self.nst)]
        c.append("R", anc)
        if p > 0:
            c.append("X_ERROR", anc, p)
        c.append("TICK")                                            # phase: reset ancillas

        xanc = [AID[k] for k in self.Xk] + [AID[k] for k in self.Mk]
        c.append("H", sorted(xanc))
        if p > 0:
            c.append("DEPOLARIZE1", sorted(xanc), p)
        c.append("TICK")                                            # phase: prep X/mixed ancillas

        for t in range(T):                                          # ONE coupling window
            cn, cz = [], []
            for k in self.Xk:
                for q, tk in tickmaps[k].items():
                    if tk == t:
                        cn += [AID[k], di[q]]
            for k in self.Zk:
                for q, tk in tickmaps[k].items():
                    if tk == t:
                        cn += [di[q], AID[k]]
            for k in self.Mk:
                ch = CHECKS[k]
                for q, tk in tickmaps[k].items():
                    if tk == t:
                        if ch['pauli'][q] == 'X':
                            cn += [AID[k], di[q]]                   # CNOT(anc -> data)
                        else:
                            cz += [di[q], AID[k]]                   # CZ(data, anc)
            if cn:
                c.append("CNOT", cn)
                if p > 0:
                    c.append("DEPOLARIZE2", cn, p)
            if cz:                                                  # disjoint qubits: same tick
                c.append("CZ", cz)
                if p > 0:
                    c.append("DEPOLARIZE2", cz, p)
            c.append("TICK")

        c.append("H", sorted(xanc))
        if p > 0:
            c.append("DEPOLARIZE1", sorted(xanc), p)
        c.append("TICK")                                            # phase: unprep X/mixed ancillas

        c.append("M", anc, p if p > 0 else 0)                       # all ancillas, fixed order
        c.append("TICK")

    def circuit(self, rounds=3, p=0.0, basis="X"):
        """Build the full bent joint-measurement circuit: ``basis``-basis init, ``rounds`` of
        syndrome extraction, ``basis``-basis data readout, detectors and the tracked observable.

        ``basis="X"`` (default) is the X-memory experiment (X-init/readout, X-type checks close, the
        observable is the X̄ logical).  ``basis="Z"`` is the Z-memory experiment, used when the tracked
        logical is Z (e.g. a pure-Z joint ``M(Z̄₁Z̄₂)``).  Returns a ``stim.Circuit``.

        Syndrome extraction uses the hook-benign schedule (orientation-aware benign
        tables + T=5 floating idle slot at orientation seams; see
        :func:`assign_hook_benign_schedule`).

        WARNING: this is a STANDALONE builder whose observable is a SINGLE logical
        (``x_logical``), NOT the ``SyndromeTracker`` joint m.  It is correct for
        single-patch memory and for checking a round's determinism / collision-
        freedom, but on a MULTI-PATCH joint its graphlike distance is NOT the real
        joint distance -- use :class:`RoutedMultiPatchLSExperiment` to measure joint
        distance / LER."""
        se_round = lambda cc, pp: self._se_round_hook_benign(cc, pp)
        di, AID, CHECKS = self.di, self.aid, self.checks
        nq, NST = self.nq, self.nst
        reset, meas, err, want = (("RX", "MX", "Z_ERROR", "X") if basis == "X"
                                  else ("RZ", "MZ", "X_ERROR", "Z"))
        close_k = self.Xk if basis == "X" else self.Zk             # same-type checks close at readout
        c = stim.Circuit()
        for i, q in enumerate(self.data):
            c.append("QUBIT_COORDS", [i], [float(q[0]), float(q[1])])
        for k in range(NST):
            c.append("QUBIT_COORDS", [AID[k]], [float(CHECKS[k]['syn'][0]), float(CHECKS[k]['syn'][1])])
        c.append(reset, range(nq))
        if p > 0:
            c.append(err, range(nq), p)
        is_det0 = lambda ch: all(P == want for P in ch['pauli'].values())   # deterministic on the init state
        for r in range(rounds):
            if p > 0:
                c.append("DEPOLARIZE1", range(nq), p)               # data idle
            se_round(c, p)
            for k in range(NST):
                tr = -(NST - k); co = [float(CHECKS[k]['syn'][0]), float(CHECKS[k]['syn'][1]), r]
                if r == 0:
                    if is_det0(CHECKS[k]):
                        c.append("DETECTOR", [stim.target_rec(tr)], co)
                else:
                    c.append("DETECTOR", [stim.target_rec(tr), stim.target_rec(tr - NST)], co)
        c.append(meas, range(nq), p if p > 0 else 0)                # basis-basis data readout
        drec = {q: -(nq - di[q]) for q in self.data}
        for k in close_k:                                          # closing detectors: last SE ancilla vs data
            ch = CHECKS[k]; tr_anc = -(nq + (NST - k))
            c.append("DETECTOR", [stim.target_rec(tr_anc)] + [stim.target_rec(drec[d]) for d in ch['pauli']],
                     [float(ch['syn'][0]), float(ch['syn'][1]), rounds])
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(drec[q]) for q in self.x_logical], 0)   # tracked logical
        return c








def se_round_chunk(system, domains=None):
    """ONE syndrome-extraction round over ``system``'s ACTIVE stabilizers as a
    stim circuit chunk for :meth:`CircuitBuilder.apply_syndrome_extraction`.

    Emits gates on GLOBAL qubit indices; the last instruction is the single
    ancilla measurement (builder contract).  Reuses the rotated bent-joint
    hook-benign engine, so merged rounds with mixed domain-wall checks inherit
    the verified hook geometry.
    """
    checks, aid = [], {}
    for k, uid in enumerate(sorted(system.active_stabilizer_indices)):
        rec = system.stabilizers[uid]
        if rec.get('kf') is not None:
            raise ValueError(
                f"se_round_chunk: stabilizer at {tuple(rec['syn_coord'])} carries "
                f"K&F relay metadata ('kf'); the bent hook-benign schedule cannot "
                f"express a stretched check and would silently emit long-range "
                f"gates. Merged rounds containing a seam wall must use "
                f"DiagonalSurfaceCodeExtractionBlock(system).circuit.")
        pauli = {tuple(system.qubit_coords[i]): P for i, P in rec['pauli'].items()}
        typ = rec.get('type', 'Z')
        checks.append({'syn': tuple(rec['syn_coord']),
                       'pauli': pauli,
                       'type': typ if typ in ('X', 'Z') else 'M',
                       'corners': sorted(pauli)})
        aid[k] = rec['syn_idx'] if rec.get('syn_idx') is not None \
            else system.index_map[tuple(rec['syn_coord'])]
    data_coords = sorted({q for ch in checks for q in ch['pauli']})
    rb = RotatedBentJointMeasurement(data_coords, checks,
                                     x_logical=[data_coords[0]],
                                     domains=domains or {})
    rb.di = {q: system.index_map[q] for q in data_coords}
    rb.aid = aid
    c = stim.Circuit()
    rb._se_round_hook_benign(c, 0.0)
    while len(c) and c[-1].name == 'TICK':      # builder contract: chunk ends on M
        c = c[:-1]
    return c
