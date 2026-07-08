"""Deterministic mixed-stabilizer construction for N-patch subset joint measurements.

Implements the documented construction rules (Handbook §10.4, Fig 33/34) **as rules** —
no randomized search anywhere.  Per checkerboard phase the construction is a
deterministic greedy pass:

1.  **Bulk** — every complete (weight-4) plaquette of the merged region is forced:
    patch bulk, bus bulk and the mixed (XZ) walls alike.  Patch-edge weight-2s at the
    bus interface are gone automatically (those plaquettes gained bus corners and became
    forced bulk).
2.  **Logical representatives** — one patch-spanning representative per measured logical
    is fixed from the forced bulk (deterministically); every boundary decision below must
    keep all of them alive (a tile anticommuting with a representative is never added).
3.  **Corners first** — exposed corner data qubits (≤2 orthogonal neighbours) are covered
    first: for each corner in lexicographic order the first incident boundary tile that
    passes the vetoes is selected.  This anchors the alternating pattern of every
    boundary chain.  An uncoverable corner is not yet an error (a corner on a mixed wall
    may be fully protected by the forced bulk) — the final acceptance is the arbiter.
4.  **Boundary sweep — alternating-spacing rule** — remaining candidates (weight-2 and
    the weight-3 concave-corner plaquettes of Fig 34 labels 1/3) are swept in
    lexicographic order and added unless a veto applies:
      * one lattice spacing from an already-selected stabilizer along the same edge
        (they share a data qubit and their centres are colinear) — the spacing rule;
      * anticommutes with a forced/selected stabilizer or a logical representative;
      * linearly dependent on the current selection;
      * would pull a proper sub-product of the measured logicals into the span.
    On straight edges this reproduces add-one-skip-one alternation; at algebraically
    vetoed positions the pattern re-synchronises, exactly as at the walls in Fig 33.
5.  **Deterministic re-anchoring** — when the final acceptance fails, the corner
    anchoring of rule 3 is refined: the corner-pass tiles are banned one at a time (in
    lexicographic order, keeping a ban only when the acceptance metric strictly
    improves), which flips the alternation of the affected chain to its other pattern.
    A rule-driven tiebreak, not a search: the whole procedure remains a pure function of
    the geometry.
6.  **Convex outer corners** (Fig 34 label 2) — if acceptance still fails and a
    bus-corner plaquette sits one lattice spacing (centre Chebyshev distance 2) from a
    selected boundary stabilizer, that corner data qubit is **cut** (removed,
    weight-4 → weight-3) and the phase is rebuilt on the reduced region; a weight-1
    leftover on a bus corner triggers the same cut.  Cut-free solutions are preferred:
    both phases are tried without cuts before cuts are allowed.
"""

import itertools

from .bent_layout import _bent_plaquettes, _symplectic
from .multi_patch import (
    _patch_rep, _logical_direction, _int_symplectic, _IntBasis, _icommute,
)

__all__ = ["rule_based_joint_checks"]

_ORTH = ((2, 0), (-2, 0), (0, 2), (0, -2))


def _corner_qubits(dset):
    """Exposed corner data qubits: ≤2 orthogonal data neighbours."""
    return {q for q in dset
            if sum(((q[0] + dx, q[1] + dy) in dset) for dx, dy in _ORTH) <= 2}


class _Builder:
    """One deterministic greedy construction on a fixed region and phase.

    ``forbidden`` is a set of syndrome sites the boundary may NOT use — the ancilla
    qubits already owned by neighbouring idle patches; vetoing them makes the
    alternating pattern interleave with the neighbour on a shared ancilla line, so
    edge-adjacent placement is physically collision-free whenever the parity allows.
    """

    def __init__(self, placed, target, orient, data, retype, phase, forbidden=frozenset()):
        self.data = sorted(data)
        self.forbidden = frozenset(forbidden)
        self.target = target
        self.patchq = set().union(*[placed[nm] for nm, _ in target])
        self.plaqs = _bent_plaquettes(self.data, retype, phase)
        self.forced = [p for p in self.plaqs if len(p["pauli"]) >= 4]
        self.pool = sorted((p for p in self.plaqs if len(p["pauli"]) < 4),
                           key=lambda p: p["syn"])
        self.isv, self.n = _int_symplectic(self.data)
        sv, _ = _symplectic(self.data)
        F = [sv(p["pauli"]) for p in self.forced]
        self.reps = {}
        for nm, P in target:
            sup = _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, self.n)
            if sup is None:
                self.fail = (f"no {P}-representative for patch {nm} commutes with the "
                             f"forced bulk (checkerboard phase {phase})")
                return
            self.reps[nm] = (P, sup)
        self.fail = None
        self.repv = [self.isv({q: P for q in sup}) for P, sup in self.reps.values()]
        N = len(target)
        self.subprods = []
        for r in range(1, N):
            for comb in itertools.combinations(range(N), r):
                x = 0
                for k in comb:
                    x ^= self.repv[k]
                self.subprods.append(x)
        self.fvecs = [self.isv(p["pauli"]) for p in self.forced]

    def build(self, banned):
        """Greedy corner-first + sweep with ``banned`` corner tiles excluded.

        Returns ``(selected, corner_picks, metric, w1)`` where metric is the acceptance
        shortfall ``(rank deficit, #weight-1, joint-missing)`` — ``(0,0,0)`` is success.
        """
        n, isv = self.n, self.isv
        B = _IntBasis()
        svecs = list(self.fvecs)
        for v in self.fvecs:
            B.add(v)
        Lres = [B.reduce(L) for L in self.subprods]
        Lset = set(Lres)
        selected, sel_qubits, corner_picks = [], {}, []

        def one_spacing(p):
            return any(s[0] == p["syn"][0] or s[1] == p["syn"][1]
                       for q in p["pauli"] for s in sel_qubits.get(q, ()))

        def try_add(p, relax=False):
            if p["syn"] in banned or p["syn"] in self.forbidden:
                return False
            if not relax and one_spacing(p):
                return False
            v = isv(p["pauli"])
            if any(not _icommute(v, s, n) for s in svecs):
                return False
            if any(not _icommute(v, L, n) for L in self.repv):
                return False
            r = B.reduce(v)
            if r == 0 or r in Lset:
                return False
            h = r.bit_length() - 1
            B.piv[h] = r
            svecs.append(v)
            selected.append(p)
            for q in p["pauli"]:
                sel_qubits.setdefault(q, []).append(p["syn"])
            for i, lr in enumerate(Lres):
                if (lr >> h) & 1:
                    Lres[i] = lr ^ r
            if True:
                Lset.clear()
                Lset.update(Lres)
            return True

        tiles_at = {}
        for p in self.pool:
            for q in p["pauli"]:
                tiles_at.setdefault(q, []).append(p)
        for q in sorted(_corner_qubits(set(self.data))):
            if q in sel_qubits:
                continue
            for p in tiles_at.get(q, ()):
                if try_add(p):
                    corner_picks.append(p["syn"])
                    break
        for p in self.pool:
            if p not in selected:
                try_add(p)

        N = len(self.target)
        deficit = (len(self.data) - B.rank) - (N - 1)
        if deficit > 0:                    # T-junction repair: relax spacing, lex order
            for p in self.pool:
                if p not in selected and try_add(p, relax=True):
                    deficit = (len(self.data) - B.rank) - (N - 1)
                    if deficit <= 0:
                        break
        w1 = []
        for q in self.data:
            for P in "XZ":
                v = isv({q: P})
                if B.reduce(v) != 0 and all(_icommute(v, s, n) for s in svecs):
                    w1.append(q)
        joint = 0
        for v in self.repv:
            joint ^= v
        metric = (max(deficit, 0), len(w1), 0 if B.contains(joint) else 1)
        return selected, corner_picks, metric, w1


def _construct_phase(placed, target, orient, dset, retype, phase, forbidden=frozenset()):
    """One deterministic construction attempt.  Returns ``("ok", checks, logicals,
    x_obs)``, ``("cut", {qubits})`` per the convex-corner rule, or ``("fail", reason)``."""
    bl = _Builder(placed, target, orient, dset, retype, phase, forbidden)
    if bl.fail is not None:
        return ("fail", bl.fail)

    banned = set()
    selected, picks, metric, w1 = bl.build(banned)
    if metric != (0, 0, 0):
        # rule 5 -- deterministic re-anchoring: ban selected tiles one at a time (lex
        # order), keep a ban only when the acceptance metric strictly improves; the
        # greedy then re-synchronises the affected chain to its other pattern.
        for _ in range(len(bl.pool) + 1):
            improved = False
            for t in sorted({p["syn"] for p in selected}):
                if t in banned:
                    continue
                s2, p2, m2, w2 = bl.build(banned | {t})
                if m2 < metric:
                    banned.add(t)
                    selected, picks, metric, w1 = s2, p2, m2, w2
                    improved = True
                    break
            if not improved or metric == (0, 0, 0):
                break

    if metric != (0, 0, 0):
        # rule 6 -- convex-corner criterion (Fig 34 label 2): a bus-corner plaquette one
        # lattice spacing (syn-Chebyshev 2) from a selected boundary stabilizer, or a
        # weight-1 leftover on a bus corner, demands the corner data qubit be cut.  The
        # full candidate set is reported; the caller applies its cut strategy (all at
        # once for symmetric bends, one at a time for minimal cuts).
        corner_q = _corner_qubits(set(bl.data)) - bl.patchq
        cuts = {q for q in w1 if q in corner_q}
        for q in corner_q:
            w4 = next((p for p in bl.forced if q in p["pauli"]), None)
            if w4 is None:
                continue
            for p in selected:
                if max(abs(w4["syn"][0] - p["syn"][0]),
                       abs(w4["syn"][1] - p["syn"][1])) == 2:
                    cuts.add(q)
                    break
        if cuts:
            return ("cut", cuts)
        return ("fail", f"acceptance shortfall {metric} (rank deficit, weight-1 count, "
                        f"joint missing) and no convex corner to cut (phase {phase}; "
                        f"weight-1 at {sorted(w1)[:4]})")

    checks = []
    for p in bl.forced + selected:
        c = dict(p)
        c["corners"] = sorted(c["pauli"])
        checks.append(c)
    logicals = [(nm, bl.reps[nm][0], bl.reps[nm][1]) for nm, _ in target]
    x_obs = next((s for nm, P, s in logicals if P == "X"), logicals[0][2])
    return ("ok", checks, logicals, x_obs)


def rule_based_joint_checks(placed, target, orient, data, retype, d, max_cut=4,
                            forbidden=frozenset()):
    """Deterministic construction on the routed region (both phases, rule-driven cuts).

    Cut-free solutions are preferred: phases 0/1 are tried without cuts first, then with
    the convex-corner rule allowed to remove up to ``max_cut`` bus-corner qubits.
    Returns ``dict(checks, data, cut, phase, logicals, x_observable, reason="")`` on
    success, else ``dict(checks=None, reason=<why per phase>)``.  Pure function of the
    geometry — no randomness.
    """
    reasons = []
    first_cuts = {}                       # phase -> the first failing state's candidates

    def attempt(phase, strategy, allowed, initial=()):
        dset = set(data)
        cut = []
        for q in initial:
            dset.discard(q)
            cut.append(q)
        while True:
            rt = {q for q in retype if q in dset}
            res = _construct_phase(placed, target, orient, dset, rt, phase, forbidden)
            if res[0] == "ok":
                return dict(checks=res[1], data=sorted(dset), cut=tuple(sorted(cut)),
                            phase=phase, logicals=res[2], x_observable=res[3], reason="")
            if res[0] == "cut":
                if not cut and phase not in first_cuts:
                    first_cuts[phase] = sorted(res[1])
                take = sorted(res[1]) if strategy == "all" else [min(res[1])]
                if len(cut) + len(take) > allowed:
                    if allowed:
                        reasons.append(f"phase {phase}/{strategy}: corner rule wants "
                                       f"more than max_cut={max_cut} cuts")
                    return None
                for q in take:
                    dset.discard(q)
                    cut.append(q)
                continue
            if allowed:
                reasons.append(f"{strategy}: {res[1]}")
            return None

    # cut-free first; then cuts: "all at once" (symmetric bends), "one at a time"
    # (minimal cuts), and finally each first-round candidate as the seed cut — a
    # bounded, deterministic sequence of rule-driven attempts.
    for allowed, phase, strategy in [(0, 0, "all"), (0, 1, "all"),
                                     (max_cut, 0, "all"), (max_cut, 1, "all"),
                                     (max_cut, 0, "one"), (max_cut, 1, "one")]:
        out = attempt(phase, strategy, allowed)
        if out is not None:
            return out
    for phase in (0, 1):
        for q in first_cuts.get(phase, ()):
            out = attempt(phase, "one", max_cut, initial=(q,))
            if out is not None:
                return out
    return dict(checks=None, data=sorted(data), cut=(), phase=None,
                logicals=None, x_observable=None, reason="; ".join(reasons))
