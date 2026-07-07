"""Rotated **N-patch** joint lattice-surgery measurement — ``M(∏ᵢ P̄ᵢ)``.

A genuinely N-patch generalization of the two-/three-patch bent joint measurements.  The user
gives one :class:`.PatchSpec` per logical patch (origin / distance / measured_logical /
orientation) and an optional ``target`` (which logical each patch contributes); the generator
**derives the whole geometry from the patch origins** — routed bus, mixed (XZ) walls, boundary
trim/replacement and the readout chain all follow from where the patches sit.  There is **no
``offset`` parameter and no hard-coded p1/p2/p3 topology**: any vertical/horizontal stagger is an
*internal* quantity read off the origins.  Changing any origin regenerates the layout; an
incompatible placement raises :class:`.BentLayoutError` with a concrete geometric reason.

**Routing layer (coordinate-derived, not a template).**  The geometry is built in two layers.
The *routing* layer turns the patch coordinates into a connected data region; the *physics* layer
(re-typing + mixed walls + selection + verification) then works on whatever region it is handed —
it is geometry-agnostic, so all of the cleverness is in routing.

* **Pairwise fusion corridor** (Phase 1) — for two same-type (X-side) patches that share a row or a
  column, :func:`_xx_fuse` derives a ``d``-wide corridor straight from their boxes: it extends the
  lower patch's columns up to the upper patch (or the right patch's rows left to the left patch).
  The corridor is **computed from the coordinates** (any even stagger works); there is no
  ``p1.right == p3.left`` template to match and no ``offset`` parameter.  A narrow corridor (not a
  wide shared band) is what keeps the two X-logicals independent.
* **Routing graph over the X-patches** (Phase 2) — fusing a *set* of X-patches is a graph: each
  edge is one pairwise corridor, and the X-region is the union of the patches and their corridors.
  The graph comes from ``routing``:
  * ``routing="auto"`` — the **bounded auto-router** (:func:`_candidate_edge_sets`) computes the
    fusable X-patch pairs (those sharing a row/column) and takes a spanning tree (greedy MST by
    centre distance) over them.  It is *bounded*: it tries that one coordinate-derived candidate
    tree, not an exhaustive search over all topologies — hence "not a general auto-router".
  * an **explicit graph** — ``routing={"type": "tree", "edges": [("p1","p3"), ...]}`` or a bare
    list of edges — uses exactly the X–X fusions you name.
* **Each** Z-side patch is reached by a band (:func:`_zband`) tapping the X-region (a vertical bus,
  :func:`_add_bus`, grows it to span every Z-patch's rows); the **mixed (XZ) domain wall** falls at
  each Z-patch's boundary automatically (``retype`` = everything except the Z-patches).  Multiple
  Z-side patches are supported — each gets its own wall.

Each candidate routing is **oracle-gated**: it is accepted only if the selection finds a boundary
set that puts the full joint in the stabilizer span with every proper sub-product excluded.  A
disconnected X-region, an unfusable explicit edge, bad parity/overlap, or an unreachable/overlapping
Z all raise :class:`.BentLayoutError` with a concrete reason — the generator never silently
mis-routes or mis-measures.

``M(X̄₁X̄₃Z̄₂)`` (one Z), ``M(Z̄₁Z̄₂X̄₃)`` (two Z onto one X-bus), and ``M(Z̄₁Z̄₂X̄₃X̄₄)`` (the XXZZ
cross: X̄₃/X̄₄ on a vertical trunk, Z̄₁/Z̄₂ on the horizontal arm) are all **examples** of this one
API, not special cases.

**Scope (the real constraint is *separation*, not patch counts).**  Both multiple X-side and
multiple Z-side patches work.  The genuine limit is homological: **two same-type patches whose
logicals run parallel collapse to a measured pair-product when they fuse into one region with no
mixed wall / junction separating them.**  So two X-patches *adjacent* on a bare trunk give
``X̄ᵢX̄ⱼ`` in the span (a sub-product), but two X-patches at opposite ends of a trunk with the
Z-walls crossing *between* them (the XXZZ cross) stay independent and the 4-body joint is clean.
Separately-walled Z-patches do not collapse (X-side material sits between their walls).  When a
routing would collapse a pair, the oracle **detects it** (no boundary selection measures the full
joint) and **raises** — it never silently mis-measures.  The auto-router only tries one
coordinate-derived candidate, so a collapsing placement is rejected; reposition so same-type
logicals are separated, or pass an explicit routing.

The construction realizes **only the full joint** ``∏ᵢ P̄ᵢ`` — no single logical and **no proper
sub-product** is measured — leaving ``N-1`` logical degrees of freedom.  Selection forbids every
proper sub-product from the stabilizer span, forces the full joint into it, and avoids leaving a
weight-1 leftover logical.  A bit-packed-int GF(2) search makes this fast (3-patch ``d=5`` and the
4-patch build in ~1-2 s).  ``verify()`` is the oracle (eleven checks).
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from .bent_layout import (
    PatchSpec, BentLayoutError, place_patch,
    _bent_plaquettes, _symplectic, _gf2_rank, _commute, _no_tick_collision,
)


# -----------------------------------------------------------------------------
# bit-packed GF(2) machinery — Pauli vectors as Python ints (X in bits 0..n-1, Z in n..2n-1).
# Int XOR / int.bit_count() are C-level, so the boundary search scales to N=5/6 and d=5.
# -----------------------------------------------------------------------------

def _int_symplectic(data):
    """Return ``(isv, n)`` where ``isv({coord: pauli})`` packs a Pauli into a 2n-bit int."""
    idx = {c: i for i, c in enumerate(sorted(data))}
    n = len(idx)

    def isv(pauli):
        v = 0
        for c, P in pauli.items():
            i = idx[c]
            if P in ("X", "Y"):
                v |= 1 << i
            if P in ("Z", "Y"):
                v |= 1 << (n + i)
        return v
    return isv, n


def _icommute(a, b, n):
    mask = (1 << n) - 1
    return (((a & mask) & (b >> n)).bit_count() + ((a >> n) & (b & mask)).bit_count()) & 1 == 0


class _IntBasis:
    """An incremental GF(2) row basis over bit-packed ints (reduction by highest set bit)."""
    __slots__ = ("piv",)

    def __init__(self):
        self.piv = {}                      # highest-set-bit -> reduced row int

    def copy(self):
        b = _IntBasis()
        b.piv = dict(self.piv)
        return b

    def reduce(self, v):
        while v:
            r = self.piv.get(v.bit_length() - 1)
            if r is None:
                return v
            v ^= r
        return v

    def add(self, v):                      # returns True iff v is independent (and adds it)
        r = self.reduce(v)
        if r:
            self.piv[r.bit_length() - 1] = r
            return True
        return False

    def contains(self, v):
        return self.reduce(v) == 0

    @property
    def rank(self):
        return len(self.piv)


# -----------------------------------------------------------------------------
# routing layer: coordinate-derived corridors (fusion / band), an explicit routing
# graph, and a bounded auto-router.  This REPLACES the old fixed staircase template:
# corridors are computed from the patch coordinates (any even offset works), and the
# old corner-touch staircase is just one possible edge set of corner fusions.
# -----------------------------------------------------------------------------

def _cols(d):
    return sorted({c for c, _ in d})


def _rows(d):
    return sorted({r for _, r in d})


def _box(d):
    xs = _cols(d); ys = _rows(d)
    return xs[0], xs[-1], ys[0], ys[-1]        # left, right, top, bottom


def _rect(x0, x1, y0, y1):
    return {(x, y) for x in range(x0, x1 + 1, 2) for y in range(y0, y1 + 1, 2)}


def _xx_fuse(A, B):
    """A d-wide fusion region joining same-type patches ``A``,``B`` that share a row or column.

    Same-type (X–X) patches are joined by **extending one patch's strip to meet the other** along
    their shared columns (vertical fusion) or shared rows (horizontal fusion) — a corner touch
    (1-cell overlap) or a wider overlap both work, and the offset is read from the coordinates (no
    ``U.right == L.left`` template).  Separated same-type patches that share **no** row or column
    are rejected: a plain d-wide corridor between them would *merge* their logicals (measure the
    sub-product).  Raises :class:`.BentLayoutError` with the concrete reason.
    """
    al, ar, at, ab = _box(A); bl, br, bt, bb = _box(B)
    col_ov = max(al, bl) <= min(ar, br)
    row_ov = max(at, bt) <= min(ab, bb)
    if col_ov and row_ov:
        raise BentLayoutError("fused X-patches overlap")
    if col_ov:                                  # vertical fusion: extend the LOWER patch's columns
        if at <= bt:                            # A upper, B lower   (canonical: order-independent)
            u_bot, l_cols, l_top = ab, (bl, br), bt
        else:                                   # B upper, A lower
            u_bot, l_cols, l_top = bb, (al, ar), at
        u_top = min(at, bt)
        gap = l_top - u_bot
        if gap < 2 or gap % 2:
            raise BentLayoutError(f"vertical gap between fused X-patches = {gap} (need even >= 2)")
        return _rect(l_cols[0], l_cols[1], u_top, l_top - 2)
    if row_ov:                                  # horizontal fusion: extend the RIGHT patch's rows
        if al <= bl:                            # A left, B right
            l_right, r_rows, r_left = ar, (bt, bb), bl
        else:                                   # B left, A right
            l_right, r_rows, r_left = br, (at, ab), al
        l_left = min(al, bl)
        gap = r_left - l_right
        if gap < 2 or gap % 2:
            raise BentLayoutError(f"horizontal gap between fused X-patches = {gap} (need even >= 2)")
        return _rect(l_left, r_left - 2, r_rows[0], r_rows[1])
    raise BentLayoutError(
        "X-patches share no row or column to fuse — a d-wide corridor between fully separated "
        "same-type patches would merge their logicals (measure a sub-product).  Reposition so they "
        "share a row or column, or connect them through a shared neighbour.")


def _zband(xregion, Z):
    """A d-wide band attaching the patch ``Z`` to the bus ``xregion`` from **any** side.

    The patch attaches **horizontally** (a left/right band) if the bus has data at the patch's rows,
    or **vertically** (an above/below band) if the bus has data at the patch's columns — whichever
    applies.  Gaps are even and ≥ 2 (a consequence of odd origins + no overlap).  Raises if the patch
    reaches the bus on neither axis, or overlaps it.
    """
    zl, zr, zt, zb = _box(Z)
    cols_at_rows = [x for (x, y) in xregion if zt <= y <= zb]   # bus columns at the patch's rows
    rows_at_cols = [y for (x, y) in xregion if zl <= x <= zr]   # bus rows at the patch's columns
    if cols_at_rows:                                            # horizontal attach (left / right)
        if zl > max(cols_at_rows):
            gap = zl - max(cols_at_rows)
            if gap < 2 or gap % 2:
                raise BentLayoutError(f"band gap = {gap} (need even >= 2)")
            return _rect(max(cols_at_rows) + 2, zl - 2, zt, zb)
        if zr < min(cols_at_rows):
            gap = min(cols_at_rows) - zr
            if gap < 2 or gap % 2:
                raise BentLayoutError(f"band gap = {gap} (need even >= 2)")
            return _rect(zr + 2, min(cols_at_rows) - 2, zt, zb)
        raise BentLayoutError("patch overlaps the bus in columns at its rows")
    if rows_at_cols:                                            # vertical attach (above / below)
        if zt > max(rows_at_cols):
            gap = zt - max(rows_at_cols)
            if gap < 2 or gap % 2:
                raise BentLayoutError(f"band gap = {gap} (need even >= 2)")
            return _rect(zl, zr, max(rows_at_cols) + 2, zt - 2)
        if zb < min(rows_at_cols):
            gap = min(rows_at_cols) - zb
            if gap < 2 or gap % 2:
                raise BentLayoutError(f"band gap = {gap} (need even >= 2)")
            return _rect(zl, zr, zb + 2, min(rows_at_cols) - 2)
        raise BentLayoutError("patch overlaps the bus in rows at its columns")
    raise BentLayoutError(f"patch (rows {zt}..{zb}, cols {zl}..{zr}) does not reach the bus on either "
                          "axis; place it adjacent to the routed region")


def _validate_placement(placed, target):
    """Routing-independent sanity of the placement: odd parity, no overlap.

    Raised before any routing is attempted so the reason is the *placement*, not the corridor.
    A joint may be pure-X (``M(X̄₁X̄₂)``), pure-Z (``M(Z̄₁Z̄₂)``) or mixed — no X/Z balance is required
    here; whether the chosen routing measures exactly the joint is decided by the GF(2) oracle."""
    reasons = []
    if any(x % 2 == 0 or y % 2 == 0 for d in placed.values() for x, y in d):
        reasons.append("all patch origins must be odd (data on the (odd,odd) lattice)")
    nm_list = list(placed)
    for i in range(len(nm_list)):
        for j in range(i + 1, len(nm_list)):
            if placed[nm_list[i]] & placed[nm_list[j]]:
                reasons.append(f"patches {nm_list[i]} and {nm_list[j]} overlap")
    if reasons:
        raise BentLayoutError("patches cannot host an N-patch joint: " + "; ".join(reasons))


def _add_bus(xregion, others):
    """Extend the trunk with a vertical d-wide **bus** so it reaches every *horizontally*-attached patch.

    A patch attaches **horizontally** (a left/right band) only if the trunk has data at the patch's
    rows; when it does not (e.g. the lone X-patch sits below both Z-patches, as in ``M(Z̄₁Z̄₂X̄₃)``),
    this grows the trunk into a vertical bus spanning that patch's rows at the trunk's columns.
    Patches that share the trunk's **columns** attach *vertically* (above/below) instead, so they are
    **not** extended into.  If every patch is already reached it is a **no-op**, so existing layouts
    are unchanged.
    """
    xl, xr, xt, xb = _box(xregion)
    reach = {y for _, y in xregion}
    need = []
    for p in others:
        pl, pr, pt, pb = _box(p)
        if max(xl, pl) <= min(xr, pr):               # shares trunk columns -> vertical (above/below) attach
            continue
        if set(_rows(p)) <= reach:                   # already reached horizontally
            continue
        need.append(p)
    if not need:
        return xregion
    tops = [xt] + [_box(p)[2] for p in need]
    bots = [xb] + [_box(p)[3] for p in need]
    return xregion | _rect(xl, xr, min(tops), max(bots))


def _route(placed, target, edges, z_sides=None):
    """Build the data set + X-side (re-type) set from a routing ``edges`` graph (X–X fusions).

    The X-patches split into fusion components (joined by the X–X edges); the largest is the **trunk**
    and is grown by a vertical bus (:func:`_add_bus`) to span every other component's rows.  Every
    other component hangs off the bus through its own band (:func:`_zband`): a **side-arm X-component**
    — *one or more* fused X-patches — by a bare X-corridor (no wall, it is X-side), each **Z-patch** by
    a band that carries its own mixed (XZ) wall (Z stays native type).  ``retype`` = everything except
    the Z-patches, so the walls fall at each Z-patch boundary automatically.  Any number of X- and
    Z-patches, and multi-patch side-arms, are allowed: **this layer never rejects a placement for
    being a "non-trunk X-group"** — whether the candidate measures *only* the joint (no same-type pair
    collapsing to a sub-product) is decided downstream by the GF(2) oracle.  Raises
    :class:`.BentLayoutError` only for hard geometry faults: parity/overlap, a broken fusion, an
    unreachable/overlapping patch, or a disconnected region.
    """
    _validate_placement(placed, target)
    tgt = dict(target)
    xnames = [nm for nm, _ in target if tgt[nm] == "X"]
    znames = [nm for nm, _ in target if tgt[nm] == "Z"]
    # The bus is X-side whenever there is any X-patch (mixed or pure-X); a pure-Z joint uses a Z-bus.
    # The bus-type patches fuse into the trunk; the opposite-type patches attach through mixed walls.
    bus_type = "X" if xnames else "Z"
    bus_names = xnames if xnames else znames
    wall_names = znames if xnames else []
    bus_edges = [(a, b) for a, b in edges if tgt.get(a) == bus_type and tgt.get(b) == bus_type]

    # group bus-patches into fusion components (union-find over the edges); the largest is the trunk
    parent = {nm: nm for nm in bus_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a, b in bus_edges:
        parent[find(a)] = find(b)
    comps = {}
    for nm in bus_names:
        comps.setdefault(find(nm), []).append(nm)
    core_names = max(comps.values(), key=len)

    def _fuse(members):                              # fused cell-region of one bus-component
        reg = set().union(*[placed[nm] for nm in members])
        for a, b in bus_edges:
            if a in members and b in members:
                reg |= _xx_fuse(placed[a], placed[b])
        if not _connected(reg):
            raise BentLayoutError(f"patches {sorted(members)} do not fuse into one connected region")
        return reg

    core = _fuse(core_names)                          # the trunk
    side_regions = [_fuse(m) for m in comps.values() if m is not core_names]   # side bus-arms (1+ each)

    # bus: extend the trunk to span every attached region's rows (side bus-arms and wall patches)
    others = side_regions + [placed[w] for w in wall_names]
    core = _add_bus(core, others)
    wall_cells = set().union(*[placed[w] for w in wall_names]) if wall_names else set()
    clash = sorted({nm for m in comps.values() if m is not core_names for nm in m if placed[nm] & core}
                   | {w for w in wall_names if placed[w] & core})
    if clash:
        raise BentLayoutError(f"patch(es) {clash} overlap the bus; move them clear of the trunk "
                              "columns (the bus runs vertically through the trunk's columns)")

    # Each non-trunk bus-arm hangs off the trunk by a bare corridor (same type, no wall); each
    # opposite-type wall patch attaches by a band that carries its own mixed (XZ) wall.  Whether any
    # same-type pair then collapses to a sub-product is left to the GF(2) oracle (not pre-judged here).
    data = set(core)
    for reg in side_regions:        # side bus-arm: bare corridor into the trunk, NO wall (same type)
        data |= _zband(core, reg) | reg
    for nm in wall_names:           # opposite-type patch: band + its own mixed (XZ) wall
        if z_sides and nm in z_sides:                # honour an explicitly-declared wall side
            zl, zr, zt, zb = _box(placed[nm])
            xa = [x for (x, y) in core if zt <= y <= zb]
            actual = "right" if xa and zl > max(xa) else ("left" if xa and zr < min(xa) else "?")
            if actual != z_sides[nm]:
                raise BentLayoutError(f"explicit routing: Z-patch {nm} attaches on the {actual!r} of "
                                      f"the bus, but z_attachments declared {z_sides[nm]!r}")
        data |= _zband(core, placed[nm]) | placed[nm]
    if not _connected(data):
        raise BentLayoutError("the routed region is not connected")
    # re-typing makes the bus the X-side: for an X-bus, flip everything except the native Z-patches;
    # a pure-Z bus is entirely native (no flip), so nothing is re-typed.
    retype = {c for c in data if c not in wall_cells} if xnames else set()
    return sorted(data), retype, xnames, znames


def _connected(region):
    """True iff ``region`` (a set of (odd,odd) coords) is connected via the plaquette graph."""
    region = set(region)
    if not region:
        return True
    start = next(iter(region))
    seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2),
                       (x + 2, y + 2), (x + 2, y - 2), (x - 2, y + 2), (x - 2, y - 2)):
            if (nx, ny) in region and (nx, ny) not in seen:
                seen.add((nx, ny)); stack.append((nx, ny))
    return len(seen) == len(region)


def _candidate_edge_sets(placed, target):
    """Auto-routing: a candidate bus–bus fusion edge set (an MST over fusable bus-patch pairs).

    The bus is the X-patches when any exist (mixed/pure-X), else the Z-patches (pure-Z).  Returns the
    maximum spanning forest of the fusable pairs (those sharing a row/column).  Bus-patches that fuse
    become the trunk; any left unfused becomes a **side-arm** that :func:`_route` attaches by a bare
    corridor — so a not-all-fusable placement is no longer rejected here (the GF(2) oracle decides
    whether the resulting layout measures only the joint).
    """
    tgt = dict(target)
    xnames = [nm for nm, _ in target if tgt[nm] == "X"]
    bus_names = xnames if xnames else [nm for nm, _ in target if tgt[nm] == "Z"]
    if len(bus_names) <= 1:
        return [[]]
    fusable = []
    for i in range(len(bus_names)):
        for j in range(i + 1, len(bus_names)):
            al, ar, at, ab = _box(placed[bus_names[i]]); bl, br, bt, bb = _box(placed[bus_names[j]])
            col_ov = max(al, bl) <= min(ar, br); row_ov = max(at, bt) <= min(ab, bb)
            if (col_ov or row_ov) and not (col_ov and row_ov):
                dist = abs(al + ar - bl - br) + abs(at + ab - bt - bb)
                fusable.append((dist, bus_names[i], bus_names[j]))
    fusable.sort()
    parent = {nm: nm for nm in bus_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    mst = []
    for _, a, b in fusable:
        if find(a) != find(b):
            parent[find(a)] = find(b); mst.append((a, b))
    return [mst]                                 # unfused X-patches become side-arms in _route


def _logical_direction(pauli, orientation):
    """The direction the measured ``pauli`` logical string must run, per the patch ``orientation``.

    ``orientation`` names the X̄ direction (``"X_horizontal"`` ⇒ X̄ runs along a row, ``"X_vertical"``
    ⇒ X̄ runs up a column); Z̄ is perpendicular to X̄.  So a measured ``X`` on an ``X_horizontal``
    patch runs ``"horizontal"``, a measured ``Z`` on an ``X_horizontal`` patch runs ``"vertical"``,
    and vice-versa for ``X_vertical``.
    """
    x_is_horizontal = orientation == "X_horizontal"
    if pauli == "X":
        return "horizontal" if x_is_horizontal else "vertical"
    return "vertical" if x_is_horizontal else "horizontal"


def _patch_rep(patch, pauli, direction, F, sv, n):
    """A patch-spanning ``pauli`` string **in the requested ``direction``** commuting with bulk ``F``.

    The orientation declared on the ``PatchSpec`` is honoured: an ``X_horizontal`` X-logical is
    represented by a horizontal string, an ``X_vertical`` X-logical by a vertical one (and Z̄
    perpendicular).  Returns the support, or ``None`` if no string in that direction commutes — the
    caller tries the other parity phase, and if both fail the placement is rejected rather than the
    logical being silently re-oriented.
    """
    c, r = _cols(patch), _rows(patch)
    cands = ([[(x, y) for y in r] for x in c] if direction == "vertical"
             else [[(x, y) for x in c] for y in r])
    for sup in cands:
        v = sv({q: pauli for q in sup})
        if all(_commute(v, f, n) for f in F):
            return sup
    return None


# -----------------------------------------------------------------------------
# check selection: forbid every proper sub-product, force the joint, avoid weight-1 leftover
# -----------------------------------------------------------------------------

def _select_joint_checks(data, plaqs, logicals, seed=0, max_trials=2500, n_seeds=20):
    """Pick a valid stabilizer code measuring exactly the joint ``∏ᵢ P̄ᵢ`` — or ``None``.

    ``logicals`` is a list of ``(pauli, support)``.  Weight-4 plaquettes (bulk + mixed wall) are
    forced; boundary plaquettes are selected (fixed-RNG order) so the code is commuting,
    GF(2)-independent, keeps every PROPER sub-product out of the span, puts the FULL joint into the
    span with residual logical count ``N-1``, and leaves **no weight-1 leftover logical**.
    When a trial code has a corner weight-1 logical, a **swap-repair** fixes it by trading the
    same-type boundary check on that corner for the opposite-type "killer" plaquette (which exists
    whenever the corner is weight-1) — keeping the code commuting, the joint in span and every
    sub-product out.  Returns the first weight-1-free (possibly repaired) selection, else a fallback.
    """
    isv, n = _int_symplectic(data)
    vecs = [isv({c: P for c in sup}) for P, sup in logicals]
    joint = 0
    for v in vecs:
        joint ^= v
    N = len(logicals)
    preserve = []
    for r in range(1, N):                               # all proper non-empty subset products
        for comb in itertools.combinations(range(N), r):
            x = 0
            for k in comb:
                x ^= vecs[k]
            preserve.append(x)

    forced = [p for p in plaqs if len(p["pauli"]) >= 4]
    boundary = [p for p in plaqs if len(p["pauli"]) < 4]
    Bforced = _IntBasis()                               # the forced weight-4 basis is the same every trial
    for p in forced:
        Bforced.add(isv(p["pauli"]))
    bvecs = [(p, isv(p["pauli"])) for p in boundary]
    qvecs = [isv({q: P}) for q in data for P in ("X", "Z")]   # single-qubit ops, for the weight-1 test
    # residuals of the 2^N-2 proper sub-products mod the forced basis, maintained incrementally so
    # the per-candidate "would this tile pull a sub-product into the span?" test is an O(1) set lookup.
    Lres0 = [Bforced.reduce(L) for L in preserve]

    # fast necessary check: if the joint is not even in the span of ALL plaquettes (forced +
    # every boundary tile, ignoring commutation) then no commuting subset can measure it — bail
    # out immediately instead of searching, so an unrealizable placement raises promptly.
    Bfull = Bforced.copy()
    for _, v in bvecs:
        Bfull.add(v)
    if not Bfull.contains(joint):
        return None

    def build(order):
        B = Bforced.copy()
        Lres = list(Lres0)
        Lset = set(Lres)
        kept = list(forced)
        for p, v in order:
            if not all(_icommute(v, b, n) for b in B.piv.values()):
                continue
            r = B.reduce(v)
            if r == 0:
                continue
            if r in Lset:                               # adding v would pull a sub-product into the span
                continue
            h = r.bit_length() - 1
            B.piv[h] = r
            kept.append(p)
            changed = False
            for i, lr in enumerate(Lres):               # update each sub-product residual
                if (lr >> h) & 1:
                    Lres[i] = lr ^ r
                    changed = True
            if changed:
                Lset = set(Lres)
        return B, kept

    def has_weight1(B):
        return any(B.reduce(v) != 0 and all(_icommute(v, b, n) for b in B.piv.values()) for v in qvecs)

    # --- weight-1 repair --------------------------------------------------------------------------
    # A corner weight-1 logical (e.g. X@q) means a same-type boundary check sits on q where the
    # opposite-type "killer" plaquette (Z on q) was available but not selected.  Swap it in — drop
    # the single selected check it anticommutes with, add the killer — which removes the weight-1
    # while keeping the code commuting (the only anticommuting partner is removed), the joint in span,
    # every proper sub-product out, and the logical count N-1.  Because a killer exists this is a
    # *selection* fix, not a distance artifact.
    qlab = [(q, P) for q in data for P in ("X", "Z")]
    qiv = {(q, P): isv({q: P}) for q in data for P in ("X", "Z")}
    KILL = {"X": "Z", "Z": "X"}

    def weight1_set(checks):
        B = _IntBasis()
        for c in checks:
            B.add(isv(c["pauli"]))
        return [(q, P) for (q, P) in qlab
                if B.reduce(qiv[(q, P)]) != 0 and all(_icommute(qiv[(q, P)], b, n) for b in B.piv.values())]

    def selection_ok(checks):
        B = _IntBasis()
        for c in checks:
            B.add(isv(c["pauli"]))
        return (len(data) - B.rank == N - 1 and B.contains(joint)
                and not any(B.contains(x) for x in preserve))

    def repair(checks):
        code = list(checks)
        for _ in range(4 * N + 4):
            w = weight1_set(code)
            if not w:
                break
            advanced = False
            for (q, P) in w:
                for kp in boundary:
                    if kp in code or kp["pauli"].get(q) not in (KILL[P], "Y"):
                        continue
                    kv = isv(kp["pauli"])
                    anti = [c for c in code if not _icommute(kv, isv(c["pauli"]), n)]
                    if len(anti) != 1:                  # a clean one-for-one swap preserves the rank
                        continue
                    cand = [c for c in code if c is not anti[0]] + [kp]
                    if selection_ok(cand) and len(weight1_set(cand)) < len(w):
                        code = cand; advanced = True; break
                if advanced:
                    break
            if not advanced:
                break
        return code

    fallback = None
    for s in range(seed, seed + n_seeds):
        rng = np.random.RandomState(s)
        for _ in range(max_trials):
            order = list(bvecs)
            rng.shuffle(order)
            B, kept = build(order)
            if B.contains(joint) and len(data) - B.rank == N - 1:
                if not has_weight1(B):
                    return kept                         # already weight-1-free
                fixed = repair(kept)                    # swap away the corner weight-1 logicals
                if not weight1_set(fixed):
                    return fixed
                if fallback is None:
                    fallback = fixed                    # best effort if repair can't fully clear it
    return fallback


def _readout_chain(data, checks, logicals):
    """Check syndromes whose GF(2) product equals the joint ``∏ᵢ P̄ᵢ`` (for highlighting/readout)."""
    from ....utils.linear_algebra import solve_linear_decomposition
    sv, n = _symplectic(data)
    basis = np.array([sv(ch["pauli"]) for ch in checks], np.uint8)
    target = np.zeros(2 * n, np.uint8)
    for P, sup in logicals:
        target ^= sv({c: P for c in sup})
    co, dep, _ = solve_linear_decomposition(basis=basis, targets=target.reshape(1, -1),
                                             reduce_weight=True)
    if not dep[0]:
        return set()
    return {checks[i]["syn"] for i in np.where(co[0])[0]}


# -----------------------------------------------------------------------------
# layout + builder
# -----------------------------------------------------------------------------

@dataclass
class MultiPatchLayout:
    """A rotated N-patch joint-measurement layout (``M(∏ᵢ P̄ᵢ)``).

    ``logicals`` is a list of ``(name, measured_pauli, support)`` — one per patch.
    ``x_observable`` is the X-type logical the X-memory circuit reads.
    """
    distance: int
    data: list
    checks: list
    logicals: list
    x_observable: list
    readout_chain: set = field(default_factory=set)
    specs: list = field(default_factory=list)
    target: list = field(default_factory=list)

    @property
    def N(self):
        return len(self.logicals)

    def build_circuit(self, rounds=None, p=0.0):
        """No-MPP gate-level syndrome-extraction circuit.  X-memory experiment when an X-logical is
        tracked (mixed / pure-X joints); Z-memory when the tracked logical is Z (a pure-Z joint)."""
        from .bent_joint_se import RotatedBentJointMeasurement
        if rounds is None:
            rounds = self.distance
        basis = "X" if any(P == "X" for _, P, _ in self.logicals) else "Z"
        return RotatedBentJointMeasurement(list(self.data), self.checks,
                                           list(self.x_observable)).circuit(rounds=rounds, p=p, basis=basis)

    def verify(self, rounds=None):
        """The eleven N-patch acceptance checks; returns a dict of booleans (all True == valid)."""
        isv, n = _int_symplectic(self.data)
        S = [isv(ch["pauli"]) for ch in self.checks]
        singles = [isv({c: P for c in sup}) for _, P, sup in self.logicals]
        joint = 0
        for v in singles:
            joint ^= v
        N = len(self.logicals)
        subs = []                                       # all proper non-empty subset products
        for r in range(1, N):
            for comb in itertools.combinations(range(N), r):
                v = 0
                for k in comb:
                    v ^= singles[k]
                subs.append(v)
        B = _IntBasis()
        for v in S:
            B.add(v)
        commute = all(_icommute(S[i], S[j], n) for i in range(len(S)) for j in range(i + 1, len(S)))
        twist = any(P == "Y" for ch in self.checks for P in ch["pauli"].values())

        def w1():
            for q in self.data:
                for P in "XZ":
                    v = isv({q: P})
                    if B.reduce(v) != 0 and all(_icommute(v, b, n) for b in B.piv.values()):
                        return True
            return False

        out = dict(
            commute=commute,
            joint=B.contains(joint),
            no_single=not any(B.contains(v) for v in singles),
            no_subjoint=not any(B.contains(v) for v in subs),
            no_twist=not twist,
            logical_count=len(self.data) - B.rank == N - 1,
            no_weight1_logical=not w1(),
        )
        circuit = self.build_circuit(rounds=rounds)
        out["no_mpp"] = "MPP" not in str(circuit)
        out["no_tick_collision"] = _no_tick_collision(circuit)
        try:
            dem = circuit.detector_error_model(decompose_errors=True)
            dem_ok = (dem.num_detectors == circuit.num_detectors
                      and dem.num_observables == circuit.num_observables)
        except Exception:
            dem_ok = False
        det, obs = circuit.compile_detector_sampler(seed=0).sample(200, separate_observables=True)
        out["dem_valid"] = dem_ok and not det.any() and not obs.any()
        return out


def _collapse_check(data, plaqs, log_pairs, target):
    """Diagnose a failed selection: which proper sub-products are homologically trivial.

    A logical product is in the FULL plaquette GF(2) span iff it is a genuine stabilizer (it commutes
    with every plaquette), i.e. it is *measured*.  A proper sub-product here is therefore a forbidden
    same-type collapse.  Returns ``(collapsed_labels, joint_in_span)``.
    """
    isv, n = _int_symplectic(data)
    B = _IntBasis()
    for p in plaqs:
        B.add(isv(p["pauli"]))
    vecs = [isv({c: P for c in sup}) for P, sup in log_pairs]
    names = [nm for nm, _ in target]
    paulis = [P for _, P in target]
    N = len(vecs)
    joint = 0
    for v in vecs:
        joint ^= v
    collapsed = []
    for r in range(1, N):
        for comb in itertools.combinations(range(N), r):
            v = 0
            for k in comb:
                v ^= vecs[k]
            if B.contains(v):
                collapsed.append("".join(f"{paulis[k]}{names[k][1:]}" for k in comb))
    return collapsed, B.contains(joint)


def _parse_explicit(routing, target):
    """Parse ``routing={"type":"explicit", "x_edges":[...], "z_attachments":[...]}``.

    ``x_edges`` are X–X fusion pairs ``(a, b)``; the sentinel ``(p, "trunk")`` (or ``("trunk", p)``)
    declares X-patch ``p`` a **side-arm** (it must then not appear in any fusion pair).  Optional
    ``z_attachments`` are ``(zname, side)`` with ``side`` in ``{"left","right","left_wall",
    "right_wall"}`` — asserted against the geometry (a mismatch raises, no silent override).
    Returns ``(edges, z_sides)`` where ``edges`` is the fusion-pair list and ``z_sides`` maps each
    declared Z to ``"left"``/``"right"``.  Honours the caller's routing exactly — no auto-search.
    """
    tgt = dict(target)
    edges, side_arms = [], set()
    for e in routing.get("x_edges", []):
        a, b = e
        if "trunk" in (a, b):
            p = a if b == "trunk" else b
            if tgt.get(p) != "X":
                raise BentLayoutError(f"explicit routing: {p!r} marked as a trunk side-arm but is "
                                      "not an X-side patch")
            side_arms.add(p)
        elif tgt.get(a) == "X" and tgt.get(b) == "X":
            edges.append((a, b))
        else:
            raise BentLayoutError(f"explicit routing: x_edge {e!r} must join two X-side patches "
                                  "(or use the (patch, 'trunk') side-arm form)")
    clash = side_arms & {x for e in edges for x in e}
    if clash:
        raise BentLayoutError(f"explicit routing: {sorted(clash)} declared as trunk side-arm(s) but "
                              "also given an explicit fusion edge")
    z_sides = {}
    for z, side in routing.get("z_attachments", []):
        if tgt.get(z) != "Z":
            raise BentLayoutError(f"explicit routing: z_attachment {z!r} is not a Z-side patch")
        s = side.replace("_wall", "")
        if s not in ("left", "right"):
            raise BentLayoutError(f"explicit routing: z_attachment side {side!r} must be "
                                  "'left'/'right' (or 'left_wall'/'right_wall')")
        z_sides[z] = s
    return edges, z_sides


def build_rotated_multi_patch_joint_layout(patches, target=None, routing="auto",
                                           seed=0, max_trials=4000):
    """Build a rotated N-patch joint-measurement layout ``M(∏ᵢ P̄ᵢ)`` from explicit ``PatchSpec``s.

    ``patches`` is a list of :class:`.PatchSpec`.  ``target`` is a list of ``(name, pauli)`` saying
    which logical each patch contributes; if omitted it is read from each patch's
    ``measured_logical``.  ``routing`` selects how the patches are connected:

    * ``"auto"`` — the bounded auto-router: it computes coordinate-derived fusion corridors between
      the X-side patches (a spanning tree over pairs that share a row or column), grows a vertical
      bus to reach the Z-patches, attaches each Z-patch by its own band, and accepts the first
      candidate the oracle verifies;
    * an **explicit routing graph** (for an upstream lattice-surgery compiler) —
      ``{"type": "explicit", "x_edges": [("p4","p3"), ("p5","trunk"), ...],
      "z_attachments": [("p1","left_wall"), ("p2","right_wall")]}``: the routing is taken **exactly
      as given** (no auto-search) — ``x_edges`` are X–X fusion pairs, ``(p,"trunk")`` marks a
      side-arm, and ``z_attachments`` assert each Z's wall side.  It still passes the full GF(2)
      oracle, and an illegal explicit routing reports the concrete reason (overlap / orientation /
      sub-product / weight-1);
    * a bare ``{"type": "tree", "edges": [...]}`` or a list of edges ``[("p1","p3"), ...]`` — the
      X–X edges you give are used as the fusion tree (a thin form of the explicit graph).

    **Mixed, pure-X and pure-Z joints** are all supported.  A joint with any X-patch is built on an
    X-side bus with the Z-patches attached through their own mixed (XZ) walls (from *any* side —
    left/right/above/below); a **pure-Z** joint uses a Z-side bus (and a Z-memory circuit).  So
    ``M(X̄₁X̄₃Z̄₂)``, ``M(Z̄₁Z̄₂X̄₃)``, a pure-X ``M(X̄₁X̄₂)`` and a pure-Z ``M(Z̄₁Z̄₂)`` are all this one
    API.  (Three or more *parallel* same-type patches over-measure their pairwise products — a clean
    ``k≥3`` same-type parity needs an opposite-type ancilla bus, which is future work.)

    The geometry is **derived from the patch origins** (any even offset works; corridors are not a
    fixed staircase).  Returns a verified :class:`MultiPatchLayout`.

    Raises :class:`.BentLayoutError` (concrete reason) when the placement cannot be routed (parity /
    overlap / unfusable / disconnected X-side / a Z-patch the bus cannot reach or that overlaps it)
    or when no candidate routing measures the full joint (e.g. four or more X-side patches on a
    single Z-band close a sub-product).  It never falls back to a fixed topology or silently
    mis-measures a sub-product.
    """
    if len(patches) < 2:
        raise ValueError(f"need at least 2 patches, got {len(patches)}")
    names = [s.name for s in patches]
    if len(set(names)) != len(names):
        raise ValueError(f"patch names must be unique, got {names}")
    d = patches[0].distance
    if any(s.distance != d for s in patches):
        raise ValueError("all patches must share the same distance")
    if target is None:
        target = [(s.name, s.measured_logical) for s in patches]
    tnames = [nm for nm, _ in target]
    if set(tnames) != set(names) or len(tnames) != len(names):
        raise ValueError(f"target names {tnames} must be a permutation of patch names {names}")
    if any(P not in ("X", "Z") for _, P in target):
        raise ValueError(f"target paulis must be 'X' or 'Z', got {[P for _, P in target]}")

    placed = {s.name: set(place_patch(s)["data"]) for s in patches}
    orient = {s.name: s.orientation for s in patches}
    _validate_placement(placed, target)            # parity/overlap/single-Z, before any routing

    z_sides = None
    if routing == "auto":
        candidates = _candidate_edge_sets(placed, target)
        if not candidates:
            raise BentLayoutError(
                "auto-routing failed: the bus patches are not all fusable (each connected pair "
                "must share a row or column).  Reposition them, or pass an explicit "
                "routing={'type': 'explicit', 'x_edges': [...]}.")
    elif isinstance(routing, dict) and routing.get("type") == "explicit":
        edges, z_sides = _parse_explicit(routing, target)    # honoured exactly; still oracle-gated
        candidates = [edges]
    elif isinstance(routing, dict) and routing.get("type") == "tree":
        candidates = [list(routing.get("edges", []))]
    elif isinstance(routing, (list, tuple)):
        candidates = [list(routing)]
    else:
        raise NotImplementedError(
            "routing must be 'auto', a list of edges, {'type':'tree','edges':[...]}, or "
            f"{{'type':'explicit','x_edges':[...],'z_attachments':[...]}}, got {routing!r}")

    last = None
    for edges in candidates:
        try:
            data, retype, xnames, znames = _route(placed, target, edges, z_sides=z_sides)
        except BentLayoutError as e:
            last = e
            continue
        from .deterministic_checks import rule_based_joint_checks
        sv, n = _symplectic(data)
        orient_fail = None
        any_reps = False
        rb = rule_based_joint_checks(placed, target, orient, sorted(data), set(retype), d)
        if rb["checks"] is None:                        # diagnose WHY (the rules are the gate)
            for phase in (0, 1):
                plaqs = _bent_plaquettes(data, retype, phase)
                F = [sv(p["pauli"]) for p in plaqs if len(p["pauli"]) >= 4]
                reps = {}
                for nm, P in target:
                    direction = _logical_direction(P, orient[nm])
                    sup = _patch_rep(placed[nm], P, direction, F, sv, n)
                    if sup is None:
                        orient_fail = (nm, P, direction, orient[nm])
                        break
                    reps[nm] = (P, sup)
                if len(reps) != len(target):
                    continue
                any_reps = True
                log_pairs = [reps[nm] for nm, _ in target]
                collapsed, joint_ok = _collapse_check(data, plaqs, log_pairs, target)
                if collapsed:
                    last = BentLayoutError(
                        f"forbidden placement: the sub-product {', '.join(collapsed)} is in the "
                        f"stabilizer span (homologically trivial).  Two same-type measured patches "
                        f"fuse with no mixed wall / junction separating them, so a local readout "
                        f"chain closes their pair-product instead of the full joint.  Separate "
                        f"same-type patches with an opposite-type wall/junction between them (see the "
                        f"placement rule); this placement is rejected, not silently mis-measured.")
                elif not joint_ok:
                    last = BentLayoutError(
                        "placement cannot host the full joint: it is not in the plaquette span (the "
                        "patches are not all tied into one routed region).")
                else:
                    last = BentLayoutError(
                        f"the deterministic construction rules cannot host this placement: "
                        f"{rb['reason']}")
                break
        if rb["checks"] is not None:
            checks = rb["checks"]
            data = rb["data"]
            logicals = rb["logicals"]
            x_obs = rb["x_observable"]
            log_pairs = [(P, sup) for _, P, sup in logicals]
            readout = _readout_chain(data, checks, log_pairs)
            return MultiPatchLayout(distance=d, data=data, checks=checks, logicals=logicals,
                                    x_observable=x_obs, readout_chain=readout,
                                    specs=list(patches), target=list(target))
        if not any_reps and orient_fail is not None:
            nm, P, direction, o = orient_fail
            last = BentLayoutError(
                f"patch {nm}: its measured {P}-logical must run {direction} (orientation={o!r}), but "
                f"no such string commutes with the routed bus here — the routing cannot host that "
                f"orientation.  Use the perpendicular orientation or reposition {nm}; the logical is "
                f"never silently re-oriented.")
    raise last or BentLayoutError("could not route these patches into a joint code.")


# -----------------------------------------------------------------------------
# diagnostic mode: build the candidate geometry and report WHY it (in)validates,
# WITHOUT raising — so a failing layout can still be drawn and explained.
# -----------------------------------------------------------------------------

@dataclass
class MultiPatchDiagnosis:
    """A non-raising report on a candidate multi-patch routing (drawable even when it fails).

    ``ok`` is ``True`` iff the full joint is in the stabilizer span and **no** proper sub-product is.
    ``failing_subproducts`` lists the proper sub-products that are homologically trivial (in the
    plaquette span) — the reason a bad routing mis-measures.  ``joint_chain`` / each sub-product's
    chain is the set of plaquette syndromes whose GF(2) product equals that operator (the readout
    loop), for highlighting.  ``plaqs`` are the candidate bent plaquettes (for drawing the tiles).
    """
    distance: int
    data: list
    retype: set
    phase: int
    plaqs: list
    logicals: list                       # (name, P, support)
    ok: bool
    reason: str
    joint_in_span: bool
    joint_chain: set
    failing_subproducts: list            # [(label, names_tuple, chain_set)]

    @property
    def syn_to_plaq(self):
        return {p["syn"]: p for p in self.plaqs}


def _label(P, name):
    return f"{P}{name[1:]}"               # ("X","p3") -> "X3"


def diagnose_multi_patch_joint(patches, target=None, routing="auto"):
    """Diagnose a candidate routing **without raising**: return a :class:`MultiPatchDiagnosis`.

    Unlike :func:`build_rotated_multi_patch_joint_layout` (which raises when the layout does not
    measure exactly the joint), this always returns the candidate geometry it could build, together
    with which proper sub-products are in the stabilizer span and the plaquette chain that closes
    each.  Use it to visualise *why* a bad placement (e.g. two X-patches fused adjacently with no
    mixed wall between them) measures a sub-product such as ``X̄₃X̄₄``.
    """
    from ....utils.linear_algebra import solve_linear_decomposition
    if target is None:
        target = [(s.name, s.measured_logical) for s in patches]
    placed = {s.name: set(place_patch(s)["data"]) for s in patches}
    orient = {s.name: s.orientation for s in patches}
    d = patches[0].distance

    # parse routing like build(), but capture errors as a (non-raising) diagnosis
    try:
        _validate_placement(placed, target)
        if routing == "auto":
            cands = _candidate_edge_sets(placed, target)
            if not cands:
                raise BentLayoutError("auto-routing failed: the X-side patches are not all fusable.")
        elif isinstance(routing, dict) and routing.get("type") == "tree":
            cands = [list(routing.get("edges", []))]
        elif isinstance(routing, (list, tuple)):
            cands = [list(routing)]
        else:
            raise BentLayoutError(f"unsupported routing {routing!r}")
        data = retype = None
        last = None
        for edges in cands:                          # first candidate that routes is the one we diagnose
            try:
                data, retype, xnames, znames = _route(placed, target, edges)
                break
            except BentLayoutError as e:
                last = e; data = None
        if data is None:
            raise last or BentLayoutError("no candidate routing could be built")
    except BentLayoutError as e:
        return MultiPatchDiagnosis(distance=d, data=[], retype=set(), phase=0, plaqs=[], logicals=[],
                                   ok=False, reason=f"routing could not be built: {e}",
                                   joint_in_span=False, joint_chain=set(), failing_subproducts=[])

    sv, n = _symplectic(data)
    # pick the phase that admits all logical reps; fall back to phase 0 for a drawable (if repless) geometry
    phase, plaqs, reps = 0, _bent_plaquettes(data, retype, 0), None
    for ph in (0, 1):
        pl = _bent_plaquettes(data, retype, ph)
        F = [sv(p["pauli"]) for p in pl if len(p["pauli"]) >= 4]
        r = {}
        ok = True
        for nm, P in target:
            sup = _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, n)
            if sup is None:
                ok = False; break
            r[nm] = (P, sup)
        if ok:
            phase, plaqs, reps = ph, pl, r; break
    if reps is None:
        return MultiPatchDiagnosis(distance=d, data=sorted(data), retype=retype, phase=phase,
                                   plaqs=plaqs, logicals=[], ok=False,
                                   reason="no parity phase admits all logical reps in their declared "
                                          "orientation", joint_in_span=False, joint_chain=set(),
                                   failing_subproducts=[])

    logicals = [(nm, reps[nm][0], reps[nm][1]) for nm, _ in target]
    L = {nm: sv({c: P for c in sup}) for nm, (P, sup) in reps.items()}
    allvecs = np.array([sv(p["pauli"]) for p in plaqs], np.uint8)

    def chain_for(vec):
        co, dep, _ = solve_linear_decomposition(basis=allvecs, targets=vec.reshape(1, -1),
                                                reduce_weight=True)
        if not dep[0]:
            return False, set()
        return True, {plaqs[i]["syn"] for i in np.where(co[0])[0]}

    names = [nm for nm, _ in target]
    joint = np.zeros(2 * n, np.uint8)
    for v in L.values():
        joint ^= v
    joint_in_span, joint_chain = chain_for(joint)

    failing = []
    for rr in range(1, len(names)):
        for comb in itertools.combinations(names, rr):
            v = np.zeros(2 * n, np.uint8)
            for nm in comb:
                v ^= L[nm]
            ins, chain = chain_for(v)
            if ins:
                lbl = "".join(_label(reps[nm][0], nm) for nm in comb)
                failing.append((lbl, comb, chain))

    ok = joint_in_span and not failing
    if ok:
        reason = "clean: the full joint is in the span and no proper sub-product is"
    elif failing:
        labs = ", ".join(f"{l}" for l, _, _ in failing)
        reason = (f"sub-product(s) [{labs}] are in the stabilizer span (homologically trivial); "
                  f"the full joint is {'in' if joint_in_span else 'NOT in'} span")
    else:
        reason = "the full joint is not in the plaquette span (the placement cannot host it)"
    return MultiPatchDiagnosis(distance=d, data=sorted(data), retype=retype, phase=phase, plaqs=plaqs,
                               logicals=logicals, ok=ok, reason=reason, joint_in_span=joint_in_span,
                               joint_chain=joint_chain, failing_subproducts=failing)
