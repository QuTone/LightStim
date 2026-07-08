"""Subset joint lattice-surgery measurement with obstacle-aware, ``d``-wide corridor routing.

Given many placed patches, measure the joint ``M(∏ᵢ P̄ᵢ)`` of only a chosen **subset**; the
non-target patches are **obstacles** the routed ancilla bus must not overlap.  This module holds the
**core routing logic** — the coordinate/coarse-cell model, the ``networkx`` corridor graph, the
``d``-wide channel + obstacle handling, the shortest-path-tree routing, and the path→corridor
conversion — so the demo notebook only imports and calls.

The router is **propose-and-verify**: it builds an obstacle-aware ``d``-wide corridor graph,
enumerates candidate shortest-path trees connecting the targets, and the **existing GF(2) oracle**
(the :mod:`.multi_patch` physics layer, reused unchanged) gates every candidate — returning the first
**verified** :class:`.MultiPatchLayout`, or an honest ``"no_verified_route"`` (it never silently
mis-measures, and it never routes a corridor through an obstacle).

**Scope.**  The current physics layer verifies a **straight X-bus (trunk) with Z-targets attached
perpendicular through mixed (XZ) walls (any side)** plus simple end-bends.  It **rejects** most
*arbitrary bent* trunks (they fail ``joint ∈ span``).  So the router routes within that family; a
general bent-bus / rectilinear-Steiner router is future work.

Geometry model: a coarse grid of ``d×d`` cells at pitch ``2d``.  Each patch occupies one cell;
corridors run through **empty** cells.  The joint code's ``data`` = target cells + corridor-tree
cells — the obstacles are **not** in ``data`` (they are separate idle patches, only forbidden
coordinates for the bus), which is exactly why the code measures the *subset* joint and not the
all-patch joint.
"""

import itertools
from dataclasses import dataclass, field

import networkx as nx

from .bent_layout import _bent_plaquettes, _symplectic, PatchSpec
from .multi_patch import (
    _patch_rep, _logical_direction, _readout_chain,
    MultiPatchLayout, _connected, _collapse_check, _int_symplectic, _IntBasis, _icommute,
)

__all__ = ["PatchSpec", "cell", "origin_of", "cell_index", "route_and_build", "complete_code",
           "acceptance", "acceptance_of_layout", "ACCEPTANCE_ITEMS", "collision_report",
           "SubsetRoute"]

NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _xor_ints(vs):
    x = 0
    for v in vs:
        x ^= v
    return x
_LR = [(1, 0), (-1, 0)]          # X-faces of an X_horizontal patch
_TB = [(0, 1), (0, -1)]          # X-faces of an X_vertical patch


def cell(a, b, d):
    """The ``d×d`` data-qubit footprint of coarse cell ``(a, b)`` at pitch ``2d`` (rotated frame)."""
    x0, y0 = 1 + 2 * d * a, 1 + 2 * d * b
    return {(x, y) for x in range(x0, x0 + 2 * d, 2) for y in range(y0, y0 + 2 * d, 2)}


def origin_of(a, b, d):
    """The data-qubit **origin** (bus-facing ``(1,1)``-corner) of coarse cell ``(a, b)`` at pitch
    ``2d`` — i.e. the :attr:`PatchSpec.origin` that places a distance-``d`` patch on that cell.
    Inverse of :func:`cell_index`.  Use it to write subset-routing patches in the same
    ``PatchSpec(name, origin, distance, measured_logical, orientation)`` form as the N-patch API."""
    return (1 + 2 * d * a, 1 + 2 * d * b)


def cell_index(origin, d):
    """The coarse cell ``(a, b)`` a distance-``d`` patch placed at ``origin`` occupies.  Inverse of
    :func:`origin_of`.  Raises ``ValueError`` if ``origin`` is off the pitch-``2d`` grid (the coarse
    router places patches only on that grid; use :func:`origin_of` to land on it)."""
    ox, oy = origin
    if (ox - 1) % (2 * d) or (oy - 1) % (2 * d):
        raise ValueError(f"origin {origin} is not on the pitch-{2 * d} coarse grid for d={d}")
    return ((ox - 1) // (2 * d), (oy - 1) // (2 * d))


def _specs_to_cells(patches, target=None):
    """Map the user-facing :class:`PatchSpec` list onto the internal coarse-cell model.

    Returns ``(patch_at, orient, d)`` where ``patch_at[name] = (a, b)`` is the coarse cell whose
    ``d×d`` footprint equals the patch's placed data qubits, ``orient[name]`` is the declared
    orientation, and ``d`` is the (shared) code distance.  This is the single adapter every public
    entry point runs first, so the whole subset API speaks ``PatchSpec`` while the routing internals
    keep working in coarse cells.

    Raises ``TypeError`` / ``ValueError`` with a concrete reason when the specs can't be placed on the
    coarse grid — a non-``PatchSpec`` element (e.g. the old ``{name:(a,b)}`` cell dict), mixed
    distances, duplicate names, an origin off the pitch-``2d`` grid, or a bad orientation — or when
    ``target`` names an unknown patch, repeats one, or uses a non-``X``/``Z`` Pauli.
    """
    patches = list(patches)
    if not patches:
        raise ValueError("need at least one PatchSpec")
    if not all(isinstance(s, PatchSpec) for s in patches):
        raise TypeError("patches must be a list of PatchSpec(name, origin, distance, "
                        "measured_logical, orientation); the old {name: (a, b)} cell-dict form is no "
                        "longer accepted — build a PatchSpec per patch (origin_of(a, b, d) places "
                        "one on coarse cell (a, b)).")
    d = patches[0].distance
    if any(s.distance != d for s in patches):
        raise ValueError(f"all patches must share one distance, got {sorted({s.distance for s in patches})}")
    names = [s.name for s in patches]
    if len(set(names)) != len(names):
        raise ValueError(f"patch names must be unique, got {names}")
    patch_at, orient = {}, {}
    for s in patches:
        if s.orientation not in ("X_horizontal", "X_vertical"):
            raise ValueError(f"patch {s.name!r}: orientation must be 'X_horizontal'|'X_vertical', "
                             f"got {s.orientation!r}")
        try:
            patch_at[s.name] = cell_index(s.origin, d)
        except ValueError:
            raise ValueError(
                f"patch {s.name!r}: origin {s.origin} is not on the coarse routing grid (pitch "
                f"{2 * d} for d={d}).  Subset routing places patches on cells whose bus-facing corner "
                f"is (1+{2 * d}·a, 1+{2 * d}·b); use origin_of(a, b, d) to place them.")
        orient[s.name] = s.orientation
    if target is not None:
        tnames = [nm for nm, _ in target]
        unknown = [nm for nm in tnames if nm not in patch_at]
        if unknown:
            raise ValueError(f"target names {unknown} are not among the patches {names}")
        if len(set(tnames)) != len(tnames):
            raise ValueError(f"target lists a patch more than once: {tnames}")
        if any(P not in ("X", "Z") for _, P in target):
            raise ValueError(f"target paulis must be 'X' or 'Z', got {[P for _, P in target]}")
    return patch_at, orient, d


def _x_faces(orientation):
    """The neighbour offsets on the patch's X̄-parallel (rough) faces — where the X-bus may leave."""
    return _LR if orientation == "X_horizontal" else _TB


# -----------------------------------------------------------------------------
# corridor graph  (d-wide channel search + obstacle handling)
# -----------------------------------------------------------------------------

def _cheb(ab, cd):
    """King-move (Chebyshev) distance between two coarse cells."""
    return max(abs(ab[0] - cd[0]), abs(ab[1] - cd[1]))


def _corridor_graph(patch_at, target, d, pad, keepout=1):
    """Build the coarse-cell corridor graph.

    A coarse cell is **corridor-eligible** iff it holds no patch and it stays clear of every obstacle
    patch by a **keep-out margin** of ``keepout`` cells (king-move / Chebyshev distance ``> keepout``).
    ``keepout=1`` (the default) forbids any cell **adjacent** to an obstacle, so the routed code never
    shares a boundary ancilla line with an idle obstacle patch (``keepout=0`` did — two edge-adjacent
    cells share the ancilla row/column between them, a physical collision).

    Returns ``(G, corridor, placed, occupied, obstacle_fp, onames)``.
    """
    tnames = [nm for nm, _ in target]
    onames = [nm for nm in patch_at if nm not in tnames]
    occupied = {ab: nm for nm, ab in patch_at.items()}
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    obstacle_fp = set().union(*[placed[nm] for nm in onames]) if onames else set()
    obstacle_cells = [patch_at[nm] for nm in onames]

    blocked = set(occupied)
    forbid_fp = obstacle_fp
    A = [a for a, b in patch_at.values()]
    B = [b for a, b in patch_at.values()]
    corridor = set()
    for a in range(min(A) - pad, max(A) + pad + 1):
        for b in range(min(B) - pad, max(B) + pad + 1):
            if (a, b) in blocked:
                continue                                  # a patch sits here (targets, or all patches)
            if cell(a, b, d) & forbid_fp:
                continue                                  # d-wide footprint would hit an obstacle
            if any(_cheb((a, b), oc) <= keepout for oc in obstacle_cells):
                continue                                  # keep-out margin: no shared boundary ancilla
            corridor.add((a, b))
    G = nx.Graph()
    G.add_nodes_from(corridor)
    for (a, b) in corridor:
        for (da, db) in NEIGH:
            if (a + da, b + db) in corridor:
                G.add_edge((a, b), (a + da, b + db))
    return G, corridor, placed, occupied, obstacle_fp, onames


def _faces_in(patch_at, corridor, nm, dirs):
    """The corridor cells adjacent to patch ``nm`` on the given face directions."""
    a, b = patch_at[nm]
    return [(a + da, b + db) for (da, db) in dirs if (a + da, b + db) in corridor]


def _candidate_arms(G, patch_at, corridor, root, root_faces, z, per_z):
    """Up to ``per_z`` shortest obstacle-aware paths from an X-face of ``root`` to a face of ``z``."""
    paths = []
    for s in _faces_in(patch_at, corridor, root, root_faces):
        for e in _faces_in(patch_at, corridor, z, NEIGH):
            if s in G and e in G and nx.has_path(G, s, e):
                paths.append(nx.shortest_path(G, s, e))
    paths.sort(key=len)
    seen, keep = set(), []
    for p in paths:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            keep.append(p)
        if len(keep) >= per_z:
            break
    return keep


def _cells_connected(cells):
    """True iff the coarse cells form ONE 4-neighbour connected component (empty = connected).

    A union of arms that meets only *through* a target patch is **not** connected: the ancilla
    bus must be a single contiguous region on its own, so such a corridor is an illegal route
    (it would be two separate buses), not a candidate for the physics layer.
    """
    cells = set(map(tuple, cells))
    if not cells:
        return True
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        a, b = stack.pop()
        for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cells)


def _steiner_trees(G, patch_at, corridor, root, zs):
    """Approximately-minimal **connected** corridors touching a face of the root and of every
    ``z`` — the "fewest corridor cells" candidates the per-arm product enumeration misses.

    Greedy Steiner approximation: seed with the shortest root-face → first-target path, then
    attach each next target via its shortest path from the *current tree* (multi-source
    Dijkstra), so arms share cells instead of duplicating them.  Several target orders are
    tried (near-first, far-first, given); connected-by-construction, smallest first.
    """
    if not zs:
        return []
    root_faces = [c for c in _faces_in(patch_at, corridor, root, NEIGH) if c in G]
    zfaces = {z: [c for c in _faces_in(patch_at, corridor, z, NEIGH) if c in G] for z in zs}
    if not root_faces or any(not v for v in zfaces.values()):
        return []
    rc = patch_at[root]
    orders = [sorted(zs, key=lambda z: _cheb(patch_at[z], rc)),
              sorted(zs, key=lambda z: -_cheb(patch_at[z], rc)),
              list(zs)]
    trees, seen = [], set()
    for order in orders:
        tree = set()
        ok = True
        for z in order:
            sources = tree if tree else set(root_faces)
            try:
                lengths, paths = nx.multi_source_dijkstra(G, sources)
            except ValueError:
                ok = False
                break
            ends = [f for f in zfaces[z] if f in lengths]
            if not ends:
                ok = False
                break
            end = min(ends, key=lambda f: lengths[f])
            tree |= set(paths[end])
        if ok and tree:
            key = frozenset(tree)
            if key not in seen:
                seen.add(key)
                trees.append(tree)
    trees.sort(key=len)
    return trees


def path_to_corridor(tree_cells, placed, target, d):
    """Convert a set of corridor **cells** into the joint code's ``(data, retype)``.

    ``data`` = the target patch cells ∪ the corridor cells' footprints.  ``retype`` (the X-side bus)
    = everything in ``data`` except the Z-target cells — so the mixed (XZ) wall falls at each
    Z-target's boundary, exactly as in the straight-corridor auto-router.
    """
    tnames = [nm for nm, _ in target]
    data = set()
    for nm in tnames:
        data |= placed[nm]
    for c in tree_cells:
        data |= cell(*c, d)
    z_cells = set().union(*[placed[nm] for nm, P in target if P == "Z"]) \
        if any(P == "Z" for _, P in target) else set()
    retype = {q for q in data if q not in z_cells}
    return data, retype


# -----------------------------------------------------------------------------
# physics-layer reuse: a routed (data, retype) region -> a verified MultiPatchLayout
# -----------------------------------------------------------------------------

def _assemble_region(placed, target, orient, data, retype, d, seed=0, max_trials=5000, max_cut=4,
                     forbidden=frozenset()):
    """Hand a routed region to the **deterministic rule-based** physics layer.

    The stabilizers are constructed by :func:`.deterministic_checks.rule_based_joint_checks`
    (the documented Handbook §10.4 / Fig 33-34 rules: forced bulk, alternating-spacing
    boundary, concave/convex corner rules with rule-driven corner cuts) — no randomized
    search.  ``seed`` / ``max_trials`` are accepted for backward compatibility and ignored.
    Returns a :class:`.MultiPatchLayout` (whose ``data`` may be smaller than the input when
    the convex-corner rule cut bus-corner qubits), or ``None`` if the rules cannot host this
    geometry.  The oracle decision itself is ``MultiPatchLayout.verify()``.
    """
    from .deterministic_checks import rule_based_joint_checks
    data = sorted(data)
    if not _connected(set(data)):
        return None
    rb = rule_based_joint_checks(placed, target, orient, data, set(retype), d, max_cut=max_cut,
                                 forbidden=forbidden)
    if rb["checks"] is None:
        return None
    log_pairs = [(P, sup) for _, P, sup in rb["logicals"]]
    return MultiPatchLayout(distance=d, data=rb["data"], checks=rb["checks"],
                            logicals=rb["logicals"], x_observable=rb["x_observable"],
                            readout_chain=_readout_chain(rb["data"], rb["checks"], log_pairs),
                            target=list(target))


# -----------------------------------------------------------------------------
# the router
# -----------------------------------------------------------------------------

@dataclass
class SubsetRoute:
    """Result of :func:`route_and_build`.  ``status == "ok"`` iff ``layout`` is a verified joint.

    On a **failure** an obstacle-free corridor may still exist but be un-hostable by the physics
    layer — ``attempted`` / ``attempted_arms`` carry the shortest such corridor (empty only when
    ``status == "no_path"``) so a caller can *draw the route that was found* and label it correctly
    (a rejected corridor is **not** "no route").
    """
    status: str    # "ok" | "no_path" (unreachable OR only disconnected corridors) | "no_verified_route"
    message: str = ""
    layout: object = None                             # MultiPatchLayout when status == "ok"
    root: str = None
    arms: dict = field(default_factory=dict)          # z-name -> corridor-cell path (VERIFIED bus)
    tree: set = field(default_factory=set)            # corridor cells of the VERIFIED bus
    attempted: set = field(default_factory=set)       # corridor cells of the shortest candidate (any status)
    attempted_arms: dict = field(default_factory=dict)  # z-name -> shortest candidate path
    data: set = field(default_factory=set)            # data qubits of the routed region
    placed: dict = field(default_factory=dict)        # name -> patch cells (ALL patches)
    target: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)     # non-target patch names
    obstacle_fp: set = field(default_factory=set)     # non-target patch data qubits
    corridor: set = field(default_factory=set)        # all corridor-eligible cells
    tried: int = 0
    how: str = "standard"                             # "standard" | "corner-cut" (route_and_build)
    cut: tuple = ()                                   # convex-corner qubits removed (corner-cut only)

    @property
    def ok(self):
        return self.status == "ok"


def _obstacle_ancillas(patches, tnames):
    """The ancilla sites actually USED by the non-target (idle obstacle) patches.

    Each idle patch keeps its standalone construction; its selected syndrome sites are
    physical qubits the routed joint code may not re-use.  These are handed to the rule
    constructor as ``forbidden`` boundary positions, so a corridor sharing an ancilla
    line with an idle neighbour interleaves with it instead of colliding — edge-adjacent
    placement (keepout=0) is then physically sound whenever the parity allows.
    """
    from .bent_layout import place_patch
    out = set()
    for s in patches:
        if s.name in tnames:
            continue
        for c in place_patch(s)["checks"]:
            out.add(tuple(int(v) for v in c["syn"]))
    return frozenset(out)


def route_and_build(patches, target, pad=1, per_z=6, max_std=48, cut_budget=4, max_cut=4,
                    keepout=0, seed=0, max_trials=5000, route=None):
    """Fully-automatic route **and** build: no hand-written corridor needed.

    ``route`` (optional): an **explicit corridor** — a list of coarse cells ``[(a, b), …]``.
    When given, NO automatic routing happens: the joint code is built on exactly these
    cells by the deterministic rule constructor and gated by the full oracle.  The cells
    must not overlap any patch; the obstacle **keep-out margin is NOT enforced** for an
    explicit route (you are overriding the router), so check ``collision_report`` if
    obstacles sit next to your corridor.  Omit ``route`` (default) for auto-routing.

    Propose-and-verify with retry: (1) candidate arms leave the X-anchor from **any** face (not just
    its X-faces), so clean below-/side-attach corridors are found; (2) arm-product unions are
    augmented with greedy **Steiner** candidates (:func:`_steiner_trees`) whose arms share a common
    trunk; (3) a candidate whose corridor is **disconnected** (its arms meet only through a target
    patch — two separate buses) is illegal and never tried; if *every* candidate is disconnected the
    result is an honest ``no_path``; (4) each surviving candidate is built by the **deterministic
    rule constructor** (:mod:`.deterministic_checks` — cut-free first, then rule-driven convex-corner
    cuts up to ``max_cut``) and gated by the full oracle.  Candidates are tried
    **fewest-corridor-cells first**, so the smallest *valid* bus wins — a shared straight trunk is
    preferred over fat multi-arm unions.  ``per_z`` / ``max_std`` / ``keepout`` shape the candidate
    pool; ``cut_budget`` / ``seed`` / ``max_trials`` are accepted for backward compatibility and
    ignored (there is no randomized search any more).  This is the routine the demo notebook calls
    instead of pasting cells.

    Returns a :class:`SubsetRoute` with ``status == "ok"`` and ``.layout`` / ``.tree`` (route cells) /
    ``.how`` (``"standard"`` | ``"corner-cut"``) / ``.cut`` set, or a failure ``SubsetRoute`` with
    status ``target_obstacle_conflict`` / ``no_path`` / ``no_verified_route``.
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    tnames = [nm for nm, _ in target]
    onames = [nm for nm in patch_at if nm not in tnames]
    placed_all = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    obstacle_fp0 = set().union(*[placed_all[nm] for nm in onames]) if onames else set()
    base0 = dict(placed=placed_all, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp0,
                 corridor=set())
    for tn in tnames:
        bad = [on for on in onames if _cheb(patch_at[tn], patch_at[on]) <= keepout]
        if bad:
            return SubsetRoute(status="target_obstacle_conflict", root=tnames[0],
                               message=(f"target {tn} is within keepout={keepout} of obstacle(s) "
                                        f"{bad}: their boundary ancillas would collide."), **base0)
    root0 = next((nm for nm, P in target if P == "X"), tnames[0])
    forbidden = _obstacle_ancillas(patches, tnames)   # idle neighbours' USED ancilla sites

    if route is not None:                  # explicit corridor: build on EXACTLY these cells
        tree = {tuple(c) for c in route}
        occupied_cells = set(patch_at.values())
        bad = sorted(tree & occupied_cells)
        if bad:
            raise ValueError(f"explicit route cells {bad} overlap patch cells; the corridor "
                             f"may only use empty coarse cells")
        data, retype = path_to_corridor(tree, placed_all, target, d)
        layout = _assemble_region(placed_all, target, orient, data, retype, d, seed,
                                  max_trials, max_cut=max_cut, forbidden=forbidden)
        if layout is not None and all(layout.verify().values()):
            cutq = tuple(sorted(set(data) - set(layout.data)))
            how = "corner-cut" if cutq else "standard"
            return SubsetRoute(status="ok",
                               message=f"verified subset joint ({how}, rule-based, "
                                       f"EXPLICIT route)",
                               layout=layout, root=root0, tree=tree, attempted=tree,
                               data=sorted(layout.data), tried=1, how=how, cut=cutq,
                               **base0)
        return SubsetRoute(status="no_verified_route", root=root0, tried=1, attempted=tree,
                           message=("the EXPLICIT route does not pass the rule-based "
                                    "construction; the physics layer cannot host this "
                                    "corridor"), **base0)

    G, corridor, placed, occupied, obstacle_fp, onames = _corridor_graph(patch_at, target, d, pad,
                                                                         keepout=keepout)
    base = dict(placed=placed, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp,
                corridor=corridor)
    root = root0
    zs = [nm for nm in tnames if nm != root]

    cand = {}
    for z in zs:
        arms = _candidate_arms(G, patch_at, corridor, root, NEIGH, z, per_z)   # any face, not just X
        if not arms:
            return SubsetRoute(status="no_path", root=root,
                               message=f"path search found no obstacle-free corridor {root} -> {z}",
                               **base)
        cand[z] = arms
    combos = list(itertools.product(*[cand[z] for z in zs])) if zs else [()]
    raw = [set().union(*[set(p) for p in c]) if c else set() for c in combos]
    raw += _steiner_trees(G, patch_at, corridor, root, zs)   # shared-trunk / fewest-cell candidates
    uniq, trees, dropped = set(), [], 0
    for t in raw:
        key = frozenset(t)
        if key in uniq:
            continue
        uniq.add(key)
        if zs and not _cells_connected(t):     # a corridor split by a target patch is illegal
            dropped += 1
            continue
        trees.append(t)
    if zs and not trees:
        return SubsetRoute(status="no_path", root=root,
                           message=(f"every candidate corridor is disconnected (the {dropped} "
                                    f"arm-unions meet only through a target patch); no single "
                                    f"connected ancilla bus exists for this placement"), **base)
    trees.sort(key=len)                        # fewest corridor cells first, not sum of arm lengths
    trees = trees[:max_std]
    shortest = trees[0] if trees else set()

    # single pass -- deterministic rule-based construction per candidate, fewest cells
    # first; the convex-corner rule cuts bus-corner qubits by itself (no search pass).
    tried = 0
    for tree in trees:
        tried += 1
        data, retype = path_to_corridor(tree, placed, target, d)
        layout = _assemble_region(placed, target, orient, data, retype, d, seed, max_trials,
                                  max_cut=max_cut, forbidden=forbidden)
        if layout is not None and all(layout.verify().values()):
            cutq = tuple(sorted(set(data) - set(layout.data)))
            how = "corner-cut" if cutq else "standard"
            return SubsetRoute(status="ok",
                               message=f"verified subset joint ({how}, rule-based)",
                               layout=layout, root=root, tree=tree, attempted=tree,
                               data=sorted(layout.data), tried=tried, how=how, cut=cutq,
                               **base)
    return SubsetRoute(status="no_verified_route", root=root, tried=tried, attempted=shortest,
                       message=("no candidate corridor passes the rule-based construction; "
                                "the physics layer cannot host this geometry"), **base)


# -----------------------------------------------------------------------------
# failure taxonomy
# -----------------------------------------------------------------------------

def complete_code(patches, target, tree_cells, phase=None):
    """Build a **complete, maximal commuting** stabilizer code on the routed region: force the
    weight-4 plaquettes, then greedily add **every** commuting-and-independent boundary plaquette
    (weight-2 and the weight-3 corner plaquettes).  The result is a *valid* code — ``k = n - rank``
    logical qubits, all stabilizers commuting — not just the forced bulk.  Use it to judge measurability
    honestly: the joint is measurable **iff it lies in the plaquette span** (``joint_in_span``), which
    is a property of *all* plaquettes, independent of which complete code you pick.

    Returns a dict: ``stabilizers`` (the complete code, with ``corners``), ``k``, ``weights``
    (count by weight), ``commute``, ``no_weight1``, ``no_twist``, ``joint_in_span``,
    ``singles_in_span``, ``subproducts_in_span``, ``readout_chain`` (the stabilizers whose product is
    the joint, when measurable), ``data``, ``reps``, ``phase``.

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_and_build`); ``tree_cells`` is the
    explicit corridor to complete, a list of coarse cells ``[(a, b), ...]``.  ``phase`` (default
    ``None`` = auto) **forces** a specific parity phase (``0``/``1``) instead of auto-picking the
    joint-in-span one — use it to probe whether a given local checkerboard coloring changes
    measurability (it does not: joint-in-span is a property of the full plaquette span, checked here
    for whichever phase(s) are considered).
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    data = sorted(data)
    sv, n = _symplectic(data)
    isv, _ = _int_symplectic(data)
    N = len(target)
    # pick the parity that hosts both reps AND (preferably) puts the joint in span -- exactly what
    # _assemble_region does; picking merely the first rep-valid phase can miss a measurable joint that
    # only closes in the OTHER parity (e.g. a single L-bend).  A forced ``phase`` restricts the search.
    cands = []
    for ph in ((0, 1) if phase is None else (phase,)):
        pl = _bent_plaquettes(data, retype, ph)
        F = [sv(p["pauli"]) for p in pl if len(p["pauli"]) >= 4]
        rp = {nm: (P, _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, n))
              for nm, P in target}
        reps_ok = all(rp[nm][1] is not None for nm, _ in target)
        jspan = False
        if reps_ok:
            Bph = _IntBasis()
            for p in pl:
                Bph.add(isv(p["pauli"]))
            jspan = Bph.contains(_xor_ints([isv({q: P for q in rp[nm][1]}) for nm, P in target]))
        cands.append((jspan, reps_ok, ph, pl, rp))
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)   # prefer joint-in-span, then rep-valid
    _, _, phase, plaqs, reps = cands[0]

    # measurability is a property of the WHOLE plaquette span (any complete code is a subset of it)
    Bpool = _IntBasis()
    for p in plaqs:
        Bpool.add(isv(p["pauli"]))
    have_reps = all(reps[nm][1] is not None for nm, _ in target)
    singles, joint_in_span, subp = {}, None, None
    if have_reps:
        svecs = [isv({c: P for c in reps[nm][1]}) for nm, P in target]
        singles = {nm: Bpool.contains(isv({c: P for c in reps[nm][1]})) for nm, P in target}
        joint_in_span = Bpool.contains(_xor_ints(svecs))
        subp = sum(Bpool.contains(_xor_ints([svecs[i] for i in comb]))
                   for r in range(1, N) for comb in itertools.combinations(range(N), r))

    # the code to PRESENT: the joint-MEASURING code (with a readout chain) when measurable, else a
    # maximal complete commuting memory code (so a failing region still shows a full, valid patch).
    # The measuring code comes from the deterministic rule constructor (cut-free, on the region
    # exactly as given) — no randomized search anywhere.
    present, chain = None, set()
    if joint_in_span:
        from .deterministic_checks import rule_based_joint_checks
        forbidden = _obstacle_ancillas(patches, {nm for nm, _ in target})
        rb = rule_based_joint_checks(placed, target, orient, data, set(retype), d, max_cut=0,
                                     forbidden=forbidden)
        if rb["checks"] is not None:
            present = rb["checks"]
            chain = _readout_chain(data, present, [reps[nm] for nm, _ in target])
    if present is None:                                    # greedy maximal commuting complete code
        # CRUCIAL: only keep stabilizers that COMMUTE with the target logicals X̄ᵢ, so the reps stay
        # *valid* logicals of the drawn code (a stabilizer that anti-commutes with X̄₁ would make X̄₁
        # not a logical -- which is exactly the odd-overlap inconsistency to avoid).  X̄ᵢ commute with
        # the weight-4 bulk by construction (_patch_rep), so this only constrains the boundary.
        repvecs = [isv({q: P for q in reps[nm][1]}) for nm, P in target if reps[nm][1] is not None]
        present, vecs, B = [], [], _IntBasis()
        for p in sorted(plaqs, key=lambda q: -len(q["pauli"])):    # weight-4 first, then boundary/corner
            v = isv(p["pauli"])
            if (all(_icommute(v, w, n) for w in vecs)
                    and all(_icommute(v, rv, n) for rv in repvecs)   # keep every target logical valid
                    and B.reduce(v) != 0):
                p2 = dict(p); p2["corners"] = sorted(p["pauli"])
                present.append(p2); vecs.append(v); B.add(v)
        # EXPLICITLY search for a readout chain on the COMPLETE code too (a linear solve over its
        # stabilizers) -- it returns empty iff the joint is not a product of them, confirming the
        # "no chain" verdict directly rather than short-circuiting on joint_in_span.
        if have_reps:
            chain = _readout_chain(data, present, [reps[nm] for nm, _ in target])

    # health of the PRESENTED code (k, commutation, no weight-1 leftover)
    Bp, pvecs = _IntBasis(), []
    for p in present:
        v = isv(p["pauli"]); pvecs.append(v); Bp.add(v)
    k = n - Bp.rank
    weights = {}
    for p in present:
        weights[len(p["pauli"])] = weights.get(len(p["pauli"]), 0) + 1
    commute = all(_icommute(pvecs[i], pvecs[j], n)
                  for i in range(len(pvecs)) for j in range(i + 1, len(pvecs)))
    twist = any(P == "Y" for p in present for P in p["pauli"].values())
    w1 = any(Bp.reduce(isv({q: P})) != 0 and all(_icommute(isv({q: P}), w, n) for w in pvecs)
             for q in data for P in "XZ")
    return dict(stabilizers=present, k=k, weights=weights, commute=commute, no_weight1=not w1,
                no_twist=not twist, joint_in_span=joint_in_span, singles_in_span=singles,
                subproducts_in_span=subp, readout_chain=chain, data=data, reps=reps, phase=phase)


ACCEPTANCE_ITEMS = (
    "target logicals commute with all stabilizers",
    "remaining logical dof == N-1",
    "full joint in span",
    "no single logical measured",
    "no proper sub-product measured",
    "no weight-1 leftover logical",
    "all stabilizers commute",
    "no Y / no twist",
    "no MPP",
    "DEM valid",
    "no tick collision",
    "readout chain exists and product == joint",
)


def acceptance(patches, target, tree_cells, seed=0, max_trials=5000):
    """The **strict** subset-joint acceptance gate: ``accept=True`` iff **all twelve**
    :data:`ACCEPTANCE_ITEMS` hold on the actually-**selected measuring code** (not the complete
    memory code).  A geometry is only feasible for a joint measurement when it *measures the joint*,
    so a region with ``remaining logical dof != N-1`` (the joint left as a logical) is a **FAIL** — it
    is never reported as passing.

    Returns a dict: ``accept`` (bool), ``has_measuring_code`` (did the physics layer build a code that
    measures the joint at all), ``items`` (an ordered ``{name: bool|None}`` over the twelve
    conditions; ``None`` = undefined because no measuring code exists), ``k`` (remaining logical dof),
    ``n_stab``, ``data``, ``readout_chain_len``.  ``patches``/``tree_cells`` are as in
    :func:`route_and_build` / :func:`complete_code`.
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    data = sorted(data)
    N = len(target)
    lay = _assemble_region(placed, target, orient, data, retype, d, seed, max_trials)
    if lay is None:                                        # no code measures the joint -> reject
        cc = complete_code(patches, target, tree_cells)
        items = {k: None for k in ACCEPTANCE_ITEMS}
        items["remaining logical dof == N-1"] = (cc["k"] == N - 1)
        items["full joint in span"] = cc["joint_in_span"]
        items["readout chain exists and product == joint"] = False
        return dict(accept=False, has_measuring_code=False, k=cc["k"],
                    n_stab=len(cc["stabilizers"]), data=len(data),
                    readout_chain_len=len(cc["readout_chain"]), items=items)

    a = acceptance_of_layout(lay)
    return dict(accept=a["accept"], has_measuring_code=True, k=a["k"], n_stab=a["n_stab"],
                data=a["data"], readout_chain_len=a["readout_chain_len"], items=a["items"])


def acceptance_of_layout(lay):
    """The twelve :data:`ACCEPTANCE_ITEMS` computed directly on an already-built
    :class:`MultiPatchLayout` — the layout-level core of :func:`acceptance`, usable on **any** built
    layout, including the convex-corner-cut ones :func:`route_and_build` returns (which
    :func:`acceptance` cannot reach — it rebuilds the standard construction).

    Returns a dict: ``accept`` (bool), ``items`` (an ordered ``{name: bool}`` over the twelve
    conditions), ``N`` (target-logical count), ``data``, ``n_stab``, ``k`` (remaining logical dof),
    ``readout_chain_len``.
    """
    v = lay.verify()
    isv, n = _int_symplectic(lay.data)
    S = [isv(c["pauli"]) for c in lay.checks]
    B = _IntBasis()
    for s in S:
        B.add(s)
    logvecs = [isv({c: P for c in sup}) for _, P, sup in lay.logicals]
    log_ok = all(_icommute(Lv, s, n) for Lv in logvecs for s in S)   # every target logical valid
    joint = _xor_ints(logvecs)
    chain = lay.readout_chain
    prod = _xor_ints([isv(c["pauli"]) for c in lay.checks if c["syn"] in chain])
    chain_ok = bool(chain) and prod == joint                        # chain product IS the joint
    items = {
        "target logicals commute with all stabilizers": log_ok,
        "remaining logical dof == N-1": v["logical_count"],
        "full joint in span": v["joint"],
        "no single logical measured": v["no_single"],
        "no proper sub-product measured": v["no_subjoint"],
        "no weight-1 leftover logical": v["no_weight1_logical"],
        "all stabilizers commute": v["commute"],
        "no Y / no twist": v["no_twist"],
        "no MPP": v["no_mpp"],
        "DEM valid": v["dem_valid"],
        "no tick collision": v["no_tick_collision"],
        "readout chain exists and product == joint": chain_ok,
    }
    return dict(accept=all(bool(x) for x in items.values()), items=items, N=len(lay.logicals),
                data=len(lay.data), n_stab=len(lay.checks), k=len(lay.data) - B.rank,
                readout_chain_len=len(chain))


def collision_report(patches, target, tree_cells, layout=None):
    """**Physical placement check**: does the routed code collide with any idle obstacle patch?

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_and_build`); ``tree_cells`` is the
    routed corridor, a list of coarse cells ``[(a, b), ...]``.

    The routed joint code and every non-target (obstacle) patch are *separate* patches on one chip, so
    they may share no physical qubit.  Returns a dict with three counts (all should be **0**):

    * ``data``               — a routed **data** qubit sits on an obstacle data qubit;
    * ``corner_uses_obstacle`` — a routed **stabilizer** uses an obstacle data qubit as a corner;
    * ``ancilla``            — a routed **ancilla** (plaquette centre) coincides with an obstacle
      patch's own boundary ancilla (the ``keepout=0`` failure: edge-adjacent cells share that line).

    Plus ``clean`` (all zero) and per-obstacle ``details``.  Run it on any routed corridor to prove the
    layout is physically placeable.
    """
    patch_at, _orient, d = _specs_to_cells(patches, target)
    if layout is not None:                 # judge the ACTUAL built code, not a re-derivation
        stabs, rdata = layout.checks, set(layout.data)
    else:
        cc = complete_code(patches, target, tree_cells)
        stabs, rdata = cc["stabilizers"], set(cc["data"])
    tnames = {nm for nm, _ in target}
    onames = [nm for nm in patch_at if nm not in tnames]
    routed_syn = {tuple(int(v) for v in p["syn"]) for p in stabs}
    routed_data = rdata
    routed_corners = (set().union(*[set(p["pauli"]) for p in stabs]) if stabs else set())
    nd, nc, na, details = 0, 0, 0, {}
    spec_of = {p.name: p for p in patches}
    for on in onames:
        Odata = cell(*patch_at[on], d)
        from .bent_layout import place_patch
        Osyn = {tuple(int(v) for v in c["syn"]) for c in place_patch(spec_of[on])["checks"]}
        a, b, c = routed_data & Odata, routed_corners & Odata, routed_syn & Osyn
        nd += len(a); nc += len(b); na += len(c)
        if a or b or c:
            details[on] = dict(data=sorted(a), corner=sorted(b), ancilla=sorted(c))
    return dict(data=nd, corner_uses_obstacle=nc, ancilla=na,
                clean=(nd == 0 and nc == 0 and na == 0), details=details)


