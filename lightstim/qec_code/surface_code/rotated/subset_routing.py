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
general bent-bus / rectilinear-Steiner router is future work.  :func:`classify_route` reports, for
any placement, exactly which stage a route reaches (see its docstring for the failure taxonomy).

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
    _patch_rep, _logical_direction, _select_joint_checks, _readout_chain,
    MultiPatchLayout, _connected, _collapse_check, _int_symplectic, _IntBasis, _icommute,
)

__all__ = ["PatchSpec", "cell", "origin_of", "cell_index", "route_subset", "classify_route",
           "trace_physics", "complete_code", "verify_report", "acceptance", "ACCEPTANCE_ITEMS",
           "local_plaquette_types", "collision_report", "selection_search", "SubsetRoute"]

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


def _corridor_graph(patch_at, target, d, pad, ignore_obstacles=False, keepout=1):
    """Build the coarse-cell corridor graph.

    A coarse cell is **corridor-eligible** iff it holds no patch and it stays clear of every obstacle
    patch by a **keep-out margin** of ``keepout`` cells (king-move / Chebyshev distance ``> keepout``).
    ``keepout=1`` (the default) forbids any cell **adjacent** to an obstacle, so the routed code never
    shares a boundary ancilla line with an idle obstacle patch (``keepout=0`` did — two edge-adjacent
    cells share the ancilla row/column between them, a physical collision).  ``ignore_obstacles=True``
    treats the obstacles as *absent* (routable, no margin) — the no-obstacle baseline for
    :func:`classify_route`.

    Returns ``(G, corridor, placed, occupied, obstacle_fp, onames)``.
    """
    tnames = [nm for nm, _ in target]
    onames = [nm for nm in patch_at if nm not in tnames]
    occupied = {ab: nm for nm, ab in patch_at.items()}
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    obstacle_fp = set().union(*[placed[nm] for nm in onames]) if onames else set()
    obstacle_cells = [] if ignore_obstacles else [patch_at[nm] for nm in onames]

    blocked = {patch_at[nm] for nm in tnames} if ignore_obstacles else set(occupied)
    forbid_fp = set() if ignore_obstacles else obstacle_fp
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

def _assemble_region(placed, target, orient, data, retype, d, seed=0, max_trials=5000):
    """Hand a routed region to the existing physics layer.

    Mirrors :func:`.build_rotated_multi_patch_joint_layout`'s parity-phase loop exactly — only the
    geometry that feeds it (``data``/``retype``) comes from the subset router.  Returns a
    :class:`.MultiPatchLayout` (checks selected + logical reps found), or ``None`` if the physics
    layer cannot host this geometry (no valid logical rep, or no boundary selection measures the
    joint).  The oracle decision itself is ``MultiPatchLayout.verify()``.
    """
    data = sorted(data)
    if not _connected(set(data)):
        return None
    sv, n = _symplectic(data)
    for phase in (0, 1):
        plaqs = _bent_plaquettes(data, retype, phase)
        F = [sv(p["pauli"]) for p in plaqs if len(p["pauli"]) >= 4]
        reps, ok = {}, True
        for nm, P in target:
            sup = _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, n)
            if sup is None:
                ok = False
                break
            reps[nm] = (P, sup)
        if not ok:
            continue
        log_pairs = [reps[nm] for nm, _ in target]
        checks = _select_joint_checks(data, plaqs, log_pairs, seed=seed, max_trials=max_trials)
        if checks is None:
            continue
        for c in checks:
            c["corners"] = sorted(c["pauli"])
        logicals = [(nm, reps[nm][0], reps[nm][1]) for nm, _ in target]
        x_obs = next((s for nm, P, s in logicals if P == "X"), logicals[0][2])
        return MultiPatchLayout(distance=d, data=data, checks=checks, logicals=logicals,
                                x_observable=x_obs, readout_chain=_readout_chain(data, checks, log_pairs),
                                target=list(target))
    return None


# -----------------------------------------------------------------------------
# the router
# -----------------------------------------------------------------------------

@dataclass
class SubsetRoute:
    """Result of :func:`route_subset`.  ``status == "ok"`` iff ``layout`` is a verified joint.

    On a **failure** an obstacle-free corridor may still exist but be un-hostable by the physics
    layer — ``attempted`` / ``attempted_arms`` carry the shortest such corridor (empty only when
    ``status == "no_path"``) so a caller can *draw the route that was found* and label it correctly
    (a rejected corridor is **not** "no route" — see :func:`classify_route`).
    """
    status: str                                       # "ok" | "no_path" | "no_verified_route"
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

    @property
    def ok(self):
        return self.status == "ok"


def route_subset(patches, target, pad=1, per_z=4, max_combo=64,
                 seed=0, max_trials=5000, keepout=1):
    """Route + verify a subset joint ``M(∏ᵢ P̄ᵢ)``.

    ``patches``: a list of :class:`PatchSpec` — **all** patches on the chip (targets *and*
    obstacles), in the same ``PatchSpec(name, origin, distance, measured_logical, orientation)`` form
    as the N-patch API.  Origins must sit on the coarse pitch-``2d`` routing grid (use
    :func:`origin_of`); distance ``d`` and each patch's orientation are read off the specs.
    ``target``: ``[(name, "X"|"Z"), ...]`` — a **subset** of the patch names; the rest are obstacles.
    ``keepout``: minimum king-move clearance (in cells) the routed code — **including the target
    patches** — must keep from every obstacle.  Default ``1``: a target or corridor cell may not be
    adjacent to an obstacle cell, so the routed code never shares a boundary ancilla with an idle
    obstacle patch (``keepout=0`` allowed that — a physical collision; see :func:`collision_report`).

    Builds the obstacle-aware ``d``-wide corridor graph, forms candidate shortest-path trees from an
    X-anchor to each other target, and **oracle-gates** every candidate (via
    :meth:`MultiPatchLayout.verify`).  Returns a :class:`SubsetRoute`.
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    tnames = [nm for nm, _ in target]
    onames = [nm for nm in patch_at if nm not in tnames]
    placed_all = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    obstacle_fp0 = set().union(*[placed_all[nm] for nm in onames]) if onames else set()
    base0 = dict(placed=placed_all, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp0,
                 corridor=set())
    # a TARGET patch must itself keep the margin from every obstacle (else its own boundary ancillas
    # collide with the obstacle's) -- this is a placement error, not something routing can fix.
    for tn in tnames:
        bad = [on for on in onames if _cheb(patch_at[tn], patch_at[on]) <= keepout]
        if bad:
            return SubsetRoute(status="target_obstacle_conflict", root=tnames[0],
                               message=(f"target {tn} is within keepout={keepout} of obstacle(s) "
                                        f"{bad}: their boundary ancillas would collide.  Move them "
                                        f"apart (target patches must clear obstacles too)."),
                               **base0)
    G, corridor, placed, occupied, obstacle_fp, onames = _corridor_graph(patch_at, target, d, pad,
                                                                         keepout=keepout)
    base = dict(placed=placed, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp,
                corridor=corridor)

    root = next((nm for nm, P in target if P == "X"), tnames[0])
    root_faces = _x_faces(orient[root])
    zs = [nm for nm in tnames if nm != root]

    cand = {}
    for z in zs:
        arms = _candidate_arms(G, patch_at, corridor, root, root_faces, z, per_z)
        if not arms:
            return SubsetRoute(status="no_path", root=root,
                               message=f"path search found no obstacle-free corridor {root} -> {z}",
                               **base)
        cand[z] = arms
    # the shortest candidate corridor — an obstacle-free route that DEMONSTRABLY exists (drawable even
    # if no candidate verifies, so a failure is shown as a *rejected* corridor, not "no route").
    shortest_arms = {z: cand[z][0] for z in zs}
    attempted = set().union(*[set(p) for p in shortest_arms.values()]) if zs else set()

    tried = 0
    for combo in (itertools.product(*[cand[z] for z in zs]) if zs else [()]):
        tried += 1
        if tried > max_combo:
            break
        tree = set().union(*[set(p) for p in combo]) if combo else set()
        data, retype = path_to_corridor(tree, placed, target, d)
        layout = _assemble_region(placed, target, orient, data, retype, d, seed, max_trials)
        if layout is not None and all(layout.verify().values()):
            arms = dict(zip(zs, combo))
            return SubsetRoute(status="ok", message="verified subset joint", layout=layout,
                               root=root, arms=arms, tree=tree, attempted=tree, attempted_arms=arms,
                               data=data, tried=tried, **base)
    return SubsetRoute(status="no_verified_route", root=root, tried=tried,
                       attempted=attempted, attempted_arms=shortest_arms,
                       message=(f"an obstacle-free corridor exists but no candidate passed the GF(2) "
                                f"oracle (tried {tried}); the physics layer cannot host this "
                                f"geometry (physics_unsupported), e.g. a bent trunk"),
                       **base)


# -----------------------------------------------------------------------------
# failure taxonomy
# -----------------------------------------------------------------------------

def classify_route(patches, target, pad=1, per_z=4, max_combo=64, keepout=1):
    """Diagnose *which stage* a subset route reaches — the four-way failure taxonomy.

    ``patches`` is a list of :class:`PatchSpec` (all patches; ``target`` selects the measured subset,
    the rest are obstacles) — see :func:`route_subset` for the argument conventions.

    Returns a dict with:

    * ``category`` — one of:
        - ``"verified"``            : a route passed the full GF(2) oracle;
        - ``"no_path"``             : path search found no obstacle-free corridor to some target;
        - ``"physics_unsupported"`` : an obstacle-free corridor exists, but the physics layer cannot
          host it (no valid logical rep, or no boundary selection measures the joint) — e.g. a bent
          trunk;
        - ``"oracle_reject"``       : the physics layer built a code, but ``verify()`` failed a check.
    * ``detour_forced`` — ``True`` iff an obstacle makes the obstacle-aware shortest corridor to some
      target **strictly longer** than the obstacle-ignoring one (the direct ``d``-wide corridor would
      overlap an obstacle, so a detour is forced; the aware router excludes those cells).
    * ``aware_path_exists``, ``assembled``, ``verified`` — the per-stage booleans.
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    tnames = [nm for nm, _ in target]
    root = next((nm for nm, P in target if P == "X"), tnames[0])
    root_faces = _x_faces(orient[root])
    zs = [nm for nm in tnames if nm != root]

    def shortest_len(G, corr, z):
        arms = _candidate_arms(G, patch_at, corr, root, root_faces, z, 1)
        return len(arms[0]) if arms else None

    # stage 0/1: obstacle-ignoring baseline vs obstacle-aware shortest corridor to each target.
    Gn, corr_n, placed, _, _, _ = _corridor_graph(patch_at, target, d, pad, ignore_obstacles=True)
    G, corr, placed, _, obs_fp, onames = _corridor_graph(patch_at, target, d, pad, keepout=keepout)
    detour_forced, aware_path = False, True
    for z in zs:
        naive_len, aware_len = shortest_len(Gn, corr_n, z), shortest_len(G, corr, z)
        if aware_len is None:
            aware_path = False
        elif naive_len is not None and aware_len > naive_len:
            detour_forced = True
    if not aware_path:
        return dict(category="no_path", detour_forced=detour_forced,
                    aware_path_exists=False, assembled=None, verified=False)

    # stages 2-3: physics-layer acceptance and oracle verdict (propose-and-verify)
    cand = {z: _candidate_arms(G, patch_at, corr, root, root_faces, z, per_z) for z in zs}
    assembled_any, verified_any = False, False
    for combo in (itertools.product(*[cand[z] for z in zs]) if zs else [()]):
        tree = set().union(*[set(p) for p in combo]) if combo else set()
        data, retype = path_to_corridor(tree, placed, target, d)
        lay = _assemble_region(placed, target, orient, data, retype, d)
        if lay is not None:
            assembled_any = True
            if all(lay.verify().values()):
                verified_any = True
                break
    if verified_any:
        category = "verified"
    elif assembled_any:
        category = "oracle_reject"          # built a code, but verify() failed a check
    else:
        category = "physics_unsupported"    # no reps / no boundary selection measured the joint
    return dict(category=category, detour_forced=detour_forced, aware_path_exists=True,
                assembled=assembled_any, verified=verified_any)


def trace_physics(patches, target, tree_cells, seed=0, max_trials=3000):
    """Feed **one specific** routed corridor (``tree_cells``) into the physics layer and report which
    **step** succeeds or fails — so ``physics_unsupported`` can be seen concretely, not just asserted.

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_subset`); ``tree_cells`` is the
    explicit corridor to test, a list of coarse cells ``[(a, b), ...]`` (the routing, not patches).

    The pipeline (mirroring :func:`_assemble_region`) is:

      1. ``connected``       — is the routed data region connected?
      2. ``patch_rep``       — does each target's declared-orientation logical string exist?
      3. ``joint_in_span``   — is the joint ``∏P̄ᵢ`` a **product of plaquettes** (a stabilizer of the
                               full-plaquette code)?  If not, it stays a *nontrivial logical* and **no**
                               boundary selection can measure it — this is where a bent trunk fails.
      4. ``selected``        — does :func:`_select_joint_checks` find a valid measuring code?

    Returns a dict with the per-step booleans, ``failing_step`` (``None`` if it verifies), the
    per-single ``singles_in_span``, and the geometry (``data``, ``retype``, ``plaqs``, ``reps``) for
    drawing the true data-qubit layout.  Reports the phase that got furthest.
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    base = dict(data=data, retype=retype, placed=placed, tree=set(tree_cells), target=list(target))
    if not _connected(set(data)):
        return dict(base, connected=False, patch_rep_ok=False, joint_in_span=False, selected=False,
                    failing_step="connected", plaqs=[], reps={}, singles_in_span={})
    sv, n = _symplectic(data)
    best = None
    for phase in (0, 1):
        plaqs = _bent_plaquettes(data, retype, phase)
        F = [sv(p["pauli"]) for p in plaqs if len(p["pauli"]) >= 4]
        reps, rep_ok = {}, True
        for nm, P in target:
            sup = _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, n)
            reps[nm] = (P, sup)
            if sup is None:
                rep_ok = False
        forced = [p for p in plaqs if len(p["pauli"]) >= 4]     # the definite bulk/wall stabilizers
        for p in forced:
            p["corners"] = sorted(p["pauli"])
        info = dict(base, phase=phase, connected=True, patch_rep_ok=rep_ok, plaqs=plaqs, reps=reps,
                    forced=forced, checks=None)
        if not rep_ok:
            info.update(joint_in_span=False, selected=False, failing_step="patch_rep", singles_in_span={})
            best = best or info
            continue
        log_pairs = [reps[nm] for nm, _ in target]
        isv, _n = _int_symplectic(data)
        B = _IntBasis()
        for p in plaqs:
            B.add(isv(p["pauli"]))
        singles = {nm: B.contains(isv({c: P for c in sup})) for nm, (P, sup) in reps.items()}
        _collapsed, joint_ok = _collapse_check(data, plaqs, log_pairs, target)
        checks = _select_joint_checks(data, plaqs, log_pairs, seed=seed, max_trials=max_trials)
        if checks is not None:
            for c in checks:
                c["corners"] = sorted(c["pauli"])
        info.update(joint_in_span=joint_ok, selected=checks is not None, singles_in_span=singles,
                    checks=checks)                          # the SELECTED code (or None) -- what to draw
        if checks is not None:
            info["failing_step"] = None
            return info
        info["failing_step"] = "joint_in_span" if not joint_ok else "boundary_selection"
        best = info                                     # a rep-valid phase beats a rep-invalid one
    return best


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

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_subset`); ``tree_cells`` is the
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
    present, chain = None, set()
    if joint_in_span:
        jchecks = _select_joint_checks(data, plaqs, [reps[nm] for nm, _ in target])
        if jchecks:
            for c in jchecks:
                c["corners"] = sorted(c["pauli"])
            present = jchecks
            chain = _readout_chain(data, jchecks, [reps[nm] for nm, _ in target])
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


def verify_report(patches, target, tree_cells, seed=0, max_trials=5000):
    """The **full 11-check verification summary** for ANY routed corridor — passing OR failing.

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_subset`); ``tree_cells`` is the
    explicit corridor to verify, a list of coarse cells ``[(a, b), ...]``.

    Returns a dict with ``items`` mapping each check to ``True`` / ``False`` / an ``int`` / ``None``
    (``None`` = the item is undefined because **no valid stabilizer selection exists**, i.e. the code
    could not be built).  Also returns construction-sanity fields (``forced_w4_commute``,
    ``qubits_covered``) so a failing layout is shown to be *well-formed but unmeasurable*, not
    malformed.  Formatting is left to the caller (see the notebook's ``print_report``).
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    data = sorted(data)
    N = len(target)
    label = "M(" + " ".join(f"{P}̄{nm[1:]}" for nm, P in target) + ")"

    def subproducts_in_span(B, vecs):
        return sum(B.contains(_xor_ints([vecs[k] for k in comb]))
                   for r in range(1, N) for comb in itertools.combinations(range(N), r))

    lay = _assemble_region(placed, target, orient, data, retype, d, seed, max_trials)
    if lay is not None:                                   # a valid measuring code exists
        v = lay.verify()
        isv, n = _int_symplectic(lay.data)
        B = _IntBasis()
        for c in lay.checks:
            B.add(isv(c["pauli"]))
        vecs = [isv({c: P for c in sup}) for _, P, sup in lay.logicals]
        items = {
            "full joint in span": v["joint"],
            "proper sub-products in span": subproducts_in_span(B, vecs),
            "remaining logical dof": len(lay.data) - B.rank,
            "no weight-1 leftover logical": v["no_weight1_logical"],
            "all stabilizers commute": v["commute"],
            "no Y / no twist": v["no_twist"],
            "no MPP": v["no_mpp"],
            "DEM valid": v["dem_valid"],
            "no tick collision": v["no_tick_collision"],
        }
        return dict(label=label, N=N, data=len(lay.data), n_stab=len(lay.checks), n_pool=None,
                    selected=True, forced_w4_commute=True, qubits_covered=True, failing_step=None,
                    items=items, all_pass=all(v.values()))

    # FAILING: build the COMPLETE code (max commuting) and report ITS properties -- so the reader sees
    # a full, valid k=1 patch, and that the joint is STILL not measurable (not just an incomplete bulk).
    tr = trace_physics(patches, target, tree_cells)
    cc = complete_code(patches, target, tree_cells)
    items = {
        "full joint in span": cc["joint_in_span"],          # measurability across ALL codes
        "proper sub-products in span": cc["subproducts_in_span"],
        "remaining logical dof": cc["k"],                    # k of the COMPLETE code (a valid patch)
        "no weight-1 leftover logical": cc["no_weight1"],
        "all stabilizers commute": cc["commute"],            # True -> the complete code IS valid
        "no Y / no twist": cc["no_twist"],
        "no MPP": None,                                      # no joint-measurement circuit exists
        "DEM valid": None,                                   #   (the joint is a logical, not a stabilizer)
        "no tick collision": None,
    }
    return dict(label=label, N=N, data=len(data), n_stab=len(cc["stabilizers"]), n_pool=len(tr["plaqs"]),
                k=cc["k"], weights=cc["weights"], selected=False, forced_w4_commute=cc["commute"],
                qubits_covered=True, failing_step=tr["failing_step"], items=items, all_pass=False)


# every one of these must hold for a routed subset joint to be ACCEPTED (the strict gate).
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
    :func:`route_subset` / :func:`complete_code`.
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
    return dict(accept=all(bool(x) for x in items.values()), has_measuring_code=True,
                k=len(lay.data) - B.rank, n_stab=len(lay.checks), data=len(lay.data),
                readout_chain_len=len(chain), items=items)


def local_plaquette_types(patches, target, tree_cells, phase, anchor):
    """The types (``"X"`` red / ``"Z"`` blue / ``"M"`` mixed) of the **weight-4 bulk** plaquettes whose
    centre sits inside patch ``anchor``'s cell, at parity ``phase`` — so a caller can compare the local
    checkerboard coloring around one patch (e.g. ``X1``) across geometries / phases.  Returns a sorted
    ``[(syn_coord, type), ...]`` (top-left first).  ``tree_cells`` is the corridor; ``phase`` selects
    the coloring (the same knob :func:`complete_code` exposes)."""
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    plaqs = _bent_plaquettes(sorted(data), retype, phase)
    acell = placed[anchor]
    xs, ys = [q[0] for q in acell], [q[1] for q in acell]
    out = [(p["syn"], p["type"]) for p in plaqs
           if len(p["pauli"]) >= 4
           and min(xs) <= p["syn"][0] <= max(xs) and min(ys) <= p["syn"][1] <= max(ys)]
    return sorted(out)


def collision_report(patches, target, tree_cells):
    """**Physical placement check**: does the routed code collide with any idle obstacle patch?

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_subset`); ``tree_cells`` is the
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
    cc = complete_code(patches, target, tree_cells)
    tnames = {nm for nm, _ in target}
    onames = [nm for nm in patch_at if nm not in tnames]
    routed_syn = {p["syn"] for p in cc["stabilizers"]}
    routed_data = set(cc["data"])
    routed_corners = (set().union(*[set(p["pauli"]) for p in cc["stabilizers"]])
                      if cc["stabilizers"] else set())
    nd, nc, na, details = 0, 0, 0, {}
    for on in onames:
        Odata = cell(*patch_at[on], d)
        Osyn = {p["syn"] for p in _bent_plaquettes(sorted(Odata), set(), 0)}
        a, b, c = routed_data & Odata, routed_corners & Odata, routed_syn & Osyn
        nd += len(a); nc += len(b); na += len(c)
        if a or b or c:
            details[on] = dict(data=sorted(a), corner=sorted(b), ancilla=sorted(c))
    return dict(data=nd, corner_uses_obstacle=nc, ancilla=na,
                clean=(nd == 0 and nc == 0 and na == 0), details=details)


def selection_search(patches, target, tree_cells, seed=0):
    """Is a bent joint unmeasurable because the SELECTION is too limited, or because the GEOMETRY needs
    a new corner domain wall?  Two rigorous tests (reusing :func:`_bent_plaquettes` read-only):

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_subset`); ``tree_cells`` is the
    explicit corridor to probe, a list of coarse cells ``[(a, b), ...]``.

    1. **Boundary reselection.**  With the checkerboard bulk fixed, is the joint in the span of the
       bulk **plus every boundary candidate of BOTH types (X and Z) that commutes with it**?  If not,
       **no** boundary reselection can measure the joint — the greedy selection is not the limitation.
    2. **Domain-wall placements.**  For every whole-cell native-Z (mixed-wall) pattern, does the
       physics layer build a fully-verified joint code?

    Returns a dict: ``joint_in_current_pool``, ``boundary_reselection_measurable``,
    ``n_boundary_candidates``, ``n_commuting_with_bulk``, ``walls_tested``, ``walls_valid``,
    ``verdict`` (``"selection_sufficient"`` or ``"needs_new_corner_construction"``).
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    data = sorted(data)
    sv, n = _symplectic(data)
    isv, _ = _int_symplectic(data)

    plaqs, reps = None, None                              # pick the parity that hosts every rep
    for phase in (0, 1):
        pl = _bent_plaquettes(data, retype, phase)
        F = [sv(p["pauli"]) for p in pl if len(p["pauli"]) >= 4]
        rp = {nm: (P, _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, n))
              for nm, P in target}
        plaqs, reps = pl, rp
        if all(rp[nm][1] is not None for nm, _ in target):
            break
    joint = _xor_ints([isv({q: P for q in reps[nm][1]}) for nm, P in target])

    B0 = _IntBasis()
    for p in plaqs:
        B0.add(isv(p["pauli"]))
    in_pool = B0.contains(joint)

    # 1. boundary reselection: forced bulk + EVERY commuting both-type boundary candidate
    fvecs = [isv(p["pauli"]) for p in plaqs if len(p["pauli"]) >= 4]
    centers = {p["syn"]: p["corners"] for p in plaqs if len(p["corners"]) < 4}
    commuting = [isv({q: t for q in cor}) for cor in centers.values() for t in ("X", "Z")
                 if all(_icommute(isv({q: t for q in cor}), w, n) for w in fvecs)]
    Bb = _IntBasis()
    for v in fvecs + commuting:
        Bb.add(v)
    boundary_measurable = Bb.contains(joint)

    # 2. whole-cell domain-wall placements (each corridor cell native-Z or not; Z-targets always native)
    z_base = set().union(*[placed[nm] for nm, P in target if P == "Z"]) \
        if any(P == "Z" for _, P in target) else set()
    walls_tested, walls_valid = 0, 0
    tree = list(tree_cells)
    for r in range(len(tree) + 1):
        for combo in itertools.combinations(tree, r):
            native = set(z_base)
            for c in combo:
                native |= cell(*c, d)
            rt = {q for q in data if q not in native}
            lay = _assemble_region(placed, target, orient, data, rt, d, seed, 4000)
            walls_tested += 1
            if lay is not None and all(lay.verify().values()):
                walls_valid += 1
    verdict = ("selection_sufficient" if (boundary_measurable or walls_valid > 0)
               else "needs_new_corner_construction")
    return dict(joint_in_current_pool=in_pool, boundary_reselection_measurable=boundary_measurable,
                n_boundary_candidates=2 * len(centers), n_commuting_with_bulk=len(commuting),
                walls_tested=walls_tested, walls_valid=walls_valid, verdict=verdict)
