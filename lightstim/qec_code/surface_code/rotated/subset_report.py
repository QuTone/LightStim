"""Report front-end for rotated **subset** joint lattice-surgery measurements.

One public entry point, :func:`report_subset_joint`: hand it the native ``PatchSpec`` array
(targets + obstacles) and the ``target`` list, and it routes, builds and verifies the joint via
:func:`.subset_routing.route_and_build`, then shows up to **four** things — **(1)** the auto-routed
path on the coarse-cell grid, **(2)** the strict acceptance checklist
(:data:`.subset_routing.ACCEPTANCE_ITEMS`), **(3)** the data-qubit layout, **(4)** the Stim
``detslice-with-ops`` diagram — each behind its own ``show_*``
toggle, and returns the verified :class:`.subset_routing.SubsetRoute` for programmatic use.

matplotlib and IPython are imported lazily inside the display paths, so importing this module (or
calling the entry point with every ``show_*`` off, e.g. in headless tests) needs neither.
"""

import numpy as np

from .subset_routing import (route_and_build, collision_report, cell, _specs_to_cells,
                             acceptance_of_layout, ACCEPTANCE_ITEMS)

__all__ = ["report_subset_joint"]


def _print_acceptance(a, label):
    """(2) The :data:`ACCEPTANCE_ITEMS` PASS checklist."""
    n_ok = sum(1 for x in a['items'].values() if x is True)
    print(f"{label}   ->  {'PASS' if a['accept'] else 'FAIL'}   "
          f"({n_ok}/{len(ACCEPTANCE_ITEMS)} conditions)")
    print(f"    data qubits = {a['data']}   independent stabilizers (rank) = {a['n_stab']}   "
          f"readout chain = {a['readout_chain_len']} stabs")
    print(f"    remaining logical dof  =  data - stabilizers  =  {a['data']} - {a['n_stab']}  =  "
          f"{a['k']}   (want N-1 = {a['N']-1})")
    for k, x in a['items'].items():
        print(f"    [{'True ' if x is True else str(x)}] {k}")


def _draw_path(patches, target, route, title):
    """(1) The routed path on the coarse-cell grid: target patches coloured (X̄ blue / Z̄ orange),
    obstacles grey-hatched, the ancilla corridor in green."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    patch_at, _o, d = _specs_to_cells(patches, target)
    tset = dict(target)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    def box(a, b, **kw):
        qs = cell(a, b, d); x0, x1 = min(q[0] for q in qs), max(q[0] for q in qs)
        y0, y1 = min(q[1] for q in qs), max(q[1] for q in qs)
        ax.add_patch(Rectangle((x0 - 1, y0 - 1), x1 - x0 + 2, y1 - y0 + 2, **kw))
        return (x0 + x1) / 2, (y0 + y1) / 2
    for (a, b) in route:                                        # ancilla corridor
        box(a, b, facecolor='#bfe6bf', edgecolor='#2ca02c', lw=1.4, zorder=1)
    for p in patches:                                           # patches
        a, b = patch_at[p.name]
        if p.name in tset:
            cx, cy = box(a, b, facecolor=('#6f9cf0' if tset[p.name] == 'X' else '#ef7a45'),
                         edgecolor='white', lw=1, zorder=2)
            ax.text(cx, cy, p.name, ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=3)
        else:
            cx, cy = box(a, b, facecolor='0.86', edgecolor='0.6', hatch='xxx', lw=0.5, zorder=1)
            ax.text(cx, cy, p.name, ha='center', va='center', color='0.4', fontsize=8, zorder=3)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.autoscale_view(); ax.margins(0.08); ax.axis('off')
    ax.set_title(f"{title}  —  routed path (coarse cells)", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color='#6f9cf0', label='target X̄'),
                       mpatches.Patch(color='#ef7a45', label='target Z̄'),
                       mpatches.Patch(facecolor='#bfe6bf', edgecolor='#2ca02c', label='ancilla corridor'),
                       mpatches.Patch(facecolor='0.86', edgecolor='0.6', hatch='xxx', label='obstacle')],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
    plt.tight_layout(); plt.show()


def _show_layout(lay, title):
    """(3) Data-qubit layout: X/Z/mixed stabilizers, data qubits, logicals, gold readout chain."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon, Circle, Rectangle
    COL = {'X': '#e23b3b', 'Z': '#2f6fd0', 'M': '#8b3fd0'}
    FILL = {'X': '#f6b8b8', 'Z': '#b8cdf0', 'M': '#d9c2f2'}; GOLD = '#f0a000'
    CHAIN = lay.readout_chain
    allc = [c for ch in lay.checks for c in ch['corners']] + list(lay.data)
    x0, x1 = min(p[0] for p in allc) - 1.6, max(p[0] for p in allc) + 1.6
    y0, y1 = min(p[1] for p in allc) - 1.6, max(p[1] for p in allc) + 1.6
    fig, ax = plt.subplots(figsize=(min(13, 0.85 + (x1 - x0) * 0.42), min(13, 0.85 + (y1 - y0) * 0.42)))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#f6f6f9', ec='none', zorder=0))
    for ch in lay.checks:
        t, pts, syn = ch['type'], ch['corners'], ch['syn']; hl = syn in CHAIN; ec = GOLD if hl else COL[t]
        if len(pts) >= 3:
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            order = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
            ax.add_patch(Polygon(order, closed=True, facecolor=FILL[t], edgecolor=ec,
                                 lw=2.4 if hl else 0.6, alpha=0.9 if hl else 0.30, zorder=3 if hl else 2))
        elif len(pts) == 2:
            (a, b), (c, dd) = pts
            ax.plot([a, syn[0], c], [b, syn[1], dd], color=ec, lw=7 if hl else 5,
                    alpha=0.7 if hl else 0.28, solid_capstyle='round', zorder=3 if hl else 2)
        if t == 'M':
            for c, P in ch['pauli'].items():
                ax.plot([syn[0], c[0]], [syn[1], c[1]], color=COL[P], lw=2.4, zorder=5, alpha=0.9, solid_capstyle='round')
        ax.add_patch(Rectangle((syn[0] - 0.18, syn[1] - 0.18), 0.36, 0.36, facecolor=COL[t],
                     edgecolor=GOLD if hl else 'white', lw=2 if hl else 0.8, zorder=6, alpha=1 if hl else 0.55))
    for q in lay.data:
        ax.add_patch(Circle(q, 0.14, facecolor='#1a1a1a', edgecolor='white', lw=0.6, zorder=8))
    xshades = ['#c01616', '#7a0d0d', '#e0552a', '#9c1b5a', '#5a1b9c']; xi = 0
    for nm, P, sup in lay.logicals:
        col = '#13346e' if P == 'Z' else xshades[xi % len(xshades)]; xi += (P == 'X')
        srt = sorted(sup)
        ax.plot([q[0] for q in srt], [q[1] for q in srt], color=col, lw=5, zorder=10, solid_capstyle='round')
        mx, my = srt[len(srt) // 2]
        ax.text(mx, my - 0.8, fr'$\bar {P}_{{{nm[1:]}}}$', color=col, fontsize=13, fontweight='bold', ha='center', zorder=11,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=col, lw=1))
    handles = [mpatches.Patch(color=COL['X'], label='X stabilizer'), mpatches.Patch(color=COL['Z'], label='Z stabilizer'),
               mpatches.Patch(color=COL['M'], label='MIXED (XZ) wall'),
               mpatches.Patch(facecolor='#fff3d6', edgecolor=GOLD, lw=2, label='readout chain (product = joint)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=9)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    ax.set_title(title, fontsize=12, pad=46); plt.tight_layout(); plt.show()


def _default_title(target):
    """``[("X1","X"), ("B2","X")]`` -> ``"Subset joint  M(X̄1 X̄_B2)"`` (notebook naming style)."""
    def term(nm, P):
        return f"{P}̄{nm[1:]}" if nm[:1] == P and nm[1:].isdigit() else f"{P}̄_{nm}"
    return "Subset joint  M(" + " ".join(term(nm, P) for nm, P in target) + ")"


def report_subset_joint(patches, target, title=None, *, route=None,
                        show_path=True, show_acceptance=True,
                        show_layout=True, show_detslice=True, rounds=2):
    """Route, build and verify a subset joint ``M(∏ᵢ P̄ᵢ)``, showing up to FOUR report artefacts.

    ``patches`` is the native ``PatchSpec`` array (targets + obstacles); ``target`` is
    ``[(name, "X"|"Z"), ...]`` naming the measured subset — the same inputs as
    :func:`.subset_routing.route_and_build`, which finds the corridor **fully automatically**
    (propose-and-verify with retry: obstacle-aware candidates, standard construction first, then
    the convex-corner cut).  No hand-written ``route``.

    ``title`` labels every artefact (default: generated from ``target``).  Each artefact has its
    own toggle, all on by default: ``show_path`` **(1)** the auto-routed path on the coarse grid,
    ``show_acceptance`` **(2)** the printed :data:`ACCEPTANCE_ITEMS` checklist (+ collision report
    when obstacles are present), ``show_layout`` **(3)** the data-qubit layout, ``show_detslice``
    **(4)** the Stim ``detslice-with-ops`` SVG of a ``rounds``-round noiseless circuit (needs
    IPython).  With every toggle off nothing is displayed and no plotting backend is touched.

    Returns the verified :class:`.subset_routing.SubsetRoute` (``.layout`` / ``.tree`` / ``.how`` /
    ``.cut``); raises ``RuntimeError`` if ``route_and_build`` finds no verified route.
    """
    if title is None:
        title = _default_title(target)
    tnames = {nm for nm, _ in target}
    r = route_and_build(patches, target, route=route)   # auto route, or EXPLICIT when route given
    if not r.ok:
        raise RuntimeError(f"route_and_build failed: {r.status} — {r.message}")
    route, lay = sorted(r.tree), r.layout
    how = 'standard construction' if r.how == 'standard' else f'convex-corner cut at {r.cut}'
    if show_path:
        _draw_path(patches, target, route, title)                   # (1) path (auto-routed)
    if show_acceptance:
        _print_acceptance(acceptance_of_layout(lay), f"{title}   [{how}]")  # (2) acceptance
        if any(p.name not in tnames for p in patches):
            print("    collision-clean (routed code vs obstacles):",
                  collision_report(patches, target, route)['clean'])
    if show_layout:
        _show_layout(lay, title)                                    # (3) layout
    if show_detslice:                                               # (4) detslice
        from IPython.display import SVG, display
        display(SVG(str(lay.build_circuit(rounds=rounds, p=0.0).diagram('detslice-with-ops-svg'))))
    return r
