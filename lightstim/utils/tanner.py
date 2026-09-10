"""Tanner-graph visualisation for any :class:`~lightstim.ir.qec_patch.QECPatch`.

``matplotlib`` is an optional (dev) dependency, so it is imported lazily inside
:func:`draw_tanner_graph`.

The drawing reads the patch's own geometry: data qubits and syndrome ancillas
are placed at their ``qubit_coords``, and each stabilizer contributes edges from
its ancilla node to every data qubit in its support, coloured by check type
(``X`` red, ``Z`` blue, mixed grey). Stabilizers with no dedicated ancilla
(``syn_idx is None``, e.g. subsystem-code stabilizer centres) get a virtual node
at the centroid of their support.
"""

from __future__ import annotations

from typing import Optional

_TYPE_COLORS = {"X": "#c1121f", "Z": "#1d4e89", "Y": "#6a4c93", None: "#666666"}
_DATA_FACE = "#f2f2f2"
_LOGICAL_COLORS = ("#2a9d8f", "#e76f51", "#e9c46a", "#8ab17d")


def draw_tanner_graph(
    patch,
    ax=None,
    *,
    show_logicals: bool = True,
    node_size: float = 0.16,
    title: Optional[str] = None,
):
    """Draw the bipartite Tanner graph of ``patch``.

    Args:
        patch: A built :class:`~lightstim.ir.qec_patch.QECPatch` (or anything
            exposing ``qubit_coords``, ``data_indices``, ``stabilizers`` and
            ``logical_ops`` with the same shapes).
        ax: A ``matplotlib`` axes to draw into; a new figure/axes is made when
            omitted.
        show_logicals: Ring the data qubits in each registered logical operator's
            support (one colour per logical operator).
        node_size: Data-node radius in data coordinates.
        title: Axes title; defaults to the patch class name.

    Returns:
        The ``matplotlib`` axes drawn into.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))

    coords = {int(i): (float(x), float(y)) for i, (x, y) in patch.qubit_coords.items()}
    data_indices = {int(i) for i in patch.data_indices}

    def _support_centroid(support):
        pts = [coords[q] for q in support if q in coords]
        n = max(len(pts), 1)
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

    # --- edges + check nodes ------------------------------------------------
    check_nodes = []  # (x, y, color, label)
    for k, stab in enumerate(patch.stabilizers):
        ctype = stab.get("type")
        color = _TYPE_COLORS.get(ctype, _TYPE_COLORS[None])
        support = list(stab["data_indices"])
        syn_idx = stab.get("syn_idx")
        if syn_idx is not None and int(syn_idx) in coords:
            cx, cy = coords[int(syn_idx)]
        else:
            cx, cy = _support_centroid(support)
        check_nodes.append((cx, cy, color, ctype or "S"))
        for q in support:
            if q in coords:
                dx, dy = coords[q]
                ax.plot([dx, cx], [dy, cy], color=color, linewidth=0.9,
                        alpha=0.7, zorder=1)

    for cx, cy, color, label in check_nodes:
        ax.add_patch(
            FancyBboxPatch(
                (cx - 1.3 * node_size, cy - node_size),
                2.6 * node_size, 2.0 * node_size,
                boxstyle="round,pad=0.01", linewidth=1.3,
                edgecolor=color, facecolor="white", zorder=3,
            )
        )
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7,
                color=color, zorder=4)

    # --- data nodes -------------------------------------------------------
    for q in sorted(data_indices):
        if q not in coords:
            continue
        x, y = coords[q]
        ax.add_patch(
            Circle((x, y), node_size, facecolor=_DATA_FACE, edgecolor="#333333",
                   linewidth=1.1, zorder=3)
        )
        ax.text(x, y, str(q), ha="center", va="center", fontsize=7, zorder=4)

    # --- logical-operator supports --------------------------------------
    if show_logicals and getattr(patch, "logical_ops", None):
        seen_labels = set()
        for i, op in enumerate(patch.logical_ops):
            color = _LOGICAL_COLORS[i % len(_LOGICAL_COLORS)]
            label = f"{op.get('type', '?')} logical {i}"
            for q in op["data_indices"]:
                if q in coords:
                    x, y = coords[q]
                    ax.add_patch(
                        Circle((x, y), node_size * 1.55, facecolor="none",
                               edgecolor=color, linewidth=1.6,
                               linestyle=(0, (1, 1)), zorder=2)
                    )
            if label not in seen_labels:
                ax.plot([], [], color=color, linewidth=1.6, linestyle=(0, (1, 1)),
                        label=label)
                seen_labels.add(label)
        if seen_labels:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                      ncol=min(len(seen_labels), 4), fontsize=7, frameon=False)

    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    pad = 4 * node_size
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or type(patch).__name__, fontsize=10)
    return ax


__all__ = ["draw_tanner_graph"]
