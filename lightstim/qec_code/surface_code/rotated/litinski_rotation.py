"""The full Litinski patch rotation (Fig 45 steps 4→9) as a single entry point.

Five deformation steps, each followed by ``d`` rounds of the DIAGONAL syndrome
extraction schedule (every N/Z zigzag orientation caps the deformed patch's
distance — see ``diagonal_se``):

1. GROW through the Z boundary onto the (2d−1)×d bar; new region |0⟩
   (the basis of the logical being extended);
2. MOVE CORNERS: the long edge flips X→Z (the same-type corner wrap slides
   from the bar's far corner to the home-side corner);
3. SHRINK: the home-side d−1 rows are measured out in X; the patch is the bar's
   far d×d — ROTATED, but displaced from its home tile;
4. GROW back through the (now) X boundary; new region |+⟩ — Fig 40c's plain
   qubit movement, convention-preserving (panel 8);
5. SHRINK: the far-side d−1 rows are measured out in X; the patch is back on
   its HOME tile, rotated (panel 9 — the same local layout as panel 7).

``direction`` selects the growth direction in coordinate terms: ``'-y'``
(default, the internal y-up convention of the tests) or ``'+y'`` (grows toward
larger y — what a stim-rendered frame shows as downward).  For ``'+y'`` every
stage geometry is the y-flipped COMPLETED layout passed fully-forced, because
the rule sweep's lexicographic order is not mirror-symmetric.

End-to-end verification (tests/test_litinski_rotation.py): p=0 determinism,
graphlike distance d for both memory bases at d = 3 and 5, the final footprint
equal to the starting one, and the final logicals orientation-swapped.
"""
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.qec_code.surface_code.rotated.litinski_layouts import (
    grown_w2_full, moved_w2, shrunk_w2, bar2_w2, flip_w2, start_w2,
    transpose_w2)


def _litinski_stage_gen(system, builder, name, rounds, direction, summaries):
    """Generator form of the 5-stage protocol: performs one stage's system
    mutation + its init/readout per ``next()`` and then YIELDS — the caller
    runs the (system-wide) SE rounds between stages, so several patches'
    protocols can advance in LOCKSTEP sharing the same SE rounds."""
    if direction not in ('-y', '+y', '-x', '+x'):
        raise ValueError(
            f"direction must be one of '-y', '+y', '-x', '+x', got {direction!r}")
    # ±x variants = the paper protocol conjugated by the transpose isometry
    # (x <-> y).  ptype is transpose-symmetric, so ALL canonical-frame logic
    # below applies verbatim: coordinates entering from the system are
    # transposed into the canonical frame, offsets/geometries leaving to the
    # system are transposed back (the map is an involution).
    ax_x = direction[1] == 'x'

    def _T(c):
        return (c[1], c[0]) if ax_x else (c[0], c[1])
    owner = system.index_to_owner_map
    retired = (system._tracker.retired_qubits
               if system._tracker is not None else frozenset())
    data = sorted(q for q in system.data_indices
                  if owner.get(q) == name and q not in retired)
    coords = [_T(system.qubit_coords[q]) for q in data]
    xs = sorted({int(round(x)) for x, _ in coords})
    ys = sorted({int(round(y)) for _, y in coords})
    d = len(xs)
    if len(ys) != d or len(data) != d * d:
        raise ValueError(f"rotate_patch_litinski({name!r}): patch is not a d×d "
                         f"({len(xs)}×{len(ys)}, {len(data)} data qubits)")
    if rounds is None:
        rounds = d
    summaries[name] = None   # filled at the end
    dy = 2 * (d - 1)
    Ybar = 2 * (2 * d - 1)
    ox, oy_tile = xs[0] - 1, ys[0] - 1          # home-tile local origin
    flip = direction[0] == '+'
    bar = (ox, oy_tile) if flip else (ox, oy_tile - dy)
    far_tile = (ox, oy_tile + dy) if flip else bar

    # checkerboard phase of the bar frame, read off one live bulk check
    bulk = next(s for u in system.active_stabilizer_indices
                for s in [system.stabilizers[u]]
                if s.get('patch_name') == name and len(s['data_indices']) == 4)

    def _phases_and_start(flip_):
        """(phase, p_gen, expected start w2 layout in global coords) for a
        growth direction.  A y-flip over the bar (2d-1 rows, odd) or the tile
        (d rows, odd) toggles the checkerboard generation phase."""
        bar_ = (ox, oy_tile) if flip_ else (ox, oy_tile - dy)
        scx, scy = (int(round(v - o))
                    for v, o in zip(_T(bulk['syn_coord']), bar_))
        ph = ((1 if bulk['type'] == 'X' else 0) - (scx + scy) // 2) % 2
        pg = (ph + 1) % 2 if flip_ else ph
        w2 = start_w2(d, pg)
        if flip_:
            w2 = flip_w2(w2, 2 * d)
        exp = {(sx + ox, sy + oy_tile): t for (sx, sy), (t, _) in w2.items()}
        return ph, pg, exp

    # PRECONDITION — the protocol only works from the paper-q2 start layout
    # (X̄ horizontal / Z̄ vertical) with the direction matching the checkerboard
    # phase.  Anything else used to fail SILENTLY (an X̄-vertical patch burns
    # 5d rounds and does not move) or blow up deep inside shrink_patch.
    live_w2 = {}
    for u in system.active_stabilizer_indices:
        st = system.stabilizers[u]
        if st.get('patch_name') == name and len(st['data_indices']) == 2:
            live_w2[tuple(int(round(v))
                          for v in _T(st['syn_coord']))] = st['type']
    phase, p_gen, expected = _phases_and_start(flip)
    if live_w2 != expected:
        raise ValueError(
            f"rotate_patch_litinski({name!r}): patch's weight-2 layout does "
            f"not match the {direction!r} start convention — got "
            f"{sorted(live_w2.items())}, needed {sorted(expected.items())}.  "
            f"±y directions need the paper-q2 start (X̄ horizontal); ±x "
            f"directions need its transpose (X̄ vertical)")
    # the growth direction is FIXED by the checkerboard phase, not a free
    # choice (the two directions' start layouts are position-identical, so
    # only the phase tells them apart; the mismatched direction breaks the
    # shrink detection-coverage handoff deep inside the protocol)
    legal = ('-' if phase == 0 else '+') + direction[1]
    if direction != legal:
        raise ValueError(
            f"rotate_patch_litinski({name!r}): checkerboard phase {phase} "
            f"requires direction {legal!r}, got {direction!r}")

    def bar_geom(w2):
        if flip:
            w2 = flip_w2(w2, Ybar)
        if ax_x:
            return RuleBasedRectPatch(distance_x=d, distance_z=2 * d - 1,
                                      forced=transpose_w2(w2), phase=phase)
        return RuleBasedRectPatch(distance_x=2 * d - 1, distance_z=d,
                                  forced=w2, phase=phase)

    def tile_geom(w2):
        if flip:
            w2 = flip_w2(w2, 2 * d)
        if ax_x:
            w2 = transpose_w2(w2)
        return RuleBasedRectPatch(distance_x=d, distance_z=d,
                                  forced=w2, phase=phase)

    def local_y(q):
        return int(round(_T(system.qubit_coords[q])[1] - bar[1]))

    def home_side(q):
        return local_y(q) < dy if flip else local_y(q) > 2 * d - 1

    def far_side(q):
        return local_y(q) > 2 * d - 1 if flip else local_y(q) < dy

    def patch_data():
        return [q for q in sorted(system.data_indices) if owner.get(q) == name
                and (system._tracker is None
                     or q not in system._tracker.retired_qubits)]

    summary = {'d': d, 'rounds_per_step': rounds, 'direction': direction}

    # ---- 1. GROW through the Z boundary (|0⟩ region) ------------------------------
    new_data, _ = system.grow_patch(name, bar_geom(grown_w2_full(d, p_gen)),
                                    offset=_T(bar))
    builder.initialize({q: 'Z' for q in new_data}, n=system.num_qubits)
    yield
    summary['grow_new'] = len(new_data)

    # ---- 2. MOVE CORNERS ----------------------------------------------------------
    system.move_corners(name, bar_geom(moved_w2(d, p_gen)), offset=_T(bar))
    yield

    # ---- 3. SHRINK to the bar's far d×d (X readout of the home-side rows) ---------
    out1 = [q for q in patch_data() if home_side(q)]
    builder.apply_data_readout(final_measurements={q: 'X' for q in out1})
    system.shrink_patch(name, tile_geom(shrunk_w2(d, p_gen)), offset=_T(far_tile))
    yield
    summary['shrink1_removed'] = len(out1)

    # ---- 4. GROW back through the X boundary (|+⟩ region) -------------------------
    new_data, _ = system.grow_patch(name, bar_geom(bar2_w2(d, p_gen)), offset=_T(bar))
    builder.initialize({q: 'X' for q in new_data}, n=system.num_qubits)
    yield
    summary['regrow_new'] = len(new_data)

    # ---- 5. SHRINK back onto the home tile (X readout of the far-side rows) -------
    out2 = [q for q in patch_data() if far_side(q)]
    builder.apply_data_readout(final_measurements={q: 'X' for q in out2})
    system.shrink_patch(name, tile_geom(shrunk_w2(d, p_gen)),
                        offset=_T((ox, oy_tile)))
    yield
    summary['shrink2_removed'] = len(out2)
    summaries[name] = summary


def rotate_patches_litinski(system, builder, names, rounds=None,
                            directions=None):
    """Rotate several patches via the Litinski protocol CONCURRENTLY.

    The five deformation stages advance in LOCKSTEP: per stage every
    patch's system mutation (+ its init/readout) is applied, then the
    system-wide diagonal SE rounds run ONCE — total 5*rounds SE rounds no
    matter how many patches rotate.  Patches whose (home tile + bar)
    footprints overlap are placed in sequential groups automatically.
    ``directions``: {name: '-y'|'+y'|'-x'|'+x'} (default '-y' each; ±y needs
    the paper-q2 X̄-horizontal start, ±x its transpose, X̄ vertical; the sign
    is fixed by the live checkerboard phase).  Returns {name: summary}."""
    directions = directions or {}
    owner = system.index_to_owner_map
    # retired data qubits linger in data_indices/owner map (a previous
    # rotation's bar, an absorbed corridor) — footprints must see LIVE only,
    # same filter as _litinski_stage_gen
    retired = (system._tracker.retired_qubits
               if system._tracker is not None else frozenset())

    def _footprint(name):
        direction = directions.get(name, '-y')
        ax_x = direction[1] == 'x'

        def _T(c):
            return (c[1], c[0]) if ax_x else (c[0], c[1])
        data = sorted(tuple(int(round(v)) for v in _T(system.qubit_coords[q]))
                      for q in system.data_indices
                      if owner.get(q) == name and q not in retired)
        d = int(round(len(data) ** 0.5))
        ox, oy = data[0]
        dy = 2 * (d - 1)
        flip = direction[0] == '+'
        y0 = oy - 1 if flip else oy - 1 - dy
        # the bar spans (2d-1) rows of the coarse frame in the growth
        # direction plus the home tile (canonical frame; mapped back below)
        return {_T((x, y)) for x in range(ox - 1, ox + 2 * d)
                for y in range(y0, y0 + 2 * (2 * d - 1) + 1)}, d

    fps = {nm: _footprint(nm) for nm in names}
    ds = {fps[nm][1] for nm in names}
    assert len(ds) == 1, f"concurrent litinski needs equal distances, got {ds}"
    n_rounds = rounds if rounds is not None else ds.pop()

    groups = []
    for nm in names:
        for g in groups:
            if all(fps[nm][0].isdisjoint(fps[o][0]) for o in g):
                g.append(nm)
                break
        else:
            groups.append([nm])

    summaries = {}
    for g in groups:
        gens = [_litinski_stage_gen(system, builder, nm, n_rounds,
                                    directions.get(nm, '-y'), summaries)
                for nm in g]
        for _stage in range(5):
            for gen in gens:
                next(gen)
            builder.apply_syndrome_extraction(
                DiagonalSurfaceCodeExtractionBlock(system).circuit,
                rounds=n_rounds)
        for gen in gens:            # exhaust (fills summaries)
            for _ in gen:
                pass
    return summaries


def rotate_patch_litinski(system, builder, name, rounds=None, direction='-y'):
    """Rotate the LIVE patch ``name`` 90° via the paper protocol, growing into
    the free space in ``direction`` ('-y'/'+y' from an X̄-horizontal start,
    '-x'/'+x' from an X̄-vertical start — the transpose-conjugated protocol;
    the sign is fixed by the live checkerboard phase).  ``rounds`` defaults
    to d per step.  Returns a summary dict."""
    return rotate_patches_litinski(
        system, builder, [name], rounds=rounds,
        directions={name: direction})[name]
