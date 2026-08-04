"""SWAP-network 90-degree patch rotation — the physically honest rotate_90.

Rotates a square d x d patch in place by physically moving the data-qubit
states with nearest-neighbour SWAPs (each SWAP = 3 CNOTs, all couplings
diagonal (+-1, +-1)).  The patch's weight-2 (lobe) POSITIONS are invariant
under the rotation, so the patch keeps its type (the user's
textbook/conjugate classification) while its logical operators swap
orientation — the counterpart to the Litinski boundary-deformation rotation,
which swaps orientation but flips the type.

Two layers, mutually asserted:

* SPEC (the user's four-edge rule): per ring, split the ring into four
  half-open edges — top (left->right, owns the top-left corner), right
  (top->bottom, owns top-right), bottom (right->left, owns bottom-right),
  left (bottom->up, owns bottom-left) — then swap element-wise top<->right,
  top<->bottom, top<->left.  This defines the target permutation (verified
  identical to the rotation formula dest(x,y) = (cx+(y-cy), cy-(x-cx))).

* EXECUTION (ring-shift hot-potato): every ring's contents advance one ring
  position per two SWAP layers (data -> hop ancilla, hop ancilla -> next
  data), all rings in parallel; ring at Chebyshev radius k shifts k times.
  Total depth 2(d-1) SWAP layers = 6(d-1) CNOT ticks.  Movement-optimal and
  congestion-free; the hop ancillas are the patch's own interior + lobe
  ancillas (junk left in them is reset by the next SE round's R layer).

The tracker is updated exactly via builder.apply_unitary_block (whole-
tableau symplectic conjugation), after which the patch is re-registered
with the rotated layout ((lr, tb, phase) -> (tb, lr, phase^1)); the first
post-rotation SE round's checks match the permuted tableau rows, giving
deterministic bridging detectors with no extra machinery.
"""
import stim

from .rule_based_patch import RuleBasedRectPatch
from .litinski_layouts import rect_boundary

_TWO_QUBIT_GATES = {"CX", "CNOT", "ZCX", "CZ", "SWAP", "XCZ", "XCX", "ZCZ",
                    "CY", "YCX", "ISWAP"}


def assert_nearest_neighbor(circuit: stim.Circuit, qubit_coords: dict,
                            chebyshev: int = 1):
    """Locality lint: every two-qubit gate must couple qubits within the
    given Chebyshev distance (the native diagonal couplings are (+-1,+-1)).
    Raises AssertionError with the offending gate otherwise."""
    for inst in circuit.flattened():
        if inst.name not in _TWO_QUBIT_GATES:
            continue
        ts = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        for a, b in zip(ts[0::2], ts[1::2]):
            ca, cb = qubit_coords[a], qubit_coords[b]
            dx, dy = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            assert max(dx, dy) <= chebyshev, (
                f"non-local 2q gate {inst.name} between {ca} and {cb} "
                f"(Chebyshev {max(dx, dy)} > {chebyshev})")


def _rings(center, d):
    """Ring paths (cyclic site lists in the content-movement direction:
    left edge up, top right, right down, bottom left), outermost first."""
    cx, cy = center
    rings = []
    for k in range(d - 1, 0, -2):
        left = [(cx - k, cy + y) for y in range(-k, k, 2)]
        top = [(cx + x, cy + k) for x in range(-k, k, 2)]
        right = [(cx + k, cy + y) for y in range(k, -k, -2)]
        bottom = [(cx + x, cy - k) for x in range(k, -k, -2)]
        rings.append((k, left + top + right + bottom))
    return rings


def spec_permutation(center, d):
    """The user's four-edge three-step rule, per ring: returns
    {source data coord: destination data coord} for every data qubit."""
    cx, cy = center
    perm = {(cx, cy): (cx, cy)}
    for k, _path in _rings(center, d):
        top = [(cx + x, cy + k) for x in range(-k, k, 2)]        # owns TL
        right = [(cx + k, cy + y) for y in range(k, -k, -2)]     # owns TR
        bottom = [(cx + x, cy - k) for x in range(k, -k, -2)]    # owns BR
        left = [(cx - k, cy + y) for y in range(-k, k, 2)]       # owns BL
        holder = {q: q for q in top + right + bottom + left}
        for other in (right, bottom, left):
            for t_site, o_site in zip(top, other):
                holder[t_site], holder[o_site] = holder[o_site], holder[t_site]
        for site, content in holder.items():
            perm[content] = site
    # cross-check against the rotation formula
    for (x, y), dst in perm.items():
        expect = (cx + (y - cy), cy - (x - cx))
        assert dst == expect, f"spec rule != rotation formula at {(x, y)}"
    return perm


def _hop_assignment(system, rings, center):
    """One hop ancilla per ring segment (site -> next site): a registered
    ancilla diagonal-adjacent to both endpoints, globally disjoint.
    Preference alternates inward/outward along the path (matches the
    resource pattern: interior ancillas and boundary lobes alternate —
    even segments hop through the inner band, odd through the outer)."""
    def registered(c):
        return c in system.index_map or (float(c[0]), float(c[1])) \
            in system.index_map

    segments = []
    for k, path in rings:
        for i, s in enumerate(path):
            t = path[(i + 1) % len(path)]
            mx, my = (s[0] + t[0]) // 2, (s[1] + t[1]) // 2
            if s[0] == t[0]:               # vertical segment
                cands = [(s[0] - 1, my), (s[0] + 1, my)]
            else:                          # horizontal segment
                cands = [(mx, s[1] - 1), (mx, s[1] + 1)]
            # inward candidate first (closer to the patch center)
            cands.sort(key=lambda c: max(abs(c[0] - center[0]),
                                         abs(c[1] - center[1])))
            if i % 2 == 1:
                cands = cands[::-1]
            segments.append(((s, t), [c for c in cands if registered(c)]))

    # the resource count is exactly tight (e.g. d=5: 24 segments, 24 usable
    # ancillas), so greedy can dead-end — use bipartite maximum matching
    match_anc = {}                          # ancilla -> segment index

    def try_assign(si, visited):
        for c in segments[si][1]:
            if c in visited:
                continue
            visited.add(c)
            if c not in match_anc or try_assign(match_anc[c], visited):
                match_anc[c] = si
                return True
        return False

    for si, (seg, cands) in enumerate(segments):
        assert try_assign(si, set()), (
            f"no hop-ancilla matching for ring segment {seg[0]}->{seg[1]}")
    return {segments[si][0]: c for c, si in match_anc.items()}


def build_swap_layers(system, center, d):
    """The full schedule: list of layers, each a list of (coord_a, coord_b)
    SWAP pairs (disjoint within a layer).  Step j (j = 1..d-1) runs rings
    with k >= j; each step = layer A (site -> hop) + layer B (hop -> next)."""
    rings = _rings(center, d)
    hops = _hop_assignment(system, rings, center)
    layers = []
    for j in range(1, d):
        a_layer, b_layer = [], []
        for k, path in rings:
            if k < j:
                continue
            for i, s in enumerate(path):
                t = path[(i + 1) % len(path)]
                h = hops[(s, t)]
                a_layer.append((s, h))
                b_layer.append((h, t))
        layers.append(a_layer)
        layers.append(b_layer)
    # disjointness within each layer
    for layer in layers:
        flat = [q for pair in layer for q in pair]
        assert len(flat) == len(set(flat)), "colliding SWAPs within a layer"
    # executed permutation == spec permutation on the data sites
    contents = {}
    for layer in layers:
        for a, b in layer:
            contents.setdefault(a, a)
            contents.setdefault(b, b)
            contents[a], contents[b] = contents[b], contents[a]
    final_pos = {src: site for site, src in contents.items()}
    for src, dst in spec_permutation(center, d).items():
        got = final_pos.get(src, src)
        assert got == dst, f"executed perm sends {src} to {got}, spec {dst}"
    return layers


def _index_of(system, coord):
    if coord in system.index_map:
        return system.index_map[coord]
    return system.index_map[(float(coord[0]), float(coord[1]))]


def swap_network_circuit(system, layers):
    """Emit the layers as CNOT triplets (SWAP = CX ab, CX ba, CX ab), one
    TICK per moment: 3 ticks per layer, 6(d-1) ticks total."""
    c = stim.Circuit()
    for layer in layers:
        pairs = [(_index_of(system, a), _index_of(system, b))
                 for a, b in layer]
        for direction in (0, 1, 0):
            args = []
            for a, b in pairs:
                args.extend([b, a] if direction else [a, b])
            c.append("CX", args)
            c.append("TICK")
    return c


def _detect_layout(system, name, d, origin):
    """(lr_type, tb_type, phase) of the live patch, read off its records."""
    ox, oy = origin
    lr = tb = None
    bulk_type = None
    for uid in sorted(system.active_stabilizer_indices):
        st = system.stabilizers[uid]
        if st.get('patch_name') != name:
            continue
        sx, sy = (int(round(st['syn_coord'][0])),
                  int(round(st['syn_coord'][1])))
        if len(st['data_indices']) == 2:
            if sx < ox or sx > ox + 2 * d - 2:
                lr = st['type']
            elif sy < oy or sy > oy + 2 * d - 2:
                tb = st['type']
        elif len(st['data_indices']) == 4 and bulk_type is None:
            bulk_type = (sx, sy, st['type'])
    assert lr and tb and bulk_type, f"could not read layout of '{name}'"
    sx, sy, t = bulk_type
    phase = 0 if t == ('X' if ((sx + sy) // 2) % 2 else 'Z') else 1
    return lr, tb, phase


def _prep_rotation(system, name):
    """Everything one patch's rotation needs: its SWAP layers and the
    rotated-layout re-registration data."""
    owner = system.index_to_owner_map
    data = sorted(tuple(int(round(v)) for v in system.qubit_coords[q])
                  for q in system.data_indices if owner.get(q) == name)
    d = int(round(len(data) ** 0.5))
    assert d * d == len(data) and d % 2 == 1, \
        f"'{name}' is not an odd square patch (|data|={len(data)})"
    ox, oy = data[0]
    assert data[-1] == (ox + 2 * d - 2, oy + 2 * d - 2), \
        f"'{name}' data footprint is not a d x d square"
    center = (ox + d - 1, oy + d - 1)
    lr, tb, phase = _detect_layout(system, name, d, (ox, oy))
    layers = build_swap_layers(system, center, d)
    return dict(name=name, layers=layers, d=d, origin=(ox, oy),
                lr=lr, tb=tb, phase=phase,
                footprint={q for layer in layers for pair in layer
                           for q in pair})


def rotate_patches_swap(system, builder, names):
    """Rotate several patches 90 degrees CONCURRENTLY.

    Each patch's SWAP network touches only its own data qubits and its own
    interior/lobe hop ancillas, so the per-patch layers are merged
    tick-for-tick: the total depth is that of ONE rotation (6(d-1) CNOT
    ticks for equal distances) no matter how many patches rotate.  Patches
    whose networks would touch a common qubit (adjacent patches sharing a
    lobe line) are placed in separate concurrent groups automatically."""
    preps = [_prep_rotation(system, name) for name in names]
    # greedy disjoint-footprint grouping (normally a single group)
    groups = []
    for p in preps:
        for g in groups:
            if all(p['footprint'].isdisjoint(q['footprint']) for q in g):
                g.append(p)
                break
        else:
            groups.append([p])
    from itertools import zip_longest
    for g in groups:
        merged = [sum((layer or [] for layer in step), [])
                  for step in zip_longest(*[p['layers'] for p in g])]
        network = swap_network_circuit(system, merged)
        assert_nearest_neighbor(network, system.qubit_coords)
        builder.apply_unitary_block(network)
        for p in g:
            # rotated layout: boundary types swap sides, bulk phase flips —
            # same weight-2 POSITIONS (type preserved), orientations swapped
            new_phase = p['phase'] ^ 1
            ox, oy = p['origin']
            rotated = RuleBasedRectPatch(
                distance_x=p['d'], distance_z=p['d'],
                forced=rect_boundary(p['d'], p['d'], p['tb'], p['lr'],
                                     new_phase),
                phase=new_phase)
            system.move_corners(p['name'], rotated, offset=(ox - 1, oy - 1))
    return [p['layers'] for p in preps]


def rotate_patch_swap(system, builder, name):
    """Physically rotate patch ``name`` by 90 degrees with a nearest-
    neighbour SWAP network, then re-register it with the rotated layout.
    Caller runs syndrome extraction before and after; the first
    post-rotation round's detectors bridge automatically (the tracker's
    tableau is conjugated exactly by the network)."""
    return rotate_patches_swap(system, builder, [name])[0]
