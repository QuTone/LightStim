"""Physical gross-code logical measurements from Tour de gross, App. A.3--A.5.

This module builds the auxiliary graph and the *physical* controlled-P circuit.
It deliberately emits neither DETECTOR nor OBSERVABLE_INCLUDE instructions.
Feed the preparation, measurement and gauge-out boundaries to CircuitBuilder
and SyndromeTracker.  In particular, prepare the Bell pair outside the syndrome
measurement block: its two individual readouts are gauge outcomes, whose parity
measures one check.

The graph, reduced cycle basis, and measurement protocol follow arXiv:2506.03094.
The deterministic scheduler below follows the five graph-coloring stages in
Appendix A.5, retaining the BB CNOT order.  It is a validated implementation of
those ordering constraints, not a claim to reproduce the authors' unpublished
edge-color assignment or their quoted 12-timestep optimized schedule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import stim

Site = tuple[str, int, int]
PauliSupport = dict[int, str]

# Share the paper convention with the patch/shift implementation.
from ..logical_presets import TOUR_DE_GROSS_CONFIG, TOUR_DE_GROSS_LOGICALS

_A = tuple(map(tuple, TOUR_DE_GROSS_CONFIG['A']))
_B = tuple(map(tuple, TOUR_DE_GROSS_CONFIG['B']))
_P = tuple(map(tuple, TOUR_DE_GROSS_LOGICALS['p']))
_Q = tuple(map(tuple, TOUR_DE_GROSS_LOGICALS['q']))
_R = tuple(map(tuple, TOUR_DE_GROSS_LOGICALS['r']))
_S = tuple(map(tuple, TOUR_DE_GROSS_LOGICALS['s']))


def _site(side: str, x: int, y: int) -> Site:
    return side, x % 12, y % 6


def _dual(q: Site) -> Site:
    """The paper's xy times transpose, including the L/R interchange."""
    return _site("R" if q[0] == "L" else "L", 1 - q[1], 1 - q[2])


def _support(p: stim.PauliString) -> PauliSupport:
    return {q: "IXYZ"[p[q]] for q in p.pauli_indices()}


def _pauli(support: Mapping[int, str], n: int) -> stim.PauliString:
    p = stim.PauliString(n)
    for q, basis in support.items():
        p[q] = basis
    return p


@dataclass(frozen=True)
class NativeCheck:
    """A check and its physical check-qubit implementation.

    ``parts`` partition ``pauli`` between one or two control ancillas. For two
    ancillas the parity of their X measurements is the check result.
    """

    name: str
    kind: str
    pauli: stim.PauliString
    ancillas: tuple[int, ...]
    parts: tuple[PauliSupport, ...]


@dataclass(frozen=True)
class FrameCorrection:
    """Apply pauli iff the listed gauge-out result bits have odd parity."""

    pauli: PauliSupport
    edge_result_indices: tuple[int, ...]


@dataclass(frozen=True)
class NativeVertexMetadata:
    index: int
    check_name: str
    label: Site
    aliases: tuple[Site, ...]
    halves: tuple[str, ...]
    data_pauli: stim.PauliString
    incident_edge_indices: tuple[int, ...]


@dataclass(frozen=True)
class NativeEdgeMetadata:
    index: int
    endpoints: tuple[Site, Site]
    qubit: int
    kind: str
    bb_check_owners: tuple[Site, ...]


@dataclass(frozen=True)
class NativeCycleMetadata:
    index: int
    check_name: str
    kind: str
    vertices: tuple[Site, ...]
    edge_indices: tuple[int, ...]
    edge_qubits: tuple[int, ...]


@dataclass(frozen=True)
class NativeConstructionMetadata:
    """Construction provenance; site labels are not hardware coordinates.

    Cycle vertex paths omit the repeated closing vertex. Edge owners list
    original BB checks related to an edge; the chosen Pauli measurement
    determines which of these checks actually gain support on that edge.
    """

    lpu_mode: str
    data_sites: Mapping[Site, int]
    bb_check_sites: Mapping[Site, int]
    original_bb_checks: Mapping[Site, stim.PauliString]
    vertices: tuple[NativeVertexMetadata, ...]
    edges: tuple[NativeEdgeMetadata, ...]
    cycles: tuple[NativeCycleMetadata, ...]
    merged_vertex: Site
    merged_aliases: tuple[Site, Site]
    gaugeout_root: Site
    gaugeout_paths: Mapping[Site, tuple[int, ...]]
    correction_vertices: tuple[Site, ...]


@dataclass(frozen=True)
class NativeMeasurement:
    p1: str
    p7: str
    num_qubits: int
    target_pauli: stim.PauliString
    graph_vertices: tuple[Site, ...]
    graph_edges: tuple[tuple[Site, Site], ...]
    edge_qubits: tuple[int, ...]
    all_lpu_qubits: tuple[int, ...]
    auxiliary_roles: Mapping[int, str]
    checks: tuple[NativeCheck, ...]
    round_reset_bases: Mapping[int, str]
    bell_pairs: tuple[tuple[int, int], ...]
    syndrome_qubits: tuple[int, ...]
    prepare_circuit: stim.Circuit
    bell_prepare_circuit: stim.Circuit
    syndrome_circuit: stim.Circuit
    gaugeout_circuit: stim.Circuit
    outcome_parity: tuple[int, ...]
    outcome_sign: int
    frame_corrections: tuple[FrameCorrection, ...]
    gate_layers: tuple[tuple[tuple[str, int, int], ...], ...]
    construction: NativeConstructionMetadata

    @property
    def round_circuit(self) -> stim.Circuit:
        """Standalone raw physical round; use separated phases with the tracker."""
        c = stim.Circuit()
        for basis in ("X", "Z"):
            qs = [q for q, b in self.round_reset_bases.items() if b == basis]
            if qs:
                c.append("RX" if basis == "X" else "R", qs)
        c.append("TICK", tag="SE_start")
        c += self.bell_prepare_circuit
        if len(self.bell_prepare_circuit):
            c.append("TICK")
        c += self.syndrome_circuit
        return c

    def validate(self) -> None:
        """Check commutation, target product, collisions and actual readout flows."""
        for a, b in combinations(self.checks, 2):
            if not a.pauli.commutes(b.pauli):
                raise ValueError(f"Noncommuting checks: {a.name}, {b.name}")
        product = stim.PauliString(self.num_qubits)
        for check in self.checks:
            if check.kind == "vertex":
                product *= check.pauli
        if product != self.target_pauli:
            raise ValueError("The vertex product is not the requested logical Pauli")
        for layer in self.gate_layers:
            used = [q for _, a, b in layer for q in (a, b)]
            if len(set(used)) != len(used):
                raise ValueError("Two scheduled gates use the same qubit")
        # Full signed back-propagation verifies even the Bell-pair parity.
        small_unitary = stim.Tableau.from_circuit(
            self.bell_prepare_circuit + self.syndrome_circuit,
            ignore_measurement=True,
        )
        padded = stim.Tableau(self.num_qubits)
        padded.append(small_unitary, list(range(len(small_unitary))))
        unitary = padded.inverse()
        for check in self.checks:
            output = stim.PauliString(self.num_qubits)
            for q in check.ancillas:
                output[q] = "X"
            measured = unitary(output)
            for q, basis in self.round_reset_bases.items():
                if measured[q] and "IXYZ"[measured[q]] != basis:
                    raise ValueError(f"Bad back-propagation for {check.name} at {q}")
                measured[q] = "I"
            if measured != check.pauli:
                raise ValueError(f"Circuit does not measure check {check.name}")


def _gross_checks(data: Mapping[Site, int], n: int):
    result = {}
    for basis in "XZ":
        for i in range(12):
            for j in range(6):
                if basis == "X":
                    sites = [_site("L", i + a, j + b) for a, b in _A]
                    sites += [_site("R", i + a, j + b) for a, b in _B]
                else:
                    sites = [_site("L", i - a, j - b) for a, b in _B]
                    sites += [_site("R", i - a, j - b) for a, b in _A]
                result[(basis, i, j)] = _pauli({data[q]: basis for q in sites}, n)
    return result


def _vertices(left: bool) -> tuple[Site, ...]:
    a, b = (_P, _Q) if left else (_R, _S)
    return tuple([_site("L", *p) for p in a] + [_site("R", *p) for p in b])


def _graph():
    """All fixed hardware incidences, before choosing a logical measurement."""
    lv, rv = _vertices(True), _vertices(False)
    # Each pair is selected by an original Z check, App. A.3. The associated
    # dual X check is determined from physical support, not guessed from labels.
    d = {_site(s, i, j): (0 if s == "L" else 72) + i * 6 + j
         for s in "LR" for i in range(12) for j in range(6)}
    checks = _gross_checks(d, 144)
    half_edges = []
    edge_checks = []
    for verts in (lv, rv):
        edges, owners = [], []
        for key, check in checks.items():
            if key[0] != "Z":
                continue
            overlap = [q for q in verts if check[d[q]]]
            if not overlap:
                continue
            if len(overlap) != 2:
                raise ValueError("The paper half graph needs paired check overlap")
            edge = tuple(sorted(overlap))
            if edge in edges:
                raise ValueError("Unexpected parallel half-graph edge")
            dual_pair = [_dual(q) for q in edge]
            xowners = [k for k, p in checks.items() if k[0] == "X"
                       and all(p[d[q]] for q in dual_pair)]
            if len(xowners) != 1:
                raise ValueError("Ambiguous dual edge check")
            edges.append(edge)
            owners.append((key, xowners[0]))
        if len(edges) != 18:
            raise ValueError("Each paper gross half graph has 18 edges")
        half_edges.append(edges)
        edge_checks.append(owners)
    merged = _site("R", 4, 0)
    other = _site("L", 9, 1)
    canon = lambda q: merged if q == other else q
    edges = [tuple(sorted((canon(a), canon(b))))
             for half in half_edges for a, b in half]
    # Eq. (39): columns are the 11 bridge edges.
    upper = (("L", 6, 5), ("L", 4, 2), ("R", 3, 5), ("L", 5, 4),
             ("L", 6, 1), ("R", 4, 2), ("L", 5, 0), ("L", 4, 0),
             ("R", 3, 2), ("R", 3, 1), ("R", 3, 0))
    lower = (("L", 11, 4), ("R", 8, 5), ("R", 8, 4), ("L", 0, 0),
             ("R", 1, 0), ("R", 4, 4), ("L", 8, 0), ("R", 9, 0),
             ("R", 0, 5), ("L", 3, 4), ("L", 1, 1))
    edges += [tuple(sorted((a, b))) for a, b in zip(upper, lower)]
    # Reduced cycles explicitly listed in A.3. The closing vertex is implicit.
    lcycles = [
        (("L", 4, 0), ("R", 3, 2), ("R", 3, 1), ("L", 5, 0)),
        (("R", 3, 5), ("R", 3, 0), ("L", 6, 5), ("L", 4, 2)),
        (("R", 3, 5), ("R", 3, 0), ("L", 6, 5), ("R", 4, 0), ("L", 5, 4)),
        (("L", 4, 0), ("R", 3, 2), ("L", 6, 1), ("L", 5, 4), ("R", 4, 0)),
        (("R", 3, 1), ("R", 3, 2), ("L", 6, 1), ("R", 4, 2), ("L", 5, 0)),
    ]
    rcycles = [
        (("L", 9, 1), ("R", 8, 4), ("R", 8, 5), ("L", 11, 4)),
        (("L", 1, 1), ("R", 1, 0), ("R", 4, 4), ("L", 3, 4)),
        (("L", 0, 0), ("R", 0, 5), ("R", 9, 0), ("L", 9, 1), ("R", 8, 4)),
    ]
    bcycles = [(upper[i], upper[i + 1], lower[i + 1], lower[i]) for i in range(10)]
    bcycles.append((merged, upper[0], lower[0]))
    cycles = []
    cycle_paths = []
    for path in lcycles + rcycles + bcycles:
        path = tuple(canon(q) for q in path)
        cycle_paths.append(path)
        cycles.append(tuple(edges.index(tuple(sorted((a, b))))
                            for a, b in zip(path, path[1:] + path[:1])))
    verts = tuple(dict.fromkeys(canon(q) for q in lv + rv))
    return (lv, rv, verts, tuple(edges), tuple(cycles), edge_checks, merged,
            canon, tuple(cycle_paths))


def _edge_colors(items):
    """Optimal deterministic bipartite edge coloring, including multiedges.

    Complete the graph to a regular bipartite multigraph with dummy edges,
    then remove a perfect matching for each color (Konig's theorem).
    Items are (left_vertex, right_vertex, payload).
    """
    if not items:
        return []
    ls = list(dict.fromkeys(a for a, _, _ in items))
    rs = list(dict.fromkeys(b for _, b, _ in items))
    n = max(len(ls), len(rs))
    li, ri = {a: i for i, a in enumerate(ls)}, {b: i for i, b in enumerate(rs)}
    pairs = {}
    dl, dr = [0] * n, [0] * n
    for a, b, payload in items:
        u, v = li[a], ri[b]
        pairs.setdefault((u, v), []).append(payload)
        dl[u] += 1
        dr[v] += 1
    degree = max(dl + dr)
    vr = 0
    for u in range(n):
        while dl[u] < degree:
            while dr[vr] == degree:
                vr += 1
            count = min(degree - dl[u], degree - dr[vr])
            pairs.setdefault((u, vr), []).extend([None] * count)
            dl[u] += count
            dr[vr] += count
    colors = []
    for _ in range(degree):
        matched = {}

        def augment(u, visited):
            for v in range(n):
                if v in visited or not pairs.get((u, v)):
                    continue
                visited.add(v)
                if v not in matched or augment(matched[v], visited):
                    matched[v] = u
                    return True
            return False

        for u in range(n):
            if not augment(u, set()):
                raise ValueError("Regular bipartite graph lacks perfect matching")
        layer = []
        for v, u in sorted(matched.items()):
            payload = pairs[(u, v)].pop(0)
            if payload is not None:
                layer.append(payload)
        if layer:
            colors.append(layer)
    return colors


def _schedule(checks, data, bb_ids, edge_set, bell_pairs):
    """Physical controlled-P gate layers satisfying A.5 ordering constraints."""
    # An event is (check index, control, data, local Pauli, phase, BB sublayer).
    events = []
    port = []
    vertex = []
    cycle = []
    extra = []
    for ci, ch in enumerate(checks):
        for anc, part in zip(ch.ancillas, ch.parts):
            for q, basis in sorted(part.items()):
                e = (ci, anc, q, basis, 0, 0)
                if ch.kind == "vertex":
                    (vertex if q in edge_set else port).append((anc, q, e))
                elif ch.kind == "cycle":
                    cycle.append((anc, q, e))
                elif q in edge_set:
                    extra.append((anc, q, e))
    for layer in _edge_colors(port):
        events += [(ci, a, q, p, 1, 0) for ci, a, q, p, _, _ in layer]
    # Fig. 4b / Bravyi et al. BB seven-CNOT-layer schedule.
    bb_schedule = [(("A", 0), None), (("A", 2), ("A", 1)),
                   (("B", 0), ("B", 1)), (("B", 1), ("B", 0)),
                   (("B", 2), ("B", 2)), (("A", 1), ("A", 0)),
                   (None, ("A", 2))]
    check_lookup = {anc: ci for ci, ch in enumerate(checks) for anc in ch.ancillas}
    for layer, (zaction, xaction) in enumerate(bb_schedule):
        for basis, action in (("Z", zaction), ("X", xaction)):
            if action is None:
                continue
            poly_name, k = action
            dx, dy = (_A if poly_name == "A" else _B)[k]
            side = ("R" if poly_name == "A" else "L") if basis == "Z" else (
                "L" if poly_name == "A" else "R")
            sign = -1 if basis == "Z" else 1
            for i in range(12):
                for j in range(6):
                    anc = bb_ids[basis, i, j]
                    q = data[_site(side, i + sign * dx, j + sign * dy)]
                    events.append((check_lookup[anc], anc, q, basis, 2, layer))
    for phase, group in ((3, vertex), (4, cycle), (5, extra)):
        for layer in _edge_colors(group):
            events += [(ci, a, q, p, phase, 0) for ci, a, q, p, _, _ in layer]
    # Preserve qubit order and the relative order at both anticommuting
    # overlaps. The stage order above is a valid sequential reference schedule.
    by_data = {}
    for idx, (ci, a, q, p, ph, bb) in enumerate(events):
        by_data.setdefault(q, []).append((idx, ci, p))
    anticommute_pairs = {}
    for q, at in by_data.items():
        for (ia, ca, pa), (ib, cb, pb) in combinations(at, 2):
            if ca != cb and pa != pb:
                key = tuple(sorted((ca, cb)))
                anticommute_pairs.setdefault(key, []).append((ia, ib))
    for key, pairs in anticommute_pairs.items():
        # Each pair represents a physical anticommuting overlap position.
        # The first check must be the same at every position.
        firsts = {events[a][0] for a, b in pairs}
        if len(firsts) != 1 or len(pairs) % 2:
            raise ValueError(f"Invalid sequential check ordering {key}")
        # At each overlap the local qubit order below preserves this sign.
        # No cross-position precedence is needed (or generally compatible
        # with the interleaved BB schedule).
    times, last = [], {}
    bell_ancillas = {q for pair in bell_pairs for q in pair}
    bb_start = None
    for idx, (ci, anc, q, p, phase, bb_layer) in enumerate(events):
        t = max(last.get(anc, -1), last.get(q, -1)) + 1
        t = max(t, 1 if anc in bell_ancillas else 0)
        if phase == 2:
            if bb_start is None:
                # BB internal gates begin after port gates; LPU-internal gates
                # can still overlap this fixed seven-layer BB schedule.
                bb_start = max(times, default=-1) + 1
            t = max(t, bb_start + bb_layer)
            if t != bb_start + bb_layer:
                raise ValueError("BB fixed schedule conflicts with precedences")
        times.append(t)
        last[anc] = last[q] = t
    layers = [[] for _ in range(max(times, default=-1) + 1)]
    for event, t in zip(events, times):
        ci, a, q, p, phase, bb = event
        layers[t].append(({"X": "CX", "Y": "CY", "Z": "CZ"}[p], a, q))
    return tuple(tuple(layer) for layer in layers)


def build_native_measurement(
    p1: str,
    p7: str,
    *,
    data_index: Mapping[Site, int] | None = None,
    bb_check_indices: Mapping[Site, int] | None = None,
    ancilla_start: int = 288,
    validate: bool = True,
) -> NativeMeasurement:
    """Construct one of the 15 native gross-code surgery measurements.

    Logical numbers are the paper's 1 and 7, not zero-based software indices.
    ``data_index`` keys are ("L"|"R", x, y), 0<=x<12, 0<=y<6;
    ``bb_check_indices`` uses ("X"|"Z", x, y). Defaults place the 144
    data qubits before 144 BB check ancillas.  Ninety fixed LPU qubits start
    at ``ancilla_start``; subsets are activated for half-LPU measurements.
    """
    if p1 not in "IXYZ" or len(p1) != 1 or p7 not in "IXYZ" or len(p7) != 1:
        raise ValueError("p1 and p7 must be I, X, Y or Z")
    if p1 == p7 == "I":
        raise ValueError("Identity is not a native measurement instruction")
    data = dict(data_index) if data_index is not None else {
        (s, i, j): (0 if s == "L" else 72) + i * 6 + j
        for s in "LR" for i in range(12) for j in range(6)}
    bb_ids = dict(bb_check_indices) if bb_check_indices is not None else {
        (s, i, j): (144 if s == "X" else 216) + i * 6 + j
        for s in "XZ" for i in range(12) for j in range(6)}
    if set(data) != {(s, i, j) for s in "LR" for i in range(12) for j in range(6)}:
        raise ValueError("data_index must map all 144 paper data sites")
    if set(bb_ids) != {(s, i, j) for s in "XZ" for i in range(12) for j in range(6)}:
        raise ValueError("bb_check_indices must map all 144 paper check sites")
    all_lpu = tuple(range(ancilla_start, ancilla_start + 90))
    used = list(data.values()) + list(bb_ids.values()) + list(all_lpu)
    if min(used) < 0 or len(set(used)) != len(used):
        raise ValueError("Data, BB checks and LPU indices must be distinct nonnegative integers")
    n = max(used) + 1
    lv, rv, vertices, edges, cycles, edge_owners, merged, canon, cycle_paths = _graph()
    x1, z1 = p1 in "XY", p1 in "ZY"
    x7, z7 = p7 in "XY", p7 in "ZY"
    active_l, active_r = x1 or z7, x7 or z1
    if active_l and active_r:
        ve = list(range(23))
        ee = list(range(47))
        cc = list(range(19))
    elif active_l:
        ve = [vertices.index(canon(q)) for q in lv]
        ee, cc = list(range(18)), list(range(5))
    else:
        ve = [vertices.index(canon(q)) for q in rv]
        ee, cc = list(range(18, 36)), list(range(5, 8))
    edge_ids = {e: ancilla_start + e for e in range(47)}
    vertex_ids = {v: ancilla_start + 47 + v for v in range(23)}
    cycle_ids = {c: ancilla_start + 70 + c for c in range(19)}
    bell_other = ancilla_start + 89
    roles = {q: "data" for q in all_lpu[:47]}
    roles.update({q: "syndrome_x" for q in all_lpu[47:70]})
    roles.update({q: "syndrome_z" for q in all_lpu[70:89]})
    roles[bell_other] = "syndrome_x"
    ports = {v: stim.PauliString(n) for v in ve}
    for enabled, verts, basis, dual in (
        (x1, lv, "X", False), (z1, rv, "Z", True),
        (x7, rv, "X", False), (z7, lv, "Z", True),
    ):
        if enabled:
            for q in verts:
                v = vertices.index(canon(q))
                physical = data[_dual(q) if dual else q]
                ports[v] *= _pauli({physical: basis}, n)
    # Y_j = i X_j Z_j. Both anticommuting physical intersections occur at
    # the merged vertex, so assign both i phases there.
    ports[vertices.index(merged)] *= 1j ** ((p1 == "Y") + (p7 == "Y"))
    target = stim.PauliString(n)
    for port in ports.values():
        if port.sign not in (1, -1):
            raise ValueError("A vertex port is not Hermitian")
        target *= port
    bb = _gross_checks(data, n)
    original_bb = {key: p.copy() for key, p in bb.items()}
    for half, enabled_x, enabled_z in ((0, x1, z7), (1, x7, z1)):
        for local, (zowner, xowner) in enumerate(edge_owners[half]):
            edge = 18 * half + local
            if edge not in ee:
                continue
            if enabled_x:
                bb[zowner][edge_ids[edge]] = "Z"
            if enabled_z:
                bb[xowner][edge_ids[edge]] = "Z"
    checks = [NativeCheck(f"{key[0]}_{key[1]}_{key[2]}", "bb", p,
                          (bb_ids[key],), (_support(p),)) for key, p in bb.items()]
    bell_pairs = []
    for v in ve:
        p = ports[v].copy()
        incident = [e for e in ee if vertices[v] in edges[e]]
        for e in incident:
            p[edge_ids[e]] = "X"
        anc = vertex_ids[v]
        if vertices[v] == merged:
            # Figure 5 Bell check: halves use their own incident graph edges
            # and one physical overlap site, so neither ancilla has degree 8.
            left = {edge_ids[e]: "X" for e in incident if e < 18}
            right = {edge_ids[e]: "X" for e in incident if e >= 18}
            for q, basis in _support(ports[v]).items():
                (left if q == data[merged] else right)[q] = basis
            if active_l and active_r:
                ancillas, parts = (anc, bell_other), (left, right)
                bell_pairs.append(ancillas)
            else:
                ancillas, parts = (anc,), (_support(p),)
        else:
            ancillas, parts = (anc,), (_support(p),)
        checks.append(NativeCheck(f"v_{v}", "vertex", p, ancillas, parts))
    for c in cc:
        p = _pauli({edge_ids[e]: "Z" for e in cycles[c]}, n)
        checks.append(NativeCheck(f"cycle_{c}", "cycle", p,
                                  (cycle_ids[c],), (_support(p),)))
    # Controlled-P primitives below measure +Paulis. The chosen paper ports
    # have positive signs; fail visibly if a future basis convention changes.
    if any(check.pauli.sign != 1 for check in checks):
        raise ValueError("This circuit requires positive Hermitian check representatives")
    reset = {q: "X" for check in checks for q in check.ancillas}
    for a, b in bell_pairs:
        reset[b] = "Z"
    syn_qubits = tuple(reset)
    bell = stim.Circuit()
    for a, b in bell_pairs:
        bell.append("CX", [a, b])
    layers = _schedule(checks, data, bb_ids, {edge_ids[e] for e in ee}, bell_pairs)
    syndrome = stim.Circuit()
    for layer in layers:
        for gate in ("CX", "CY", "CZ"):
            targets = [q for name, a, b in layer if name == gate for q in (a, b)]
            if targets:
                syndrome.append(gate, targets)
        syndrome.append("TICK")
    syndrome.append("MX", syn_qubits)
    prepare = stim.Circuit()
    active_edges = tuple(edge_ids[e] for e in ee)
    prepare.append("R", active_edges)
    gaugeout = stim.Circuit()
    gaugeout.append("M", active_edges)
    outcome = tuple(syn_qubits.index(q) for ch in checks if ch.kind == "vertex"
                    for q in ch.ancillas)
    # Deterministic spanning tree specifies App. A.4's otherwise arbitrary
    # root and paths. Each edge index below indexes gaugeout's measurement list.
    adjacency = {v: [] for v in ve}
    for edge_result, e in enumerate(ee):
        a, b = [vertices.index(q) for q in edges[e]]
        adjacency[a].append((b, edge_result))
        adjacency[b].append((a, edge_result))
    root = min(ve)
    paths = {root: ()}
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b, e in sorted(adjacency[a]):
            if b not in paths:
                paths[b] = paths[a] + (e,)
                queue.append(b)
    if len(paths) != len(ve):
        raise ValueError("Active LPU graph is disconnected")
    corrections = tuple(FrameCorrection(_support(ports[v]), paths[v])
                        for v in sorted(ve) if v != root and ports[v].weight)
    construction = NativeConstructionMetadata(
        lpu_mode="full" if active_l and active_r else ("left" if active_l else "right"),
        data_sites=dict(data),
        bb_check_sites=dict(bb_ids),
        original_bb_checks=original_bb,
        vertices=tuple(NativeVertexMetadata(
            index=v,
            check_name=f"v_{v}",
            label=vertices[v],
            aliases=tuple(dict.fromkeys(q for q in lv + rv if canon(q) == vertices[v])),
            halves=tuple(name for name, half in (("left", lv), ("right", rv))
                         if any(canon(q) == vertices[v] for q in half)),
            data_pauli=ports[v].copy(),
            incident_edge_indices=tuple(e for e in ee if vertices[v] in edges[e]),
        ) for v in ve),
        edges=tuple(NativeEdgeMetadata(
            index=e, endpoints=edges[e], qubit=edge_ids[e],
            kind="left" if e < 18 else ("right" if e < 36 else "bridge"),
            bb_check_owners=tuple(edge_owners[e // 18][e % 18]) if e < 36 else (),
        ) for e in ee),
        cycles=tuple(NativeCycleMetadata(
            index=c, check_name=f"cycle_{c}",
            kind="left" if c < 5 else ("right" if c < 8 else "bridge"),
            vertices=cycle_paths[c], edge_indices=cycles[c],
            edge_qubits=tuple(edge_ids[e] for e in cycles[c]),
        ) for c in cc),
        merged_vertex=merged,
        merged_aliases=(merged, _site("L", 9, 1)),
        gaugeout_root=vertices[root],
        gaugeout_paths={vertices[v]: tuple(ee[i] for i in paths[v]) for v in sorted(ve)},
        correction_vertices=tuple(vertices[v] for v in sorted(ve) if v != root and ports[v].weight),
    )
    result = NativeMeasurement(
        p1, p7, n, target, tuple(vertices[v] for v in ve),
        tuple(edges[e] for e in ee), active_edges, all_lpu, roles,
        tuple(checks), reset, tuple(bell_pairs), syn_qubits,
        prepare, bell, syndrome, gaugeout, outcome, 1, corrections, layers,
        construction,
    )
    if validate:
        result.validate()
    return result
