"""Flow-solved processing of ancilla-relay rounds.

A relay round hands stabilizer checks between ancillas mid-round (e.g. the
Y-basis transition round of arXiv:2302.07395, Fig. 5/7): individual
measurements are not data-only Pauli observables — only GF(2) COMBINATIONS
of the round's measurements are. The standard SE path's per-measurement
contract (each back-propagated measurement touches other ancillas only in
their reset basis) is unsound there, so this module solves the round's
stabilizer flows with stim.Circuit.flow_generators() and merges them with
the tracker's record-augmented account book, emulating WriteBack semantics:

  1. CANONICAL FRESH ROWS — for every registered target stabilizer S' of the
     post-round configuration, solve   1 -> S' (xor) rec[..]   over the flow
     group (allowing pinned input rows on the left): the new tracked row is
     (S', fresh records).
  2. PER-INPUT-ROW ACCOUNTING — for every tracked input row (S, R), solve
     S -> q (xor) rec[..]; if the image q is spanned by the fresh rows the
     relation closes: DETECTOR = R (xor) rec (xor) fresh pinning. Otherwise
     the row is transported AS IS (no basis blending — surviving free-DOF
     representatives such as a logical coset row keep pristine records).
  3. COMPLETION — leftover flow directions (e.g. the Y_L preparation flow of
     the init transition round) become additional rows with their own
     records, ready for tracker.declare_logical.

Global signs of flows are ignored: stim's detector/observable sampling is
relative to the noiseless reference, matching the tracker's convention of
storing parity relations without a sign column.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import stim

from ..utils.linear_algebra import solve_linear_decomposition

_MEAS_NAMES = {"M", "MX", "MY"}


def measured_qubits_in_order(chunk: stim.Circuit) -> List[int]:
    """Global qubit index of every measurement in the chunk, in record order."""
    out: List[int] = []
    for inst in chunk.flattened():
        if inst.name in _MEAS_NAMES:
            out.extend(t.value for t in inst.targets_copy())
    return out


def _pauli_to_vec(ps: stim.PauliString, n: int) -> np.ndarray:
    xs, zs = ps.to_numpy()
    v = np.zeros(2 * n, dtype=np.uint8)
    k = min(len(xs), n)
    v[:k] = xs[:k].astype(np.uint8)
    v[n:n + k] = zs[:k].astype(np.uint8)
    return v


def _intersection_basis(A: np.ndarray, B: np.ndarray,
                        preferred: Optional[np.ndarray] = None) -> np.ndarray:
    """Basis of rowspan(A) ∩ rowspan(B) over GF(2) (Zassenhaus).

    ``preferred`` rows that lie in the intersection are put FIRST in the
    returned basis (canonical representatives, e.g. registered stabilizers),
    with the Zassenhaus remainder completing the span.
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((0, A.shape[1] if A.size else B.shape[1]),
                        dtype=np.uint8)
    w = A.shape[1]
    top = np.hstack([A, A])
    bot = np.hstack([B, np.zeros_like(B)])
    m = np.vstack([top, bot]).astype(np.uint8)
    # RREF on the left block; rows whose left block vanishes carry an
    # intersection element in the right block.
    rank = 0
    for col in range(w):
        piv = None
        for r in range(rank, m.shape[0]):
            if m[r, col]:
                piv = r
                break
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
        if rank == m.shape[0]:
            break
    inter = [row[w:] for row in m if not row[:w].any() and row[w:].any()]
    W = (np.array(inter, dtype=np.uint8) if inter
         else np.zeros((0, w), dtype=np.uint8))
    if preferred is None or preferred.shape[0] == 0 or W.shape[0] == 0:
        return W
    # reorder: preferred members of the intersection first
    chosen: List[np.ndarray] = []
    for p in preferred:
        cA = solve_linear_decomposition(basis=A, targets=p.reshape(1, -1),
                                        reduce_weight=False)[1][0]
        cB = solve_linear_decomposition(basis=B, targets=p.reshape(1, -1),
                                        reduce_weight=False)[1][0]
        if cA and cB:
            cur = (np.array(chosen, dtype=np.uint8) if chosen
                   else np.zeros((0, w), dtype=np.uint8))
            if _rank(np.vstack([cur, p.reshape(1, -1)])) > _rank(cur):
                chosen.append(p.astype(np.uint8))
    for row in W:
        cur = (np.array(chosen, dtype=np.uint8) if chosen
               else np.zeros((0, w), dtype=np.uint8))
        if _rank(np.vstack([cur, row.reshape(1, -1)])) > _rank(cur):
            chosen.append(row)
    return np.array(chosen, dtype=np.uint8)


def _left_nullspace_basis(mat: np.ndarray) -> np.ndarray:
    """Basis of {y : y @ mat = 0} over GF(2), via RREF on [mat | I]."""
    g = mat.shape[0]
    if g == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    aug = np.hstack([mat.copy().astype(np.uint8),
                     np.eye(g, dtype=np.uint8)])
    w = mat.shape[1]
    kept: List[np.ndarray] = []
    pivots: List[int] = []
    out: List[np.ndarray] = []
    for row in aug:
        row = row.copy()
        for prow, pcol in zip(kept, pivots):
            if row[pcol]:
                row ^= prow
        nz = np.flatnonzero(row[:w])
        if nz.size == 0:
            out.append(row[w:])
        else:
            kept.append(row)
            pivots.append(int(nz[0]))
    if out:
        return np.array(out, dtype=np.uint8)
    return np.zeros((0, g), dtype=np.uint8)


def _rank(mat: np.ndarray) -> int:
    if mat.shape[0] == 0:
        return 0
    m = mat.copy()
    rank = 0
    for col in range(m.shape[1]):
        piv = None
        for r in range(rank, m.shape[0]):
            if m[r, col]:
                piv = r
                break
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
        if rank == m.shape[0]:
            break
    return rank


@dataclass
class RelayResult:
    measured_qubits: List[int]
    # each detector: (absolute record set, global qubit index used for coords)
    detectors: List[Tuple[Set[int], int]]
    new_stab_matrix: np.ndarray = None
    new_stab_records: List[List[int]] = field(default_factory=list)
    new_log_matrix: np.ndarray = None
    new_log_records: List[List[int]] = field(default_factory=list)
    # logicals MEASURED OUT by this chunk (transport flow ends at identity):
    # (original logical row index, absolute record set of the observable —
    # the row's banked seed records XOR the chunk's measuring records)
    # absorbed-relation ledger, transported exactly like logical rows:
    new_absorbed_matrix: np.ndarray = None
    new_absorbed_records: List[List[int]] = field(default_factory=list)
    # absorbed relations MEASURED OUT by this chunk (resolved: their value
    # is the parity of banked seed records XOR the measuring records):
    measured_absorbed: List[Tuple[int, List[int]]] = field(
        default_factory=list)
    measured_logicals: List[Tuple[int, List[int]]] = field(
        default_factory=list)


def solve_relay_round(chunk: stim.Circuit,
                      num_qubits: int,
                      stab_matrix: np.ndarray,
                      stab_records: List[List[int]],
                      log_matrix: np.ndarray,
                      log_records: List[List[int]],
                      base_record: int,
                      canonical_stabs: Optional[np.ndarray] = None,
                      absorbed_matrix: Optional[np.ndarray] = None,
                      absorbed_records: Optional[List[List[int]]] = None
                      ) -> RelayResult:
    n = num_qubits
    measured = measured_qubits_in_order(chunk)
    n_meas = len(measured)
    n_in = stab_matrix.shape[0]

    # --- 0. chunk flows over the full system width -----------------------
    p_rows: List[np.ndarray] = []
    q_rows: List[np.ndarray] = []
    m_sets: List[Set[int]] = []
    for fl in chunk.flow_generators():
        p_rows.append(_pauli_to_vec(fl.input_copy(), n))
        q_rows.append(_pauli_to_vec(fl.output_copy(), n))
        m_sets.append(set(fl.measurements_copy()))
    for q in range(chunk.num_qubits, n):        # untouched system qubits
        for half in (q, n + q):
            v = np.zeros(2 * n, dtype=np.uint8)
            v[half] = 1
            p_rows.append(v.copy())
            q_rows.append(v.copy())
            m_sets.append(set())
    P = np.array(p_rows, dtype=np.uint8)
    Q = np.array(q_rows, dtype=np.uint8)
    g = P.shape[0]

    zeros_in = np.zeros((n_in, 2 * n), dtype=np.uint8)
    # joint solve space: [ p | q ] for generators, [ S | 0 ] for input rows
    joint = np.hstack([P, Q])
    if n_in:
        joint = np.vstack([joint, np.hstack([stab_matrix, zeros_in])])

    def _combo_records(coeff: np.ndarray) -> Set[int]:
        recs: Set[int] = set()
        for j in np.flatnonzero(coeff[:g]):
            recs.symmetric_difference_update(
                base_record + k for k in m_sets[int(j)])
        for k in np.flatnonzero(coeff[g:]):
            recs.symmetric_difference_update(stab_records[int(k)])
        return recs

    # --- 1. canonical fresh rows: solve [0 | S'] for registered targets --
    fresh_rows: List[np.ndarray] = []
    fresh_recs: List[Set[int]] = []
    if canonical_stabs is not None and canonical_stabs.shape[0]:
        targets = np.hstack([np.zeros_like(canonical_stabs), canonical_stabs])
        c, dep, _ = solve_linear_decomposition(
            basis=joint, targets=targets, reduce_weight=True)
        for i in range(canonical_stabs.shape[0]):
            if not dep[i]:
                continue                    # not established by this round
            fresh_rows.append(canonical_stabs[i].copy())
            fresh_recs.append(_combo_records(c[i]))
    F = (np.array(fresh_rows, dtype=np.uint8) if fresh_rows
         else np.zeros((0, 2 * n), dtype=np.uint8))

    def _reduce_over_fresh(vec: np.ndarray) -> Tuple[bool, Set[int]]:
        """Is vec in span(fresh rows)? If so return the pinning records."""
        if F.shape[0] == 0:
            return (not vec.any()), set()
        c, dep, _ = solve_linear_decomposition(
            basis=F, targets=vec.reshape(1, -1), reduce_weight=True)
        if not dep[0]:
            return False, set()
        recs: Set[int] = set()
        for i in np.flatnonzero(c[0]):
            recs.symmetric_difference_update(fresh_recs[int(i)])
        return True, recs

    # --- 2. intersection accounting -----------------------------------------
    # W = span(input rows) ∩ span(flow inputs): exactly the pinned information
    # this round can either re-pin (detector) or carry through (transport).
    # Input DOFs outside W were randomized by the round's measurements and
    # vanish silently (the SE path's Case-A replacement).
    detectors: List[Set[int]] = []
    seen_detectors: Set[frozenset] = set()
    transported_rows: List[np.ndarray] = []
    transported_recs: List[Set[int]] = []
    if n_in:
        W = _intersection_basis(stab_matrix, P, preferred=canonical_stabs)
        for w_vec in W:
            ch, dh, _ = solve_linear_decomposition(
                basis=stab_matrix, targets=w_vec.reshape(1, -1),
                reduce_weight=True)
            hist: Set[int] = set()
            for s in np.flatnonzero(ch[0]):
                hist.symmetric_difference_update(stab_records[int(s)])
            cy, dy, _ = solve_linear_decomposition(
                basis=P, targets=w_vec.reshape(1, -1), reduce_weight=True)
            yg = np.flatnonzero(cy[0])
            img = (np.bitwise_xor.reduce(Q[yg], axis=0)
                   if yg.size else np.zeros(2 * n, dtype=np.uint8))
            recs = set(hist)
            for j in yg:
                recs.symmetric_difference_update(
                    base_record + m for m in m_sets[int(j)])
            in_span, pin_recs = _reduce_over_fresh(img)
            if in_span:
                det = recs ^ pin_recs
                key = frozenset(det)
                if det and key not in seen_detectors:
                    seen_detectors.add(key)
                    detectors.append(det)
            else:
                transported_rows.append(img)
                transported_recs.append(recs)

    # --- 3. completion: pure preparation flows (e.g. Y_L birth) -------------
    completion_rows: List[np.ndarray] = []
    completion_recs: List[Set[int]] = []
    for j in np.flatnonzero(~P.any(axis=1)):
        img = Q[int(j)]
        if not img.any():
            continue
        recs = {base_record + m for m in m_sets[int(j)]}
        completion_rows.append(img)
        completion_recs.append(recs)

    # --- 4. assemble rows: fresh, then transports, then completions ------
    def _strip_measured(row: np.ndarray, rec: Set[int]) -> Optional[
            Tuple[np.ndarray, Set[int]]]:
        row = row.copy()
        rec = set(rec)
        pos = 0
        for inst in chunk.flattened():
            if inst.name not in _MEAS_NAMES:
                continue
            basis = {"M": "Z", "MX": "X", "MY": "Y"}[inst.name]
            for t in inst.targets_copy():
                qb = t.value
                has_x, has_z = bool(row[qb]), bool(row[n + qb])
                if has_x or has_z:
                    comp = ("Y" if has_x and has_z
                            else "X" if has_x else "Z")
                    if comp != basis:
                        return None
                    row[qb] = 0
                    row[n + qb] = 0
                    rec.symmetric_difference_update({base_record + pos})
                pos += 1
        return row, rec

    def _clean_by_fresh(row: np.ndarray, rec: Set[int]
                        ) -> Tuple[np.ndarray, Set[int]]:
        """If the row's records are exactly a combination of fresh rows'
        records, XOR those fresh rows in: the surviving coset keeps a
        records-less (pristine) representative, which downstream promotion
        machinery requires. Fresh rows themselves are never modified."""
        if not rec or F.shape[0] == 0:
            return row, rec
        universe = sorted(set().union(rec, *fresh_recs))
        col = {r: i for i, r in enumerate(universe)}

        def _v(s: Set[int]) -> np.ndarray:
            out = np.zeros(len(universe), dtype=np.uint8)
            for r in s:
                out[col[r]] = 1
            return out

        basis = np.array([_v(fr) for fr in fresh_recs], dtype=np.uint8)
        c, dep, _ = solve_linear_decomposition(
            basis=basis, targets=_v(rec).reshape(1, -1), reduce_weight=True)
        if not dep[0]:
            return row, rec
        new_row = row.copy()
        new_rec = set(rec)
        for i in np.flatnonzero(c[0]):
            new_row ^= F[int(i)]
            new_rec.symmetric_difference_update(fresh_recs[int(i)])
        if new_rec or not new_row.any():
            return row, rec
        return new_row, new_rec

    rows: List[np.ndarray] = list(F)
    recs_out: List[Set[int]] = [set(r) for r in fresh_recs]
    for row, rec in list(zip(transported_rows, transported_recs)) + \
            list(zip(completion_rows, completion_recs)):
        stripped = _strip_measured(row, rec)
        if stripped is None:
            continue
        row, rec = stripped
        row, rec = _clean_by_fresh(row, rec)
        if not row.any():
            key = frozenset(rec)
            if rec and key not in seen_detectors:
                seen_detectors.add(key)
                detectors.append(rec)
            continue
        cur = (np.array(rows, dtype=np.uint8) if rows
               else np.zeros((0, 2 * n), dtype=np.uint8))
        if _rank(np.vstack([cur, row.reshape(1, -1)])) > _rank(cur):
            rows.append(row)                # kept AS IS: no record blending
            recs_out.append(rec)
        else:
            ok, pin = _reduce_over_fresh(row)
            if not ok:
                c2, d2, _ = solve_linear_decomposition(
                    basis=cur, targets=row.reshape(1, -1),
                    reduce_weight=True)
                pin = set()
                if d2[0]:
                    for i in np.flatnonzero(c2[0]):
                        pin.symmetric_difference_update(recs_out[int(i)])
            det = rec ^ pin
            key = frozenset(det)
            if det and key not in seen_detectors:
                seen_detectors.add(key)
                detectors.append(det)

    # --- 5. transport logical AND absorbed rows; measured-out -> resolved --
    # The absorbed ledger rows are Pauli operators with banked records,
    # exactly like logical rows - transport them through the same flows
    # (review blocker: a Clifford relay must move the ledger to the new
    # frame; the census cannot catch the corruption, rank is frame-
    # invariant).
    n_log_real = log_matrix.shape[0] if log_matrix is not None else 0
    if absorbed_matrix is not None and absorbed_matrix.shape[0]:
        log_matrix = (np.vstack([log_matrix, absorbed_matrix])
                      if n_log_real else absorbed_matrix)
        log_records = (list(log_records) if log_records else []) + [
            list(r) for r in absorbed_records]
    new_log_rows: List[np.ndarray] = []
    new_log_recs: List[List[int]] = []
    measured_logicals: List[Tuple[int, List[int]]] = []
    new_absorbed_rows: List[np.ndarray] = []
    new_absorbed_recs: List[List[int]] = []
    measured_absorbed: List[Tuple[int, List[int]]] = []
    if log_matrix is not None and log_matrix.shape[0] > 0:
        basis = np.vstack([P, stab_matrix]) if n_in else P
        c, d, _ = solve_linear_decomposition(
            basis=basis, targets=log_matrix, reduce_weight=True)
        for i in range(log_matrix.shape[0]):
            if not d[i]:
                raise NotImplementedError(
                    "relay chunk destroys a tracked logical row (no "
                    "transport or measurement flow exists for it)")
            yg = np.flatnonzero(c[i][:g])
            img = (np.bitwise_xor.reduce(Q[yg], axis=0)
                   if yg.size else np.zeros(2 * n, dtype=np.uint8))
            rec: Set[int] = set(log_records[i])
            for j in yg:
                rec.symmetric_difference_update(
                    base_record + m for m in m_sets[int(j)])
            for s in np.flatnonzero(c[i][g:]):
                rec.symmetric_difference_update(stab_records[int(s)])
            if img.any():
                stripped = _strip_measured(img, rec)
                if stripped is None:
                    raise NotImplementedError(
                        "logical transport image anticommutes with a chunk "
                        "measurement")
                img, rec = stripped
            if not img.any():
                # The chunk MEASURED this row: its value is the parity of
                # the banked seed records XOR the chunk's measuring records —
                # a logical observable / a resolved absorbed relation.
                if i < n_log_real:
                    measured_logicals.append((i, sorted(rec)))
                else:
                    measured_absorbed.append((i - n_log_real, sorted(rec)))
                continue
            if i < n_log_real:
                new_log_rows.append(img)
                new_log_recs.append(sorted(rec))
            else:
                new_absorbed_rows.append(img)
                new_absorbed_recs.append(sorted(rec))

    # --- 5b. completeness: the FULL deterministic-parity space ------------
    # Combos (y over gens, c over input rows) with Σy·p ⊕ Σc·S = 0 AND
    # Σy·q = 0 form the complete space of deterministic record parities this
    # round; the canonical detectors above are a subset. Complete the basis
    # so no coverage dimension is lost (a missing dimension silently reduces
    # the circuit's fault distance).
    stacked = np.hstack([P, Q])
    if n_in:
        stacked = np.vstack([stacked,
                             np.hstack([stab_matrix, zeros_in])])
    null = _left_nullspace_basis(stacked)
    if null.shape[0]:
        universe = sorted(
            {r for det in detectors for r in det}
            | {base_record + m for ms in m_sets for m in ms}
            | {r for rs in stab_records for r in rs})
        col = {r: i for i, r in enumerate(universe)}

        def _rvec(s: Set[int]) -> np.ndarray:
            v = np.zeros(len(universe), dtype=np.uint8)
            for r in s:
                v[col[r]] = 1
            return v

        chosen = [_rvec(det) for det in detectors]
        for y in null:
            recs: Set[int] = set()
            for j in np.flatnonzero(y[:g]):
                recs.symmetric_difference_update(
                    base_record + m for m in m_sets[int(j)])
            for k in np.flatnonzero(y[g:]):
                recs.symmetric_difference_update(stab_records[int(k)])
            if not recs:
                continue
            cur = (np.array(chosen, dtype=np.uint8) if chosen
                   else np.zeros((0, len(universe)), dtype=np.uint8))
            v = _rvec(recs)
            if _rank(np.vstack([cur, v.reshape(1, -1)])) > _rank(cur):
                chosen.append(v)
                key = frozenset(recs)
                if key not in seen_detectors:
                    seen_detectors.add(key)
                    detectors.append(recs)

    # --- 6. detector coordinate hints -------------------------------------
    dets_out: List[Tuple[Set[int], int]] = []
    for recs in detectors:
        local = [r - base_record for r in recs
                 if base_record <= r < base_record + n_meas]
        coord_q = measured[max(local)] if local else -1
        dets_out.append((recs, coord_q))

    res = RelayResult(measured_qubits=measured, detectors=dets_out)
    res.new_stab_matrix = (np.array(rows, dtype=np.uint8) if rows
                           else np.zeros((0, 2 * n), dtype=np.uint8))
    res.new_stab_records = [sorted(r) for r in recs_out]
    res.new_log_matrix = (np.array(new_log_rows, dtype=np.uint8)
                          if new_log_rows
                          else np.zeros((0, 2 * n), dtype=np.uint8))
    res.new_log_records = new_log_recs
    res.measured_logicals = measured_logicals
    res.new_absorbed_matrix = (np.array(new_absorbed_rows, dtype=np.uint8)
                               if new_absorbed_rows
                               else np.zeros((0, 2 * n), dtype=np.uint8))
    res.new_absorbed_records = new_absorbed_recs
    res.measured_absorbed = measured_absorbed
    return res
