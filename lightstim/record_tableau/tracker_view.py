"""SyndromeTracker-compatible, read-only state for the record-tableau backend.

With ``LIGHTSTIM_DETECTOR_BACKEND=record_tableau`` the builder never updates
its SyndromeTracker.  Code that inspects the tracker (``tracker.stabilizers``,
``tracker.logicals``, ``validate_logical_count``, ``record_parity`` ...) reads
this view instead: the record tableau's tracked generators, exported only when
the view is read and cached until the circuit changes, so the annotation hot
path is untouched.

The engine's stabilizer bank and logical bank (generators with the logical
flag) become ``tracker.stabilizers`` and ``tracker.logicals``.  The rows span
the same group as the tracker's rows, but individual rows may be other
representatives.  Records drop the symbolic logical values (the
tracker's logical rows carry only measurement records); a stabilizer whose
records carried one is listed in ``stabilizer_with_logical_components``.
"""
from typing import List, Set

import numpy as np

from ..ir.tableau import PauliTableau
from .annotator import LAMBDA_BASE

_STATE = ("stabilizers", "logicals", "stabilizer_with_logical_components")
_VIEW_CLASSES: dict = {}


def attach_tracker_view(tracker, builder) -> None:
    """Make ``tracker`` read its state from ``builder``'s record tableau."""
    if isinstance(tracker, _TrackerView):          # tracker reused by a new builder
        tracker._record_view = _RecordState(builder)
        return
    cls = type(tracker)
    if cls not in _VIEW_CLASSES:
        _VIEW_CLASSES[cls] = type(f"RecordTableau{cls.__name__}View", (_TrackerView, cls), {})
    tracker._record_view = _RecordState(builder)
    tracker._declared_logicals = tracker.__dict__.pop("expected_num_logicals")
    for name in _STATE:
        tracker.__dict__.pop(name, None)
    tracker.__class__ = _VIEW_CLASSES[cls]


def _gf2_rank(vectors: List[Set[int]]) -> int:
    basis = {}
    for v in vectors:
        v = set(v)
        while v and max(v) in basis:
            v ^= basis[max(v)]
        if v:
            basis[max(v)] = v
    return len(basis)


class _RecordState:
    def __init__(self, builder):
        self.builder = builder
        self._key = None
        self._state = None

    def get(self) -> dict:
        builder = self.builder
        _ = builder.circuit                         # annotate everything appended so far
        annotator = builder._annotator
        annotator.flush_state()
        engine = annotator.engine
        key = (builder.tracker.num_qubits, annotator._synced, annotator._last_targets,
               engine.num_measurements,
               engine.rebase_ops, annotator.num_lambdas, len(annotator.lambda_basis))
        if key != self._key:
            self._state = self._build(annotator, engine)
            self._key = key
        return self._state

    def _build(self, annotator, engine) -> dict:
        rows, gens, recs, logical, deps, consumed = engine.snapshot()
        rows = np.asarray(rows, dtype=np.uint8)
        n = max(self.builder.tracker.num_qubits, engine.n)
        if engine.n < n:                            # widen [X|Z] halves to the tracker width
            wide = np.zeros((rows.shape[0], 2 * n), dtype=np.uint8)
            wide[:, :engine.n] = rows[:, :engine.n]
            wide[:, n:n + engine.n] = rows[:, engine.n:]
            rows = wide
        else:
            rows = rows.copy()
        recs = [list(r) for r in recs]
        logical = list(logical)
        slot = {k: i for i, k in enumerate(gens)}
        extra_rows = []
        for c, r in deps:                           # dependent rows: products of generators
            if r is None or any(k not in slot for k in c):
                continue
            row = np.zeros(2 * n, dtype=np.uint8)
            for k in c:
                row ^= rows[slot[k]]
            extra_rows.append(row)
            recs.append(list(r))
            logical.append(False)
        if extra_rows:
            rows = np.vstack([rows, np.array(extra_rows, dtype=np.uint8)])
        # Like the tracker, keep only data-supported stabilizer rows: a
        # single-qubit constraint on a non-data qubit is multiplied out of the
        # other rows (its records move with it) and dropped, and so is a row
        # still correlated with an in-flight ancilla (e.g. a Bell partner).
        data = set(getattr(self.builder.system, "data_indices", ()))
        support = rows[:, :n] | rows[:, n:]
        ancilla = {}
        for i in np.flatnonzero(support.sum(axis=1) == 1):
            q = int(np.flatnonzero(support[i])[0])
            if q not in data and not logical[i]:
                ancilla[int(i)] = q
        for i, q in ancilla.items():
            factor = (rows[i, q], rows[i, n + q])
            for j in np.flatnonzero(support[:, q]):
                if j != i and j not in ancilla and (rows[j, q], rows[j, n + q]) == factor:
                    rows[j] ^= rows[i]
                    recs[j] = sorted(set(recs[j]) ^ set(recs[i]))
        stabilizers, logicals = PauliTableau(n), PauliTableau(n)
        stab_rows, stab_recs, with_logical, log_items = [], [], set(), []
        live_lambdas = set()
        off_data = np.ones(n, dtype=bool)
        off_data[[q for q in data if q < n]] = False
        for i in range(rows.shape[0]):
            if i in ancilla:
                continue
            if data and not logical[i] and (rows[i, :n] | rows[i, n:])[off_data].any():
                continue
            lam = [r for r in recs[i] if r <= LAMBDA_BASE]
            live_lambdas.update(lam)
            plain = [r for r in recs[i] if r > LAMBDA_BASE]
            if logical[i]:
                log_items.append((min((LAMBDA_BASE - r for r in lam), default=rows.shape[0] + i),
                                  rows[i], plain))
            else:
                if lam:
                    with_logical.add(len(stab_rows))
                stab_rows.append(rows[i])
                stab_recs.append(plain)
        if stab_rows:
            stabilizers.add_stabilizers(np.array(stab_rows, dtype=np.uint8), stab_recs)
        log_items.sort(key=lambda item: item[0])
        if log_items:
            logicals.add_stabilizers(np.array([row for _, row, _ in log_items], dtype=np.uint8),
                                     [rec for _, _, rec in log_items])
        # An absorbed joint relation keeps a logical DOF trapped while one of
        # its logical symbols is still carried by a tracked row.
        alive = [p for p in annotator.absorbed_patterns if p & live_lambdas]
        return dict(stabilizers=stabilizers, logicals=logicals,
                    stabilizer_with_logical_components=with_logical,
                    absorbed=_gf2_rank(alive), consumed=consumed)


def _read_only(name):
    def get(self):
        return self._record_view.get()[name]

    def set_(self, value):
        raise AttributeError(
            f"tracker.{name} is read-only on the record_tableau backend "
            f"(the record tableau owns the tracked state)")
    return property(get, set_)


class _TrackerView:
    """Mixin placed in front of the tracker's class by attach_tracker_view."""
    stabilizers = _read_only("stabilizers")
    logicals = _read_only("logicals")
    stabilizer_with_logical_components = _read_only("stabilizer_with_logical_components")

    @property
    def expected_num_logicals(self) -> int:
        """The declared logical count minus logicals consumed by collapses
        (the tracker's own decrement)."""
        return self._declared_logicals - self._record_view.get()["consumed"]

    @expected_num_logicals.setter
    def expected_num_logicals(self, value: int) -> None:
        self._declared_logicals = value + self._record_view.get()["consumed"]

    def process_unitary_block(self, circuit_chunk) -> None:
        """The record tableau applies the gates from the builder's circuit;
        only relations banked on the tracker itself move with the frame."""
        if self.absorbed_ops.count:
            m = self.get_forward_symplectic_matrix(circuit_chunk, self.num_qubits)
            self.absorbed_ops.matrix = ((self.absorbed_ops.matrix @ m) % 2).astype(np.uint8)

    def num_absorbed_dof(self) -> int:
        """Relations banked on the tracker itself plus the record tableau's
        absorbed joint measurements that still trap a logical DOF."""
        return super().num_absorbed_dof() + self._record_view.get()["absorbed"]
