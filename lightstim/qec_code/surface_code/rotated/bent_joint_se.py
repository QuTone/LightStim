import stim

from .SE_block import RotatedSurfaceCodeExtractionBlock

# -----------------------------------------------------------------------------
# Rotated bent (XZ) joint lattice-surgery measurement — circuit builder
# -----------------------------------------------------------------------------
#
# The bent joint-measurement layout is transcribed from `rotated.png` as an
# explicit list of data-qubit coordinates plus checks (pure X, pure Z, and MIXED
# domain-wall checks carrying an explicit per-data Pauli). This module turns that
# layout into a circuit-level syndrome-extraction circuit so the notebook does
# NOT define the syndrome-extraction circuit inline — it lives here in
# `lightstim/qec_code/`, next to the other surface-code extraction blocks.
#
# Pure X/Z checks use the rotated code's fault-tolerant `perpendicular` schedule
# (reused from `RotatedSurfaceCodeExtractionBlock.SCHEDULES`). MIXED checks are
# measured with a single X-basis ancilla — X terms couple via CNOT(ancilla->data)
# and Z terms via CZ(ancilla, data) — scheduled in the SAME direction-by-tick
# slots the pure checks use, so the mixed checks inherit the pure checks'
# hook-error-benign (logical-perpendicular) orientation.


def _sgn(a):
    return (a > 0) - (a < 0)


class RotatedBentJointMeasurement:
    """Circuit builder for the rotated bent (XZ) joint lattice-surgery measurement.

    Args:
        data:      list of data-qubit ``(col, row)`` coordinates (rotated frame).
        checks:    list of stabilizer dicts ``{'syn', 'type', 'pauli', 'corners'}``
                   where ``type`` is ``'X'`` / ``'Z'`` / ``'M'`` and ``pauli`` maps
                   each corner ``(col, row)`` to ``'X'`` / ``'Z'``.
        x_logical: data-qubit coords supporting the measured logical X̄ (read out
                   from the final X-basis data measurement as observable 0).

    The qubit index layout is: data qubits ``0 .. nq-1`` (in ``data`` order),
    then one ancilla per check ``nq + k`` (in ``checks`` order ``k``).
    """

    #: pure-check entangling schedule, reused from the rotated SE block
    SCHED = RotatedSurfaceCodeExtractionBlock.SCHEDULES['perpendicular']

    def __init__(self, data, checks, x_logical):
        self.data = list(data)
        self.checks = list(checks)
        self.x_logical = list(x_logical)
        self.nq = len(self.data)
        self.nst = len(self.checks)
        self.di = {q: i for i, q in enumerate(self.data)}
        self.aid = {k: self.nq + k for k in range(self.nst)}
        self.Xk = [k for k, c in enumerate(self.checks) if c['type'] == 'X']
        self.Zk = [k for k, c in enumerate(self.checks) if c['type'] == 'Z']
        self.Mk = [k for k, c in enumerate(self.checks) if c['type'] == 'M']

    def _dvec(self, syn, d):
        return (_sgn(d[0] - syn[0]), _sgn(d[1] - syn[1]))   # diagonal direction ancilla -> data

    def _se_round(self, c, p):
        """Append ONE syndrome-extraction round (pure X/Z block, then the mixed
        checks in their own H..H blocks). Each phase gets its own TICK so the
        detslice never mixes reset / basis-change / measurement with the
        two-qubit gate layers (one clean op-type per slice)."""
        di, AID, CHECKS = self.di, self.aid, self.checks
        anc = [AID[k] for k in range(self.nst)]
        c.append("R", anc)
        if p > 0:
            c.append("X_ERROR", anc, p)
        c.append("TICK")                                            # phase: reset ancillas

        xanc = [AID[k] for k in self.Xk]
        c.append("H", xanc)
        if p > 0:
            c.append("DEPOLARIZE1", xanc, p)
        c.append("TICK")                                            # phase: prep X-ancillas (H)

        for dxx, dxz in self.SCHED:                                 # pure X/Z block: one tick per CNOT layer
            pairs = []
            for k in self.Xk:
                for d in CHECKS[k]['pauli']:
                    if self._dvec(CHECKS[k]['syn'], d) == dxx:
                        pairs += [AID[k], di[d]]
            for k in self.Zk:
                for d in CHECKS[k]['pauli']:
                    if self._dvec(CHECKS[k]['syn'], d) == dxz:
                        pairs += [di[d], AID[k]]
            if pairs:
                c.append("CNOT", pairs)
                if p > 0:
                    c.append("DEPOLARIZE2", pairs, p)
            c.append("TICK")

        c.append("H", xanc)
        if p > 0:
            c.append("DEPOLARIZE1", xanc, p)
        c.append("TICK")                                            # phase: unprep X-ancillas (H)

        for k in self.Mk:                                           # mixed checks: own H..H blocks
            ch = CHECKS[k]; a = AID[k]
            c.append("H", [a])
            if p > 0:
                c.append("DEPOLARIZE1", [a], p)
            c.append("TICK")                                        # phase: prep mixed ancilla (H)
            for dxx, dxz in self.SCHED:
                cn = []; cz = []
                for d, P in ch['pauli'].items():
                    if P == 'X' and self._dvec(ch['syn'], d) == dxx:
                        cn += [a, di[d]]
                    if P == 'Z' and self._dvec(ch['syn'], d) == dxz:
                        cz += [a, di[d]]
                # CNOT and CZ go in SEPARATE ticks: they share the ancilla, so emitting
                # them in one slice would touch the ancilla twice in the same tick (a tick
                # collision / two op-types per slice). One clean gate per slice instead.
                if cn:
                    c.append("CNOT", cn)
                    if p > 0:
                        c.append("DEPOLARIZE2", cn, p)
                    c.append("TICK")
                if cz:
                    c.append("CZ", cz)
                    if p > 0:
                        c.append("DEPOLARIZE2", cz, p)
                    c.append("TICK")
            c.append("H", [a])
            if p > 0:
                c.append("DEPOLARIZE1", [a], p)
            c.append("TICK")                                        # phase: unprep mixed ancilla (H)

        c.append("M", anc, p if p > 0 else 0)                       # all ancillas, Z basis, fixed order k=0..nst-1
        c.append("TICK")                                            # phase: measure (separates from next round)

    def circuit(self, rounds=3, p=0.0, basis="X"):
        """Build the full bent joint-measurement circuit: ``basis``-basis init, ``rounds`` of
        syndrome extraction, ``basis``-basis data readout, detectors and the tracked observable.

        ``basis="X"`` (default) is the X-memory experiment (X-init/readout, X-type checks close, the
        observable is the X̄ logical).  ``basis="Z"`` is the Z-memory experiment, used when the tracked
        logical is Z (e.g. a pure-Z joint ``M(Z̄₁Z̄₂)``).  Returns a ``stim.Circuit``."""
        di, AID, CHECKS = self.di, self.aid, self.checks
        nq, NST = self.nq, self.nst
        reset, meas, err, want = (("RX", "MX", "Z_ERROR", "X") if basis == "X"
                                  else ("RZ", "MZ", "X_ERROR", "Z"))
        close_k = self.Xk if basis == "X" else self.Zk             # same-type checks close at readout
        c = stim.Circuit()
        for i, q in enumerate(self.data):
            c.append("QUBIT_COORDS", [i], [float(q[0]), float(q[1])])
        for k in range(NST):
            c.append("QUBIT_COORDS", [AID[k]], [float(CHECKS[k]['syn'][0]), float(CHECKS[k]['syn'][1])])
        c.append(reset, range(nq))
        if p > 0:
            c.append(err, range(nq), p)
        is_det0 = lambda ch: all(P == want for P in ch['pauli'].values())   # deterministic on the init state
        for r in range(rounds):
            if p > 0:
                c.append("DEPOLARIZE1", range(nq), p)               # data idle
            self._se_round(c, p)
            for k in range(NST):
                tr = -(NST - k); co = [float(CHECKS[k]['syn'][0]), float(CHECKS[k]['syn'][1]), r]
                if r == 0:
                    if is_det0(CHECKS[k]):
                        c.append("DETECTOR", [stim.target_rec(tr)], co)
                else:
                    c.append("DETECTOR", [stim.target_rec(tr), stim.target_rec(tr - NST)], co)
        c.append(meas, range(nq), p if p > 0 else 0)                # basis-basis data readout
        drec = {q: -(nq - di[q]) for q in self.data}
        for k in close_k:                                          # closing detectors: last SE ancilla vs data
            ch = CHECKS[k]; tr_anc = -(nq + (NST - k))
            c.append("DETECTOR", [stim.target_rec(tr_anc)] + [stim.target_rec(drec[d]) for d in ch['pauli']],
                     [float(ch['syn'][0]), float(ch['syn'][1]), rounds])
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(drec[q]) for q in self.x_logical], 0)   # tracked logical
        return c


def build_bent_joint_circuit(data, checks, x_logical, rounds=3, p=0.0):
    """Convenience wrapper: build the rotated bent (XZ) joint-measurement circuit.

    See :class:`RotatedBentJointMeasurement` for the layout contract."""
    return RotatedBentJointMeasurement(data, checks, x_logical).circuit(rounds=rounds, p=p)
