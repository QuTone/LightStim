"""Exact most-likely-error (MLE) decoder via integer programming.

Solves, for each shot, the exact most-likely-error problem

    minimize   sum_j w_j e_j        with w_j = log((1 - p_j) / p_j)
    subject to H e == s   (mod 2),  e in {0,1}^n

The GF(2) parity is linearised with one bounded integer slack per detector:

    sum_j H[i,j] e_j - 2 k_i == s_i,   0 <= k_i <= floor(rowweight_i / 2)

This is the decoder papers mean by "exact MLE with Gurobi" (more precisely
*most-likely-error*, not degenerate maximum-likelihood: it finds the single
highest-probability error, it does not sum over the logical coset). It is
exponential in the worst case and is intended as a **ground-truth reference**
for small instances, not for production sampling throughput.

The MILP is the fallback, not the first move. Each shot is first attempted with
adaptive-LP (ALP) decoding -- Feldman's fundamental-polytope relaxation with
forbidden-set cuts generated on demand, in the manner of Taghavi & Siegel --
and only escalated to the MILP when that leaves a fractional pseudocodeword.
This stays exact; see :meth:`MleIlpDecoder._adaptive_lp`. ALP settles 85-100%
of shots and is worth 2.0-11.0x over calling the MILP directly, with the
largest gains on QLDPC codes.

The solver is ``scipy.optimize.milp``, which needs no dependency beyond the
scipy LightStim already requires. Alternatives were benchmarked and rejected;
median ms/shot on identical syndromes with an identical formulation, all
cross-checked to return the same optimal cost:

    instance              scipy   highspy    SCIP   CP-SAT (8 threads)
    surface d=5            11.8      21.5      26                  60
    surface d=7            41.4      76.5     271                 616
    BB [[72,12,6]] r=2     56.7     132.6      79                 119
    BB [[72,12,6]] r=4    113.7     270.6     271                5488

Two results there are worth not re-deriving. First, scipy and ``highspy`` are
*not* the same binary — scipy vendors its own HiGHS (1.8.0 in scipy 1.15)
rather than importing ``highspy`` (1.15.1), and the older vendored build wins
everywhere. The gap is not explained by threads, matrix format, presolve,
``mip_rel_gap``, incremental-vs-rebuild model updates, or ``passModel``; all
were tested and are a wash. Second, CP-SAT degrades far worse than the MILP
solvers as the DEM densifies (a BB detector row averages ~214 error mechanisms
against a handful for the surface code), so its native ``AddBoolXOr`` handling
of the parity constraints does not pay off. At BB r=4 it left 2 of 10 shots
unproven at a 60 s cap.

Single-threaded is the right comparison throughout: :class:`SimulationPipeline`
already parallelises across shots, so intra-solve threads only oversubscribe.

If a future scipy vendors a slower HiGHS, the alternatives are worth re-testing
-- ``git log`` for this file has a working ``highspy`` backend. Note that
``highspy`` and ``ortools`` cannot share an environment: both vendor HiGHS and
clash on symbols at import, in either order.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from ..external import ExternalDecoder
from ..registry import register_decoder

_INT_TOL = 1e-6        # treat an LP component as integral within this
_CUT_TOL = 1e-7        # minimum violation before a cut is worth adding

_DEFAULTS = {
    "max_prior": 0.5,      # clamp: priors >= 0.5 give non-positive weights
    "time_limit": 0.0,     # seconds per shot; 0 = unlimited. A shot that hits
                           # the limit is reported as a decode failure, so
                           # DecoderConfig(on_decode_failure=...) can herald it.
    "max_cut_rounds": 200,  # give up on cut generation and fall back to MILP
}


def _weights(priors: np.ndarray, max_prior: float) -> np.ndarray:
    """Negative-log-likelihood-ratio cost per error mechanism."""
    q = np.clip(np.asarray(priors, dtype=float), 1e-15, max_prior - 1e-15)
    return np.log((1.0 - q) / q)


class MleIlpDecoder(ExternalDecoder):
    """Exact MLE decoder backed by ``scipy.optimize.milp`` (HiGHS)."""

    output_type = "correction"

    def setup(self, *, H, priors, **_):
        params = {**_DEFAULTS, **self.params}
        self._time_limit = float(params["time_limit"])

        H = sp.csr_matrix(H, dtype=np.uint8)
        self._m, self._n = H.shape
        self._w = _weights(priors, float(params["max_prior"]))

        self._H = H
        self._indptr = H.indptr
        self._indices = H.indices
        self._starts = H.indptr[:-1]
        self._nonempty = H.indptr[1:] > H.indptr[:-1]
        self._max_rounds = int(params["max_cut_rounds"])

        rowsum = np.asarray(H.sum(axis=1)).ravel()
        lower = np.zeros(self._n + self._m)
        upper = np.concatenate([np.ones(self._n), np.floor(rowsum / 2.0)])
        self._bounds = Bounds(lower, upper)
        self._c = np.concatenate([self._w, np.zeros(self._m)])
        self._integrality = np.ones(self._n + self._m)
        # [H | -2I] -- one slack column per detector row.
        self._A = sp.hstack(
            [H.astype(float), -2.0 * sp.eye(self._m, format="csr")],
            format="csr",
        )
        self._options = ({"time_limit": self._time_limit}
                         if self._time_limit > 0 else {})

    def _violated_checks(self, e, s):
        """Screen every check for a violated forbidden-set cut, vectorised.

        For a check with support N, writing
        ``g(S) = sum_{j in S}(1 - e_j) + sum_{j not in S} e_j``, the cut's
        violation is ``1 - g(S)``. So the most-violated cut minimises g: take
        ``S = {j : e_j > 0.5}``, and if ``|S|`` has the wrong parity flip the
        single element whose ``e_j`` is nearest 0.5 (each flip costs
        ``|2 e_j - 1|``). Returns the indices of checks whose best cut is
        violated.
        """
        v = e[self._indices]
        in_s = v > 0.5
        # Per element: cost of its cheapest side, and of being flipped.
        base = np.where(in_s, 1.0 - v, v)
        flip = np.abs(2.0 * v - 1.0)

        starts = self._starts[self._nonempty]
        g = np.full(self._m, np.inf)
        count = np.zeros(self._m, dtype=np.int64)
        cheapest = np.zeros(self._m)
        g[self._nonempty] = np.add.reduceat(base, starts)
        count[self._nonempty] = np.add.reduceat(in_s.astype(np.int64), starts)
        cheapest[self._nonempty] = np.minimum.reduceat(flip, starts)

        # |S| must have parity s_i + 1; where it matches s_i, pay one flip.
        needs_flip = (count & 1) == (s & 1)
        g = g + np.where(needs_flip & self._nonempty, cheapest, 0.0)
        return np.flatnonzero(self._nonempty & (g < 1.0 - _CUT_TOL))

    def _build_cut(self, e, s, i):
        """Materialise the most-violated cut for check ``i``.

        Returns ``(columns, coefficients, rhs)`` for ``coeffs . e[cols] <= rhs``.
        """
        cols = self._indices[self._indptr[i]:self._indptr[i + 1]]
        v = e[cols]
        in_s = v > 0.5
        if (int(in_s.sum()) & 1) == (int(s[i]) & 1):
            in_s = in_s.copy()
            in_s[np.argmin(np.abs(2.0 * v - 1.0))] ^= True
        return cols, np.where(in_s, 1.0, -1.0), float(in_s.sum()) - 1.0

    def _adaptive_lp(self, s):
        """Adaptive-LP decode (Taghavi-Siegel). Returns the error or None.

        Solves over the fundamental polytope, generating forbidden-set cuts on
        demand. Terminating with no violated cut and an integral solution means
        every parity holds and the solution is optimal over a relaxation that
        contains every valid error, so it is the exact MLE answer.
        """
        rows, cols, vals, rhs = [], [], [], []
        e = np.zeros(self._n)

        for _ in range(self._max_rounds):
            violated = self._violated_checks(e, s)
            if violated.size == 0:
                break
            for i in violated:
                c_cols, c_vals, c_rhs = self._build_cut(e, s, i)
                rows.append(np.full(c_cols.size, len(rhs), dtype=np.int64))
                cols.append(c_cols.astype(np.int64))
                vals.append(c_vals)
                rhs.append(c_rhs)
            a_ub = sp.csr_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(len(rhs), self._n))
            lp = linprog(c=self._w, A_ub=a_ub, b_ub=np.asarray(rhs),
                         bounds=(0, 1), method="highs")
            if not lp.success:
                return None
            e = lp.x
        else:
            return None                     # cut generation did not converge

        if np.abs(e - np.round(e)).max() >= _INT_TOL:
            return None                     # fractional: a pseudocodeword
        ei = np.round(e).astype(np.uint8)
        # Verify the parity exactly rather than trusting _INT_TOL.
        return ei if np.array_equal((self._H @ ei) % 2, s) else None

    def decode_single(self, syndrome):
        """Solve one shot: adaptive LP first, MILP only if it doesn't settle.

        Both paths return the same answer -- see :meth:`_adaptive_lp` for why
        the LP result, when integral, is provably the MILP optimum. Only the
        cost of getting there differs.

        The relaxation is tight enough on these models that HiGHS reports a
        branch-and-bound node count of 1 on essentially every shot, so the win
        is skipping MIP machinery that changes nothing. Measured against
        calling the MILP directly: 3.9x on surface d=5, 2.0x on d=7, 11.0x on
        BB [[72,12,6]] r=2 and 7.9x on r=4, with the LP settling 34/40 to 40/40
        of shots. Totals gain least where a few hard shots dominate, and those
        shots pay only the discarded cut-generation work.

        Against the plain (slack-formulation) relaxation this replaced, the
        end-to-end gain is larger still -- a 100-shot BB r=4 run went from 727
        to 38 ms/shot -- because ALP settles hard shots that the slack
        relaxation left fractional, so they never reach the MILP at all.
        """
        s = np.asarray(syndrome, dtype=np.uint8).ravel()
        e = self._adaptive_lp(s)
        if e is not None:
            return e, True

        sf = s.astype(float)
        res = milp(
            c=self._c,
            constraints=LinearConstraint(self._A, sf, sf),
            integrality=self._integrality,
            bounds=self._bounds,
            options=self._options,
        )
        if not res.success:
            return np.zeros(self._n, dtype=np.uint8), False
        return np.round(res.x[:self._n]).astype(np.uint8), True


register_decoder("mle-ilp", MleIlpDecoder, aliases=["mle", "ilp"],
                 backend="cpu")
