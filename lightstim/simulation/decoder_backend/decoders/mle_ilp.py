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
from scipy.optimize import Bounds, LinearConstraint, milp

from ..external import ExternalDecoder
from ..registry import register_decoder

_DEFAULTS = {
    "max_prior": 0.5,      # clamp: priors >= 0.5 give non-positive weights
    "time_limit": 0.0,     # seconds per shot; 0 = unlimited. A shot that hits
                           # the limit is reported as a decode failure, so
                           # DecoderConfig(on_decode_failure=...) can herald it.
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

        rowsum = np.asarray(H.sum(axis=1)).ravel()
        self._bounds = Bounds(
            np.zeros(self._n + self._m),
            np.concatenate([np.ones(self._n), np.floor(rowsum / 2.0)]),
        )
        self._c = np.concatenate([self._w, np.zeros(self._m)])
        self._integrality = np.ones(self._n + self._m)
        # [H | -2I] -- one slack column per detector row.
        self._A = sp.hstack(
            [H.astype(float), -2.0 * sp.eye(self._m, format="csr")],
            format="csr",
        )
        self._options = ({"time_limit": self._time_limit}
                         if self._time_limit > 0 else {})

    def decode_single(self, syndrome):
        s = np.asarray(syndrome, dtype=np.uint8).ravel().astype(float)
        res = milp(
            c=self._c,
            constraints=LinearConstraint(self._A, s, s),
            integrality=self._integrality,
            bounds=self._bounds,
            options=self._options,
        )
        if not res.success:
            return np.zeros(self._n, dtype=np.uint8), False
        return np.round(res.x[:self._n]).astype(np.uint8), True


register_decoder("mle-ilp", MleIlpDecoder, aliases=["mle", "ilp"],
                 backend="cpu")
