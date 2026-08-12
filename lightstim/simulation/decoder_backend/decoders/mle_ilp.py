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

Both backends are HiGHS (MIT-licensed) and return identical optima
(``DecoderConfig(params={"solver": ...})``):

    "auto"  (default) -- currently always ``scipy``; see below.
    "scipy"           -- ``scipy.optimize.milp``. No dependency beyond
                         LightStim's own scipy.
    "highs"            -- the standalone ``highspy`` package.

**Prefer "scipy".** These are not the same binary: scipy vendors its own HiGHS
(1.8.0 in scipy 1.15) rather than importing ``highspy`` (1.15.1), and the older
vendored build is consistently ~2x faster on these models. Median ms/shot,
same syndromes, same formulation:

    instance                    scipy   highspy
    surface d=5                  11.8      21.5
    surface d=7                  41.4      76.5
    BB [[72,12,6]] r=2           56.7     132.6
    BB [[72,12,6]] r=4          113.7     270.6

That ordering held on every instance tried, and is not explained by threads,
matrix format, presolve, ``mip_rel_gap``, incremental-vs-rebuild model
updates, or ``passModel`` — all were tested and are a wash. So ``highspy``
is kept only as an escape hatch if a future scipy vendors a slower HiGHS.

Against other solver families, on BB [[72,12,6]] r=4 (10 shots, 60 s cap):
SCIP via ``pyscipopt`` 271 ms median, and OR-Tools CP-SAT 5.5 s median with
8 threads (2 of 10 shots hit the cap). CP-SAT degrades far worse than the MILP
solvers as the DEM gets denser -- BB detector rows average ~214 error
mechanisms against the surface code's handful -- so its native XOR handling
does not pay off here. Single-threaded is the right comparison regardless:
:class:`SimulationPipeline` already parallelises across shots, so intra-solve
threads only oversubscribe.

Do not install ``highspy`` and ``ortools`` into the same environment: both
vendor HiGHS and clash on symbols at import time, in either order.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..external import ExternalDecoder
from ..registry import register_decoder

_DEFAULTS = {
    "solver": "auto",
    "max_prior": 0.5,      # clamp: priors >= 0.5 give non-positive weights
    "time_limit": 0.0,     # seconds per shot; 0 = unlimited. A shot that hits
                           # the limit is reported as a decode failure, so
                           # DecoderConfig(on_decode_failure=...) can herald it.
}


def _resolve_solver(name: str) -> str:
    """Resolve the ``solver`` param to a concrete backend.

    ``"auto"`` picks scipy unconditionally: it is both the faster HiGHS build
    (see module docstring) and always present, so there is nothing to detect.
    """
    name = str(name).lower()
    if name == "auto":
        return "scipy"
    if name not in ("highs", "scipy"):
        raise ValueError(
            f"Unknown solver {name!r}; expected 'auto', 'highs' or 'scipy'.")
    return name


def _weights(priors: np.ndarray, max_prior: float) -> np.ndarray:
    """Negative-log-likelihood-ratio cost per error mechanism."""
    q = np.clip(np.asarray(priors, dtype=float), 1e-15, max_prior - 1e-15)
    return np.log((1.0 - q) / q)


class MleIlpDecoder(ExternalDecoder):
    """Exact MLE decoder backed by an open-source MILP solver."""

    output_type = "correction"

    def setup(self, *, H, priors, **_):
        params = {**_DEFAULTS, **self.params}
        self._solver = _resolve_solver(params["solver"])
        self._time_limit = float(params["time_limit"])

        H = sp.csr_matrix(H, dtype=np.uint8)
        self._m, self._n = H.shape
        self._w = _weights(priors, float(params["max_prior"]))

        rowsum = np.asarray(H.sum(axis=1)).ravel()
        self._lb = np.zeros(self._n + self._m)
        self._ub = np.concatenate([np.ones(self._n), np.floor(rowsum / 2.0)])
        self._c = np.concatenate([self._w, np.zeros(self._m)])
        # [H | -2I] -- one slack column per detector row.
        self._A = sp.hstack(
            [H.astype(float), -2.0 * sp.eye(self._m, format="csr")],
            format="csr",
        )

        if self._solver == "highs":
            self._setup_highs()

    # ------------------------------------------------------------- HiGHS
    def _setup_highs(self):
        import highspy

        self._highspy = highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        # One solver thread: the pipeline already parallelises across shots
        # with multiple worker processes, so intra-solve threads oversubscribe.
        h.setOptionValue("threads", 1)
        if self._time_limit > 0:
            h.setOptionValue("time_limit", self._time_limit)

        nvars = self._n + self._m
        idx = np.arange(nvars, dtype=np.int32)
        h.addVars(nvars, self._lb, self._ub)
        h.changeColsCost(nvars, idx, self._c)
        h.changeColsIntegrality(
            nvars, idx, np.full(nvars, highspy.HighsVarType.kInteger))

        # Row bounds are placeholders; decode_single rewrites them per shot.
        z = np.zeros(self._m)
        A = self._A
        h.addRows(self._m, z, z, A.nnz,
                  A.indptr[:-1].astype(np.int32),
                  A.indices.astype(np.int32), A.data)
        self._h = h
        self._rows = np.arange(self._m, dtype=np.int32)

    def _decode_highs(self, syndrome):
        s = syndrome.astype(float)
        # Only the RHS changes between shots, so the model stays resident.
        # Measured as a wash against rebuilding it per shot -- kept because it
        # is no more code, not because it buys speed.
        self._h.changeRowsBounds(self._m, self._rows, s, s)
        self._h.run()
        optimal = (self._h.getModelStatus()
                   == self._highspy.HighsModelStatus.kOptimal)
        if not optimal:
            return np.zeros(self._n, dtype=np.uint8), False
        sol = np.asarray(self._h.getSolution().col_value)
        return np.round(sol[:self._n]).astype(np.uint8), True

    # ------------------------------------------------------------- scipy
    def _decode_scipy(self, syndrome):
        from scipy.optimize import Bounds, LinearConstraint, milp

        s = syndrome.astype(float)
        options = {"time_limit": self._time_limit} if self._time_limit > 0 else {}
        res = milp(
            c=self._c,
            constraints=LinearConstraint(self._A, s, s),
            integrality=np.ones(self._n + self._m),
            bounds=Bounds(self._lb, self._ub),
            options=options,
        )
        if not res.success:
            return np.zeros(self._n, dtype=np.uint8), False
        return np.round(res.x[:self._n]).astype(np.uint8), True

    def decode_single(self, syndrome):
        syndrome = np.asarray(syndrome, dtype=np.uint8).ravel()
        if self._solver == "highs":
            return self._decode_highs(syndrome)
        return self._decode_scipy(syndrome)


register_decoder("mle-ilp", MleIlpDecoder, aliases=["mle", "ilp"],
                 backend="cpu")
