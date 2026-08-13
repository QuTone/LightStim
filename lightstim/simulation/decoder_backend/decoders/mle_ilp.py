"""Exact most-likely-error (MLE) decoder via integer programming.

Solves, for each shot, the exact most-likely-error problem

    minimize   sum_j w_j e_j        with w_j = log((1 - p_j) / p_j)
    subject to H e == s   (mod 2),  e in {0,1}^n

The GF(2) parity is linearised with one bounded integer slack per detector:

    sum_j H[i,j] e_j - 2 k_i == s_i,   0 <= k_i <= floor(rowweight_i / 2)

This is the decoder papers mean by "exact MLE" (more precisely
*most-likely-error*, not degenerate maximum-likelihood: it finds the single
highest-probability error, it does not sum over the logical coset). It is
exponential in the worst case and is intended as a **ground-truth reference**
for small instances, not for production sampling throughput.

The MILP is the fallback, not the first move. Each shot is first attempted with
ACG-ALP: Feldman's fundamental-polytope relaxation with forbidden-set cuts
generated on demand (Taghavi & Siegel), plus redundant-parity-check cuts when
that stalls (Zhang & Siegel). Only a shot that survives both escalates to the
MILP. The valid cuts already generated are retained as MILP constraints instead
of being discarded; they tighten the branch-and-bound relaxation without
changing the integer feasible set. This stays exact; see
:meth:`MleIlpDecoder._adaptive_lp`.

ALP settles many shots without invoking mixed-integer branch-and-bound, while
RPC cuts can settle additional fractional cases. ``max_rpc_rounds=0`` disables
the RPC stage when its overhead is not useful for a particular code family.

Both stages go through scipy -- ``linprog`` for the cuts, ``milp`` for the
fallback. The deterministic benchmark in ``benchmarks/mle`` compares the
optimised path with a direct SciPy MILP on identical surface-code and BB-code
syndromes, checks objective agreement, and records the local environment.

Single-threaded is the right comparison throughout: :class:`SimulationPipeline`
already parallelises across shots, so intra-solve threads only oversubscribe.

Supplied priors are first validated to be in [0, 0.5]; values above 0.5 are
rejected, not clamped. ``max_prior`` is a separate compatibility/model-tuning
cap. Its default of 0.5 leaves every accepted prior unchanged. Setting
``max_prior=c<0.5`` replaces each positive prior ``p`` with ``min(p, c)``, so
accepted priors above ``c`` are intentionally decoded under an altered model.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from ..external import ExternalDecoder
from ..registry import register_decoder

_INT_TOL = 1e-6        # treat an LP component as integral within this
_CUT_TOL = 1e-7        # minimum violation before a cut is worth adding
_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)
_BIT_MASKS = np.left_shift(np.uint8(1), np.arange(8, dtype=np.uint8))
_RPC_XOR_CHUNK_BYTES = 8 << 20

_DEFAULTS = {
    "max_prior": 0.5,      # Applied only after priors are validated in
                            # [0, 0.5]; values above 0.5 are rejected. The
                            # default changes nothing. Setting c < 0.5 maps
                            # each positive p to min(p, c), while p=0 stays
                            # impossible/fixed off.
    "time_limit": 0.0,     # soft wall-clock budget per shot; 0 = unlimited.
                           # Cut generation and the MILP fallback share one
                           # deadline, and a shot that exhausts it is reported
                           # as a decode failure so
                           # DecoderConfig(on_decode_failure=...) can herald it
                           # instead of silently recording a wrong answer.
                           # Worth setting for large DEMs with long solve tails.
                           # Soft, not hard: HiGHS polices its own limit only
                           # at checkpoints, so expect overshoot up to ~2x.
    "max_cut_rounds": 200,  # give up on cut generation and fall back to MILP
    "max_rpc_rounds": 8,    # redundant-parity-check rounds once ALP stalls;
                            # 0 disables ACG and leaves plain ALP
    "max_rpc_pivots": 64,   # Gaussian-elimination pivots per RPC round
    "max_rpc_weight": 256,  # skip derived checks denser than this
    "max_rpc_memory_mb": 512.0,
                            # cap the main packed RPC workspace (cached H,
                            # elimination copy, and bounded temporaries). If the
                            # estimate exceeds this, skip RPC and let the exact
                            # MILP handle the shot. 0 means unlimited.
}


def _weights(priors: np.ndarray, max_prior: float) -> np.ndarray:
    """Return LLR costs after validation and the optional ``max_prior`` cap.

    Priors above 0.5 are rejected. For an accepted prior p, a non-default cap
    c replaces p with min(p, c); the default c=0.5 leaves p unchanged.
    """
    q = np.asarray(priors, dtype=float)
    if not np.all(np.isfinite(q)) or np.any(q < 0.0) or np.any(q > 0.5):
        raise ValueError("priors must be finite probabilities in [0, 0.5]")
    if not 0.0 < max_prior <= 0.5:
        raise ValueError("max_prior must be in (0, 0.5]")
    q = np.minimum(q, max_prior)
    weights = np.zeros_like(q)
    positive = q > 0.0
    weights[positive] = np.log1p(-q[positive]) - np.log(q[positive])
    # Avoid leaving a platform-dependent rounding residue at the exactly
    # uninformative prior. Zero-prior variables are fixed off in setup, so their
    # placeholder weight is never part of a feasible solution.
    weights[q == 0.5] = 0.0
    return weights


class MleIlpDecoder(ExternalDecoder):
    """Exact MLE decoder backed by ``scipy.optimize.milp`` (HiGHS)."""

    output_type = "correction"

    def setup(self, *, H, priors, **_):
        params = {**_DEFAULTS, **self.params}
        self._time_limit = float(params["time_limit"])
        if not np.isfinite(self._time_limit) or self._time_limit < 0:
            raise ValueError("time_limit must be finite and non-negative")

        H = sp.csr_matrix(H, dtype=np.uint8)
        self._m, self._n = H.shape
        priors = np.asarray(priors, dtype=float)
        self._w = _weights(priors, float(params["max_prior"]))
        possible = priors > 0.0

        self._H = H
        self._indptr = H.indptr
        self._indices = H.indices
        self._starts = H.indptr[:-1]
        self._nonempty = H.indptr[1:] > H.indptr[:-1]
        self._max_rounds = int(params["max_cut_rounds"])
        self._max_rpc_rounds = int(params["max_rpc_rounds"])
        self._max_rpc_pivots = int(params["max_rpc_pivots"])
        self._max_rpc_weight = int(params["max_rpc_weight"])
        max_rpc_memory_mb = float(params["max_rpc_memory_mb"])
        if max_rpc_memory_mb < 0:
            raise ValueError("max_rpc_memory_mb must be non-negative")
        self._max_rpc_memory_bytes = (
            int(max_rpc_memory_mb * (1 << 20))
            if max_rpc_memory_mb > 0 else None
        )
        packed_row_bytes = (self._n + 7) // 8
        packed_bytes = self._m * packed_row_bytes
        self._rpc_xor_chunk_rows = max(
            1, _RPC_XOR_CHUNK_BYTES // max(1, packed_row_bytes))
        max_h_row_weight = int(np.diff(self._indptr).max(initial=0))
        build_workspace = packed_bytes + 16 * max_h_row_weight
        eliminate_workspace = (
            2 * packed_bytes
            + min(self._m, self._rpc_xor_chunk_rows) * packed_row_bytes
            + packed_row_bytes + 16 * self._max_rpc_weight
        )
        self._rpc_workspace_bytes = max(
            build_workspace, eliminate_workspace)
        self._packed = None            # built lazily on the first stalled shot

        rowsum = np.asarray(H.sum(axis=1)).ravel()
        lower = np.zeros(self._n + self._m)
        error_upper = possible.astype(float)
        upper = np.concatenate([error_upper, np.floor(rowsum / 2.0)])
        self._bounds = Bounds(lower, upper)
        self._lp_bounds = np.column_stack(
            [np.zeros(self._n, dtype=float), error_upper]
        )
        self._c = np.concatenate([self._w, np.zeros(self._m)])
        self._integrality = np.ones(self._n + self._m)
        # [H | -2I] -- one slack column per detector row.
        self._A = sp.hstack(
            [H.astype(float), -2.0 * sp.eye(self._m, format="csr")],
            format="csr",
        )

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

    def _packed_rows(self):
        """Bit-pack H's rows (little bitorder), built once on first use.

        Packed directly from the CSR indices -- an (m, n) dense intermediate
        would be hundreds of MB on a large QLDPC detector error model.
        """
        if self._packed is None:
            packed = np.zeros((self._m, (self._n + 7) // 8), dtype=np.uint8)
            # Work one CSR row at a time. Constructing a global row-index array
            # would add O(nnz) transient memory before elimination even starts.
            for i in range(self._m):
                indices = self._indices[self._indptr[i]:self._indptr[i + 1]]
                if indices.size:
                    np.bitwise_or.at(
                        packed[i], indices >> 3,
                        np.left_shift(
                            np.uint8(1),
                            (indices & 7).astype(np.uint8)))
            self._packed = packed
        return self._packed

    def _rpc_cuts(self, e, s):
        """Derive redundant parity checks and return any violated cuts.

        When ALP stalls, every original check already satisfies its cuts, so no
        further cut exists from H's own rows. But any GF(2) combination of rows
        is also a valid check -- with syndrome the XOR of the combined bits --
        and one of those may still be violated. Following Zhang & Siegel, the
        combinations come from Gaussian-eliminating the augmented [H | s] with
        the most fractional columns as pivots, so the derived checks isolate
        the fractional variables.

        Elimination stays bit-packed because row XOR on scipy sparse matrices
        creates substantial fill-in and allocation churn. Derived rows are
        popcounted while packed, and the sparse support of a qualifying row is
        extracted directly from its nonzero bytes. If the estimated main
        workspace exceeds ``max_rpc_memory_mb``, no packed cache is allocated
        and the caller falls back to the exact MILP.
        """
        frac = np.abs(e - np.round(e))
        order = np.flatnonzero(frac > _CUT_TOL)
        if order.size == 0:
            return []
        if (self._max_rpc_memory_bytes is not None
                and self._rpc_workspace_bytes > self._max_rpc_memory_bytes):
            return []
        order = order[np.argsort(-frac[order])]

        a = self._packed_rows().copy()
        sy = np.asarray(s, dtype=np.uint8).copy()
        used = np.zeros(self._m, dtype=bool)
        n_piv = 0
        for col in order:
            bits = (a[:, col >> 3] >> (col & 7)) & 1
            ones = np.flatnonzero(bits)
            if ones.size == 0:
                continue
            free = ones[~used[ones]]
            if free.size == 0:
                continue
            p = free[0]
            used[p] = True
            others = ones[ones != p]
            if others.size:
                # Advanced indexing creates a temporary. Chunk it so dense
                # pivot columns cannot briefly allocate another full matrix.
                for start in range(0, others.size,
                                   self._rpc_xor_chunk_rows):
                    chunk = others[start:start + self._rpc_xor_chunk_rows]
                    a[chunk] ^= a[p]
                sy[others] ^= sy[p]
            n_piv += 1
            if n_piv >= self._max_rpc_pivots:
                break

        cuts = []
        for i in range(self._m):
            # Dense derived checks give weak cuts for a lot of work.
            weight = int(_POPCOUNT[a[i]].sum())
            if weight == 0 or weight > self._max_rpc_weight:
                continue
            # Extract the support directly from this row's nonzero bytes. This
            # materialises O(weight) data instead of an m-by-n or length-n
            # dense uint8 array.
            nonzero_bytes = np.flatnonzero(a[i])
            byte_pos, bit_pos = np.nonzero(
                a[i, nonzero_bytes, None] & _BIT_MASKS)
            cols = (nonzero_bytes[byte_pos] << 3) + bit_pos
            # The final packed byte can have padding bits, although H itself
            # never sets them. Keep the bound explicit for defensive safety.
            cols = cols[cols < self._n]
            v = e[cols]
            in_s = v > 0.5
            if (int(in_s.sum()) & 1) == (int(sy[i]) & 1):
                in_s = in_s.copy()
                in_s[np.argmin(np.abs(2.0 * v - 1.0))] ^= True
            g = (1.0 - v[in_s]).sum() + v[~in_s].sum()
            if g < 1.0 - _CUT_TOL:
                cuts.append((cols, np.where(in_s, 1.0, -1.0),
                             float(in_s.sum()) - 1.0))
        return cuts

    def _adaptive_lp(self, s, deadline=None):
        """ACG-ALP decode. Returns the error, or None to escalate to the MILP.

        Solves over the fundamental polytope, generating forbidden-set cuts on
        demand (Taghavi & Siegel). If that stalls on a fractional
        pseudocodeword, redundant-parity-check cuts are added to break it
        (Zhang & Siegel) and plain cut generation resumes.

        Terminating with no violated cut and an integral solution means every
        parity holds and the solution is optimal over a relaxation containing
        every valid error, so it is the exact MLE answer.

        ``deadline`` is an absolute :func:`time.monotonic` value; cut generation
        gives up once past it so the caller's per-shot budget is respected.
        """
        rows, cols, vals, rhs = [], [], [], []
        e = np.zeros(self._n)
        last_a_ub = None
        self._fallback_cuts = None

        def add(cut_list):
            for c_cols, c_vals, c_rhs in cut_list:
                rows.append(np.full(c_cols.size, len(rhs), dtype=np.int64))
                cols.append(c_cols.astype(np.int64))
                vals.append(c_vals)
                rhs.append(c_rhs)

        def resolve():
            nonlocal e, last_a_ub
            a_ub = sp.csr_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(len(rhs), self._n))
            last_a_ub = a_ub
            # Hand the LP whatever budget is left, so one pathological solve
            # cannot blow the per-shot limit on its own.
            options = {}
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                options["time_limit"] = remaining
            lp = linprog(c=self._w, A_ub=a_ub, b_ub=np.asarray(rhs),
                         bounds=self._lp_bounds, method="highs", options=options)
            if not lp.success:
                return False
            e = lp.x
            return True

        def fallback():
            """Preserve valid ALP/RPC cuts to tighten the exact MILP."""
            if rhs and last_a_ub is not None:
                self._fallback_cuts = (
                    last_a_ub, np.asarray(rhs, dtype=float))
            return None

        def out_of_time():
            return deadline is not None and time.monotonic() >= deadline

        def separate_to_convergence():
            for _ in range(self._max_rounds):
                if out_of_time():
                    return False
                violated = self._violated_checks(e, s)
                if violated.size == 0:
                    return True
                add([self._build_cut(e, s, i) for i in violated])
                if not resolve():
                    return False
            return False                    # cut generation did not converge

        if not separate_to_convergence():
            return fallback()

        for _ in range(self._max_rpc_rounds):
            if np.abs(e - np.round(e)).max() < _INT_TOL:
                break
            if out_of_time():
                return fallback()
            rpc = self._rpc_cuts(e, s)
            if not rpc:
                break                       # no derived check is violated
            add(rpc)
            # RPC cuts move the solution, which can re-violate original checks.
            if not resolve() or not separate_to_convergence():
                return fallback()

        if np.abs(e - np.round(e)).max() >= _INT_TOL:
            return fallback()               # fractional: a pseudocodeword
        ei = np.round(e).astype(np.uint8)
        # Verify the parity exactly rather than trusting _INT_TOL.
        return ei if np.array_equal((self._H @ ei) % 2, s) else fallback()

    def decode_single(self, syndrome):
        """Solve one shot: ACG-ALP first, MILP only if it doesn't settle.

        Both paths return the same answer -- see :meth:`_adaptive_lp` for why
        the LP result, when integral, is provably the MILP optimum. Only the
        cost of getting there differs.

        The LP shortcut skips mixed-integer machinery when the relaxation is
        already integral. Fractional shots reuse all generated cuts in the
        exact MILP fallback. See ``benchmarks/mle`` for reproducible timings.
        """
        raw_s = np.asarray(syndrome)
        if raw_s.size != self._m:
            raise ValueError(
                f"syndrome has {raw_s.size} entries; expected {self._m}")
        if not np.all((raw_s == 0) | (raw_s == 1)):
            raise ValueError("syndrome entries must be binary")
        s = raw_s.astype(np.uint8, copy=False).ravel()
        deadline = (time.monotonic() + self._time_limit
                    if self._time_limit > 0 else None)

        e = self._adaptive_lp(s, deadline)
        if e is not None:
            return e, True

        # The MILP gets what is *left* of the budget, not a fresh copy of it --
        # otherwise a shot could spend the full limit twice over.
        options = {}
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return np.zeros(self._n, dtype=np.uint8), False
            options["time_limit"] = remaining

        sf = s.astype(float)
        constraints = [LinearConstraint(self._A, sf, sf)]
        if self._fallback_cuts is not None:
            cut_a, cut_rhs = self._fallback_cuts
            cut_a = sp.hstack(
                [cut_a, sp.csr_matrix((cut_a.shape[0], self._m))],
                format="csr",
            )
            constraints.append(LinearConstraint(
                cut_a, np.full(cut_rhs.size, -np.inf), cut_rhs))

        res = milp(
            c=self._c,
            constraints=constraints,
            integrality=self._integrality,
            bounds=self._bounds,
            options=options,
        )
        if not res.success or res.x is None:
            return np.zeros(self._n, dtype=np.uint8), False
        raw_error = np.asarray(res.x[:self._n], dtype=float)
        if (not np.all(np.isfinite(raw_error))
                or np.abs(raw_error - np.round(raw_error)).max(initial=0.0)
                >= _INT_TOL):
            return np.zeros(self._n, dtype=np.uint8), False
        error = np.round(raw_error).astype(np.uint8)
        # A successful optimizer status should already imply both properties,
        # but never let a backend/tolerance regression become a silent logical
        # miscorrection.  Failed validation is heralded to the decoder chain.
        if not np.array_equal((self._H @ error) % 2, s):
            return np.zeros(self._n, dtype=np.uint8), False
        return error, True


register_decoder("mle-ilp", MleIlpDecoder, aliases=["mle", "ilp"],
                 backend="cpu")
