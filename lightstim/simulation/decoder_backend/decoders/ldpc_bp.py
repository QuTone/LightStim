"""Plain belief-propagation decoder (CPU) backed by ``ldpc.BpDecoder``.

This is *pure* BP — no OSD, no relay/ensemble post-processing. It runs belief
propagation on the un-decomposed DEM parity-check matrix (BP handles hyperedges
natively), returns the correction over error mechanisms, and reports per-shot
convergence so ``DecoderConfig(on_decode_failure=...)`` can herald or count
non-converged shots.

It is registered via the :class:`ExternalDecoder` facade (Pattern D), so
bit-packing, the correction→observable multiply, and multi-process workers are
all handled by :class:`SimulationPipeline`.

Params (flow through ``DecoderConfig(params={...})``; see ldpc for the rest):
    max_iter          : int    = 100      -- BP iterations per shot
    bp_method         : str    = 'minimum_sum'  ('minimum_sum' | 'product_sum')
    ms_scaling_factor : float  = 0.0      -- min-sum scaling; 0 = ldpc's
                                             dynamic scaling schedule
    schedule          : str    = 'serial' ('serial' | 'parallel')

The defaults are serial schedule + dynamic min-sum scaling: on dense QLDPC
detector error models the parallel flooding schedule oscillates (convergence
can drop from ~99% to ~1% on the same instance), and fixed scaling factors
underperform dynamic scaling.

Note: OSD is deliberately *not* applied. On high-rate QLDPC codes (e.g. Kasai)
plain BP alone has a high logical error floor; pair it with BP+OSD (the
``bposd`` decoder) if you want competitive thresholds.
"""

from __future__ import annotations

import numpy as np

from ..external import ExternalDecoder
from ..registry import register_decoder

try:
    from ldpc import BpDecoder
    _LDPC_AVAILABLE = True
except ImportError:  # pragma: no cover - guarded by find_spec in __init__
    BpDecoder = None  # type: ignore
    _LDPC_AVAILABLE = False


_DEFAULTS = {
    "max_iter": 100,
    "bp_method": "minimum_sum",
    "ms_scaling_factor": 0.0,
    "schedule": "serial",
}


class LdpcBpDecoder(ExternalDecoder):
    """Plain BP from the ``ldpc`` library, over the DEM error mechanisms."""

    output_type = "correction"

    def setup(self, *, H, priors, **_):
        params = {**_DEFAULTS, **self.params}
        self._bp = BpDecoder(
            H,
            error_channel=list(priors),
            max_iter=int(params["max_iter"]),
            bp_method=params["bp_method"],
            ms_scaling_factor=float(params["ms_scaling_factor"]),
            schedule=params["schedule"],
            # We always decode detector syndromes; state it so ldpc doesn't
            # error out trying to infer it on small/square check matrices.
            input_vector_type="syndrome",
        )

    def decode_single(self, syndrome):
        correction = self._bp.decode(syndrome.astype(np.uint8))
        # ``converge`` is True when BP settled on a valid solution; report a
        # False flag otherwise so on_decode_failure policy can act on it.
        return correction, bool(self._bp.converge)


if _LDPC_AVAILABLE:
    register_decoder(
        "ldpc-bp",
        LdpcBpDecoder,
        aliases=["ldpc_bp", "bp"],
        backend="cpu",
    )
