"""Relay-BP decoder (CPU) — wraps relay_bp's sinter integration.

relay_bp (https://github.com/trmue/relay) already ships sinter-compatible
decoders: ``SinterDecoder_RelayBP`` is a ``sinter.Decoder`` that implements
both ``compile_decoder_for_dem`` and ``decode_via_files``. So this is a
Pattern A integration — we register the upstream class directly, no subclass
or CompiledDecoder of our own. Registration is skipped silently if relay_bp
is not installed.

User-facing params flow straight through ``DecoderConfig(params={...})`` to
``SinterDecoder_RelayBP``. The important ones (see upstream for the rest):

    alpha               : float | None  -- disordered memory strength (None disables)
    gamma0              : float = 0.1    -- ordered memory parameter
    pre_iter            : int   = 60     -- BP iterations for the first ensemble
    num_sets            : int   = 60     -- number of Relay-BP ensemble members
    set_max_iter        : int   = 60     -- iterations per ensemble member
    gamma_dist_interval : (float, float) = (-0.24, 0.66)  -- memory-weight range
    stop_nconv          : int   = 5      -- stop after this many converged solutions
    parallel            : bool  = False  -- relay_bp-internal threading

``gamma_dist_interval`` is highly sensitive and should be tuned per code/DEM
using relay_bp's analysis notebooks.

Note: Relay-BP consumes the *un-decomposed* DEM (it handles hyperedges
natively), which is what the pipeline produces by default — we deliberately do
not set ``decompose_errors`` on the decoder.
"""

from __future__ import annotations

from ..registry import register_decoder

try:
    from relay_bp.stim import SinterDecoder_RelayBP

    register_decoder(
        "relay-bp",
        SinterDecoder_RelayBP,
        aliases=["relay_bp", "relaybp"],
        backend="cpu",
    )
except ImportError:
    pass
