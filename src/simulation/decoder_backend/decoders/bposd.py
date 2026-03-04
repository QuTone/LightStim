"""BP+OSD decoder (CPU) for quantum LDPC codes.

Prefers stimbposd; falls back to ldpc package if stimbposd not installed.
Accepts unified parameter names shared with the GPU backend (see cudaqx.py).
"""

import sinter

from ..registry import register_decoder

_BPOSD_AVAILABLE = False
_BPOSD_SOURCE = None

# Prefer stimbposd (pip install stimbposd)
try:
    from stimbposd import SinterDecoder_BPOSD
    _BPOSD_AVAILABLE = True
    _BPOSD_SOURCE = "stimbposd"
except ImportError:
    SinterDecoder_BPOSD = None  # type: ignore

# Fallback to ldpc (pip install ldpc)
if not _BPOSD_AVAILABLE:
    try:
        from ldpc.sinter_decoders.sinter_bposd_decoder import SinterBpOsdDecoder
        SinterDecoder_BPOSD = SinterBpOsdDecoder
        _BPOSD_AVAILABLE = True
        _BPOSD_SOURCE = "ldpc"
    except ImportError:
        SinterDecoder_BPOSD = None  # type: ignore


# ---------------------------------------------------------------------------
# Unified → stimbposd/ldpc param translation
# ---------------------------------------------------------------------------

_BP_METHOD_TO_CPU = {
    "min_sum":     "minimum_sum",
    "minimum_sum": "minimum_sum",
    "product_sum": "product_sum",
    "sum_product": "product_sum",
}

_OSD_METHOD_NORM = {
    "osd_0": "osd_0", "OSD_0": "osd_0", "osd0": "osd_0",
    "osd_e": "osd_e", "OSD_E": "osd_e",
    "osd_cs": "osd_cs", "OSD_CS": "osd_cs",
}


def _unified_to_cpu(params: dict) -> dict:
    """Translate unified parameter names to stimbposd/ldpc parameter names.

    Unified → CPU mappings:
      max_iterations    → max_bp_iters
      bp_method         → bp_method  ('min_sum' → 'minimum_sum', etc.)
      ms_scaling_factor → ms_scaling_factor  (unchanged)
      osd_order         → osd_order  (unchanged)
      osd_method        → osd_method  (case-normalised to lowercase)
      use_osd           → (dropped; BpOsdDecoder always performs OSD)
    """
    out = {}
    for k, v in params.items():
        if k == "max_iterations":
            out["max_bp_iters"] = v
        elif k == "bp_method":
            out["bp_method"] = _BP_METHOD_TO_CPU.get(v, v)
        elif k == "osd_method":
            out["osd_method"] = _OSD_METHOD_NORM.get(v, v)
        elif k == "use_osd":
            pass  # BpOsdDecoder always performs OSD; this param is a no-op on CPU
        else:
            out[k] = v  # ms_scaling_factor, osd_order, etc. pass through unchanged
    return out


# ---------------------------------------------------------------------------
# Wrapper decoder
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "max_iterations":    1000,
    "osd_order":         10,
    "bp_method":         "min_sum",
    "osd_method":        "osd_cs",
    "ms_scaling_factor": 0,
}


class BpOsdCpuDecoder(sinter.Decoder):
    """Thin wrapper around SinterDecoder_BPOSD that accepts unified parameter names."""

    def __init__(self, **params):
        translated = _unified_to_cpu({**_DEFAULTS, **params})
        self._inner = SinterDecoder_BPOSD(**translated)

    def compile_decoder_for_dem(self, *, dem):
        return self._inner.compile_decoder_for_dem(dem=dem)


if _BPOSD_AVAILABLE:
    register_decoder("bposd", BpOsdCpuDecoder, aliases=["bp_osd"], backend="cpu")
