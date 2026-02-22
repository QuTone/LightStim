"""BP+OSD decoder (CPU) for quantum LDPC codes.

Prefers stimbposd; falls back to ldpc package if stimbposd not installed.
"""

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


def _create_bposd_decoder(**kwargs):
    """Factory for BP+OSD decoder. Params: max_bp_iters, bp_method, osd_order, osd_method, etc."""
    if not _BPOSD_AVAILABLE:
        raise ImportError(
            "BP+OSD decoder requires stimbposd or ldpc. Install with: pip install stimbposd"
        )
    return SinterDecoder_BPOSD(**kwargs)


if _BPOSD_AVAILABLE:
    register_decoder("bposd", SinterDecoder_BPOSD, aliases=["bp_osd"])
