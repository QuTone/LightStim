"""MWPF (Minimum-Weight Parity Factor) decoder for QEC, via mwpf package."""

from ..registry import register_decoder

_MWPF_IMPORT_ERROR = None
try:
    from mwpf import SinterMWPFDecoder
    _MWPF_AVAILABLE = True
except ImportError as e1:
    try:
        # mwpf base install may not expose SinterMWPFDecoder (sinter_decoders import
        # can fail silently). Try direct import. Requires: pip install mwpf[stim]
        # (adds stim, sinter, frozendict, frozenlist).
        from mwpf.sinter_decoders import SinterMWPFDecoder
        _MWPF_AVAILABLE = True
    except ImportError as e2:
        SinterMWPFDecoder = None  # type: ignore
        _MWPF_AVAILABLE = False
        _MWPF_IMPORT_ERROR = e2


def _create_mwpf_decoder(**kwargs):
    """Factory for MWPF decoder. Params: cluster_node_limit, timeout, etc."""
    if not _MWPF_AVAILABLE:
        raise ImportError(
            "MWPF decoder requires mwpf, frozendict, frozenlist. "
            "Install with: pip install mwpf frozendict frozenlist"
        )
    return SinterMWPFDecoder(**kwargs)


if _MWPF_AVAILABLE:
    register_decoder("mwpf", SinterMWPFDecoder, aliases=[])
