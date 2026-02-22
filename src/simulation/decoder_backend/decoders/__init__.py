"""Decoder implementations."""

try:
    from .pymatching import PyMatchingDecoder
except Exception:
    PyMatchingDecoder = None  # type: ignore

try:
    from . import bposd  # noqa: F401 - registers bposd decoder
except Exception:
    pass

try:
    from . import mwpf  # noqa: F401 - registers mwpf decoder
except Exception:
    pass

__all__ = ["PyMatchingDecoder"]
