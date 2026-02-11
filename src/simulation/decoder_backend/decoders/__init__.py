"""Decoder implementations."""

try:
    from .pymatching import PyMatchingDecoder
except Exception:
    PyMatchingDecoder = None  # type: ignore

__all__ = ["PyMatchingDecoder"]
