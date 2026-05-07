"""Decoder implementations."""

import warnings

try:
    from .pymatching import PyMatchingDecoder
except Exception as exc:
    PyMatchingDecoder = None  # type: ignore
    warnings.warn(f"PyMatching decoder unavailable: {exc}", RuntimeWarning, stacklevel=2)

try:
    from . import bposd  # noqa: F401 - registers bposd decoder
except Exception as exc:
    warnings.warn(f"BPOSD decoder unavailable: {exc}", RuntimeWarning, stacklevel=2)

try:
    from . import mwpf  # noqa: F401 - registers mwpf decoder
except Exception as exc:
    warnings.warn(f"MWPF decoder unavailable: {exc}", RuntimeWarning, stacklevel=2)

try:
    from . import cudaqx  # noqa: F401 - registers nv-qldpc-decoder + bposd gpu
except Exception as exc:
    warnings.warn(f"CUDA-QX decoder unavailable: {exc}", RuntimeWarning, stacklevel=2)

__all__ = ["PyMatchingDecoder"]
