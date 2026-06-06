"""Decoder implementations — soft-import optional backends."""
from __future__ import annotations

import importlib.util
import logging

_log = logging.getLogger(__name__)

# PyMatching is a hard runtime dependency — always available.
try:
    from .pymatching import PyMatchingDecoder
except ImportError as exc:
    PyMatchingDecoder = None  # type: ignore
    _log.debug("pymatching not available: %s", exc)

# stimbposd — optional CPU BP+OSD backend.
if importlib.util.find_spec("stimbposd") is not None:
    try:
        from . import bposd  # noqa: F401 — registers bposd/cpu
    except ImportError as exc:
        _log.debug("stimbposd import failed: %s", exc)
else:
    _log.debug("stimbposd not installed; skipping CPU BP+OSD decoder")

# mwpf — optional MWPF backend.
if importlib.util.find_spec("mwpf") is not None:
    try:
        from . import mwpf  # noqa: F401 — registers mwpf/cpu
    except ImportError as exc:
        _log.debug("mwpf import failed: %s", exc)
else:
    _log.debug("mwpf not installed; skipping MWPF decoder")

# relay_bp — optional Relay-BP backend (sinter-native).
if importlib.util.find_spec("relay_bp") is not None:
    try:
        from . import relay_bp  # noqa: F401 — registers relay-bp/cpu
    except ImportError as exc:
        _log.debug("relay_bp import failed: %s", exc)
else:
    _log.debug("relay_bp not installed; skipping Relay-BP decoder")

# tesseract_decoder — optional beam-search MLE decoder (sinter-native).
# Imported unconditionally on purpose: decoders/tesseract.py is import-safe —
# it does NOT import tesseract_decoder at module load. It self-guards with
# find_spec and registers a lazy wrapper that imports the native extension only
# when the decoder is constructed, so a prebuilt wheel that doesn't match the
# host CPU affects only the Tesseract decoder, not the whole registry.
try:
    from . import tesseract  # noqa: F401 — registers tesseract/cpu if installed
except ImportError as exc:
    _log.debug("tesseract registration skipped: %s", exc)

# cudaq_qec — optional NVIDIA GPU backend.
# Import cudaqx unconditionally: it no longer eager-imports cudaq_qec at module
# scope. The actual `import cudaq_qec` is deferred to first use of the GPU
# decoder, which avoids forking nvidia-smi (and grabbing the NVML global lock)
# in every CPU-decoder worker.
try:
    from . import cudaqx  # noqa: F401 — registers nv-qldpc-decoder + bposd/gpu
except ImportError as exc:
    _log.debug("cudaqx import failed: %s", exc)

__all__ = ["PyMatchingDecoder"]
