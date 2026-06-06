"""Tesseract decoder (CPU) — Google's beam-search most-likely-error decoder.

tesseract_decoder (https://github.com/quantumlib/tesseract-decoder) ships a
sinter-compatible decoder, ``TesseractSinterDecoder``. This is a Pattern A
integration — we register it under the LightStim name ``"tesseract"`` and let
user params flow straight through ``DecoderConfig(params={...})``. Common ones:

    det_beam       : int   -- beam width (accuracy/speed trade-off, e.g. 50)
    beam_climbing  : bool  -- cost-based beam climbing
    det_penalty    : float -- detector penalty heuristic
    pqlimit        : int   -- priority-queue size limit

See upstream for the full list, and ``tesseract-long-beam`` /
``tesseract-short-beam`` for the paper's reference parameter sets.

Unlike :mod:`relay_bp`, the import is **deferred to construction time** rather
than done at module load. ``tesseract_decoder`` is a native extension and a
prebuilt wheel may not match every CPU; deferring the import means that if it
fails to load on a given host, only an explicit ``DecoderConfig("tesseract")``
is affected rather than the whole LightStim registry. (If it fails to import,
build tesseract_decoder from source — the repo's ``CMakeLists.txt`` uses
``-march=native``.)
"""

from __future__ import annotations

import importlib.util

import sinter
import stim

from ..registry import register_decoder


class TesseractDecoder(sinter.Decoder):
    """Lazy wrapper around ``tesseract_decoder.TesseractSinterDecoder``.

    The upstream class duck-types the sinter interface (it is not a
    ``sinter.Decoder`` subclass), so we delegate ``compile_decoder_for_dem`` to
    it. The ``tesseract_decoder`` import happens in ``__init__`` so a broken
    wheel can't crash the registry at import time.
    """

    def __init__(self, **params: object) -> None:
        from tesseract_decoder import TesseractSinterDecoder

        self._inner = TesseractSinterDecoder(**params)

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel):
        return self._inner.compile_decoder_for_dem(dem=dem)


# find_spec checks availability without importing the native extension. The
# actual import is deferred to TesseractDecoder.__init__.
if importlib.util.find_spec("tesseract_decoder") is not None:
    register_decoder("tesseract", TesseractDecoder, backend="cpu")
