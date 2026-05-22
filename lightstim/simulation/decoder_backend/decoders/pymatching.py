"""PyMatching MWPM decoder (CPU) — wraps pymatching directly."""
from __future__ import annotations

import numpy as np
import sinter
import stim

from ..registry import register_decoder

try:
    import pymatching

    class _CompiledPyMatching(sinter.CompiledDecoder):
        def __init__(self, matching: pymatching.Matching) -> None:
            self._m = matching

        def decode_shots_bit_packed(
            self, *, bit_packed_detection_event_data: np.ndarray
        ) -> np.ndarray:
            return self._m.decode_batch(
                bit_packed_detection_event_data,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            )

    class PyMatchingDecoder(sinter.Decoder):
        """MWPM decoder backed by pymatching."""

        def compile_decoder_for_dem(
            self, *, dem: stim.DetectorErrorModel
        ) -> sinter.CompiledDecoder:
            return _CompiledPyMatching(
                pymatching.Matching.from_detector_error_model(dem)
            )

        def decode_shots_bit_packed(
            self,
            *,
            bit_packed_detection_event_data: np.ndarray,
            pre_compiled_decoder: sinter.CompiledDecoder | None,
            dem: stim.DetectorErrorModel,
        ) -> np.ndarray:
            if pre_compiled_decoder is not None:
                return pre_compiled_decoder.decode_shots_bit_packed(
                    bit_packed_detection_event_data=bit_packed_detection_event_data
                )
            return pymatching.Matching.from_detector_error_model(dem).decode_batch(
                bit_packed_detection_event_data,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            )

    register_decoder("pymatching", PyMatchingDecoder, aliases=["mwpm"])

except ImportError:
    pass
