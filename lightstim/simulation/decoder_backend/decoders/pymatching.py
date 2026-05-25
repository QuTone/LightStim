"""PyMatching decoder with decompose_errors + enable_correlations for proper hyperedge handling."""

import numpy as np
import sinter

from ..registry import register_decoder


class _PyMatchingCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, matching):
        self._matching = matching
        self._num_detectors = matching.num_detectors

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")
        dets = dets[:, :self._num_detectors]
        predictions = self._matching.decode_batch(dets)
        return np.packbits(predictions, axis=1, bitorder="little")


class PyMatchingDecoder(sinter.Decoder):
    """PyMatching decoder using stim decompose_errors + pymatching enable_correlations.

    This is the correct way to use pymatching for circuits with hyperedges.
    The worker passes decompose_errors=True to stim (triggered by the class attribute),
    and compile_decoder_for_dem passes enable_correlations=True to pymatching so that
    the residual correlated errors after stim decomposition are handled exactly.
    """

    # Tells the simulation worker to call circuit.detector_error_model(decompose_errors=True)
    decompose_errors = True

    def compile_decoder_for_dem(self, *, dem: "stim.DetectorErrorModel") -> _PyMatchingCompiledDecoder:
        try:
            import pymatching
        except ImportError as ex:
            raise ImportError(
                "The decoder 'pymatching' isn't installed. "
                "Fix with: pip install pymatching"
            ) from ex
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        return _PyMatchingCompiledDecoder(matching)


try:
    import pymatching  # noqa: F401
    register_decoder("pymatching", PyMatchingDecoder, aliases=["mwpm"])
except ImportError:
    pass
