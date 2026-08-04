"""PyMatching MWPM decoder (CPU) — wraps pymatching directly.

Matching requires a graphlike error model, so this decoder declares
``decompose_errors = True``: the pipeline/worker must hand it the DEM with
stim's suggested hyperedge decomposition.  Building the matcher from the raw
undecomposed DEM silently mangles hyperedges and costs real effective
distance (measured on the diagonal-schedule d=3 memory: LER 1.03e-2 vs
5.9e-3 at p=2e-3, LER-slope d_eff ~1.8 vs ~2.7).

``enable_correlations=True`` (default, matching the Kishony-Fowler reference
benchmarks) additionally reweights the second matching pass with the
decomposed hyperedges' correlation structure; pass
``DecoderConfig('pymatching', params={'enable_correlations': False})`` for
plain MWPM.
"""
from __future__ import annotations

import numpy as np
import sinter
import stim

from ..registry import register_decoder

try:
    import pymatching

    class _CompiledPyMatching(sinter.CompiledDecoder):
        def __init__(
            self, matching: pymatching.Matching, enable_correlations: bool
        ) -> None:
            self._m = matching
            self._corr = enable_correlations

        def decode_shots_bit_packed(
            self, *, bit_packed_detection_event_data: np.ndarray
        ) -> np.ndarray:
            return self._m.decode_batch(
                bit_packed_detection_event_data,
                bit_packed_shots=True,
                bit_packed_predictions=True,
                enable_correlations=self._corr,
            )

    class PyMatchingDecoder(sinter.Decoder):
        """MWPM decoder backed by pymatching."""

        decompose_errors = True

        def __init__(self, enable_correlations: bool = True) -> None:
            self.enable_correlations = enable_correlations

        def _build(
            self, dem: stim.DetectorErrorModel
        ) -> tuple[pymatching.Matching, bool]:
            """Build the matcher; returns (matching, correlations_active).

            Falls back to plain MWPM when correlations are unsupported for
            this DEM (pymatching rejects error probabilities > 0.5 with
            correlations enabled)."""
            if self.enable_correlations:
                try:
                    return (
                        pymatching.Matching.from_detector_error_model(
                            dem, enable_correlations=True
                        ),
                        True,
                    )
                except ValueError:
                    pass
            return (
                pymatching.Matching.from_detector_error_model(dem),
                False,
            )

        def compile_decoder_for_dem(
            self, *, dem: stim.DetectorErrorModel
        ) -> sinter.CompiledDecoder:
            return _CompiledPyMatching(*self._build(dem))

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
            matching, corr = self._build(dem)
            return matching.decode_batch(
                bit_packed_detection_event_data,
                bit_packed_shots=True,
                bit_packed_predictions=True,
                enable_correlations=corr,
            )

    register_decoder("pymatching", PyMatchingDecoder, aliases=["mwpm"])

except ImportError:
    pass
