"""GPU BP+OSD decoder via cudaq_qec nv-qldpc-decoder.

Wraps cudaq_qec's H-matrix API in the sinter CompiledDecoder interface.
Registration is skipped silently if cudaq_qec is not installed.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import sinter
import stim

from ..registry import register_decoder

# ---------------------------------------------------------------------------
# DEM → matrix parser
# ---------------------------------------------------------------------------


def _dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a stim DEM to the three arrays cudaq_qec needs.

    Returns:
        H          -- uint8 parity-check matrix (n_detectors, n_error_mechanisms)
        obs_matrix -- uint8 observable-flip matrix (n_observables, n_error_mechanisms)
        probs      -- float64 prior error probabilities (n_error_mechanisms,)
    """
    n_dets = dem.num_detectors
    n_obs = dem.num_observables

    error_cols: list[dict] = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        dets: list[int] = []
        obs: list[int] = []
        for t in instruction.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        error_cols.append({"p": p, "dets": dets, "obs": obs})

    n_err = len(error_cols)
    # Explicitly C-contiguous (row-major) to match cudaq_qec's expected layout,
    # equivalent to scipy sparse_matrix.todense(order='C').
    H = np.zeros((n_dets, n_err), dtype=np.uint8, order='C')
    obs_matrix = np.zeros((n_obs, n_err), dtype=np.uint8, order='C')
    probs = np.zeros(n_err, dtype=np.float64)

    for e, col in enumerate(error_cols):
        probs[e] = col["p"]
        for d in col["dets"]:
            H[d, e] = 1
        for o in col["obs"]:
            obs_matrix[o, e] = 1

    return np.ascontiguousarray(H), obs_matrix, probs


# ---------------------------------------------------------------------------
# sinter CompiledDecoder wrapper
# ---------------------------------------------------------------------------


class CudaQxCompiledDecoder(sinter.CompiledDecoder):
    """Compiled decoder that wraps a cudaq_qec decoder instance."""

    def __init__(
        self,
        decoder,
        obs_matrix: np.ndarray,
        n_detectors: int,
        n_observables: int,
    ) -> None:
        self._decoder = decoder
        self._obs_matrix = obs_matrix  # (n_obs, n_err) uint8
        self._n_detectors = n_detectors
        self._n_observables = n_observables

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        """Decode bit-packed detector data; return bit-packed observable predictions.

        Args:
            bit_packed_detection_event_data: uint8 array of shape
                (shots, ceil(n_detectors/8)), little-endian bit order.

        Returns:
            uint8 array of shape (shots, ceil(n_observables/8)), little-endian.
        """
        shots = bit_packed_detection_event_data.shape[0]

        # Unpack to (shots, n_dets) float64 for cudaq_qec.
        # Pipeline packs with little-endian to match sinter.Decoder convention.
        syndromes_bits = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little"
        )[:, : self._n_detectors]
        syndromes = syndromes_bits.astype(np.float64)

        # cudaq_qec decode_batch returns list[DecoderResult]
        results = self._decoder.decode_batch(syndromes)

        # Stack corrections: (shots, n_err) uint8
        corrections = np.array(
            [np.asarray(r.result, dtype=np.uint8) for r in results]
        )

        # Logical flips: (shots, n_obs) uint8
        obs_preds = (corrections @ self._obs_matrix.T) % 2

        # Pack to little-endian to match sinter.Decoder convention.
        n_obs_bytes = math.ceil(self._n_observables / 8)
        packed = np.packbits(obs_preds, axis=1, bitorder="little")
        # packbits may produce more bytes than needed; trim to exact size
        return packed[:, :n_obs_bytes]


# ---------------------------------------------------------------------------
# Unified → cudaq_qec param translation
# ---------------------------------------------------------------------------

_BP_METHOD_TO_GPU = {
    "product_sum": 0,
    "sum_product": 0,
    "min_sum":     1,
    "minimum_sum": 1,
}

# cudaq_qec requires integer codes for osd_method
_OSD_METHOD_TO_GPU = {
    "osd_0": 1, "OSD_0": 1, "osd0": 1,
    "osd_e": 2, "OSD_E": 2,
    "osd_cs": 3, "OSD_CS": 3,
}


def _unified_to_gpu(params: dict) -> dict:
    """Translate unified parameter names to cudaq_qec parameter names.

    Unified → GPU mappings:
      max_iterations    → max_iterations  (unchanged)
      bp_method         → bp_method  ('min_sum'→1, 'product_sum'→0)
      ms_scaling_factor → scale_factor
      osd_order         → osd_order  (unchanged)
      osd_method        → osd_method  (case-normalised to lowercase)
      use_osd           → use_osd  (unchanged)
    """
    out = {}
    for k, v in params.items():
        if k == "ms_scaling_factor":
            out["scale_factor"] = v
        elif k == "bp_method" and isinstance(v, str):
            out["bp_method"] = _BP_METHOD_TO_GPU.get(v, v)
        elif k == "osd_method" and isinstance(v, str):
            out["osd_method"] = _OSD_METHOD_TO_GPU.get(v, v)
        else:
            out[k] = v  # max_iterations, osd_order, use_osd, scale_factor, etc.
    return out


# ---------------------------------------------------------------------------
# sinter Decoder (picklable, stores only primitives)
# ---------------------------------------------------------------------------


class CudaQxDecoder(sinter.Decoder):
    """GPU-accelerated BP+OSD via cudaq_qec.

    Accepts unified parameter names shared with the CPU backend (see bposd.py):
      max_iterations, bp_method, ms_scaling_factor, osd_order, osd_method, use_osd.

    Args:
        decoder_name: cudaq_qec decoder name (default: ``"nv-qldpc-decoder"``).
        **params: Unified or cudaq_qec-native keyword arguments.
                  ``error_rate_vec`` is always derived from the DEM probabilities.
    """

    def __init__(
        self,
        decoder_name: str = "nv-qldpc-decoder",
        **params: Any,
    ) -> None:
        self._decoder_name = decoder_name
        self._params: Dict[str, Any] = params

    # Required by sinter.Decoder
    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CudaQxCompiledDecoder:
        import cudaq_qec as qec  # noqa: F401

        H, obs_matrix, probs = _dem_to_matrices(dem)
        # Defaults match the reference configuration; user params override them.
        params = {
            "error_rate_vec": probs,
            "use_sparsity":   True,
            "use_osd":        True,
            "bp_method":      1,          # min_sum
            "osd_method":     3,          # osd_cs
            "scale_factor":   0,
            "max_iterations": 1000,
            "osd_order":      10,
            "bp_batch_size":  1_000,
            **_unified_to_gpu(self._params),
        }
        decoder = qec.get_decoder(self._decoder_name, H, **params)
        return CudaQxCompiledDecoder(
            decoder, obs_matrix, dem.num_detectors, dem.num_observables
        )

    # sinter calls this when pickling workers; we only store primitives
    def __repr__(self) -> str:
        return f"CudaQxDecoder(decoder_name={self._decoder_name!r}, **{self._params!r})"


# ---------------------------------------------------------------------------
# Registration (skipped if cudaq_qec not installed)
# ---------------------------------------------------------------------------

_CUDAQX_AVAILABLE = False
try:
    import cudaq_qec  # noqa: F401
    _CUDAQX_AVAILABLE = True
except ImportError:
    pass

if _CUDAQX_AVAILABLE:
    register_decoder("nv-qldpc-decoder", CudaQxDecoder, backend="gpu")
    # GPU override for the "bposd" name so DecoderConfig(name="bposd", backend="gpu") works
    register_decoder("bposd", CudaQxDecoder, backend="gpu")
