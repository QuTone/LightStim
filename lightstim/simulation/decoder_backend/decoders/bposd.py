"""BP+OSD decoder (CPU) for hypergraph detector error models.

Prefers stimbposd; falls back to ldpc package if stimbposd not installed.
Accepts unified parameter names shared with the GPU backend (see cudaqx.py).
"""

import numpy as np
import scipy.sparse as sp
import sinter
import stim

from ..dem_matrices import dem_to_matrices
from ..registry import register_decoder

_BPOSD_AVAILABLE = False
_BPOSD_SOURCE = None

# Prefer stimbposd (pip install stimbposd)
try:
    from stimbposd import SinterDecoder_BPOSD
    _BPOSD_AVAILABLE = True
    _BPOSD_SOURCE = "stimbposd"
except ImportError:
    SinterDecoder_BPOSD = None  # type: ignore

# Fallback to ldpc (pip install ldpc)
if not _BPOSD_AVAILABLE:
    try:
        from ldpc.sinter_decoders.sinter_bposd_decoder import SinterBpOsdDecoder
        SinterDecoder_BPOSD = SinterBpOsdDecoder
        _BPOSD_AVAILABLE = True
        _BPOSD_SOURCE = "ldpc"
    except ImportError:
        SinterDecoder_BPOSD = None  # type: ignore


# ---------------------------------------------------------------------------
# Unified → stimbposd/ldpc param translation
# ---------------------------------------------------------------------------

_BP_METHOD_TO_CPU = {
    "min_sum":     "minimum_sum",
    "minimum_sum": "minimum_sum",
    "product_sum": "product_sum",
    "sum_product": "product_sum",
}

_OSD_METHOD_NORM = {
    "osd_0": "osd_0", "OSD_0": "osd_0", "osd0": "osd_0",
    "osd_e": "osd_e", "OSD_E": "osd_e",
    "osd_cs": "osd_cs", "OSD_CS": "osd_cs",
}


def _unified_to_cpu(params: dict) -> dict:
    """Translate unified parameter names to stimbposd/ldpc parameter names.

    Unified → CPU mappings:
      max_iterations    → max_bp_iters
      bp_method         → bp_method  ('min_sum' → 'minimum_sum', etc.)
      ms_scaling_factor → ms_scaling_factor  (unchanged)
      osd_order         → osd_order  (unchanged)
      osd_method        → osd_method  (case-normalised to lowercase)
      use_osd           → (dropped; BpOsdDecoder always performs OSD)
    """
    out = {}
    for k, v in params.items():
        if k == "max_iterations":
            out["max_bp_iters"] = v
        elif k == "bp_method":
            out["bp_method"] = _BP_METHOD_TO_CPU.get(v, v)
        elif k == "osd_method":
            out["osd_method"] = _OSD_METHOD_NORM.get(v, v)
        elif k == "use_osd":
            pass  # BpOsdDecoder always performs OSD; this param is a no-op on CPU
        else:
            out[k] = v  # ms_scaling_factor, osd_order, etc. pass through unchanged
    return out


# ---------------------------------------------------------------------------
# Wrapper decoder
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "max_iterations":    1000,
    "osd_order":         10,
    "bp_method":         "min_sum",
    "osd_method":        "osd_cs",
    "ms_scaling_factor": 0,
}


class BpOsdCpuDecoder(sinter.Decoder):
    """Thin wrapper around SinterDecoder_BPOSD that accepts unified parameter names."""

    def __init__(self, **params):
        self._translated = _unified_to_cpu({**_DEFAULTS, **params})
        self._inner = SinterDecoder_BPOSD(**self._translated)

    def compile_decoder_for_dem(self, *, dem):
        # stimbposd limits the requested OSD order to n_columns - n_rows.
        # Detector-rich, low-rank DEMs (notably injection-only protocols) can
        # have more detector rows than error mechanisms, making that value
        # negative and causing ldpc to reject the decoder at compile time.
        # Keep the normal path untouched, and row-reduce only when it would
        # otherwise be impossible for stimbposd to select a valid OSD order.
        if dem.num_detectors > dem.num_errors:
            reduced_dem, detector_rows = _independent_detector_dem(dem)
            compiled = self._inner.compile_decoder_for_dem(dem=reduced_dem)
            return _DetectorSubsetCompiledDecoder(
                compiled=compiled,
                detector_rows=detector_rows,
                num_detectors=dem.num_detectors,
            )
        return self._inner.compile_decoder_for_dem(dem=dem)


class _DetectorSubsetCompiledDecoder(sinter.CompiledDecoder):
    """Project packed syndromes onto independent original detector rows."""

    def __init__(self, *, compiled, detector_rows, num_detectors):
        self._compiled = compiled
        self._detector_rows = np.asarray(detector_rows, dtype=np.int64)
        self._num_detectors = num_detectors

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        syndromes = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            bitorder="little",
        )[:, :self._num_detectors]
        reduced = syndromes[:, self._detector_rows]
        reduced_packed = np.packbits(reduced, axis=1, bitorder="little")
        return self._compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=reduced_packed
        )


def _independent_detector_dem(dem):
    """Return an equivalent DEM containing only independent detector rows."""
    check_matrix, observables_matrix, priors = dem_to_matrices(
        dem,
        sparse=True,
        merge_duplicates=False,
    )
    detector_rows = _independent_row_indices(check_matrix)
    reduced_checks = check_matrix[detector_rows].tocsc()
    observables = observables_matrix.tocsc()

    reduced_dem = stim.DetectorErrorModel()
    for error_index, probability in enumerate(priors):
        targets = [
            stim.target_relative_detector_id(int(row))
            for row in reduced_checks.indices[
                reduced_checks.indptr[error_index]:
                reduced_checks.indptr[error_index + 1]
            ]
        ]
        targets.extend(
            stim.target_logical_observable_id(int(observable))
            for observable in observables.indices[
                observables.indptr[error_index]:
                observables.indptr[error_index + 1]
            ]
        )
        reduced_dem.append("error", float(probability), targets)

    # An observable with no error mechanism still contributes an output bit.
    # Preserve the original DEM's output width for the sinter contract.
    if reduced_dem.num_observables < dem.num_observables:
        reduced_dem.append(
            "logical_observable",
            [],
            [stim.target_logical_observable_id(dem.num_observables - 1)],
        )

    return reduced_dem, detector_rows


def _independent_row_indices(check_matrix):
    """Select original rows forming a GF(2) basis without densifying H."""
    matrix = sp.csr_matrix(check_matrix, dtype=np.uint8)
    pivots = {}
    selected = []

    for row in range(matrix.shape[0]):
        value = 0
        for column in matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]:
            value ^= 1 << int(column)

        while value:
            pivot = value.bit_length() - 1
            basis_row = pivots.get(pivot)
            if basis_row is None:
                pivots[pivot] = value
                selected.append(row)
                break
            value ^= basis_row

    return np.asarray(selected, dtype=np.int64)


if _BPOSD_AVAILABLE:
    register_decoder("bposd", BpOsdCpuDecoder, aliases=["bp_osd"], backend="cpu")
