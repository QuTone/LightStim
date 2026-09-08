"""Optional IonQ BP-guided beam search; see docs/ionq_beam_search.md.

Build upstream separately. No native code or build dependency is vendored.
"""
from __future__ import annotations

import importlib
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from ..external import ExternalDecoder
from ..registry import register_decoder


def _load_native():
    # Support upstream/decoder and upstream root on PYTHONPATH, respectively.
    errors = []
    for name in ("beam_search_decoder", "decoder.beam_search_decoder"):
        try:
            return importlib.import_module(name).BeamSearchDecoder
        except (ImportError, OSError, AttributeError) as exc:
            errors.append(exc)
    raise ImportError(
        "ionq-beam-search requires the separately built IonQ BeamSearchDecoder. "
        "Run 'python setup.py build_ext --inplace' in its decoder directory "
        "and add that directory to PYTHONPATH. See docs/ionq_beam_search.md. "
        f"Import errors: {'; '.join(str(e) for e in errors)}"
    ) from errors[0]


class IonqBeamSearchDecoder(ExternalDecoder):
    """Paper's beam8_230iters defaults; returns error-mechanism corrections.

    Use on_decode_failure='ignore' in DecoderConfig for the upstream Sinter
    wrapper's paper-comparison policy (always score the returned prediction).
    """

    output_type = "correction"
    decompose_errors = False

    def __init__(self, *, max_rounds=10, beam_width=8, num_results=1,
                 initial_iters=30, iters_per_round=20):
        params = dict(max_rounds=max_rounds, beam_width=beam_width,
                      num_results=num_results, initial_iters=initial_iters,
                      iters_per_round=iters_per_round)
        for name, value in params.items():
            # Zero iterations leave upstream search scores undefined; other
            # zero values silently select unexpected defaults upstream.
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        super().__init__(**{k: int(v) for k, v in params.items()})
        _load_native()  # fail in the parent before workers start

    def setup(self, *, H, priors, **_):
        self._H = sp.csr_matrix(H, dtype=np.uint8, copy=True)
        self._H.eliminate_zeros()
        priors = np.asarray(priors, dtype=float)
        if priors.shape != (self._H.shape[1],):
            raise ValueError("priors must have one entry per error mechanism")
        if not np.all(np.isfinite(priors) & (priors > 0) & (priors < 1)):
            raise ValueError("ionq-beam-search requires finite priors strictly between 0 and 1")
        if np.any(self._H.data != 1):
            raise ValueError("H must be a binary parity-check matrix")
        # Avoid upstream's LLR_sums[0] access for an empty/noiseless DEM.
        self._inner = (
            _load_native()(pcm=self._H, error_channel=priors.tolist(), **self.params)
            if self._H.shape[1] else None
        )

    def decode_single(self, syndrome):
        original = np.asarray(syndrome)
        if original.shape != (self._H.shape[0],) or not np.all(
            (original == 0) | (original == 1)
        ):
            raise ValueError("syndrome must be a binary vector with one entry per detector")
        if self._inner is None:
            return np.zeros(0, dtype=np.uint8), not bool(np.any(original))
        correction = np.array(
            self._inner.decode(np.array(original, dtype=np.uint8, copy=True)),
            dtype=np.uint8, copy=True,
        )
        if correction.shape != (self._H.shape[1],) or np.any(correction > 1):
            raise ValueError("IonQ decoder returned an invalid correction vector")
        # converge refers to the last explored branch, not necessarily the
        # best correction returned after a multi-result search.
        valid = np.array_equal((self._H @ correction) & 1, original)
        return correction, bool(valid)

    def decode_batch(self, syndromes):
        # Explicit shape handles zero shots in correction output mode too.
        corrections = np.empty((len(syndromes), self._H.shape[1]), dtype=np.uint8)
        flags = np.empty(len(syndromes), dtype=bool)
        for i, syndrome in enumerate(syndromes):
            corrections[i], flags[i] = self.decode_single(syndrome)
        return corrections, flags


# Registration never imports native code. Selecting the name without an
# installed extension produces an actionable dependency error.
register_decoder("ionq-beam-search", IonqBeamSearchDecoder, backend="cpu")
