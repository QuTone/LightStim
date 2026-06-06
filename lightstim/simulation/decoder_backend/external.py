"""Friendly base class for integrating an external decoder.

Most third-party / research decoders are *not* ``sinter``-compatible: they want
a parity-check matrix and priors, they operate on plain (unpacked) syndrome
arrays, and some of them can *fail to converge* (BP) — a fact the standard
``sinter`` bit-packed contract has no way to express.

:class:`ExternalDecoder` is a thin facade over that contract. A subclass only
has to:

1. declare ``output_type`` (``"correction"`` or ``"observables"``),
2. build its decoder in :meth:`setup` (called once per DEM), and
3. implement :meth:`decode_batch` *or* :meth:`decode_single` (whichever is
   natural — lightstim bridges the other).

Everything else — bit-packing, the observable-matrix multiply, and routing
per-shot convergence flags to the pipeline — is handled here.

Example::

    from lightstim.simulation.decoder_backend.external import ExternalDecoder
    from lightstim.simulation.decoder_backend.registry import register_decoder

    class MyDecoder(ExternalDecoder):
        output_type = "correction"

        def setup(self, *, H, priors, **_):
            self._inner = my_lib.Decoder(H, priors, **self.params)

        def decode_single(self, syndrome):
            correction, converged = self._inner.decode(syndrome)
            return correction, converged   # flag=None means "always converged"

    register_decoder("my-decoder", MyDecoder, backend="cpu")
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import sinter
import stim

from .dem_matrices import dem_to_matrices

# Permitted values for ``ExternalDecoder.output_type``.
_OUTPUT_CORRECTION = "correction"
_OUTPUT_OBSERVABLES = "observables"
_OUTPUT_TYPES = (_OUTPUT_CORRECTION, _OUTPUT_OBSERVABLES)


class ExternalDecoder(sinter.Decoder):
    """Base class for user-supplied decoders. Subclass and override.

    Class attribute:
        output_type: **Required.** ``"correction"`` if :meth:`decode_batch` /
            :meth:`decode_single` return a correction over error mechanisms
            (length ``n_error_mechanisms``) — lightstim then computes the
            observable flips for you. ``"observables"`` if they already return
            logical-observable flips (length ``n_observables``).
    """

    # Subclasses MUST set this; left None so a forgotten declaration fails loudly
    # rather than silently decoding on the wrong axis.
    output_type: Optional[str] = None

    def __init__(self, **params: Any) -> None:
        # Stored as primitives so the instance is picklable across workers.
        self.params = dict(params)

    # ------------------------------------------------------------------ #
    # Override hooks
    # ------------------------------------------------------------------ #

    def setup(
        self,
        *,
        dem: stim.DetectorErrorModel,
        H: np.ndarray,
        obs_matrix: np.ndarray,
        priors: np.ndarray,
        num_detectors: int,
        num_observables: int,
    ) -> None:
        """Build the underlying decoder. Called once per DEM, before any decode.

        Override and keep whatever you need on ``self``. Accept ``**_`` to
        ignore the arguments you don't use.
        """

    def decode_batch(
        self, syndromes: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Decode a batch of syndromes.

        Args:
            syndromes: uint8 array, shape ``(n_shots, n_detectors)``, unpacked.

        Returns:
            ``(predictions, flags)`` where ``predictions`` is a uint8 array of
            shape ``(n_shots, k)`` — ``k = n_error_mechanisms`` if
            ``output_type == "correction"`` else ``n_observables`` — and
            ``flags`` is either ``None`` (every shot converged) or a boolean
            array of shape ``(n_shots,)`` that is ``False`` for shots the
            decoder failed on.
        """
        raise NotImplementedError

    def decode_single(
        self, syndrome: np.ndarray
    ) -> tuple[np.ndarray, Optional[bool]]:
        """Decode one syndrome.

        Args:
            syndrome: uint8 array, shape ``(n_detectors,)``, unpacked.

        Returns:
            ``(prediction, flag)`` where ``prediction`` is a uint8 array of
            shape ``(k,)`` and ``flag`` is ``True``/``None`` (converged) or
            ``False`` (decode failed).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # sinter.Decoder glue
    # ------------------------------------------------------------------ #

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> "_ExternalCompiledDecoder":
        if self.output_type not in _OUTPUT_TYPES:
            raise ValueError(
                f"{type(self).__name__}.output_type must be one of {_OUTPUT_TYPES}, "
                f"got {self.output_type!r}. Set it as a class attribute."
            )

        prefers_batch = self._overrides("decode_batch")
        if not prefers_batch and not self._overrides("decode_single"):
            raise NotImplementedError(
                f"{type(self).__name__} must override decode_batch or decode_single."
            )

        H, obs_matrix, priors = dem_to_matrices(dem)
        self.setup(
            dem=dem,
            H=H,
            obs_matrix=obs_matrix,
            priors=priors,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )
        return _ExternalCompiledDecoder(
            decoder=self,
            obs_matrix=obs_matrix,
            n_detectors=dem.num_detectors,
            n_observables=dem.num_observables,
            output_type=self.output_type,
            prefers_batch=prefers_batch,
        )

    def _overrides(self, method_name: str) -> bool:
        """True if a subclass overrides ``method_name`` (vs. the base stub)."""
        return getattr(type(self), method_name) is not getattr(
            ExternalDecoder, method_name
        )


class _ExternalCompiledDecoder(sinter.CompiledDecoder):
    """Adapts an :class:`ExternalDecoder` to ``sinter.CompiledDecoder``.

    Owns all bit-(un)packing and the correction → observable-flip multiply.
    Per-shot convergence flags ride a side channel: after each
    :meth:`decode_shots_bit_packed` call, ``self.last_flags`` holds either
    ``None`` (all shots converged) or a boolean array (``False`` = failed). The
    pipeline reads it in-process; it is never serialised through the bit-packed
    return value.
    """

    def __init__(
        self,
        *,
        decoder: ExternalDecoder,
        obs_matrix: np.ndarray,
        n_detectors: int,
        n_observables: int,
        output_type: str,
        prefers_batch: bool,
    ) -> None:
        self._decoder = decoder
        self._obs_matrix = obs_matrix  # (n_obs, n_err) uint8
        self._n_detectors = n_detectors
        self._n_observables = n_observables
        self._output_type = output_type
        self._prefers_batch = prefers_batch
        # Side channel read by the pipeline after each decode call.
        self.last_flags: Optional[np.ndarray] = None

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        syndromes = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little"
        )[:, : self._n_detectors].astype(np.uint8)
        n_shots = syndromes.shape[0]

        if self._prefers_batch:
            predictions, flags = self._decoder.decode_batch(syndromes)
            predictions = np.asarray(predictions, dtype=np.uint8)
            flags = self._normalize_flags(flags, n_shots)
        else:
            preds: list[np.ndarray] = []
            single_flags: list[bool] = []
            any_flag = False
            for i in range(n_shots):
                pred, flag = self._decoder.decode_single(syndromes[i])
                preds.append(np.asarray(pred, dtype=np.uint8))
                if flag is None:
                    single_flags.append(True)
                else:
                    single_flags.append(bool(flag))
                    any_flag = True
            predictions = (
                np.stack(preds, axis=0)
                if preds
                else np.zeros((0, self._n_observables), dtype=np.uint8)
            )
            flags = np.asarray(single_flags, dtype=bool) if any_flag else None

        self.last_flags = flags

        if self._output_type == _OUTPUT_CORRECTION:
            # corrections (n_shots, n_err) @ (n_err, n_obs) -> (n_shots, n_obs) mod 2
            obs_preds = (
                predictions.astype(np.int64) @ self._obs_matrix.T.astype(np.int64)
            ) % 2
            obs_preds = obs_preds.astype(np.uint8)
        else:
            obs_preds = (predictions % 2).astype(np.uint8)

        n_obs_bytes = math.ceil(self._n_observables / 8) if self._n_observables else 0
        packed = np.packbits(obs_preds, axis=1, bitorder="little")
        return packed[:, :n_obs_bytes]

    @staticmethod
    def _normalize_flags(
        flags: Optional[np.ndarray], n_shots: int
    ) -> Optional[np.ndarray]:
        """Coerce a batch flag return into ``None`` or a length-``n_shots`` bool array."""
        if flags is None:
            return None
        flags = np.asarray(flags, dtype=bool).reshape(-1)
        if flags.shape[0] != n_shots:
            raise ValueError(
                f"decode_batch returned {flags.shape[0]} flags for {n_shots} shots."
            )
        # All-converged is equivalent to no flags; collapse so the pipeline can skip work.
        return None if flags.all() else flags
