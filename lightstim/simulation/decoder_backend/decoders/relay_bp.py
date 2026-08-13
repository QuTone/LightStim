"""Relay-BP decoder with convergence-aware chain integration.

The upstream :mod:`relay_bp` sinter adapter returns observable predictions but
does not expose per-shot convergence. That is fine when Relay-BP is the final
decoder, but it prevents the three-tier hierarchy used by arXiv:2604.16209 and
arXiv:2608.07431: the chain cannot tell which Relay-BP shots must escalate to
the exact MLE fallback.

This adapter uses Relay-BP's detailed batch API and publishes ``last_flags``
with the same convention as LightStim's external decoders (``False`` means
non-converged). It otherwise follows the upstream sinter implementation,
including DEM pruning and bias handling.

User-facing params flow straight through ``DecoderConfig(params={...})``:

    alpha                           : float | None = None (0 = dynamic ramp)
    gamma0                          : float = 0.1
    alpha_iteration_scaling_factor : float = 1.0
    pre_iter                        : int = 60
    num_sets                        : int = 60
    set_max_iter                    : int = 60
    gamma_dist_interval             : tuple = (-0.24, 0.66)
    explicit_gammas                 : array | None = None
    stop_nconv                      : int = 5
    stopping_criterion              : str = "nconv"
    precision                       : str = "f64" ("f32" matches the papers)
    seed                            : int = 0
    parallel                        : bool = False

Relay-BP consumes the un-decomposed DEM and handles hyperedges natively.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import sinter
import stim

from ..registry import register_decoder

try:
    import relay_bp
    from relay_bp.stim.sinter import CheckMatrices
except ImportError:  # pragma: no cover - registration is dependency-gated
    relay_bp = None  # type: ignore
    CheckMatrices = None  # type: ignore


class RelayBpDecoder(sinter.Decoder):
    """Relay-BP sinter decoder that preserves per-shot convergence flags."""

    def __init__(
        self,
        alpha: float | None = None,
        gamma0: float = 0.1,
        alpha_iteration_scaling_factor: float = 1.0,
        pre_iter: int = 60,
        num_sets: int = 60,
        set_max_iter: int = 60,
        gamma_dist_interval: tuple[float, float] = (-0.24, 0.66),
        explicit_gammas: Optional[np.ndarray] = None,
        stop_nconv: int = 5,
        stopping_criterion: str = "nconv",
        precision: str = "f64",
        seed: int = 0,
        logging: bool = False,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
    ) -> None:
        precision = str(precision).lower()
        if precision not in {"f32", "f64"}:
            raise ValueError("precision must be 'f32' or 'f64'")
        self.alpha = alpha
        self.gamma0 = float(gamma0)
        self.alpha_iteration_scaling_factor = float(
            alpha_iteration_scaling_factor)
        self.pre_iter = int(pre_iter)
        self.num_sets = int(num_sets)
        self.set_max_iter = int(set_max_iter)
        self.gamma_dist_interval = tuple(gamma_dist_interval)
        self.explicit_gammas = explicit_gammas
        self.stop_nconv = int(stop_nconv)
        self.stopping_criterion = stopping_criterion
        self.precision = precision
        self.seed = int(seed)
        self.logging = logging
        self.parallel = bool(parallel)
        self.decomposed_hyperedges = decomposed_hyperedges
        self.prune_decided_errors = bool(prune_decided_errors)
        self.threshold = float(threshold)

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> "_CompiledRelayBpDecoder":
        check_matrices = CheckMatrices.from_dem(
            dem,
            decomposed_hyperedges=self.decomposed_hyperedges,
            prune_decided_errors=self.prune_decided_errors,
            threshold=self.threshold,
        )
        decoder_class = (
            relay_bp.RelayDecoderF32
            if self.precision == "f32"
            else relay_bp.RelayDecoderF64
        )
        decoder = decoder_class(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            # Relay-BP assigns special meaning to exactly zero: it enables the
            # iteration-dependent ramp 1 - 2**(-iteration / scaling_factor).
            # Do not apply the ldpc wrapper convention that maps 0 to None.
            alpha=self.alpha,
            gamma0=self.gamma0,
            alpha_iteration_scaling_factor=(
                self.alpha_iteration_scaling_factor),
            pre_iter=self.pre_iter,
            num_sets=self.num_sets,
            set_max_iter=self.set_max_iter,
            gamma_dist_interval=self.gamma_dist_interval,
            explicit_gammas=self.explicit_gammas,
            stop_nconv=self.stop_nconv,
            stopping_criterion=self.stopping_criterion,
            logging=self.logging,
            seed=self.seed,
        )
        runner = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )
        return _CompiledRelayBpDecoder(
            runner=runner,
            check_matrices=check_matrices,
            parallel=self.parallel,
        )


class _CompiledRelayBpDecoder(sinter.CompiledDecoder):
    def __init__(self, *, runner, check_matrices, parallel: bool) -> None:
        self._runner = runner
        self._check_matrices = check_matrices
        self._parallel = parallel
        self._num_detectors = check_matrices.check_matrix.shape[0]
        self._num_observables = check_matrices.observables_matrix.shape[0]
        self.last_flags: Optional[np.ndarray] = None
        self.last_iterations: Optional[np.ndarray] = None

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        syndromes = np.unpackbits(
            bit_packed_detection_event_data,
            bitorder="little",
            axis=1,
            count=self._num_detectors,
        ).astype(np.uint8)
        if self._check_matrices.syndrome_bias is not None:
            syndromes = (
                syndromes + self._check_matrices.syndrome_bias
            ) % 2

        detailed = self._runner.decode_observables_detailed_batch(
            syndromes,
            parallel=self._parallel,
            progress_bar=False,
            leave_progress_bar_on_finish=False,
        )
        if detailed:
            predictions = np.stack(
                [np.asarray(result.observables, dtype=np.uint8)
                 for result in detailed]
            )
            flags = np.fromiter(
                (bool(result.converged) for result in detailed),
                dtype=bool,
                count=len(detailed),
            )
            self.last_iterations = np.fromiter(
                (int(result.physical_decode_result.iterations)
                 for result in detailed),
                dtype=np.int64,
                count=len(detailed),
            )
        else:
            predictions = np.zeros(
                (0, self._num_observables), dtype=np.uint8)
            flags = np.zeros(0, dtype=bool)
            self.last_iterations = np.zeros(0, dtype=np.int64)

        if self._check_matrices.observables_bias is not None:
            predictions = (
                predictions + self._check_matrices.observables_bias
            ) % 2

        self.last_flags = None if flags.all() else flags
        n_bytes = math.ceil(self._num_observables / 8)
        return np.packbits(
            predictions, axis=1, bitorder="little"
        )[:, :n_bytes]


if relay_bp is not None:
    register_decoder(
        "relay-bp",
        RelayBpDecoder,
        aliases=["relay_bp", "relaybp"],
        backend="cpu",
    )
