"""Acceptance and decoded loss for one observed physical branch."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from numbers import Integral

import numpy as np
import stim

from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.registry import get_decoder


BranchLoss = Callable[[tuple[bool, ...], tuple[float, ...]], float]


def indices(values: Sequence[int] | None, size: int, name: str) -> tuple[int, ...]:
    result = () if values is None else tuple(values)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < size for i in result):
        raise ValueError(f"{name} must contain indices in range({size})")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    return tuple(map(int, result))


def bit_mask(values: Sequence[int]) -> int:
    return sum(1 << i for i in values)


def reference_bits(bits: Sequence[bool] | None, width: int, name: str) -> tuple[bool, ...]:
    if bits is None:
        return (False,) * width
    result = tuple(bits)
    if len(result) != width or any(b not in (False, True) for b in result):
        raise ValueError(f"{name} must contain {width} bits")
    return tuple(map(bool, result))


class BranchScorer:
    def __init__(
        self, circuit: stim.Circuit, *, physical_clifford: bool, allow_gauge_detectors: bool,
        decoder_config: DecoderConfig | None, decoder_model: stim.DetectorErrorModel | None,
        detectors: Sequence[int], raw_observables: Sequence[int] | None,
        corrected_observables: Sequence[int] | None, targets: Sequence[int] | None,
        loss: BranchLoss | None,
    ):
        nd, no = circuit.num_detectors, circuit.num_observables
        self.detector_mask = bit_mask(indices(detectors, nd, "post_select_detector_indices"))
        self.raw_mask = bit_mask(indices(raw_observables, no, "post_select_observable_indices"))
        self.corrected_mask = bit_mask(indices(corrected_observables, no, "post_select_corrected_observable_indices"))
        self.targets = range(no) if targets is None else indices(targets, no, "target_observable_indices")
        self.target_mask = (1 << no) - 1 if targets is None else bit_mask(self.targets)
        self.loss = loss
        self.compiled = None
        self.failure_policy = "ignore"
        self.predictions: dict[int, tuple[int, bool]] = {}
        self.detector_bytes = (nd + 7) // 8
        self.observable_bytes = (no + 7) // 8

        if decoder_model is not None and (decoder_model.num_detectors != nd or decoder_model.num_observables != no):
            raise ValueError("decoder_model must match the physical detector and observable counts")
        if decoder_config is None:
            if decoder_model is not None:
                raise ValueError("decoder_model requires decoder_config")
            return
        if decoder_model is None and not physical_clifford:
            if nd or no:
                raise ValueError(
                    "A decoder for a Clifford+T circuit with detector or frame bits requires an explicit decoder_model"
                )
            decoder_model = stim.DetectorErrorModel()
        decoder = get_decoder(decoder_config.name, backend=decoder_config.backend, **decoder_config.params)
        if decoder_model is None:
            decoder_model = circuit.detector_error_model(
                decompose_errors=getattr(decoder, "decompose_errors", False) or allow_gauge_detectors,
                allow_gauge_detectors=allow_gauge_detectors,
                ignore_decomposition_failures=allow_gauge_detectors,
            )
        self.compiled = decoder.compile_decoder_for_dem(dem=decoder_model)
        self.failure_policy = decoder_config.on_decode_failure

    def prediction(self, syndrome: int) -> tuple[int, bool]:
        if self.compiled is None:
            return 0, True
        if syndrome not in self.predictions:
            packed = np.frombuffer(syndrome.to_bytes(self.detector_bytes, "little"), dtype=np.uint8)
            output = self.compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed.reshape(1, -1))
            if output.shape != (1, self.observable_bytes) or output.dtype != np.uint8:
                raise ValueError("Decoder returned an invalid packed observable prediction")
            flags = getattr(self.compiled, "last_flags", None)
            if flags is not None and np.shape(flags) != (1,):
                raise ValueError("Decoder returned invalid success flags")
            self.predictions[syndrome] = int.from_bytes(output[0].tobytes(), "little"), flags is None or bool(flags[0])
        return self.predictions[syndrome]

    def score(self, syndrome: int, observables: int, expectations: tuple[float, ...]) -> tuple[bool, float]:
        if syndrome & self.detector_mask or observables & self.raw_mask:
            return False, 0.
        prediction, success = self.prediction(syndrome)
        if not success and self.failure_policy != "ignore":
            keep = self.failure_policy == "error"
            return keep, float(keep)
        corrected = observables ^ prediction
        if corrected & self.corrected_mask:
            return False, 0.
        if self.loss is None:
            return True, float(bool(corrected & self.target_mask))
        bits = tuple(bool(corrected & (1 << i)) for i in self.targets)
        value = float(self.loss(bits, expectations))
        if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
            raise ValueError(f"Branch loss must be finite and in [0, 1], received {value}")
        return True, min(1., max(0., value))
