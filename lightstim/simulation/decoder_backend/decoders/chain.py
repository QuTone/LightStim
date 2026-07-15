"""Multi-level decoder chain — hierarchical decoding à la arXiv:2604.16209.

Chen et al. decode with a hierarchy: T1 plain BP handles almost every shot,
the rare non-converged shots escalate to T2 Relay-BP, and the remainder go to
T3 MLE. :class:`ChainDecoder` generalises that pattern to any sequence of
registered decoders: stage k+1 re-decodes only the shots stage k flagged as
failed (``last_flags == False``); shots every stage fails on surface through
the chain's own ``last_flags``, so ``DecoderConfig(on_decode_failure=...)``
applies to the chain as a whole. Decoders that never emit failure flags
(pymatching, relay-bp, ...) resolve every shot handed to them, so they only
make sense as the final stage.

Configure by name::

    DecoderConfig("chain", params={"stages": [
        {"name": "ldpc-bp", "params": {"max_iter": 200}},
        {"name": "relay-bp", "params": {"num_sets": 300, "stop_nconv": 1}},
    ]}, on_decode_failure="discard")

or hand ``SimulationPipeline`` a list of per-stage configs (equivalent to
``DecoderConfig.chain([...])``)::

    SimulationPipeline(decoder_config=[
        DecoderConfig("ldpc-bp", params={"max_iter": 200}),
        DecoderConfig("relay-bp", params={"num_sets": 300}),
    ])

A stage entry may be a decoder name, a ``{"name", "backend", "params"}``
dict, or a :class:`DecoderConfig`. A stage's own ``on_decode_failure`` is
ignored: inside the chain, "failure" means "escalate to the next stage", and
only the chain-level policy (taken from the *last* config in the list form)
decides what happens to shots the final stage cannot resolve.

All stages are compiled against the same DEM. The chain advertises
``decompose_errors=True`` if any stage requires it, so mixing hyperedge-native
decoders (BP) with decomposition-requiring ones shares a decomposed DEM —
harmless for BP, whose parity matrix ignores the decomposition hints.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import sinter
import stim

from ..config import DecoderConfig
from ..registry import get_decoder, register_decoder

StageSpec = Union[str, Dict[str, Any], DecoderConfig]


def _normalize_stage(spec: StageSpec) -> Dict[str, Any]:
    """Coerce a stage spec into a ``{"name", "backend", "params"}`` dict."""
    if isinstance(spec, DecoderConfig):
        return {
            "name": spec.name,
            "backend": spec.backend,
            "params": dict(spec.params),
        }
    if isinstance(spec, str):
        return {"name": spec, "backend": "cpu", "params": {}}
    if isinstance(spec, dict):
        # on_decode_failure is tolerated (DecoderConfig-shaped dicts) but has
        # no per-stage meaning — escalation *is* the intra-chain policy.
        unknown = set(spec) - {"name", "backend", "params", "on_decode_failure"}
        if unknown:
            raise ValueError(
                f"Unknown chain stage keys {sorted(unknown)}; "
                "expected 'name', 'backend', 'params'."
            )
        if "name" not in spec:
            raise ValueError(f"Chain stage {spec!r} is missing a 'name'.")
        return {
            "name": spec["name"],
            "backend": spec.get("backend", "cpu"),
            "params": dict(spec.get("params") or {}),
        }
    raise TypeError(
        f"Invalid chain stage {spec!r}: expected a decoder name, a dict, "
        "or a DecoderConfig."
    )


class ChainDecoder(sinter.Decoder):
    """Escalating multi-level decoder over other registered decoders."""

    def __init__(self, stages: Optional[Sequence[StageSpec]] = None):
        if not stages:
            raise ValueError(
                "chain decoder needs at least one stage, e.g. "
                "DecoderConfig('chain', params={'stages': ['ldpc-bp', 'relay-bp']})"
            )
        self._stage_specs = [_normalize_stage(s) for s in stages]
        # Resolve eagerly so an unknown stage name fails at construction —
        # the pipeline validates the config in the parent process this way.
        self._stage_decoders = [
            get_decoder(s["name"], backend=s["backend"], **s["params"])
            for s in self._stage_specs
        ]

    @property
    def decompose_errors(self) -> bool:
        return any(
            getattr(d, "decompose_errors", False) for d in self._stage_decoders
        )

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> "_CompiledChain":
        return _CompiledChain(
            stages=[d.compile_decoder_for_dem(dem=dem) for d in self._stage_decoders],
            num_observables=dem.num_observables,
        )


class _CompiledChain(sinter.CompiledDecoder):
    """Runs compiled stages in order, escalating only the failed shots.

    Side channels after each :meth:`decode_shots_bit_packed` call:
        last_flags          : None (all shots resolved by some stage) or a
                              bool array, False where even the last stage
                              failed — same contract the pipeline already
                              reads for ``on_decode_failure``.
        last_stage_attempts : shots handed to each stage (index-aligned with
                              the configured stages; 0 = never reached).
                              E.g. Chen-style conv statistics per level.
    """

    def __init__(
        self, *, stages: List[sinter.CompiledDecoder], num_observables: int
    ) -> None:
        self._stages = stages
        self._n_obs_bytes = math.ceil(num_observables / 8) if num_observables else 0
        self.last_flags: Optional[np.ndarray] = None
        self.last_stage_attempts: List[int] = [0] * len(stages)

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        dets = bit_packed_detection_event_data
        n_shots = dets.shape[0]
        predictions = np.zeros((n_shots, self._n_obs_bytes), dtype=np.uint8)
        attempts = [0] * len(self._stages)
        pending = np.arange(n_shots)  # shots no stage has resolved yet

        for k, compiled in enumerate(self._stages):
            if pending.size == 0:
                break
            attempts[k] = int(pending.size)
            preds = np.asarray(
                compiled.decode_shots_bit_packed(
                    bit_packed_detection_event_data=dets[pending]
                ),
                dtype=np.uint8,
            )
            if preds.shape != (pending.size, self._n_obs_bytes):
                raise ValueError(
                    f"chain stage {k} returned predictions of shape "
                    f"{preds.shape}, expected {(pending.size, self._n_obs_bytes)}."
                )
            predictions[pending] = preds
            flags = getattr(compiled, "last_flags", None)
            if flags is None:  # stage resolves everything it is handed
                pending = pending[:0]
            else:
                pending = pending[~np.asarray(flags, dtype=bool).reshape(-1)]

        self.last_stage_attempts = attempts
        if pending.size:
            residual = np.ones(n_shots, dtype=bool)
            residual[pending] = False
            self.last_flags = residual
        else:
            self.last_flags = None
        return predictions


register_decoder(
    "chain",
    ChainDecoder,
    aliases=["decoder-chain", "multi-level"],
    backend="cpu",
)
