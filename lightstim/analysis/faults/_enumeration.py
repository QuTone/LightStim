"""Bounded enumeration of independent physical circuit locations."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, islice, product
import math
from typing import Literal

import stim

from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.post_select import get_post_select_detector_indices

from ._catalogue import FaultLocation, catalogue
from ._clifford import clifford_signatures
from ._nonclifford import BornEvaluator
from ._phase import EXP_VAL_TAG, T_DAG_TAG, T_TAG, QuantumEnumerationLimitError, QuantumLimits, parse_quantum_circuit
from ._scoring import BranchLoss, BranchScorer, reference_bits


FaultSet = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FaultResponse:
    """Acceptance probability and acceptance-weighted loss for a fixed fault set."""

    acceptance: float
    accepted_loss: float


@dataclass(frozen=True)
class FaultWitness:
    faults: FaultSet
    response: FaultResponse


@dataclass(frozen=True)
class FaultOrder:
    """Masses at exactly this order, including all unselected sites staying quiet."""

    order: int
    configurations: int
    probability_mass: float
    accepted_mass: float
    loss_mass: float


@dataclass(frozen=True)
class FaultReport:
    backend: str
    locations: tuple[FaultLocation, ...]
    per_order: tuple[FaultOrder, ...]
    omitted_probability: float
    witnesses: tuple[FaultWitness, ...]

    @property
    def accepted_mass(self) -> float:
        return math.fsum(row.accepted_mass for row in self.per_order)

    @property
    def loss_mass(self) -> float:
        return math.fsum(row.loss_mass for row in self.per_order)

    @property
    def conditional_loss(self) -> float | None:
        """Expected loss conditional on acceptance and on the enumerated orders."""
        mass = self.accepted_mass
        return self.loss_mass / mass if mass else None

    @property
    def full_loss_bounds(self) -> tuple[float, float] | None:
        """Bounds including all fault orders for losses in [0, 1]."""
        possible_acceptance = self.accepted_mass + self.omitted_probability
        if not possible_acceptance:
            return None
        lower = self.loss_mass / possible_acceptance
        upper = (self.loss_mass + self.omitted_probability) / possible_acceptance
        return max(0., min(1., lower)), max(0., min(1., upper))


def _order_masses(locations: tuple[FaultLocation, ...], order: int, limit: int):
    """Poisson-binomial mass, overflow and distinct-site alternative counts."""
    counts = [1] + [0] * order
    masses = [1.] + [0.] * order
    overflow = 0.
    for i, location in enumerate(locations):
        q = location.probability
        overflow += masses[order] * q
        for k in range(min(i + 1, order), 0, -1):
            counts[k] += counts[k - 1] * len(location.alternatives)
            masses[k] = (1. - q) * masses[k] + q * masses[k - 1]
        masses[0] *= 1. - q
        if sum(counts) > limit:
            raise ValueError(f"Physical configuration count exceeds max_configurations={limit}")
    return counts, masses, overflow


class FaultEnumerator:
    """Compile a physical fault experiment for repeated enumeration and witness replay.

    A native Clifford circuit uses Stim-relative detector and observable flips.
    Extended text containing T, T_DAG or EXP_VAL selects exhaustive Born branches.
    Born uses Stim references for ordinary Clifford input and supplied reference
    bits (default zero) for extended input. EXP_VAL needs a loss function
    mapping corrected target bits and probe expectations to a value in [0, 1].
    With no decoder_config, recovery is the identity. See docs/api/simulation.md.
    """

    def __init__(
        self,
        circuit: stim.Circuit | str,
        *,
        backend: Literal["auto", "clifford", "born"] = "auto",
        decoder_config: DecoderConfig | None = None,
        decoder_model: stim.DetectorErrorModel | None = None,
        allow_gauge_detectors: bool = False,
        post_select_detector_indices: Sequence[int] | None = None,
        post_select_observable_indices: Sequence[int] | None = None,
        post_select_corrected_observable_indices: Sequence[int] | None = None,
        target_observable_indices: Sequence[int] | None = None,
        detector_reference: Sequence[bool] | None = None,
        observable_reference: Sequence[bool] | None = None,
        loss: BranchLoss | None = None,
        observable_failure_mode: Literal["any", "parity"] = "any",
        limits: QuantumLimits | None = None,
    ):
        if backend not in ("auto", "clifford", "born"):
            raise ValueError("backend must be 'auto', 'clifford', or 'born'")
        if observable_failure_mode not in ("any", "parity"):
            raise ValueError("observable_failure_mode must be 'any' or 'parity'")
        limits = limits or QuantumLimits()
        flat = parse_quantum_circuit(circuit, max_instructions=limits.max_instructions, max_targets=limits.max_work)
        if flat.num_observables > limits.max_work:
            raise QuantumEnumerationLimitError("Observable width exceeds max_work")
        tags = {op.tag for op in flat}
        extended = not tags.isdisjoint((T_TAG, T_DAG_TAG, EXP_VAL_TAG))
        if backend == "auto":
            backend = "born" if extended else "clifford"
        if backend == "clifford" and extended:
            raise ValueError("Physical T, T_DAG and EXP_VAL require the Born backend")
        if backend == "clifford" and (detector_reference is not None or observable_reference is not None):
            raise ValueError("Clifford fault signatures use Stim's noiseless reference")
        if EXP_VAL_TAG in tags and loss is None:
            raise ValueError("Provide loss(corrected_target_bits, expectations) for EXP_VAL")
        if post_select_detector_indices is None:
            post_select_detector_indices = get_post_select_detector_indices(flat)
        self.backend = backend
        self.locations = catalogue(flat)
        self._signatures = None
        self._born = None
        self._witness_tolerance = limits.roundoff_tolerance if backend == "born" else 0.
        self._detector_count = flat.num_detectors
        self._detector_mask = (1 << flat.num_detectors) - 1
        if backend == "clifford":
            self._signatures = clifford_signatures(flat, self.locations)
        else:
            missing_reference = (
                (flat.num_detectors and detector_reference is None)
                or (flat.num_observables and observable_reference is None)
            )
            if not extended and missing_reference:
                det_signs, obs_signs = flat.reference_detector_and_observable_signs()
                if detector_reference is None:
                    detector_reference = det_signs
                if observable_reference is None:
                    observable_reference = obs_signs
            ref = reference_bits(observable_reference, flat.num_observables, "observable_reference")
            self._observable_ref = sum(int(bit) << i for i, bit in enumerate(ref))
            det_ref = reference_bits(detector_reference, flat.num_detectors, "detector_reference")
            self._born = BornEvaluator(flat, self.locations, limits=limits, detector_reference=det_ref)
        self._scorer = BranchScorer(
            flat, physical_clifford=not extended, allow_gauge_detectors=allow_gauge_detectors,
            decoder_config=decoder_config, decoder_model=decoder_model,
            detectors=post_select_detector_indices,
            raw_observables=post_select_observable_indices,
            corrected_observables=post_select_corrected_observable_indices,
            targets=target_observable_indices, loss=loss,
            score_parity=observable_failure_mode == "parity",
        )

    def _clifford_response(self, signature: int) -> FaultResponse:
        keep, loss = self._scorer.score(signature & self._detector_mask, signature >> self._detector_count, ())
        return FaultResponse(float(keep), loss)

    def _evaluate(self, chosen: FaultSet) -> FaultResponse:
        if self._signatures is not None:
            signature = 0
            for site, alt in chosen:
                signature ^= self._signatures[site][alt]
            return self._clifford_response(signature)
        accepted = []
        losses = []
        for probability, detectors, observables, expectations in self._born.responses(chosen):
            keep, loss = self._scorer.score(detectors, observables ^ self._observable_ref, expectations)
            if keep:
                accepted.append(probability)
                losses.append(probability * loss)
        return FaultResponse(math.fsum(accepted), math.fsum(losses))

    def evaluate(self, faults: Sequence[tuple[int, int]]) -> FaultResponse:
        """Replay (location index, alternative index) pairs with all other sites ideal."""
        chosen = tuple(faults)
        sites = set()
        for site, alt in chosen:
            if type(site) is not int or not 0 <= site < len(self.locations):
                raise ValueError(f"Invalid fault location {site}")
            if site in sites:
                raise ValueError(f"Fault alternatives at location {site} are mutually exclusive")
            if type(alt) is not int or not 0 <= alt < len(self.locations[site].alternatives):
                raise ValueError(f"Invalid alternative {alt} for location {site}")
            sites.add(site)
        return self._evaluate(chosen)

    def _candidates(self, order: int) -> Iterator[FaultSet]:
        # A pair passes the detector veto exactly when its two masked signatures agree.
        # The report still accounts for the full physical mass, including vetoed sets.
        if order == 2 and self._signatures is not None and self._scorer.detector_mask:
            buckets = defaultdict(list)
            for site, signatures in enumerate(self._signatures):
                for alt, signature in enumerate(signatures):
                    key = signature & self._scorer.detector_mask
                    for first in buckets[key]:
                        yield first, (site, alt)
                for alt, signature in enumerate(signatures):
                    buckets[signature & self._scorer.detector_mask].append((site, alt))
            return
        for sites in combinations(range(len(self.locations)), order):
            for alts in product(*(range(len(self.locations[i].alternatives)) for i in sites)):
                yield tuple(zip(sites, alts))

    def _contributions(self, order, log_ratios, all_quiet, certain, witnesses, witness_limit):
        found = 0
        for chosen in self._candidates(order):
            if certain and not certain.issubset(i for i, _ in chosen):
                continue
            response = self._evaluate(chosen)
            if response.acceptance == 0:
                continue
            threshold = self._witness_tolerance * response.acceptance
            if response.accepted_loss > threshold and found < witness_limit:
                witnesses.append(FaultWitness(chosen, response))
                found += 1
            log_weight = all_quiet + math.fsum(log_ratios[i][a] for i, a in chosen)
            weight = math.exp(log_weight)
            yield weight * response.acceptance, weight * response.accepted_loss

    def enumerate(self, *, max_order: int = 2, max_configurations: int = 5_000_000,
                  witness_limit: int = 8) -> FaultReport:
        """Sum all physical configurations through max_order at the supplied rates.

        max_configurations bounds the total combinatorial space through this order,
        including the no-fault set. witness_limit is per order. If a budget is
        exceeded the call raises and returns no partial report.
        """
        for name, value, minimum in (("max_order", max_order, 0), ("max_configurations", max_configurations, 1),
                                     ("witness_limit", witness_limit, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        order = min(len(self.locations), max_order)
        counts, masses, tail = _order_masses(self.locations, order, max_configurations)
        # The product of quiet-site probabilities is common to all configurations,
        # except for the selected sites. Handle probability-one channels separately.
        certain = {i for i, loc in enumerate(self.locations) if loc.probability == 1}
        quiet_logs = [0. if i in certain else math.log1p(-loc.probability) for i, loc in enumerate(self.locations)]
        all_quiet = math.fsum(quiet_logs)
        log_ratios = [tuple(math.log(a.probability) - quiet_logs[i] for a in loc.alternatives)
                      for i, loc in enumerate(self.locations)]
        rows = []
        witnesses = []
        for k in range(order + 1):
            contributions = self._contributions(k, log_ratios, all_quiet, certain, witnesses, witness_limit)
            # Sum in chunks to keep precision when configurations span many orders of magnitude.
            accepted, lost = [], []
            while chunk := list(islice(contributions, 2048)):
                acceptance_weights, loss_weights = zip(*chunk)
                accepted.append(math.fsum(acceptance_weights))
                lost.append(math.fsum(loss_weights))
            rows.append(FaultOrder(k, counts[k], masses[k], math.fsum(accepted), math.fsum(lost)))
        return FaultReport(self.backend, self.locations, tuple(rows), tail, tuple(witnesses))
