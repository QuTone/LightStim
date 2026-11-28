"""Gross-code PPM experiments with automatic checks and a classical Pauli frame.

The graph/measurement protocol follows Tour de gross (arXiv:2506.03094).
The scheduler is a checked implementation of its constraints, not a claim to
reproduce the unpublished optimized schedule. All detector/observable policy
and sampling are supplied by the existing LightStim infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.coupler import LogicalCouplerProtocol
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.injector import NoiseInjector
from lightstim.qec_code.BB_code import BBCodeExtractionBlock
from lightstim.qec_code.BB_code.ppm.shift import (
    build_tour_de_gross_code, logical_shift_tableaus, decompose_logical_shift,
)
from lightstim.qec_code.BB_code.ppm.lowering import (
    BBPPMCompiler, BBPPMPlan, Measure, Shift, PauliFrameUpdate, RecordParity,
)
from lightstim.qec_code.BB_code.ppm.native import build_native_measurement
from lightstim.qec_code.BB_code.ppm.frame import BBFrameAdapter, strip_frame_activity
from lightstim.qec_code.BB_code.ppm.native_round import (
    apply_native_round, finalize_native_basis, rewrite_parity,
)


@dataclass(frozen=True)
class BBPPMStep:
    """One logical Pauli product; indices 1..12 refer to the paper basis."""
    targets: tuple[tuple[int, str], ...]

    def __post_init__(self):
        normalized = tuple((q, str(p).upper()) for q, p in self.targets)
        if any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or not 1 <= q <= 12
               for q, _ in normalized):
            raise ValueError('Logical qubit indices must be integers in 1..12.')
        if len({q for q, _ in normalized}) != len(normalized):
            raise ValueError('A logical qubit may appear only once in a PPM.')
        if any(p not in 'IXYZ' or len(p) != 1 for _, p in normalized):
            raise ValueError('PPM letters must be I, X, Y, or Z.')
        normalized = tuple(sorted((int(q), p) for q, p in normalized if p != 'I'))
        if not normalized:
            raise ValueError('A PPM must contain at least one nonidentity Pauli.')
        object.__setattr__(self, 'targets', normalized)


@dataclass(frozen=True)
class BBPPMOutcome:
    """A protocol result, separate from automatic evaluation observables.

    XOR the absolute measurement records and ``constant`` to obtain the bit
    (0 = +1, 1 = -1). Random PPM outcomes are legitimate, not failures.
    """
    step: int
    targets: tuple[tuple[int, str], ...]
    records: tuple[int, ...]
    constant: bool = False

    def evaluate(self, samples):
        samples = np.asarray(samples, dtype=bool)
        return np.logical_xor.reduce(samples[..., self.records], axis=-1,
                                     initial=self.constant)


@lru_cache(maxsize=1)
def _gross_compiler():
    return BBPPMCompiler(logical_shift_tableaus())


def _aux_coord(q):
    return (100.0 + (q - 288) % 15, 100.0 + (q - 288) // 15)


class _BBNativeCoupler(LogicalCouplerProtocol):
    """Attach one native LPU measurement to its gross-code patch."""

    EXPECTED_PATCH_COUNT = 1

    def _build_coupler_geometry(self, coupler_patch, patches, *, native, index_map):
        patch, = patches
        # Native checks use system-wide qubit IDs. Compare them to the patch
        # stabilizers in canvas coordinates, which both patches share.
        coords = {index_map[coord]: coord for coord in patch.index_map}
        for q, role in sorted(native.auxiliary_roles.items()):
            coords[q] = _aux_coord(q)
            coupler_patch.add_qubit(*coords[q], role=role)
        originals = {
            stab['syn_coord']: {patch.qubit_coords[q]: p for q, p in stab['pauli'].items()}
            for stab in patch.stabilizers
        }
        for check in native.checks:
            support = {coords[q]: 'IXYZ'[check.pauli[q]] for q in check.pauli.pauli_indices()}
            syn_coord = coords[check.ancillas[0]]
            if check.kind == 'bb':
                if originals[syn_coord] == support:
                    continue  # Keep unchanged checks on the original patch.
                coupler_patch.conflicting_stabilizer_coords.add(syn_coord)
            letters = set(support.values())
            kind = next(iter(letters)) if len(letters) == 1 and 'Y' not in letters else 'Mixed'
            coupler_patch.stabilizers.append({'pauli': support, 'type': kind, 'syn_coord': syn_coord})


class BBPPMExperiment:
    """Build a single gross-module PPM sequence, then use SimulationPipeline.

    ``initial_states`` and ``final_measure_states`` are uniform physical X/Z
    bases, not arbitrary per-logical-qubit state preparation. General PPMs
    use logical qubit 1 as a disposable pivot; qubits 2..12 carry data.
    Native measurements may also target qubit 1. Sequences need not commute.
    No atom loss, T gate, inter-module measurement or manual OBS is used.

    Pass ``plans`` to execute previously compiled BBPPMPlan objects verbatim,
    without running the logical compiler again. Their targets must match the
    sequence in order, and measurement keys must be unique across the plans.
    """
    def __init__(self, code, ppm_sequence, *, initial_states='Z',
                 final_measure_states=None, rounds=10, rounds_init=1,
                 rounds_final=1, idle_rounds=0, noiseless_boundaries=True, noise_params=None,
                 noise_model='circuit_level_with_idling',
                 logical_tracking='slabs', plans=None):
        self.code = build_tour_de_gross_code() if code is None else code
        if getattr(self.code, 'logical_preset', None) != 'tour_de_gross':
            raise ValueError('Use build_tour_de_gross_code() for the paper logical basis.')
        if self.code.shift != (0, 0):
            raise ValueError('BB PPM requires the gross-code patch at the canvas origin (shift=(0, 0)).')
        self.ppm_sequence = tuple(ppm_sequence)
        if any(not isinstance(s, BBPPMStep) for s in self.ppm_sequence):
            raise TypeError('ppm_sequence must contain BBPPMStep objects.')
        self.initial_states = str(initial_states).upper()
        self.final_measure_states = str(final_measure_states or initial_states).upper()
        if self.initial_states not in ('X', 'Z') or self.final_measure_states not in ('X', 'Z'):
            raise ValueError('Uniform initialization/readout bases must be X or Z.')
        for name, value, minimum in (('rounds', rounds, 1), ('rounds_init', rounds_init, 1),
                                     ('rounds_final', rounds_final, 1), ('idle_rounds', idle_rounds, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
            setattr(self, name, value)
        if logical_tracking != 'slabs':
            raise ValueError('BB PPM requires the record-aware slabs tracker.')
        self.logical_tracking = logical_tracking
        self.noiseless_boundaries = bool(noiseless_boundaries)
        self.noise_params = noise_params
        self.noise_model = noise_model
        if plans is None:
            self.plans = tuple(_gross_compiler().compile(s.targets, prefix=f'ppm{i}')
                               for i, s in enumerate(self.ppm_sequence))
        else:
            self.plans = tuple(plans)
            self._validate_precompiled_plans()
        self.ppm_outcomes = {}
        self._noise_schedule = None

    def _validate_precompiled_plans(self):
        """Validate plan/sequence identity and record wiring without searching."""
        if len(self.plans) != len(self.ppm_sequence):
            raise ValueError('plans must contain one plan per PPM in ppm_sequence.')
        measured_keys = set()

        def check_parity(parity):
            if not isinstance(parity, RecordParity):
                raise TypeError('Plan result and correction parities must be RecordParity objects.')
            missing = set(parity.keys) - measured_keys
            if missing:
                raise ValueError(f'Plan references measurement keys before they are defined: {sorted(missing)}')

        for i, (step, plan) in enumerate(zip(self.ppm_sequence, self.plans)):
            if not isinstance(plan, BBPPMPlan):
                raise TypeError('plans must contain BBPPMPlan objects.')
            expected = stim.PauliString(12)
            for q, p in step.targets:
                expected[q - 1] = p
            try:
                target = stim.PauliString(plan.target)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Plan {i} has an invalid target Pauli.') from exc
            if target != expected:
                raise ValueError(f'Plan {i} target does not match ppm_sequence[{i}].')
            for operation in plan.operations:
                if isinstance(operation, Measure):
                    if not isinstance(operation.key, str) or not operation.key:
                        raise ValueError('Plan measurement keys must be nonempty strings.')
                    if operation.key in measured_keys:
                        raise ValueError(f'Duplicate measurement key {operation.key!r}; compile plans with distinct prefixes.')
                    measured_keys.add(operation.key)
                elif isinstance(operation, PauliFrameUpdate):
                    check_parity(operation.parity)
                elif not isinstance(operation, Shift):
                    raise TypeError(f'Unknown plan operation {operation!r}')
            check_parity(plan.output)

    def _site_maps(self):
        result = {}
        for kind, offset in {'L': (0, 1), 'R': (1, 0), 'X': (0, 0), 'Z': (1, 1)}.items():
            for x in range(12):
                for y in range(6):
                    ox, oy = self.code.shift
                    result[kind, x, y] = self.patch.index_map[
                        (ox + 2 * x + offset[0], oy + 2 * y + offset[1])]
        return ({k: v for k, v in result.items() if k[0] in 'LR'},
                {k: v for k, v in result.items() if k[0] in 'XZ'})

    def _register_native(self, p1, p7):
        data, bb_checks = self._site_maps()
        native = build_native_measurement(p1, p7, data_index=data,
                                          bb_check_indices=bb_checks, ancilla_start=288)
        self.system.register_coupler(
            _BBNativeCoupler(), patch_names=['bb'], name=f'lpu_{p1}{p7}',
            native=native, index_map=self.system.index_map,
        )
        if any(self.system.index_map[_aux_coord(q)] != q for q in native.all_lpu_qubits):
            raise RuntimeError('Unexpected LPU physical allocation.')
        self._natives[p1, p7] = native

    def _standalone_se(self, rounds, *, noiseless=False):
        if noiseless and rounds and len(self.builder.circuit) and self.builder.circuit[-1].name != 'TICK':
            self.builder.circuit.append('TICK')
        for r in range(rounds):
            if r:
                self.builder.circuit.append('TICK')
            self.builder.apply_syndrome_extraction(self._se, rounds=1, noiseless=noiseless)

    def _fix_initial_syndrome_frame(self):
        """Choose the +1 code sector assumed by the paper's logical maps.

        Product-state preparation plus syndrome extraction gives known but
        random signs for half the checks. A physical shift represents its
        logical quotient only up to those checks. Fix their initial signs
        using pure-error Pauli feedback, preserving all prepared logical
        eigenvalues. This is preparation-frame bookkeeping, not decoding each
        noisy round or replacing automatic evaluation observables.
        """
        from lightstim.utils.subsystem_algebra import independent_row_indices
        from lightstim.qec_code.surface_code.rotated.ppm.lowering import record_parity
        data = sorted(self.patch.data_indices)
        compact = {q: i for i, q in enumerate(data)}
        checks = []
        rows = []
        for record in self.patch.stabilizers:
            p = stim.PauliString(len(data))
            for q, letter in record['pauli'].items():
                p[compact[q]] = letter
            checks.append(p)
            x, z = p.to_numpy()
            rows.append(np.concatenate((x, z)))
        independent = independent_row_indices(np.asarray(rows, dtype=np.uint8))
        independent_checks = [checks[i] for i in independent]
        logicals = []
        for record in self.patch.logical_ops:
            if record['type'] == self.initial_states:
                p = stim.PauliString(len(data))
                for q, letter in record['pauli'].items():
                    p[compact[q]] = letter
                logicals.append(p)
        encoding = stim.Tableau.from_stabilizers(independent_checks + logicals)
        for i, check in enumerate(independent_checks):
            vector = np.zeros(2 * self.system.num_qubits, dtype=np.uint8)
            for j in check.pauli_indices():
                letter = 'IXYZ'[check[j]]
                vector[data[j]] = letter in 'XY'
                vector[self.system.num_qubits + data[j]] = letter in 'ZY'
            records = record_parity(self.tracker, vector)
            if records is None:
                raise RuntimeError(f'Initial stabilizer {i} sign was not established; tracker S={self.tracker.stabilizers.count} L={self.tracker.logicals.count}, check={check}.')
            correction = encoding.x_output(i)
            self.builder.apply_pauli_feedback(
                {data[j]: 'IXYZ'[correction[j]] for j in correction.pauli_indices()}, records)

    def _logical_support(self, letters):
        result = stim.PauliString(self.system.num_qubits)
        xs = [op for op in self.patch.logical_ops if op['type'] == 'X']
        zs = [op for op in self.patch.logical_ops if op['type'] == 'Z']
        for i, p in enumerate(letters):
            for ops, use in ((xs, p in 'XY'), (zs, p in 'ZY')):
                if use:
                    factor = stim.PauliString(self.system.num_qubits)
                    for q, letter in ops[i]['pauli'].items():
                        factor[q] = letter
                    result *= factor
        return {q: 'IXYZ'[result[q]] for q in result.pauli_indices()}

    def _resolve(self, parity):
        records, constant = set(), bool(parity.constant)
        for key in parity.keys:
            indices, bit = self._records[key]
            records.symmetric_difference_update(indices)
            constant ^= bit
        return tuple(sorted(records)), constant

    def _shift(self, x, y):
        mapping = self.system.local_to_global_map['bb']
        for local_recipe in decompose_logical_shift(self.code, x, y):
            recipe = local_recipe.remap(mapping)
            # Check qubits are zero targets for the first transfer. The
            # previous QEC round read them; preparation is an explicit phase.
            self.builder.circuit.append('TICK')
            self.builder.initialize({q: 'Z' for _, q in recipe.first_pairs}, self.system.num_qubits)
            for transfer_index, pairs in enumerate((recipe.first_pairs, recipe.second_pairs)):
                unitary = stim.Circuit()
                unitary.append('CX', [q for s, t in pairs for q in (s, t)])
                unitary.append('TICK')
                unitary.append('CX', [q for s, t in pairs for q in (t, s)])
                self.builder.apply_unitary_block(unitary)
                sources = sorted(s for s, _ in pairs)
                start = self.tracker.total_measurements
                if transfer_index == 0:
                    self.builder.apply_mid_data_readout({q: 'Z' for q in sources})
                else:
                    self.builder.circuit.append('TICK')
                    readout = stim.Circuit()
                    readout.append('M', sources)
                    # Retain the transported data state. These syndrome sources
                    # are measured during a shift, not an experiment boundary.
                    self.builder.circuit.append('M', sources)
                    n = self.tracker.num_qubits
                    rows = np.zeros((len(sources), 2 * n), dtype=np.uint8)
                    for j, q in enumerate(sources):
                        rows[j, n + q] = 1
                    self.tracker.process_mid_measurement(
                        circuit=self.builder.circuit,
                        forward_symplectic_matrix=np.eye(2 * n, dtype=np.uint8),
                        back_propagated_paulis=rows,
                        reset_paulis=None,
                        measurement_qubit_indices=sources,
                        measurement_bases=['Z'] * len(sources),
                        measurement_coords=[self.system.qubit_coords[q] for q in sources],
                        discarded_measurement_qubit_indices=set(),
                        circuit_chunk=readout,
                    )
                    self.system.active_qubit_indices.difference_update(sources)
                for i, q in enumerate(sources):
                    self.builder.apply_pauli_feedback({q: 'X'}, (start + i,))
            self.system.active_qubit_indices.update(self.patch.data_indices)
            self._standalone_se(1)

    def _measure(self, p1, p7):
        native = self._natives[p1, p7]
        name = f'lpu_{p1}{p7}'
        self.builder.activate_coupler(name)
        self.builder.circuit.append('TICK')
        self.builder.initialize({q: 'Z' for q in native.edge_qubits}, self.system.num_qubits)
        for _ in range(self.rounds):
            self.builder.circuit.append('TICK')
            result = apply_native_round(self, native)

        self.builder.deactivate_coupler(name)
        start_edge = self.tracker.total_measurements
        self.builder.apply_data_readout({q: 'Z' for q in native.edge_qubits},
                                        resolve_absorbed=False)
        for correction in native.frame_corrections:
            self.builder.apply_pauli_feedback(correction.pauli,
                (start_edge + i for i in correction.edge_result_indices))
        self._standalone_se(1)
        return result, native.outcome_sign < 0

    def build(self):
        """Rebuild from specifications; return a tracked (optionally noisy) circuit."""
        self.basis_blocks = {}
        self.system = QECSystem()
        self.patch = self.system.add_patch(self.code, name='bb')
        self._natives = {}
        pairs = sorted({(op.p1, op.p7) for plan in self.plans for op in plan.operations
                        if isinstance(op, Measure)})
        for p1, p7 in pairs:
            self._register_native(p1, p7)
        self._se = BBCodeExtractionBlock(self.system).circuit
        self.physical_num_qubits = self.system.num_qubits
        self.tracker = SyndromeTracker(self.system.num_qubits,
            expected_num_logicals=12, logical_tracking=self.logical_tracking)
        self.builder = BBFrameAdapter(CircuitBuilder(self.tracker, self.system))
        self.system.register_tracker(self.tracker)
        self.system.register_builder(self.builder)
        self.builder.write_coordinates()
        self.builder.initialize({q: self.initial_states for q in self.patch.data_indices},
                                self.system.num_qubits, noiseless=self.noiseless_boundaries)
        self._standalone_se(self.rounds_init, noiseless=self.noiseless_boundaries)
        self._fix_initial_syndrome_frame()
        self._records = {}
        self.ppm_outcomes = {}
        for i, (step, plan) in enumerate(zip(self.ppm_sequence, self.plans)):
            self._standalone_se(self.idle_rounds)
            for operation in plan.operations:
                if isinstance(operation, Shift):
                    self._shift(operation.x, operation.y)
                elif isinstance(operation, Measure):
                    self._records[operation.key] = self._measure(operation.p1, operation.p7)
                elif isinstance(operation, PauliFrameUpdate):
                    records, invert = self._resolve(operation.parity)
                    self.builder.apply_pauli_feedback(self._logical_support(operation.pauli),
                                                      records, invert=invert)
                else:
                    raise TypeError(f'Unknown operation {operation!r}')
            records, bit = self._resolve(plan.output)
            records, bit = self.builder.effective_parity(records, bit)
            self.ppm_outcomes[i] = BBPPMOutcome(i, step.targets, records, bit)
        self._standalone_se(self.rounds_final, noiseless=self.noiseless_boundaries)
        self.builder.apply_data_readout({q: self.final_measure_states for q in self.patch.data_indices},
                                        noiseless=self.noiseless_boundaries)
        scheduled = finalize_native_basis(self.builder.finalize(), self.basis_blocks)
        for i, old in self.ppm_outcomes.items():
            records, bit = rewrite_parity(old.records, old.constant, self.basis_blocks)
            self.ppm_outcomes[i] = BBPPMOutcome(old.step, old.targets, records, bit)
        self._noise_schedule = scheduled
        if self.noise_params is not None:
            return self.build_noisy_circuit(self.noise_params)
        return strip_frame_activity(scheduled)

    def build_noisy_circuit(self, noise_params, noise_model=None):
        """Apply LightStim noise to the completed BB schedule.

        Use this method after ``build()`` when sweeping noise points. The
        returned clean circuit intentionally omits BB's temporary activity
        markers; the retained schedule preserves historical idle locations.
        Automatic observables have already been exported and mapped, so this
        method must not ask the tracker to export virtual record IDs again.
        """
        if self._noise_schedule is None:
            raise RuntimeError('Call build() before adding noise.')
        noise_model = self.noise_model if noise_model is None else noise_model
        factory = getattr(NoiseInjector, f'from_{noise_model}', None)
        if factory is None:
            raise ValueError(f'Unknown LightStim noise model {noise_model!r}.')
        target_indices = (list(range(self._noise_schedule.num_qubits))
                          if noise_model == 'circuit_level_with_idling'
                          else sorted(self.system.index_map[coord]
                                      for coord in self.system.data_coords))
        noisy = factory(noise_params, target_indices).inject_noise(self._noise_schedule)
        return strip_frame_activity(noisy)

    @property
    def noise_schedule(self):
        """Copy of the clean schedule including BB idle-activity metadata.

        External LightStim noise injection must consume this schedule instead
        of the marker-free return from ``build()`` to reproduce the audited
        ``circuit_level_with_idling`` model. Prefer ``build_noisy_circuit``.
        """
        if self._noise_schedule is None:
            raise RuntimeError('Call build() before requesting the noise schedule.')
        return self._noise_schedule.copy()
