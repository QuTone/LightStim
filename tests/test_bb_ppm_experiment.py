"""Physical BB PPM experiments use the existing tracker and simulation API."""
import contextlib
import io
from dataclasses import replace

import numpy as np
import pytest
import stim

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.bb_ppm import BBPPMExperiment, BBPPMOutcome, BBPPMStep
from lightstim.qec_code.BB_code.ppm.frame import unpack, strip_frame_activity
from lightstim.qec_code.BB_code.ppm.native_round import rewrite_parity
from lightstim.qec_code.BB_code.ppm.shift import build_tour_de_gross_code, logical_shift_tableaus
from lightstim.qec_code.BB_code.ppm.lowering import (
    BBPPMCompiler, Measure, PauliFrameUpdate, RecordParity, Shift,
)


def _experiment(targets=(), *, basis='Z', final_basis=None, rounds=1, **kwargs):
    return BBPPMExperiment(None, [BBPPMStep(tuple(target)) for target in targets],
                           initial_states=basis, final_measure_states=final_basis or basis,
                           rounds=rounds, rounds_init=1, rounds_final=1, **kwargs)


def _build(experiment):
    with contextlib.redirect_stdout(io.StringIO()):
        return experiment.build()


def _assert_clean_physical(circuit):
    assert 288 <= circuit.num_qubits <= 378
    assert circuit.num_detectors > 0
    assert circuit.num_observables > 0
    assert all(inst.name != 'MPP' for inst in circuit.flattened())
    assert not any(inst.name in ('CX', 'CY', 'CZ') and
                   any(target.is_measurement_record_target for target in inst.targets_copy())
                   for inst in circuit.flattened())
    assert not circuit.compile_detector_sampler(seed=491).sample(32, append_observables=True).any()
    circuit.detector_error_model()


def _frame_expectation(experiment, simulator, pauli):
    """Compare the effective logical state with an independent Stim simulation."""
    support = {q: 'IXYZ'[pauli[q]] for q in pauli.pauli_indices()}
    records, constant = unpack(experiment.builder.frame.anticommutes(support))
    records, constant = rewrite_parity(records, constant, experiment.basis_blocks)
    raw = np.asarray(simulator.current_measurement_record(), dtype=bool)
    assert not records or max(records) < len(raw)
    sign_flip = bool(np.logical_xor.reduce(raw[list(records)], initial=constant))
    return simulator.peek_observable_expectation(pauli) * (-1 if sign_flip else 1)


def _effective_readout(experiment, records, raw):
    records, constant = experiment.builder.effective_parity(records)
    records, constant = rewrite_parity(records, constant, experiment.basis_blocks)
    return np.logical_xor.reduce(raw[:, records], axis=1, initial=constant)


@pytest.mark.parametrize('targets', [(), ((0, 'X'),), ((13, 'Z'),),
    ((1.5, 'X'),), ((1, 'A'),), ((1, 'XX'),), ((1, 'X'), (1, 'Z')),
    ((1, 'I'),)])
def test_invalid_step_is_rejected(targets):
    with pytest.raises(ValueError):
        BBPPMStep(targets)


def test_step_normalizes_and_drops_identity():
    assert BBPPMStep(((7, 'z'), (3, 'i'), (1, 'x'))).targets == ((1, 'X'), (7, 'Z'))


@pytest.mark.parametrize('options', [{'rounds': 0}, {'rounds': 1.5},
    {'rounds_init': 0}, {'rounds_final': 0}, {'idle_rounds': -1},
    {'initial_states': 'Y'}, {'final_measure_states': 'Y'},
    {'logical_tracking': 'legacy'}])
def test_invalid_experiment_options_are_rejected(options):
    with pytest.raises(ValueError):
        BBPPMExperiment(None, [], **options)


def test_experiment_requires_steps_and_paper_preset():
    with pytest.raises(TypeError):
        BBPPMExperiment(None, [[(1, 'Z')]])
    code = build_tour_de_gross_code()
    code.logical_preset = None
    with pytest.raises(ValueError, match='paper logical basis'):
        BBPPMExperiment(code, [])


def test_experiment_rejects_translated_patch_during_setup():
    code = build_tour_de_gross_code(shift=(2, 4))
    with pytest.raises(ValueError, match='canvas origin'):
        BBPPMExperiment(code, [BBPPMStep(((1, 'Y'),))])


@pytest.fixture(scope='module')
def mixed_ppm_plan():
    return BBPPMCompiler(logical_shift_tableaus()).compile(
        ((2, 'X'), (7, 'Z'), (12, 'X')), prefix='notebook0')


def _forbid_recompilation(monkeypatch):
    def unexpected():
        pytest.fail('Supplying plans must not obtain or run the logical compiler.')
    monkeypatch.setattr('lightstim.qec_code.BB_code.ppm.experiment._gross_compiler', unexpected)


def test_precompiled_plan_is_used_without_searching(monkeypatch, mixed_ppm_plan):
    _forbid_recompilation(monkeypatch)
    experiment = _experiment([((2, 'X'), (7, 'Z'), (12, 'X'))],
                             plans=[mixed_ppm_plan])
    assert experiment.plans[0] is mixed_ppm_plan


@pytest.mark.parametrize('case', ['count', 'type', 'target', 'negative_target'])
def test_precompiled_plan_rejects_sequence_mismatches(monkeypatch, mixed_ppm_plan, case):
    _forbid_recompilation(monkeypatch)
    plans = {
        'count': [],
        'type': [object()],
        'target': [replace(mixed_ppm_plan, target='XIIIIIIIIIII')],
        'negative_target': [replace(mixed_ppm_plan, target='-' + mixed_ppm_plan.target)],
    }[case]
    expected_error = TypeError if case == 'type' else ValueError
    with pytest.raises(expected_error):
        _experiment([((2, 'X'), (7, 'Z'), (12, 'X'))], plans=plans)


def test_precompiled_plans_reject_duplicate_record_keys(monkeypatch, mixed_ppm_plan):
    _forbid_recompilation(monkeypatch)
    target = ((2, 'X'), (7, 'Z'), (12, 'X'))
    with pytest.raises(ValueError, match='Duplicate measurement key'):
        _experiment([target, target], plans=[mixed_ppm_plan, mixed_ppm_plan])


@pytest.mark.parametrize('location', ['correction', 'output'])
def test_precompiled_plan_rejects_undefined_record_references(monkeypatch, mixed_ppm_plan, location):
    _forbid_recompilation(monkeypatch)
    missing = RecordParity(('not-yet-measured',))
    if location == 'correction':
        bad = replace(mixed_ppm_plan, operations=(
            PauliFrameUpdate('IXIIIIIIIIII', missing),) + mixed_ppm_plan.operations)
    else:
        bad = replace(mixed_ppm_plan, output=missing)
    with pytest.raises(ValueError, match='before they are defined'):
        _experiment([((2, 'X'), (7, 'Z'), (12, 'X'))], plans=[bad])


@pytest.mark.slow
def test_precompiled_mixed_ppm_build_preserves_joint_projection(monkeypatch, mixed_ppm_plan):
    _forbid_recompilation(monkeypatch)
    experiment = _experiment([((2, 'X'), (7, 'Z'), (12, 'X'))],
                             plans=[mixed_ppm_plan], basis='Z')
    circuit = _build(experiment)
    assert experiment.plans[0] is mixed_ppm_plan
    _assert_clean_physical(circuit)
    raw = circuit.compile_sampler(seed=316).sample(64)
    assert set(experiment.ppm_outcomes[0].evaluate(raw)) == {False, True}

    # Inspect the state immediately before destructive readout, so this test
    # checks backaction/coherence as well as the measurement result's parity.
    data = set(experiment.patch.data_indices)
    instructions = list(circuit.flattened())
    final_readout = max(i for i, inst in enumerate(instructions)
        if inst.name == 'M' and len(inst.targets_copy()) == len(data)
        and {target.value for target in inst.targets_copy()} == data)
    prefix = stim.Circuit()
    for inst in instructions[:final_readout]:
        prefix.append(inst)

    def logical_product(factors):
        result = stim.PauliString(experiment.system.num_qubits)
        for index, basis in factors:
            record = [op for op in experiment.patch.logical_ops if op['type'] == basis][index - 1]
            factor = stim.PauliString(experiment.system.num_qubits)
            for q, letter in record['pauli'].items():
                factor[q] = letter
            result *= factor
        return result

    target = logical_product(((2, 'X'), (7, 'Z'), (12, 'X')))
    correlations = [logical_product(((2, 'Z'), (12, 'Z')))] + [
        logical_product(((q, 'Z'),)) for q in range(3, 12)]
    for seed in (31, 72):
        sim = stim.TableauSimulator(seed=seed)
        sim.do(prefix)
        outcome = bool(experiment.ppm_outcomes[0].evaluate(sim.current_measurement_record()))
        assert _frame_expectation(experiment, sim, target) == (-1 if outcome else 1)
        for observable in correlations:
            assert _frame_expectation(experiment, sim, observable) == 1
        assert sim.peek_observable_expectation(logical_product(((2, 'Z'),))) == 0
        assert sim.peek_observable_expectation(logical_product(((12, 'Z'),))) == 0


def test_outcome_bit_is_record_parity_not_a_failure_definition():
    samples = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=bool)
    outcome = BBPPMOutcome(0, ((1, 'Z'),), (0, 2), True)
    assert outcome.evaluate(samples).tolist() == [True, True, False]
    assert bool(outcome.evaluate(samples[0]))


def test_empty_sequence_uses_memory_path_and_build_is_repeatable():
    experiment = _experiment()
    first = _build(experiment)
    _assert_clean_physical(first)
    assert first.num_qubits == 288
    assert experiment.ppm_outcomes == {}
    assert _build(experiment) == first


@pytest.mark.parametrize('targets,basis', [(((1, 'Z'), (7, 'Z')), 'Z'),
                                           (((1, 'X'),), 'X'),
                                           (((2, 'Z'),), 'Z')])
def test_native_and_shifted_ppm_outcomes_match_known_logical_eigenvalues(targets, basis):
    experiment = _experiment([targets], basis=basis)
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    if targets[0][0] == 2:
        assert any(isinstance(op, Shift) for op in experiment.plans[0].operations)
    raw = circuit.compile_sampler(seed=712).sample(64)
    assert not experiment.ppm_outcomes[0].evaluate(raw).any()


def test_repeated_native_measurement_reuses_lpu_and_is_repeatable():
    targets = ((1, 'Z'), (7, 'Z'))
    experiment = _experiment([targets, targets], basis='X', final_basis='Z')
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    raw = circuit.compile_sampler(seed=872).sample(128)
    first = experiment.ppm_outcomes[0].evaluate(raw)
    second = experiment.ppm_outcomes[1].evaluate(raw)
    assert set(first) == {False, True}
    assert np.array_equal(first, second)
    # Read the logical Z1 Z7 support and account for the classical frame;
    # this does not add or redefine any OBSERVABLE_INCLUDE.
    z_ops = [op for op in experiment.patch.logical_ops if op['type'] == 'Z']
    support = set(z_ops[0]['pauli']) ^ set(z_ops[6]['pauli'])
    data = sorted(experiment.patch.data_indices)
    offset = circuit.num_measurements - len(data)
    columns = [offset + i for i, q in enumerate(data) if q in support]
    assert np.array_equal(second, _effective_readout(experiment, columns, raw))
    assert _build(experiment) == circuit


@pytest.mark.slow
def test_general_three_logical_ppm_compiles_and_preserves_known_eigenvalue():
    experiment = _experiment([((2, 'X'), (8, 'X'), (9, 'X'))], basis='X')
    assert experiment.plans[0].rotations
    assert sum(isinstance(op, Measure) for op in experiment.plans[0].operations) > 1
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    raw = circuit.compile_sampler(seed=12).sample(64)
    assert not experiment.ppm_outcomes[0].evaluate(raw).any()
    x_ops = [op for op in experiment.patch.logical_ops if op['type'] == 'X']
    support = set(x_ops[1]['pauli']) ^ set(x_ops[7]['pauli']) ^ set(x_ops[8]['pauli'])
    data = sorted(experiment.patch.data_indices)
    offset = circuit.num_measurements - len(data)
    columns = [offset + i for i, q in enumerate(data) if q in support]
    assert not _effective_readout(experiment, columns, raw).any()


def test_native_ten_round_measurement_is_noiseless():
    experiment = _experiment([((1, 'Z'), (7, 'Z'))], rounds=10)
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    blocks = list(experiment.basis_blocks.values())
    assert len(blocks) == 10
    previous_starts = {
        block['base'] + block['physical'].num_measurements: previous['base']
        for previous, block in zip(blocks, blocks[1:])
    }
    cursor = 0
    checked = 0
    for instruction in circuit.flattened():
        if instruction.name == 'DETECTOR' and cursor in previous_starts:
            records = [cursor + target.value for target in instruction.targets_copy()]
            assert records and min(records) >= previous_starts[cursor]
            checked += 1
        cursor += instruction.num_measurements
    assert checked > 0


def test_noisy_ppm_uses_existing_bposd_pipeline():
    pytest.importorskip('stimbposd')
    from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline
    p = 0.001
    experiment = _experiment([((1, 'Z'), (7, 'Z'))],
        noise_params=NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p))
    circuit = _build(experiment)
    dem = circuit.detector_error_model()
    assert dem.num_errors > 0
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig('bposd', params={'max_iterations': 10, 'osd_order': 0}),
        max_shots=32, max_errors=32, batch_size=16, num_workers=1, print_progress=False)
    stats = pipeline.run(circuit, {'p': p, 'code': 'gross', 'ppm': 'Z1Z7'})
    assert stats.shots == stats.post_selected_shots == 32
    assert 0 <= stats.errors <= stats.shots
    assert stats.logical_error_rate == stats.errors / stats.shots


class _ReachedFirstPPM(Exception):
    pass


class _CapturePreparedCode(BBPPMExperiment):
    def _measure(self, p1, p7):
        self.prepared_circuit = self.builder.circuit.copy()
        raise _ReachedFirstPPM


@pytest.mark.parametrize('basis', ['X', 'Z'])
def test_baseline_prepares_positive_codespace_preserving_logical_inputs(basis):
    experiment = _CapturePreparedCode(None, [BBPPMStep(((1, basis),))],
        initial_states=basis, rounds=1, rounds_init=1, rounds_final=1)
    with pytest.raises(_ReachedFirstPPM):
        _build(experiment)
    # Independently simulate the physical preparation and apply its classical
    # Pauli frame to expectations before any logical PPM is applied.
    # Both the stabilizer eigenvalues and initial logical state must be fixed.
    operators = experiment.patch.stabilizers + [
        op for op in experiment.patch.logical_ops if op['type'] == basis]
    for seed in (17, 42, 91):
        sim = stim.TableauSimulator(seed=seed)
        sim.do(experiment.prepared_circuit)
        for operator in operators:
            pauli = stim.PauliString(experiment.system.num_qubits)
            for q, letter in operator['pauli'].items():
                pauli[q] = letter
            assert _frame_expectation(experiment, sim, pauli) == 1


@pytest.mark.slow
def test_general_ppm_projects_joint_parity_preserving_ghz_coherence(tmp_path):
    experiment = _experiment([((2, 'X'), (8, 'X'), (9, 'X'))], basis='Z')
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    # Keep the complete logical operation and all of its frame corrections,
    # stopping only before the last destructive readout of the 144 data qubits.
    instructions = list(circuit.flattened())
    data = set(experiment.patch.data_indices)
    final_readout = max(i for i, inst in enumerate(instructions)
        if inst.name == 'M' and len(inst.targets_copy()) == len(data)
        and {target.value for target in inst.targets_copy()} == data)
    prefix = stim.Circuit()
    for inst in instructions[:final_readout]:
        prefix.append(inst)
    (tmp_path / 'general_ppm_before_readout.stim').write_text(str(prefix), encoding='utf-8')

    def logical_product(kind, indices):
        ops = [op for op in experiment.patch.logical_ops if op['type'] == kind]
        result = stim.PauliString(experiment.system.num_qubits)
        for index in indices:
            factor = stim.PauliString(experiment.system.num_qubits)
            for q, letter in ops[index - 1]['pauli'].items():
                factor[q] = letter
            result *= factor
        return result

    target = logical_product('X', (2, 8, 9))
    preserved = [logical_product('Z', indices) for indices in
        [(2, 8), (2, 9)] + [(q,) for q in (3, 4, 5, 6, 7, 10, 11, 12)]]
    for seed in (4, 19, 54, 79):
        sim = stim.TableauSimulator(seed=seed)
        sim.do(prefix)
        outcome = bool(experiment.ppm_outcomes[0].evaluate(sim.current_measurement_record()))
        assert _frame_expectation(experiment, sim, target) == (-1 if outcome else 1)
        # Independent single-X measurements would destroy these ZZ
        # correlations. They distinguish a joint projector from merely
        # multiplying three independently obtained measurement results.
        for observable in preserved:
            assert _frame_expectation(experiment, sim, observable) == 1


def test_noiseless_boundaries_tag_only_preparation_and_terminal_readout():
    p = 0.001
    cfg = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    noise_counts = []
    for ideal_boundaries in (True, False):
        experiment = _experiment([((1, 'Z'), (7, 'Z'))],
            noiseless_boundaries=ideal_boundaries, noise_params=cfg)
        circuit = _build(experiment)
        data = set(experiment.patch.data_indices)
        clean = experiment.builder.circuit
        init = next(inst for inst in clean if inst.name == 'R'
                    and data.issubset({target.value for target in inst.targets_copy()}))
        final = [inst for inst in clean if inst.name == 'M'
                 and {target.value for target in inst.targets_copy()} == data][-1]
        expected_tag = 'noiseless' if ideal_boundaries else ''
        assert init.tag == final.tag == expected_tag
        # Even with ideal boundaries the actual LPU instruction remains
        # noisy; an accidentally globally ideal experiment is not accepted.
        assert circuit.detector_error_model().num_errors > 0
        noise_counts.append(sum(inst.name in ('DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR')
                                for inst in circuit.flattened()))
    assert 0 < noise_counts[0] < noise_counts[1]


@pytest.mark.parametrize('basis', ['X', 'Z'])
def test_shifted_parity_with_multiple_boundary_and_idle_rounds(basis):
    class CaptureShiftChecks(BBPPMExperiment):
        in_shift = False

        def _shift(self, x, y):
            self.in_shift = True
            super()._shift(x, y)
            self.in_shift = False

        def _standalone_se(self, *args, **kwargs):
            before = self.builder.circuit.num_detectors
            super()._standalone_se(*args, **kwargs)
            if self.in_shift:
                # Discarding these intermediate records loses 132 checks.
                assert self.builder.circuit.num_detectors - before == 144

    experiment = CaptureShiftChecks(None, [BBPPMStep(((2, basis),))],
        initial_states=basis, rounds=2, rounds_init=3, rounds_final=2, idle_rounds=1)
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    raw = circuit.compile_sampler(seed=345).sample(128)
    assert not experiment.ppm_outcomes[0].evaluate(raw).any()


def test_noncommuting_native_sequence_has_expected_random_outcomes():
    targets = [((1, 'Z'),), ((1, 'X'),), ((1, 'Z'),)]
    experiment = BBPPMExperiment(None, [BBPPMStep(target) for target in targets],
        initial_states='Z', rounds=1, rounds_init=2, rounds_final=2)
    circuit = _build(experiment)
    _assert_clean_physical(circuit)
    raw = circuit.compile_sampler(seed=710).sample(128)
    assert not experiment.ppm_outcomes[0].evaluate(raw).any()
    assert set(experiment.ppm_outcomes[1].evaluate(raw)) == {False, True}
    assert set(experiment.ppm_outcomes[2].evaluate(raw)) == {False, True}


def test_noise_sweep_preserves_frame_slot_activity_and_is_repeatable():
    from lightstim.noise.injector import NoiseInjector

    experiment = _experiment([((2, 'Z'),)])
    clean = _build(experiment)
    schedule = experiment.noise_schedule
    assert strip_frame_activity(schedule) == clean
    assert schedule != clean
    schedule.clear()
    assert experiment.noise_schedule.num_measurements == clean.num_measurements

    for p in (0.001, 0.0005):
        config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
        noisy = experiment.build_noisy_circuit(config)
        assert noisy == experiment.build_noisy_circuit(config)
        assert noisy.without_noise() == clean
        assert not any(inst.name == 'I' and inst.tag == 'noiseless'
                       for inst in noisy.flattened())
        noisy.detector_error_model()
        # Omitting the schedule metadata changes which targets receive idle
        # noise, even though the omitted identities enact no physical gate.
        direct = NoiseInjector.from_circuit_level_with_idling(
            config, list(range(clean.num_qubits))).inject_noise(clean)
        assert direct != noisy
