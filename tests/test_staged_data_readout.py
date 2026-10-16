"""Explicit tracking boundaries preserve one physical readout/noise schedule."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder, DataReadoutStage
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector


def _physical(circuit):
    result = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name not in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            result.append(instruction)
    return result


def _fake_builder(*, mode='slabs', tracking=True):
    events = []
    tracker = SimpleNamespace(num_qubits=5, logical_tracking=mode,
                              total_measurements=7, meas_rec_to_idx_map={})
    def track(**kwargs):
        events.append(('track', tracker.total_measurements, kwargs))
    tracker.process_data_measurement = track
    system = SimpleNamespace(
        active_stabilizers=[], active_gauges=[], active_qubit_indices=set(range(5)),
        coupler_patches={'bridge': object()}, syndrome_indices=[],
        qubit_coords={q: (q, 0) for q in range(5)},
        deactivate_coupler=lambda name: events.append(('deactivate', name)),
    )
    builder = CircuitBuilder(tracker, system, if_detector=tracking)
    builder.circuit = stim.Circuit('R 0 1 2 3 4')
    return builder, events


def test_stages_use_absolute_records_in_physical_xyz_order():
    # Stage 1 spans physical Y and Z records; stage 2 includes earlier X
    # records. Tracking order must not be mistaken for record order.
    measurements = {4: 'Z', 1: 'X', 3: 'Y', 0: 'X', 2: 'Z'}
    builder, events = _fake_builder()
    builder.apply_staged_data_readout(measurements, (
        DataReadoutStage((2, 3), ('bridge',)),
        DataReadoutStage((4, 1, 0)),
    ))
    assert str(builder.circuit) == 'R 0 1 2 3 4\nTICK\nMX 1 0\nMY 3\nM 4 2'
    assert events[0] == ('deactivate', 'bridge')
    calls = [event for event in events if event[0] == 'track']
    assert [event[1] for event in calls] == [12, 12]
    assert [event[2]['measurement_record_indices'] for event in calls] == [[9, 11], [7, 8, 10]]
    assert builder.tracker.meas_rec_to_idx_map == {7: 1, 8: 0, 9: 3, 10: 4, 11: 2}
    assert builder.system.active_qubit_indices == set()
    rows = calls[0][2]['final_paulis']
    np.testing.assert_array_equal(rows[0], np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
    np.testing.assert_array_equal(rows[1], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))


@pytest.mark.parametrize('stages', [
    (DataReadoutStage((0,)),),
    (DataReadoutStage((0, 1)), DataReadoutStage((1,))),
    (DataReadoutStage((0, 2)),),
])
def test_invalid_stage_partition_has_no_side_effects(stages):
    builder, events = _fake_builder()
    original = builder.circuit.copy()
    with pytest.raises(ValueError, match='partition'):
        builder.apply_staged_data_readout({0: 'X', 1: 'Z'}, stages)
    assert builder.circuit == original
    assert builder.tracker.total_measurements == 7
    assert events == []


def test_legacy_control_retains_single_tracking_call():
    builder, events = _fake_builder(mode='legacy')
    builder.apply_staged_data_readout({0: 'X', 1: 'Z'}, (
        DataReadoutStage((0,), ('bridge',)), DataReadoutStage((1,)),
    ))
    calls = [event for event in events if event[0] == 'track']
    assert len(calls) == 1
    assert 'measurement_record_indices' not in calls[0][2]


def test_staging_preserves_original_measurement_layer_and_idle_noise():
    measurements = {4: 'Z', 1: 'X', 3: 'Y', 0: 'X', 2: 'Z'}
    original, _ = _fake_builder(tracking=False)
    staged, _ = _fake_builder(tracking=False)
    original.apply_data_readout(measurements)
    staged.apply_staged_data_readout(measurements, (
        DataReadoutStage((2, 3), ('bridge',)), DataReadoutStage((4, 1, 0)),
    ))
    assert staged.circuit == original.circuit
    injector = NoiseInjector.from_circuit_level_with_idling(
        NoiseConfig(p_idle=0.01, p_meas=0.02), qubit_indices=range(5))
    assert injector.inject_noise(staged.circuit) == injector.inject_noise(original.circuit)


NATIVE_CASES = ('two_patch', 'cnot_ls', 'bell_zz', 'bell_xx',
                'gate_teleport', 'ls_distillation', 'cross_ls')


def _native(name):
    if name == 'two_patch':
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
        return TwoPatchLSExperiment(
            patch1_config={'distance': 3}, patch2_config={'distance': 3},
            offset=(0, 6), interaction_type='ZZ', rounds=2,
            initial_state_patch1='Z', initial_state_patch2='Z',
            measure_state_patch1='Z', measure_state_patch2='Z',
        ).build()
    if name == 'cnot_ls':
        from lightstim.protocols.cnot_ls import CNOTLSExperiment
        return CNOTLSExperiment(
            patch_configs={name: {'distance': 3} for name in ('a', 'c', 't')},
            offset_ta=(6., 0.), offset_ca=(0., 6.), rounds=2,
        ).build()
    if name in ('bell_zz', 'bell_xx'):
        from lightstim.protocols.bell_teleportation import BellTeleportZZLS, BellTeleportXXLS
        cls = BellTeleportZZLS if name == 'bell_zz' else BellTeleportXXLS
        return cls(distance=3, rounds_pre=2, rounds_ls=2,
                   teleport_state='Z' if name == 'bell_zz' else 'X').build()
    if name == 'gate_teleport':
        from lightstim.protocols.gate_teleport import SGateTeleportExperiment
        return SGateTeleportExperiment(distance=3, method='ZZ',
                                       state_prep='logical_gate', rounds_prep=2,
                                       rounds_gate=2).build()
    if name == 'ls_distillation':
        from lightstim.protocols.ls_distillation import build_distillation_circuit
        return build_distillation_circuit(3, 1)[0]
    if name == 'cross_ls':
        from lightstim.protocols.cross_ls import CrossLSExperiment
        return CrossLSExperiment(PQRM_para=[1, 2, 4], d_surf=3, rounds=2,
                                 PQRM_state='Y', surf_state='X',
                                 post_select_hybrid=True).build()
    raise AssertionError(name)


@pytest.mark.parametrize('name', NATIVE_CASES)
def test_native_staged_readout_preserves_physical_and_idle_noise(monkeypatch, name):
    original_init = SyndromeTracker.__init__
    circuits = {}
    for mode in ('legacy', 'slabs'):
        created = []
        def initialize(self, *args, **kwargs):
            kwargs['logical_tracking'] = mode
            original_init(self, *args, **kwargs)
            created.append(self)
        with monkeypatch.context() as scoped:
            scoped.setattr(SyndromeTracker, '__init__', initialize)
            with redirect_stdout(StringIO()):
                circuits[mode] = _physical(_native(name))
        assert created and all(t.logical_tracking == mode for t in created)
    assert circuits['slabs'] == circuits['legacy']
    injector = NoiseInjector.from_circuit_level_with_idling(
        NoiseConfig(p_idle=0.003, p_meas=0.001, p_reset=0.001,
                    p_1q=0.002, p_2q=0.004),
        qubit_indices=range(circuits['legacy'].num_qubits),
    )
    assert injector.inject_noise(circuits['slabs']) == injector.inject_noise(circuits['legacy'])
