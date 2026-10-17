"""The merged CrossLS code declares its unmeasured PQRM X constraints."""
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.protocols.cross_ls import CrossLSExperiment
from lightstim.utils.linear_algebra import check_commutativity, solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic


@pytest.mark.parametrize('state', ['Z', 'X', 'Y'])
def test_first_merged_round_preserves_pqrm_x_checks_as_bare(monkeypatch, state):
    original_finish = CircuitBuilder._finish_measurement_block_group
    snapshots = []
    coupler_name = 'surface_pqrm_coupler'

    def finish(builder, *args, **kwargs):
        original_finish(builder, *args, **kwargs)
        system = builder.system
        if snapshots or coupler_name not in system.paused_stabilizer_indices:
            return
        tracker = kwargs.get('tracker') or builder.tracker
        assert tracker.logical_tracking == 'slabs'
        n = tracker.num_qubits
        active = stabilizers_to_symplectic(system, system.active_stabilizers, n)
        assert not check_commutativity(active, active).any()

        original_x = {
            uid for uid, row in enumerate(system.stabilizers)
            if row.get('patch_name') == 'pqrm' and row.get('type') == 'X'
        }
        z_checks = stabilizers_to_symplectic(system, system.active_stabilizers_z, n)
        original_rows = stabilizers_to_symplectic(
            system, [system.stabilizers[i] for i in sorted(original_x)], n)
        compatible = ~check_commutativity(original_rows, z_checks).any(axis=1)
        replaced = {uid for uid, keep in zip(sorted(original_x), compatible) if not keep}
        assert replaced, 'This merge must exercise a nontrivial boundary extension'
        assert replaced <= system.paused_stabilizer_indices[coupler_name]
        assert not replaced & system.active_stabilizer_indices
        assert original_x - replaced <= system.active_stabilizer_indices

        bridge = system.coupler_patches[coupler_name]
        extensions = {
            uid for uid in bridge._registered_stabilizer_uids
            if system.stabilizers[uid].get('type') == 'X'
            and system.stabilizers[uid].get('syn_idx') is None
        }
        assert len(extensions) == len(replaced)
        assert extensions <= system.active_stabilizer_indices
        expected_uids = sorted(extensions | (original_x - replaced))
        expected = stabilizers_to_symplectic(
            system, [system.stabilizers[i] for i in expected_uids], n)
        bare_indices = [i for i in range(tracker.stabilizers.count)
                        if i not in tracker._dressed_stabilizer_rows]
        bare = tracker.stabilizers.matrix[bare_indices]
        _, found, _ = solve_linear_decomposition(
            basis=bare, targets=expected, reduce_weight=False)
        assert found.all(), 'A pure PQRM code check must not become a dressed logical direction'

        # Independent exact physical check at the FIRST merged SE boundary,
        # before terminal postselection or readout can repair any classification.
        for row in expected:
            pauli = stim.PauliString.from_numpy(
                xs=row[:n].astype(bool), zs=row[n:].astype(bool))
            assert builder.circuit.has_flow(stim.Flow(output=pauli), unsigned=False)
        snapshots.append((original_x, replaced, extensions))

    monkeypatch.setattr(CircuitBuilder, '_finish_measurement_block_group', finish)
    experiment = CrossLSExperiment(PQRM_para=[1, 2, 4], d_surf=3, rounds=2,
                                   PQRM_state=state, surf_state='X')
    with redirect_stdout(StringIO()):
        experiment.build()
    assert len(snapshots) == 1
    original_x, replaced, extensions = snapshots[0]
    assert coupler_name not in experiment.system.paused_stabilizer_indices
    assert original_x <= experiment.system.active_stabilizer_indices
    assert not extensions & experiment.system.active_stabilizer_indices
