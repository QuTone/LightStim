"""Distillation task selection must survive extra historical observables."""
from contextlib import redirect_stdout
import io
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import stim

from lightstim.protocols import ls_distillation, tg_distillation
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
    logical_history_observable_indices,
)


def test_patch_matrix_uses_annotation_time_and_complete_observable_ids():
    circuit = stim.Circuit('''
        R 0 1
        M 0
        OBSERVABLE_INCLUDE(2) rec[-1]
        M 1
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-2]
        REPEAT 2 {
            M 0
            OBSERVABLE_INCLUDE(1) rec[-1]
        }
    ''')
    system = SimpleNamespace(index_to_owner_map={0: 'A', 1: 'B'})
    matrix, names = build_obs_patch_matrix(circuit, system)
    assert names == ['A', 'B']
    # OBS2's two references are to the very same record and cancel. OBS1's
    # references are to distinct records of A and must both be resolved.
    np.testing.assert_array_equal(matrix, [[0, 1], [1, 0], [0, 0]])


@pytest.mark.parametrize('protocol, output', [
    (tg_distillation, 'W0'), (ls_distillation, 'W4'),
])
def test_history_target_copy_is_not_silently_a_postselection_check(protocol, output):
    circuit = stim.Circuit('''
        R 0 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        OBSERVABLE_INCLUDE[logical-history](2) rec[-2]
    ''')
    system = SimpleNamespace(index_to_owner_map={0: output, 1: 'W1'})
    with redirect_stdout(io.StringIO()):
        transform, target, postselect, matrix, names = protocol.analyze_observables(
            circuit, system
        )
    assert target == [0]
    assert postselect == [1]
    assert logical_history_observable_indices(circuit) == [2]
    np.testing.assert_array_equal(transform, np.eye(3, dtype=int))
    # The historical copy of the output is still exported and testable; making
    # it a postselection check would reject exactly the output errors measured.
    assert matrix[2, names.index(output)] == 1


def test_distillation_rejects_unknown_excluded_observable():
    with pytest.raises(ValueError, match='Excluded observable'):
        identify_distillation_observables(
            np.array([[1]]), ['W0'], ['W0'], excluded_observable_indices=[1]
        )


@pytest.mark.parametrize('kind', ['tg', 'ls'])
def test_distillation_all_seven_preparation_fault_patterns(kind):
    """Exact outer-protocol audit: seven weight-three accepted output errors.

    TG uses its seven physical corner injection sites. LS uses seven logical
    preparation errors, so p here is calibrated logical p_in, NOT its physical
    per-data-qubit reset probability. No fit or rare-event sampling is needed.
    """
    protocol = tg_distillation if kind == 'tg' else ls_distillation
    with redirect_stdout(io.StringIO()):
        if kind == 'tg':
            circuit, _, system = protocol.build_distillation_circuit(3, rounds_init=3)
        else:
            circuit, _, system = protocol.build_distillation_circuit(3, rounds=3)
        transform, target, postselect, _, _ = protocol.analyze_observables(circuit, system)
    assert len(target) == 1
    assert len(postselect) == 3
    flat = circuit.flattened()
    sites = []
    for location, instruction in enumerate(flat):
        if kind == 'tg' and instruction.name == 'RY':
            for qubit in instruction.targets_copy():
                if system.index_to_owner_map[qubit.value] in protocol.TG_MAGIC_NAMES:
                    sites.append((location, [qubit.value]))
        elif kind == 'ls' and instruction.name == 'RX':
            resets = {qubit.value for qubit in instruction.targets_copy()}
            for name in sorted(protocol.LS_MAGIC_NAMES):
                patch = system.patches[name][0]
                mapping = system.local_to_global_map[name]
                support = [mapping[q] for q in patch.logical_ops_z['data_indices']]
                if set(support) <= resets:
                    sites.append((location, support))
    assert len(sites) == 7
    responses = []
    for location, support in sites:
        fault = stim.Circuit()
        for index, instruction in enumerate(flat):
            fault.append(instruction)
            if index == location:
                fault.append('E', [stim.target_z(q) for q in support], 1)
        detectors, observables = fault.compile_detector_sampler(seed=83).sample(
            8, separate_observables=True
        )
        assert not detectors.any()
        assert np.all(observables == observables[0])
        responses.append(observables[0])
    patterns = np.array(list(itertools.product([0, 1], repeat=7)), dtype=np.uint8)
    transformed = (patterns @ np.asarray(responses, dtype=np.uint8) @ transform.T) % 2
    accepted = ~transformed[:, postselect].any(axis=1)
    errors = accepted & transformed[:, target].any(axis=1)
    weights = patterns.sum(axis=1)
    assert [int(np.sum(accepted & (weights == w))) for w in range(8)] == [1, 0, 0, 7, 7, 0, 0, 1]
    assert [int(np.sum(errors & (weights == w))) for w in range(8)] == [0, 0, 0, 7, 0, 0, 0, 1]
