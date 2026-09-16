"""Adapter contracts run without IonQ; native tests run when separately built."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import stim

from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline
from lightstim.simulation.decoder_backend.decoders import ionq_beam_search as ionq
from lightstim.simulation.decoder_backend.registry import get_decoder
from lightstim.simulation.decoder_backend._accounting import count_batch


class NativeStub:
    def __init__(self, *, pcm, error_channel, **params):
        self.output = np.ones(pcm.shape[1], dtype=np.uint8)
        self.converge = False  # deliberately not the returned correction's flag

    def decode(self, syndrome):
        syndrome[:] = 0  # test isolation even for a backend that mutates inputs
        return self.output


@pytest.fixture
def fake_native(monkeypatch):
    monkeypatch.setattr(ionq, '_load_native', lambda: NativeStub)


def test_registry_does_not_import_native():
    code = """
import sys
from lightstim.simulation.decoder_backend.registry import list_decoders
assert 'ionq-beam-search' in list_decoders()
assert 'beam_search_decoder' not in sys.modules
assert 'decoder.beam_search_decoder' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_missing_native_gives_install_hint(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(ionq.importlib, 'import_module', missing)
    with pytest.raises(ImportError, match='PYTHONPATH'):
        get_decoder('ionq-beam-search')


@pytest.mark.parametrize('name', ['max_rounds', 'beam_width', 'num_results',
                                 'initial_iters', 'iters_per_round'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_parameters_fail_before_native_import(name, value):
    with pytest.raises(ValueError, match=name):
        ionq.IonQBeamSearchDecoder(**{name: value})


def test_copy_and_returned_correction_validity(fake_native):
    decoder = get_decoder('ionq-beam-search', num_results=2)
    decoder.setup(H=sp.csr_matrix([[1], [1], [1]]), priors=[0.1])
    syndrome = np.ones(3, dtype=np.uint8)
    correction, valid = decoder.decode_single(syndrome)
    assert valid  # despite native converge=False
    assert np.all(syndrome == 1)
    decoder._inner.output[:] = 0
    assert correction.tolist() == [1]  # no alias to native memory
    decoder._inner.converge = True
    assert decoder.decode_single(syndrome)[1] is False
    with pytest.raises(ValueError, match='syndrome'):
        decoder.decode_single(np.ones(4, dtype=np.uint8))


@pytest.mark.parametrize('prior', [0, 1, -0.1, np.nan, np.inf])
def test_unsafe_priors_rejected(fake_native, prior):
    decoder = get_decoder('ionq-beam-search')
    with pytest.raises(ValueError, match='priors'):
        decoder.setup(H=sp.eye(1), priors=[prior])


def test_bitpacking_multiple_observables_empty_batch_and_noiseless(fake_native):
    decoder = get_decoder('ionq-beam-search')
    dem = stim.DetectorErrorModel('error(0.1) D0 D1 D2 L0 L8')
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.array([[7]], dtype=np.uint8))
    assert result.tolist() == [[1, 1]]
    assert compiled.last_flags is None
    empty = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.empty((0, 1), dtype=np.uint8))
    assert empty.shape == (0, 2)
    compiled = decoder.compile_decoder_for_dem(
        dem=stim.DetectorErrorModel('detector D0\nlogical_observable L0'))
    out = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.array([[0], [1]], dtype=np.uint8))
    assert out.tolist() == [[0], [0]]
    assert compiled.last_flags.tolist() == [True, False]


def test_invalid_correction_with_correct_observable_is_policy_dependent(fake_native):
    decoder = get_decoder('ionq-beam-search')
    compiled = decoder.compile_decoder_for_dem(
        dem=stim.DetectorErrorModel('error(0.1) D0\nlogical_observable L0'))
    # The stub returns correction [1] for both shots: valid for syndrome 1,
    # invalid for syndrome 0. Neither correction changes the observable.
    prediction = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.array([[1], [0]], dtype=np.uint8))
    assert compiled.last_flags.tolist() == [True, False]
    for policy, expected in [('ignore', (2, 0)), ('error', (2, 1)), ('discard', (1, 0))]:
        assert count_batch(
            obs_filtered=np.zeros((2, 1), dtype=np.uint8),
            pred_packed=prediction,
            flags=compiled.last_flags,
            post_select_corrected_observable_indices=None,
            target_observable_indices=None,
            on_decode_failure=policy,
        ) == expected


def _require_native():
    try:
        return ionq._load_native()
    except ImportError:
        pytest.skip('Build the optional IonQ extension; see docs/decoder/ionq_beam_search.md')


def test_native_matches_upstream_matrices_and_predictions():
    _require_native()
    stimbposd = pytest.importorskip('stimbposd.dem_to_matrices')
    dem = stim.DetectorErrorModel('''
        error(0.1) D0 D1 D2 L0
        error(0.02) D0
        error(0.02) D1
        error(0.02) D2
    ''')
    matrices = stimbposd.detector_error_model_to_check_matrices(dem)
    for num_results in (1, 2):
        decoder = get_decoder('ionq-beam-search', num_results=num_results)
        compiled = decoder.compile_decoder_for_dem(dem=dem)
        reference = _require_native()(pcm=matrices.check_matrix,
                                      error_channel=list(matrices.priors),
                                      **decoder.params)
        syndromes = np.array([[0,0,0], [1,1,1], [0,1,0], [1,0,1]], dtype=np.uint8)
        expected = np.array([
            np.asarray(matrices.observables_matrix @ reference.decode(s.copy())).ravel() & 1
            for s in syndromes])
        actual = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.packbits(syndromes, axis=1, bitorder='little'))
        assert np.array_equal(np.unpackbits(actual, axis=1, bitorder='little')[:, :1], expected)


@pytest.mark.parametrize('workers', [1, 2])
def test_native_pipeline_workers(workers):
    _require_native()
    circuit = stim.Circuit('''
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    ''')
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig('ionq-beam-search'), max_shots=40,
        max_errors=40, batch_size=10, num_workers=workers, print_progress=False)
    stats = pipeline.run(circuit)
    assert stats.shots == 40
    assert stats.errors == 0


@pytest.mark.parametrize('workload, args, output_flag', [
    pytest.param('memory', ['--codes', 'rotated_sc'], '--output', id='memory'),
    pytest.param('logical_ops', ['--gate', 'TwoPatchLS_rotated_ZZ'],
                 '--output', id='rotated-ls'),
    pytest.param('logical_circuits', ['--experiment', 'bell_tele', '--protocols', 'tg',
                                     '--states', 'Z'], '--output-dir', id='teleportation'),
    pytest.param('logical_circuits', ['--experiment', 'distill_ls', '--p-injected', '0.01',
                                     '--noise-mode', 'injection', 'full'],
                 '--output-dir', id='ls-distillation'),
    pytest.param('logical_circuits', ['--experiment', 'distill_tg', '--p-injected', '0.01',
                                     '--noise-mode', 'injection', 'full'],
                 '--output-dir', id='tg-distillation'),
    pytest.param('state_injection', ['--inject-states', 'Z', '--inject-protocols', 'corner',
                                    '--inject-modes', 'hybrid', '--rounds', '1'],
                 '--output', id='state-injection'),
    pytest.param('cross_ls', ['--experiment', 'sweep', '--states', 'Z', '--pqrm', '1,2,4'],
                 '--output-dir', id='cross-ls'),
])
def test_native_workload_cli(tmp_path, workload, args, output_flag):
    _require_native()
    repo = Path(__file__).resolve().parent.parent
    runner = repo / 'benchmarks' / workload / f'run_{workload}.py'
    output = tmp_path / ('result.csv' if output_flag == '--output' else 'results')
    result = subprocess.run(
        [sys.executable, str(runner), *args, '--distances', '3', '--p-values', '0.001',
         '--decoder', 'ionq-beam-search', '--max-shots', '8', '--max-errors', '8',
         '--batch-size', '8', '--num-workers', '1', output_flag, str(output)],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = [output] if output_flag == '--output' else list(output.glob('*.csv'))
    assert files
    for path in files:
        rows = pd.read_csv(path)
        decoder_column = 'decoder_name' if workload == 'memory' else 'decoder'
        assert not rows.empty
        assert (rows[decoder_column] == 'ionq-beam-search').all()
        assert (rows['shots'] == 8).all()
        assert rows['errors'].between(0, rows['shots']).all()
        if workload == 'cross_ls':
            assert (rows['n_ps'] > 0).all()
