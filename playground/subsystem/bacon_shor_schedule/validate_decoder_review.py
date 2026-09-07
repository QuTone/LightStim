"""Reproduce saved first batches and verify scientific result provenance."""
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import json
import hashlib
import numpy as np
import stim
import nbformat
from playground.subsystem.bacon_shor_schedule.decoder_review import OUT, detector_rows
from playground.subsystem.bacon_shor_schedule.run import select_basis_detectors
from playground.subsystem.subsystem_crosscheck.common import interval, save_json, sha256, physical
from lightstim.simulation.decoder_backend import get_decoder

manifest = json.loads((OUT / 'manifest.json').read_text())
for path, expected in manifest['source_sha256'].items():
    source = ROOT / path
    if source.exists():
        assert sha256(source) == expected, path
    else:
        # Preserve the historical manifest. Reverse only the directory/import
        # relocation and require byte-for-byte equality with the old source.
        source = ROOT / path.replace('benchmarks/memory/', 'playground/subsystem/', 1)
        original = source.read_text().replace('playground.subsystem.', 'benchmarks.memory.')
        assert hashlib.sha256(original.encode()).hexdigest() == expected, path
verified = []
for path in sorted(OUT.glob('d*.json')):
    if path.name.endswith('_config.json'):
        continue
    state = json.loads(path.read_text())
    assert state['status'] in {'error_target', 'shot_cap', 'time_cap'}
    circuit_path = path.with_suffix('.stim')
    assert sha256(circuit_path) == state['circuit_sha256']
    full = stim.Circuit.from_file(circuit_path)
    selected = select_basis_detectors(full, 'Z')
    assert physical(full) == physical(selected)
    keep = detector_rows(full)
    assert len(keep) == selected.num_detectors
    model = selected if state['config']['profile'] == 'z_pair' else full
    dem = stim.DetectorErrorModel.from_file(path.with_suffix('.dem'))
    assert dem == model.detector_error_model()
    assert sum(b['shots'] for b in state['batches']) == state['shots']
    for name, result in state['results'].items():
        assert sum(b['counts'][name] for b in state['batches']) == result['errors'] == state['counts'][name]
        assert result['ler'] == result['errors']/state['shots']
        assert np.allclose(result['ci95'], interval(result['errors'], state['shots']))
    saved = np.load(OUT / (path.stem + '_first_batch.npz'))
    dets, obs = full.compile_detector_sampler(seed=int(saved['seed'])).sample(
        1000, separate_observables=True, bit_packed=True)
    assert np.array_equal(dets, saved['full_detectors']) and np.array_equal(obs, saved['observables'])
    ds = np.packbits(np.unpackbits(dets, axis=1, bitorder='little')[:, keep], axis=1, bitorder='little') if state['config']['profile'] == 'z_pair' else dets
    assert np.array_equal(ds, saved['decoder_detectors'])
    for name, (decoder, params) in state['decoder_configs'].items():
        compiled = get_decoder(decoder, **params).compile_decoder_for_dem(dem=dem)
        pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=ds)
        assert np.array_equal(pred, saved[name])
        assert int(((pred ^ obs) & 1).any(axis=1).sum()) == state['batches'][0]['counts'][name]
    verified.append(path.stem)
    print('reproduced', path.stem, flush=True)

nb = nbformat.read(ROOT / 'playground/subsystem/bacon_shor_decoder_review.ipynb', as_version=4)
cells = [c for c in nb.cells if c.cell_type == 'code']
assert len(cells) == 4 and all(c.execution_count and not any(o.output_type == 'error' for o in c.outputs) for c in cells)
sources = ['playground/subsystem/bacon_shor_schedule/' + name for name in
           ['decoder_review.py', 'audit_decoders.py', 'make_decoder_review.py', 'validate_decoder_review.py']]
save_json(OUT / 'validation.json', dict(verified_jobs=verified, reproduced_first_batches=len(verified),
    verified_executed_notebook_cells=len(cells),
    source_sha256={p: sha256(ROOT / p) for p in sources},
    artifact_sha256={str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*'))
                     if p.is_file() and p.name != 'validation.json'}))
print('All source, circuit, DEM, sample, decoder, count and notebook checks passed.')
