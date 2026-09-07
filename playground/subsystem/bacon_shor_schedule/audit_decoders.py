"""Decode each individual full-DEM mechanism with four CPU configurations."""
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import stim
from playground.subsystem.bacon_shor_schedule.decoder_review import build, detector_rows, PARAMS, OUT
from playground.subsystem.bacon_shor_schedule.run import select_basis_detectors
from playground.subsystem.subsystem_crosscheck.common import save_json
from lightstim.simulation.decoder_backend import get_decoder
from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices


def run():
    results = []
    for d in [3, 5, 7, 9]:
        c = build(d, .001)
        dem, keep = c.detector_error_model(), detector_rows(c)
        h, l, p = dem_to_matrices(dem, sparse=True, merge_duplicates=False)
        dets, obs = np.asarray(h.T.todense(), dtype=np.uint8), np.asarray(l.T.todense(), dtype=np.uint8)
        item = dict(d=d, error_mechanisms=len(p), profiles={})
        for name, decoder, params, zonly in [
            ('mwpm_z', 'pymatching', {}, True),
            ('bposd_z_serial', 'bposd', dict(PARAMS, schedule='serial'), True),
            ('bposd_full_serial', 'bposd', dict(PARAMS, schedule='serial'), False),
            ('bposd_full_parallel', 'bposd', dict(PARAMS, schedule='parallel'), False),
        ]:
            model = select_basis_detectors(c, 'Z').detector_error_model() if zonly else dem
            packed = np.packbits(dets[:, keep] if zonly else dets, axis=1, bitorder='little')
            comp = get_decoder(decoder, **params).compile_decoder_for_dem(dem=model)
            pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
            bad = np.flatnonzero(((pred & 1) ^ obs).any(axis=1))
            item['profiles'][name] = dict(bad_count=len(bad), bad_indices=bad.tolist(),
                                         bad_mechanism_probability_sum=float(p[bad].sum()))
            if len(bad):
                all_errors = [op for op in dem.flattened() if op.type == 'error']
                filt = stim.DetectorErrorModel()
                for i in bad[:3]:
                    filt.append(all_errors[i])
                explanations = c.explain_detector_error_model_errors(
                    dem_filter=filt, reduce_to_one_representative_error=True)
                (OUT / f'd{d}_{name}_single_fault_locations.txt').write_text('\n\n'.join(map(str, explanations)) + '\n')
        results.append(item)
        save_json(OUT / 'single_fault_audit.json', results)
        print(d, {k: v['bad_count'] for k, v in item['profiles'].items()}, flush=True)


if __name__ == '__main__':
    run()
