"""
Test: Verify no syndrome cross-talk in back-propagated Paulis.

For the 6-tick SE schedule, each syndrome qubit's back-propagated Pauli
should have weight = (original stabilizer weight) + 1 (for the syndrome itself).
Any increase beyond +1 indicates syndrome cross-talk.

Result: ALL configurations pass — no cross-talk detected.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import stim
import numpy as np
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedMultiPatchCoupler
)
from lightstim.ir.qec_system import QECSystem


def check_no_crosstalk(system, label):
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    chunk = se.circuit
    tab = stim.Tableau.from_circuit(chunk, ignore_noise=True, ignore_measurement=True, ignore_reset=True)
    tab_inv = tab.inverse()
    syn_indices = [t.value for t in chunk[-1].targets_copy() if t.is_qubit_target]
    n = system.num_qubits
    x2x, x2z, z2x, z2z, _, _ = tab_inv.to_numpy()

    stab_weights = {}
    for uid in system.active_stabilizer_indices:
        s = system.stabilizers[uid]
        stab_weights[s['syn_idx']] = len(s['data_indices'])

    failures = []
    for syn_idx in syn_indices:
        bp_x = z2x[syn_idx]
        bp_z = z2z[syn_idx]
        weight = sum(1 for q in range(n) if bp_x[q] or bp_z[q])
        orig = stab_weights.get(syn_idx, None)
        if orig is not None and weight != orig + 1:
            failures.append((syn_idx, orig, weight))

    if failures:
        print(f"  {label}: FAIL — {len(failures)} syndromes with cross-talk")
        for syn_idx, orig, w in failures[:3]:
            print(f"    syn {syn_idx}: stab_weight={orig}, bp_weight={w}")
    else:
        print(f"  {label}: PASS — all {len(syn_indices)} syndromes clean (weight = stab+1)")
    return len(failures) == 0


def test_single_patch():
    for d in [3, 5]:
        system = QECSystem()
        system.add_patch(UnrotatedSurfaceCode(distance=d), name='p')
        assert check_no_crosstalk(system, f"single patch d={d}")


def test_two_patch_coupler():
    d = 3
    system = QECSystem()
    for name, offset in [('p1', (-2, 0)), ('p2', (10, 0))]:
        p = UnrotatedSurfaceCode(distance=d); p.transpose_coords()
        system.add_patch(p, name=name, offset=offset)
    system.register_coupler(UnrotatedMultiPatchCoupler(), patch_names=['p1', 'p2'],
                            name='c', path_axis='vertical', center_axis=6.0)
    system.activate_coupler('c')
    assert check_no_crosstalk(system, "2-patch coupler")


def test_four_patch_coupler():
    d = 3
    system = QECSystem()
    for name, offset in [('W1',(-2,0)), ('W3',(10,0)), ('W2',(-2,8)), ('W4',(10,8)), ('W5',(-2,16))]:
        p = UnrotatedSurfaceCode(distance=d); p.transpose_coords()
        system.add_patch(p, name=name, offset=offset)
    system.register_coupler(UnrotatedMultiPatchCoupler(),
        patch_names=['W1','W2','W3','W5'], name='c',
        path_axis='vertical', center_axis=6.0)
    system.activate_coupler('c')
    assert check_no_crosstalk(system, "4-patch ZZZZ coupler (distillation layout)")


