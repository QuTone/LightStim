"""CrossLS must export and consume the protocol's PQRM-X postselection checks.

The oracle comes from the code operators and the measured merged Z checks, not
tracker row IDs, detector coordinates, or the implementation's selection helper.
Each PQRM X stabilizer has a unique bridge-X dressing commuting with the merged
Z measurements. Its final MX parity must be selected, even when the tracker uses
a different basis for those constraints. These checks existed before PR #75;
that revision's unmeasured-record guard silently removed them, and merely
restoring detector output without its postselection tags is insufficient.
"""
from functools import lru_cache
from itertools import product

import numpy as np
import pytest
import stim

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.cross_ls import CrossLSExperiment
from lightstim.simulation.decoder_backend.post_select import (
    apply_post_selection,
    get_post_select_detector_indices,
)


# Existing *extra* surface-selection parities at e0750dd7, before the PQRM fix.
# Source: CrossLSExperiment(PQRM_para=[1,2,4], d_surf=3, rounds=2 or 3,
# PQRM_state=state, surf_state='X') with the indicated flag. Terminal rows use
# offsets from the final measurement count, so the same fixture covers r2/r3.
# PQRM directions were removed from this fixture; the geometry oracle above is
# the authority for those. This deliberately preserves the current Y policy,
# rather than reinstating older coordinate-dependent hybrid choices from #75.
_EXTRA_TERMINAL_CHECKS_E075 = {
    ("Z", "hybrid"): (
        (-53, -8, -5, -3), (-48, -13, -10, -8),
        (-47, -12, -10, -9, -7), (-46, -11, -9, -6)),
    ("Z", "surface_se"): (),
    ("X", "hybrid"): (
        (-55, -5, -3, -2), (-50, -10, -8, -7, -5),
        (-45, -15, -13, -12, -10), (-44, -14, -12, -11, -9)),
    ("X", "surface_se"): (
        (-55, -5, -3, -2), (-54, -4, -2, -1),
        (-50, -10, -8, -7, -5), (-49, -9, -7, -6, -4)),
    ("Y", "hybrid"): (
        (-55, -53, -5, -3, -2), (-52, -50, -10, -8, -7, -5),
        (-54, -48, -4, -2, -1), (-49, -47, -9, -7, -6, -4),
        (-51, -45, -15, -13, -12, -10), (-46, -44, -14, -12, -11, -9)),
    ("Y", "surface_se"): ((-52, -50, -10, -8, -7, -5),),
}


@lru_cache(maxsize=None)
def _build(params=(1, 2, 4), state="Z", rounds=2, mode="pqrm_only", noisy=False,
           distance=3):
    exp = CrossLSExperiment(
        PQRM_para=list(params), d_surf=distance, rounds=rounds,
        PQRM_state=state, surf_state="X",
        post_select_hybrid=mode == "hybrid",
        post_select_surface_se=mode == "surface_se",
        noise_params=(NoiseConfig(p_1q=1e-6, p_2q=1e-3,
                                  p_meas=1e-3, p_reset=1e-3)
                      if noisy else None),
    )
    return exp, exp.build()


def _basis(rows):
    """Small independent GF(2) row-space oracle over absolute record indices."""
    pivots = {}
    for row in rows:
        row = set(row)
        while row:
            pivot = max(row)
            if pivot not in pivots:
                pivots[pivot] = frozenset(row)
                break
            row.symmetric_difference_update(pivots[pivot])
    return pivots


def _assert_contains(rows, expected):
    basis = _basis(rows)
    for relation in expected:
        remainder = set(relation)
        while remainder and max(remainder) in basis:
            remainder.symmetric_difference_update(basis[max(remainder)])
        assert not remainder, f"missing parity direction: {sorted(relation)}"


def _assert_same_space(actual, expected):
    _assert_contains(actual, expected)
    _assert_contains(expected, actual)


def _detector_rows(circuit):
    rows = []
    measurement_count = 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            row = set()
            for target in instruction.targets_copy():
                assert target.is_measurement_record_target
                row.symmetric_difference_update([measurement_count + target.value])
            rows.append(frozenset(row))
        measurement_count += instruction.num_measurements
    return rows


def _selected_rows(circuit):
    rows = _detector_rows(circuit)
    return [rows[i] for i in get_post_select_detector_indices(circuit)]


def _expected_pqrm_checks(exp, circuit):
    """Derive surviving +1 X constraints using only the code and merge geometry.

    The bridge starts in |+>. Multiplying its X operators into an encoded PQRM X
    check preserves its known eigenvalue. The correct dressing commutes with all
    measured merged Z checks, so it survives SE and closes over terminal MX.
    Undressed PQRM checks would be an incorrect oracle across the merge boundary.
    """
    system = exp.system
    bridge = system.coupler_patches["surface_pqrm_coupler"]
    local_to_global = system.local_to_global_map["surface_pqrm_coupler"]
    bridge_qubits = sorted(local_to_global[q] for q in bridge.data_indices)
    measured_z = [set(row["pauli"]) for row in system.active_stabilizers_z
                  if row.get("syn_idx") is not None]
    x_checks = [set(row["pauli"]) for row in system.stabilizers
                if row.get("patch_name") == "pqrm" and row.get("type") == "X"]

    last_mx = {}
    measurement_count = 0
    for instruction in circuit.flattened():
        if instruction.name == "MX":
            for offset, target in enumerate(instruction.targets_copy()):
                last_mx[target.value] = measurement_count + offset
        measurement_count += instruction.num_measurements

    expected = []
    for check in x_checks:
        dressings = []
        for bits in product((0, 1), repeat=len(bridge_qubits)):
            support = check | {q for q, bit in zip(bridge_qubits, bits) if bit}
            if all(len(support & z_check) % 2 == 0 for z_check in measured_z):
                dressings.append(support)
        assert len(dressings) == 1, "merge must uniquely dress each PQRM X check"
        expected.append(frozenset(last_mx[q] for q in dressings[0]))
    assert len(_basis(expected)) == len(x_checks)
    return expected


@pytest.mark.parametrize("state", ["Z", "X", "Y"])
@pytest.mark.parametrize("rounds", [2, 3])
@pytest.mark.parametrize("mode", ["pqrm_only", "hybrid", "surface_se"])
def test_cross_ls_selects_entire_pqrm_x_constraint_space(state, rounds, mode):
    exp, circuit = _build(state=state, rounds=rounds, mode=mode)
    expected = _expected_pqrm_checks(exp, circuit)
    # Validate the oracle against the physical circuit independently of its DEM.
    assert all(circuit.has_flow(stim.Flow(measurements=sorted(row)))
               for row in expected)
    circuit.detector_error_model()
    selected = _selected_rows(circuit)
    if mode == "pqrm_only":
        _assert_same_space(selected, expected)
    else:
        extra = [frozenset(circuit.num_measurements + offset for offset in row)
                 for row in _EXTRA_TERMINAL_CHECKS_E075[state, mode]]
        if mode == "hybrid":
            extra += [frozenset([i]) for i in (0, 1, 5, 10)]
        # Preserve existing extras exactly while restoring all mandatory checks.
        _assert_same_space(selected, expected + extra)


@pytest.mark.parametrize("params", [(1, 3, 5), (1, 4, 6)])
@pytest.mark.parametrize("state", ["Z", "X", "Y"])
def test_cross_ls_larger_pqrm_uses_all_x_checks(params, state):
    exp, circuit = _build(params=params, state=state, rounds=2)
    expected = _expected_pqrm_checks(exp, circuit)
    assert all(circuit.has_flow(stim.Flow(measurements=sorted(row)))
               for row in expected)
    _assert_same_space(_selected_rows(circuit), expected)


@pytest.mark.parametrize("distance,state", [(5, "Z"), (7, "Y")])
def test_cross_ls_larger_surface_preserves_pqrm_selection(distance, state):
    exp, circuit = _build(distance=distance, state=state)
    expected = _expected_pqrm_checks(exp, circuit)
    assert all(circuit.has_flow(stim.Flow(measurements=sorted(row)))
               for row in expected)
    _assert_same_space(_selected_rows(circuit), expected)


@pytest.mark.parametrize("mode", ["pqrm_only", "hybrid", "surface_se"])
def test_cross_ls_postselection_accepts_clean_and_rejects_pqrm_fault(mode):
    exp, circuit = _build(mode=mode)
    selected = get_post_select_detector_indices(circuit)
    clean_d, clean_o = circuit.compile_detector_sampler(seed=73).sample(
        16, separate_observables=True)
    assert not np.any(clean_d)
    assert not np.any(clean_o)
    accepted_d, accepted_o = apply_post_selection(clean_d, clean_o, selected)
    assert len(accepted_d) == len(accepted_o) == 16

    # A late Z on a PQRM data qubit flips a known X check. Inject it as noise so
    # Stim continues to use the clean ideal reference when sampling detectors.
    pqrm = exp.system.patches["pqrm"][0]
    local_to_global = exp.system.local_to_global_map["pqrm"]
    flat = circuit.flattened()
    for local_qubit in pqrm.data_indices:
        qubit = local_to_global[local_qubit]
        readout = [i for i, op in enumerate(flat)
                   if op.name == "MX"
                   and qubit in [t.value for t in op.targets_copy()]]
        assert len(readout) == 1
        faulty = flat[:readout[0]]
        faulty.append("Z_ERROR", [qubit], 1)
        faulty += flat[readout[0]:]
        fault_d, fault_o = faulty.compile_detector_sampler(seed=73).sample(
            16, separate_observables=True)
        assert np.all(np.any(fault_d[:, selected], axis=1)), qubit
        accepted_d, accepted_o = apply_post_selection(fault_d, fault_o, selected)
        assert len(accepted_d) == len(accepted_o) == 0


@pytest.mark.parametrize("state", ["Z", "Y"])
@pytest.mark.parametrize("mode", ["pqrm_only", "hybrid"])
def test_cross_ls_noise_export_preserves_selected_checks(state, mode):
    _, clean = _build(state=state, rounds=3, mode=mode)
    exp, noisy = _build(state=state, rounds=3, mode=mode, noisy=True)
    assert noisy.without_noise() == clean
    _assert_same_space(_selected_rows(noisy), _selected_rows(clean))
    _assert_contains(_selected_rows(noisy), _expected_pqrm_checks(exp, noisy))
    noisy.detector_error_model(approximate_disjoint_errors=True)


@pytest.mark.parametrize("canonical", [False, True])
def test_cross_ls_optional_canonicalization_preserves_selection(canonical):
    exp = CrossLSExperiment(
        PQRM_para=[1, 2, 4], d_surf=3, rounds=2, PQRM_state="Z",
        canonical_pqrm_logical=canonical,
    )
    circuit = exp.build()
    _assert_same_space(_selected_rows(circuit),
                       _expected_pqrm_checks(exp, circuit))
    circuit.detector_error_model()
