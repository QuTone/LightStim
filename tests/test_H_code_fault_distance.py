"""H-family memory audits with all-detector postselection.

The undecomposed DEM includes all single-fault signatures, including hyperedges.
Absence of an undetected logical term gives a lower bound of two. An actual
two-fault graphlike witness gives the upper bound. This applies to this memory
schedule with independent Pauli gate/SPAM noise, with and without idle noise.
"""

import numpy as np
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode


def _noisy_memory(basis, rounds=2, p=1e-3, n=6, idle=False):
    return MemoryExperiment(
        qec_patch=HCode(n=n), rounds=rounds, basis=basis,
        noise_params=NoiseConfig(
            p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p if idle else 0
        ),
        noise_model="circuit_level",
    ).build()


@pytest.mark.smoke
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("n", [6, 8, 12, 20, 32, 64])
@pytest.mark.parametrize("idle", [False, True])
def test_all_single_faults_detected_and_two_fault_witness(basis, rounds, n, idle):
    noisy = _noisy_memory(basis, rounds, n=n, idle=idle)
    errors = [
        inst for inst in noisy.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=False
        ).flattened() if inst.type == "error"
    ]
    assert errors
    assert any(any(t.is_logical_observable_id() for t in e.targets_copy()) for e in errors)
    for error in errors:
        targets = error.targets_copy()
        assert not any(t.is_separator() for t in targets)
        if any(t.is_logical_observable_id() for t in targets):
            assert any(t.is_relative_detector_id() for t in targets), str(error)

    witness = noisy.shortest_graphlike_error(canonicalize_circuit_errors=True)
    assert len(witness) == 2
    assert all(error.circuit_error_locations for error in witness)
    # Batched Stim instructions contain independent channels on each target
    # (or target pair). Exclude mutually exclusive outcomes of one channel.
    locations = [error.circuit_error_locations[0] for error in witness]
    if locations[0].stack_frames == locations[1].stack_frames:
        a, b = (loc.instruction_targets for loc in locations)
        assert a.target_range_end <= b.target_range_start or b.target_range_end <= a.target_range_start


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("n", [6, 10])
def test_baseline_memory_conditional_ler_is_quadratic(basis, n):
    rates = (0.002, 0.004, 0.008, 0.016)
    lers = []
    for index, p in enumerate(rates):
        noisy = _noisy_memory(basis, p=p, n=n)
        dets, obs = noisy.compile_detector_sampler(seed=98 + index).sample(
            400_000, separate_observables=True
        )
        accepted = ~dets.any(axis=1)
        failures = int(obs[accepted].any(axis=1).sum())
        assert accepted.sum() > 2_000 and failures > 0
        ler = failures / accepted.sum()
        lers.append(ler)
        print(f"H{n} {basis}: p={p:g}, accepted={accepted.sum()}, failures={failures}, conditional_ler={ler:.8g}")
    slope = float(np.polyfit(np.log(rates), np.log(lers), 1)[0])
    print(f"H{n} {basis}: fitted slope={slope:.4f}")
    assert 1.6 <= slope <= 2.6, f"baseline slope {slope:.2f} not near 2"


def _frame_after_cnot(patch, block_class, pair, error_qubit, error_basis):
    """Independent Pauli propagation from one physical fault to SE output."""
    import stim
    from lightstim.ir.qec_system import QECSystem

    system = QECSystem()
    system.add_patch(patch, name="h")
    circuit = block_class(system).circuit
    suffix = stim.Circuit()
    found = False
    for inst in circuit:
        if inst.name == "CX":
            targets = [t.value for t in inst.targets_copy()]
            pairs = list(zip(targets[::2], targets[1::2]))
            if pair in pairs:
                assert not found
                found = True
                # Other gates in this layer are disjoint from the faulty pair.
                continue
        if found and inst.name in ("CX", "H"):
            suffix.append(inst)
    assert found
    frame = stim.PauliString(patch.num_qubits)
    frame[error_qubit] = error_basis
    return frame.after(suffix)


def _check_paulis(patch):
    import stim
    checks = []
    for record in patch.stabilizers:
        p = stim.PauliString(patch.n)
        for q, basis in record["pauli"].items():
            p[q] = basis
        checks.append(p)
    return checks


@pytest.mark.smoke
def test_h8_coloration_has_an_undetected_single_fault_logical():
    import stim
    from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

    patch = HCode(8)
    # X on XB ancilla after CX(XB, data 5). This is one outcome of a
    # two-qubit depolarizing fault, not two independently inserted errors.
    frame = _frame_after_cnot(patch, GenericCSSColorationExtractionBlock,
                             pair=(9, 5), error_qubit=9, error_basis="X")
    assert all(frame[q] not in (1, 2) for q in patch.syndrome_indices)
    data = frame[:8]
    checks = _check_paulis(patch)
    assert all(data.commutes(check) for check in checks)
    # Up to the XB stabilizer the residual is X5 X6, a nontrivial logical.
    assert data * checks[1] == stim.PauliString("_____XX_")
    assert any(not data.commutes(stim.PauliString(
        ''.join(op['pauli'].get(q, 'I') for q in range(8))
    )) for op in patch.logical_ops)


@pytest.mark.smoke
@pytest.mark.parametrize("n", [6, 8, 12, 64])
def test_dedicated_memory_result_does_not_certify_open_output_extraction(n):
    from itertools import product
    import stim
    from lightstim.qec_code.H_code import HCodeExtractionBlock

    patch = HCode(n)
    # Z on ZA after CX(data 0, ZA) spreads to private data 1 and final shared
    # data 3, after all relevant X-check interactions. This round accepts.
    frame = _frame_after_cnot(patch, HCodeExtractionBlock,
                             pair=(0, n + 2), error_qubit=n + 2, error_basis="Z")
    assert all(frame[q] not in (1, 2) for q in patch.syndrome_indices)
    data = frame[:n]
    assert data == stim.PauliString("_Z_Z" + "_" * (n - 4))
    checks = _check_paulis(patch)
    weights = []
    for choices in product((0, 1), repeat=4):
        residual = data.copy()
        for include, check in zip(choices, checks):
            if include:
                residual *= check
        weights.append(residual.weight)
    assert min(weights) == 2  # Cannot reduce this output error to weight one.
    # The error has a nontrivial final data syndrome, so a subsequent perfect
    # round or final X readout detects it. It is not an undetected logical.
    assert [not data.commutes(s) for s in checks] == [False, True, False, False]
