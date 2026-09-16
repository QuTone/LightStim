"""H-family memory audits with all-detector postselection.

The undecomposed DEM includes all single-fault signatures, including hyperedges.
Absence of an undetected logical term gives a lower bound of two. An actual
two-fault graphlike witness gives the upper bound. This applies to this memory
schedule with independent Pauli gate/SPAM noise, with and without idle noise.

Reproduce the full dedicated/coloration comparison from the repository root:
  PYTHONPATH=. python tests/test_H_code_fault_distance.py --max-n 64 --output /tmp/h-se-audit.json
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


@pytest.mark.smoke
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("use_coloration", [False, True])
def test_terminal_readout_fault_needs_final_detector(basis, use_coloration):
    """SE-only acceptance cannot protect a later noisy logical readout."""
    from lightstim.qec_code.H_code import HCodeExtractionBlock
    from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

    block = GenericCSSColorationExtractionBlock if use_coloration else HCodeExtractionBlock
    clean = MemoryExperiment(
        qec_patch=HCode(6), extraction_block_class=block, rounds=2, basis=basis,
    ).build().flattened()
    boundaries = [i for i, inst in enumerate(clean)
                  if inst.name in ("M", "MX")
                  and {t.value for t in inst.targets_copy()} == set(range(6))]
    assert len(boundaries) == 1
    boundary = boundaries[0]
    num_se_detectors = sum(inst.name == "DETECTOR" for inst in clean[:boundary])
    assert clean.num_detectors - num_se_detectors == 2

    # All operations are perfect except a single terminal readout-bit flip.
    probe = clean[:boundary]
    probe.append("X_ERROR" if basis == "Z" else "Z_ERROR", [0], 0.001)
    probe += clean[boundary:]
    errors = [e for e in probe.detector_error_model() if e.type == "error"]
    assert len(errors) == 1
    assert errors[0].args_copy()[0] == pytest.approx(0.001)
    targets = errors[0].targets_copy()
    detectors = [t.val for t in targets if t.is_relative_detector_id()]
    assert any(t.is_logical_observable_id() for t in targets)
    assert detectors  # The full memory acceptance rule rejects this event.
    assert all(d >= num_se_detectors for d in detectors)  # SE-only accepts it.


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


def audit(max_n=64):
    """Reproduce the full SE comparison, including the coloration counterexample."""
    import stim
    from lightstim.ir.qec_system import QECSystem
    from lightstim.qec_code.H_code import HCodeExtractionBlock
    from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

    rows = []
    counterexample = None
    for n in range(6, max_n + 1, 2):
        for schedule, block in (("dedicated", HCodeExtractionBlock),
                                ("coloration", GenericCSSColorationExtractionBlock)):
            system = QECSystem()
            system.add_patch(HCode(n), name="h")
            depth = block(system).cnot_depth
            for basis in ("X", "Z"):
                for rounds in (1, 2, 3):
                    for idle in (False, True):
                        p = .001
                        c = MemoryExperiment(
                            qec_patch=HCode(n), extraction_block_class=block,
                            basis=basis, rounds=rounds,
                            noise_params=NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                                                     p_idle=p if idle else 0),
                        ).build()
                        dem = c.detector_error_model(decompose_errors=False)
                        errors = [e for e in dem.flattened() if e.type == "error"]
                        bad = [e for e in errors
                               if any(t.is_logical_observable_id() for t in e.targets_copy())
                               and not any(t.is_relative_detector_id() for t in e.targets_copy())]
                        witness = c.shortest_graphlike_error(canonicalize_circuit_errors=True)
                        assert all(e.circuit_error_locations for e in witness)
                        if len(witness) == 2:
                            a, b = [e.circuit_error_locations[0] for e in witness]
                            if a.stack_frames == b.stack_frames:
                                ta, tb = a.instruction_targets, b.instruction_targets
                                assert (ta.target_range_end <= tb.target_range_start
                                        or tb.target_range_end <= ta.target_range_start)
                        assert len(witness) in (1, 2)
                        distance = 1 if bad else 2
                        assert distance == len(witness)
                        rows.append(dict(n=n, schedule=schedule, basis=basis, rounds=rounds,
                                         idle=idle, cnot_depth=depth,
                                         undetected_single_fault_signatures=len(bad),
                                         circuit_fault_distance=distance))
                        if schedule == "coloration" and n == 8 and basis == "Z" and bad and counterexample is None:
                            counterexample = str(c.explain_detector_error_model_errors(
                                dem_filter=stim.DetectorErrorModel(str(bad[0])),
                                reduce_to_one_representative_error=True,
                            )[0])
        summary = {name: sorted({r['circuit_fault_distance'] for r in rows
                                if r['n'] == n and r['schedule'] == name})
                   for name in ('dedicated', 'coloration')}
        print(f"n={n}: {summary}", flush=True)
    return dict(stim_version=stim.__version__, max_n=max_n, rows=rows,
                h8_coloration_counterexample=counterexample)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    HCode(args.max_n)  # Validate the size before starting.
    result = audit(args.max_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(result['rows'])} circuits; results: {args.output}", flush=True)
