"""Native unrotated held-merge fixture with independent logical-output frames.

The final global XXX logical includes
both physical X-read corridor strings when the merges remain active. This
frame is derived from physical supports, not from tracker OBS annotations.
"""

import numpy as np
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedTwoPatchCoupler,
)
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _contains, _full_record_parities, _mask,
)


def last_records(circuit):
    result, record = {}, 0
    for instruction in circuit.flattened():
        if instruction.name in ("M", "MZ", "MX", "MY"):
            for offset, target in enumerate(instruction.targets_copy()):
                result[target.value] = record + offset
        record += instruction.num_measurements
    return result


def build_case(initials, finals):
    system = QECSystem()
    for i, name in enumerate("ABC"):
        patch = UnrotatedSurfaceCode(distance=3)
        if name in "AC":
            patch.rotate_coords(np.pi)
            patch.reset_rotation_angle()
        system.add_patch(patch, name=name, offset=(0, 6*i))
    for pair in ("AB", "BC"):
        system.register_coupler(UnrotatedTwoPatchCoupler(), patch_names=list(pair),
                                name=pair, interaction_type="ZZ")
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=3,
                              logical_tracking="slabs")
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    owner = system.index_to_owner_map
    patch_data = {q for q in system.data_indices if owner[q] in tuple("ABC")}
    coupler_data = {pair: {q for q in system.data_indices if owner[q] == pair}
                    for pair in ("AB", "BC")}
    patch_bases = dict(zip("ABC", initials))
    builder.initialize({q: patch_bases[owner[q]] for q in patch_data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)
    operators = {(op["patch_name"], op["type"]): op
                 for op in system.logical_ops if op["patch_name"] in tuple("ABC")}
    outcomes, stages = [], []
    for pair in ("AB", "BC"):
        builder.activate_coupler(pair)
        builder.initialize({q: "X" for q in coupler_data[pair]}, n=system.num_qubits)
        active_checks = list(system.active_stabilizers)
        active = stabilizers_to_symplectic(system, active_checks, system.num_qubits)
        n = system.num_qubits
        commutators = (active[:, :n] @ active[:, n:].T
                       + active[:, n:] @ active[:, :n].T) % 2
        assert not commutators.any(), "Declared merged code contains anticommuting checks"
        builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)
        target = np.zeros((1, 2*n), dtype=np.uint8)
        for name in pair:
            target ^= stabilizers_to_symplectic(system, [operators[name, "Z"]], n)
        coefficients, known, _ = solve_linear_decomposition(
            basis=active, targets=target, reduce_weight=False)
        assert known.all(), "Joint ZZ is not in the declared active check span"
        recs = last_records(builder.circuit)
        selected = np.flatnonzero(coefficients[0])
        output = _mask(recs[active_checks[int(i)]["syn_idx"]] for i in selected)
        outcomes.append(output)
        stages.append(dict(new_merge=pair, active_check_count=len(active_checks),
                           active_checks_commute=True, joint_records=list(_bits(output)),
                           new_history=[list(h.records) for h in tracker.logical_history]))
    patch_bases = dict(zip("ABC", finals))
    readout = {q: patch_bases[owner[q]] for q in patch_data}
    readout.update({q: "X" for data in coupler_data.values() for q in data})
    builder.apply_data_readout(readout)
    circuit = builder.to_stim_circuit()
    physical, detectors, observables = _annotations(circuit)
    recs = last_records(physical)
    outcomes += [_mask(recs[q] for q in operators[name, basis]["pauli"])
                 for name, basis in zip("ABC", finals)]
    terminal_x_frame = []
    if finals == "XXX":
        # In the connected ZZ-merged code, the only commuting X-type product
        # of patch logicals is global XXX. Complete its physical representative
        # across both corridors. Attribute this known readout frame to A; only
        # the global XXX parity can enter a deterministic ideal output relation.
        corridor_qubits = sorted(set().union(*coupler_data.values()))
        global_x = np.zeros(2*system.num_qubits, dtype=np.uint8)
        for name in "ABC":
            global_x ^= stabilizers_to_symplectic(system, [operators[name, "X"]], system.num_qubits)[0]
        global_x[corridor_qubits] ^= 1
        assert not ((active[:, :n] @ global_x[n:]
                     + active[:, n:] @ global_x[:n]) % 2).any()
        terminal_x_frame = [recs[q] for q in corridor_qubits]
        outcomes[2] ^= _mask(terminal_x_frame)

    oracle = stim.Circuit()
    for q, basis in enumerate(initials):
        oracle.append("R" if basis == "Z" else "RX", [q])
    oracle.append("MPP", [stim.target_z(0), stim.target_combiner(), stim.target_z(1)])
    oracle.append("MPP", [stim.target_z(1), stim.target_combiner(), stim.target_z(2)])
    for q, basis in enumerate(finals):
        oracle.append("M" if basis == "Z" else "MX", [q])
    ideal = _full_record_parities(oracle)
    targets = []
    for row in ideal:
        parity = 0
        for i in _bits(row):
            parity ^= outcomes[i]
        targets.append(parity)
    complete = _full_record_parities(physical)
    full_basis = _basis(complete)
    checks = dict(
        all_emitted_and_ideal_parities_valid=all(
            _contains(full_basis, row) for row in detectors + observables + targets),
        full_parity_space=len(_basis(detectors + observables)) == len(complete),
        independent_output_mapping=len(_basis(targets)) == len(ideal),
        every_ideal_output_direction_protected=(
            len(_basis(detectors + targets)) - len(_basis(detectors)) == len(ideal)),
        protected_rank_equals_ideal=(
            len(_basis(detectors + observables)) - len(_basis(detectors)) == len(ideal)))
    signed = physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row)))
        for row in detectors + observables + targets], unsigned=False)
    circuit.detector_error_model()
    return dict(initials=initials, finals=finals, native_unrotated=True,
                AB_remains_active_during_BC=True, combined_final_readout=True,
                stages=stages, ideal_logical_rank=len(ideal),
                observed_protected_rank=len(_basis(detectors + observables)) - len(_basis(detectors)),
                full_parity_rank=len(complete), detector_rank=len(_basis(detectors)),
                measurements=physical.num_measurements, qubits=physical.num_qubits,
                checks=checks, all_classification_checks_pass=all(checks.values()),
                complete_signed_flows=signed, DEM_valid=True,
                terminal_global_XXX_corridor_records=terminal_x_frame,
                physical_protocol_output_records=[list(_bits(row)) for row in outcomes],
                ideal_deterministic_relations=[list(_bits(row)) for row in targets],
                observable_rows=[list(_bits(row)) for row in observables])
