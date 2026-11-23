"""Protocol-level regression against an independently located physical witness."""

from lightstim.analysis.faults import FaultEnumerator
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode
from lightstim.simulation.decoder_backend import SimulationPipeline


def test_h_code_memory_includes_stims_distinct_site_double_fault():
    circuit = MemoryExperiment(
        qec_patch=HCode(n=6),
        rounds=1,
        basis="Z",
        noise_params=NoiseConfig(p_1q=.001, p_2q=.001, p_meas=.001, p_reset=.001),
        noise_model="circuit_level",
    ).build().flattened()
    selected = list(range(circuit.num_detectors))
    result = SimulationPipeline(post_select_detector_indices=selected).enumerate_faults(
        circuit, max_order=2
    )
    assert result.per_order[1].loss_mass == 0
    assert result.per_order[2].loss_mass > 0

    # Identify the two actual Pauli insertions independently via Stim. The
    # H(6) circuit has two logical observables; its witness flips the second.
    witness = circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
    assert len(witness) == 2
    chosen = []
    for error in witness:
        (origin,) = error.circuit_error_locations
        (frame,) = origin.stack_frames
        paulis = tuple(
            (term.gate_target.value, term.gate_target.pauli_type)
            for term in origin.flipped_pauli_product
        )
        matches = [
            (i, j)
            for i, site in enumerate(result.locations)
            if site.instruction_index == frame.instruction_offset
            and site.target_start == origin.instruction_targets.target_range_start
            for j, alternative in enumerate(site.alternatives)
            if alternative.paulis == paulis
        ]
        assert len(matches) == 1
        chosen.append(matches[0])
    audit = FaultEnumerator(circuit, post_select_detector_indices=selected)
    for fault in chosen:
        assert audit.evaluate((fault,)).acceptance == 0
    pair = audit.evaluate(tuple(sorted(chosen)))
    assert pair.acceptance == pair.accepted_loss == 1
