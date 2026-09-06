"""SHYPS algebra and memory contracts, including all protected logicals."""

from itertools import product

import numpy as np
import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.generic_css.gauge_SE_block import GenericCSSGaugeExtractionBlock
from lightstim.qec_code.repetition import RepetitionCode
from lightstim.qec_code.shyps import SHYPSCode


def _binary_rank(matrix):
    # Integer XOR elimination keeps this independent of the constructor's
    # column-elimination algorithm and of floating-point numerical rank.
    pivots = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        vector = int.from_bytes(np.packbits(row).tobytes(), "big")
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                break
            vector ^= pivots[pivot]
    return len(pivots)


def _matrix_from_records(records, basis, data_order):
    column_by_qubit = {qubit: column for column, qubit in enumerate(data_order)}
    rows = []
    for record in records:
        if record["type"] != basis:
            continue
        row = np.zeros(len(data_order), dtype=np.uint8)
        assert set(record["pauli"].values()) == {basis}
        for qubit in record["pauli"]:
            row[column_by_qubit[qubit]] = 1
        rows.append(row)
    return np.array(rows, dtype=np.uint8)


@pytest.mark.parametrize("r", [3, 4])
def test_simplex_seed_has_correct_kernel_and_weight_distribution(r):
    code = SHYPSCode(r)
    m = 2**r - 1
    h = code.simplex_parity_check
    generator = code.simplex_generator

    assert h.shape == (m, m)
    assert np.all(h.sum(axis=0) == 3)
    assert np.all(h.sum(axis=1) == 3)
    assert _binary_rank(h) == m - r
    assert _binary_rank(generator) == r
    assert not np.any(h @ generator.T % 2)
    words = np.array(list(product((0, 1), repeat=r)), dtype=np.uint8) @ generator % 2
    assert not words[0].any()
    assert np.all(words[1:].sum(axis=1) == 2 ** (r - 1))
    assert np.array_equal(
        code.simplex_pivot_matrix @ generator.T % 2,
        np.eye(r, dtype=np.uint8),
    )
    if r == 3:
        # Exact circulant starting row from Malcolm et al., Example VIII.7.
        assert h[0].tolist() == [1, 0, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("r", [3, 4])
def test_gauge_center_and_complete_bare_logical_algebra(r):
    code = SHYPSCode(r)
    m = code.simplex_length
    gx, gz = code.gauge_matrix_x, code.gauge_matrix_z
    sx, sz = code.center_matrix_x, code.center_matrix_z
    lx, lz = code.logical_matrix_x, code.logical_matrix_z

    assert code.num_data_qubits == m**2
    assert code.num_qubits == 3 * m**2
    assert code.num_logicals == r**2
    assert code.num_gauge_qubits == (m - r) ** 2
    assert code.code_distance == 2 ** (r - 1)
    assert _binary_rank(gx) == _binary_rank(gz) == m * (m - r)
    assert _binary_rank(gx @ gz.T % 2) == code.num_gauge_qubits
    assert _binary_rank(sx) == _binary_rank(sz) == r * (m - r)
    assert _binary_rank(np.vstack([gx, sx])) == _binary_rank(gx)
    assert _binary_rank(np.vstack([gz, sz])) == _binary_rank(gz)
    assert not np.any(sx @ gz.T % 2)
    assert not np.any(sz @ gx.T % 2)

    # These commute with every gauge, not merely with the centre.
    assert not np.any(lx @ gz.T % 2)
    assert not np.any(lz @ gx.T % 2)
    assert np.array_equal(lx @ lz.T % 2, np.eye(r**2, dtype=np.uint8))
    assert _binary_rank(np.vstack([sx, lx])) - _binary_rank(sx) == r**2
    assert _binary_rank(np.vstack([sz, lz])) - _binary_rank(sz) == r**2
    assert np.all(lx.sum(axis=1) == code.code_distance)
    assert np.all(lz.sum(axis=1) == code.code_distance)

    for records, expected_x, expected_z in (
        (code.gauges, gx, gz),
        (code.stabilizers, sx, sz),
        (code.logical_ops, lx, lz),
    ):
        assert np.array_equal(_matrix_from_records(records, "X", code.data_index_order), expected_x)
        assert np.array_equal(_matrix_from_records(records, "Z", code.data_index_order), expected_z)
    assert all(record["syn_idx"] is None for record in code.stabilizers)
    assert len({record["syn_idx"] for record in code.gauges}) == 2 * m**2

    for logical_id, pair in enumerate(code.logical_pairs):
        a, b = divmod(logical_id, r)
        assert pair["logical_id"] == logical_id
        assert pair["simplex_indices"] == (a, b)
        assert set(pair["x"]["data_indices"]) & set(pair["z"]["data_indices"]) == {pair["pivot_index"]}
        assert pair["pivot_index"] == code.data_qubits[
            (code.simplex_pivots[a], code.simplex_pivots[b])
        ]


def test_shifted_multi_patch_registration_keeps_gauge_and_logical_supports():
    code = SHYPSCode(3, shift=(20, 30))
    system = QECSystem()
    system.add_patch(RepetitionCode(distance=3), name="first")
    system.add_patch(code, offset=(10, -5), name="shyps")
    mapping = system.local_to_global_map["shyps"]

    assert mapping[0] != 0
    assert system.qubit_coords[mapping[0]] == (30, 25)
    for local, global_record in zip(
        code.gauges,
        [record for record in system.gauges if record["patch_name"] == "shyps"],
    ):
        assert global_record["pauli"] == {mapping[q]: b for q, b in local["pauli"].items()}
        assert global_record["syn_idx"] == mapping[local["syn_idx"]]
        assert global_record["syn_coord"] == system.qubit_coords[global_record["syn_idx"]]
    for local, global_record in zip(
        code.logical_ops,
        [record for record in system.logical_ops if record["patch_name"] == "shyps"],
    ):
        assert global_record["pauli"] == {mapping[q]: b for q, b in local["pauli"].items()}
        assert global_record["logical_id"] == local["logical_id"]


@pytest.mark.parametrize("r", [3, 4])
def test_extraction_measures_declared_gauges_and_preserves_bare_logicals(r):
    system = QECSystem()
    system.add_patch(SHYPSCode(r), name="shyps")
    extraction = GenericCSSGaugeExtractionBlock(system)
    assert extraction.depth_x == extraction.depth_z == 3

    for basis, block in zip(("X", "Z"), extraction.measurement_blocks):
        readout_order = [
            target.value
            for instruction in block
            if instruction.name in ("M", "MX")
            for target in instruction.targets_copy()
        ]
        for gauge in system.active_gauges:
            if gauge["type"] != basis:
                continue
            pauli = stim.PauliString(system.num_qubits)
            for qubit, factor in gauge["pauli"].items():
                pauli[qubit] = factor
            flow = stim.Flow(
                input=pauli,
                measurements=[readout_order.index(gauge["syn_idx"])],
            )
            assert block.has_flow(flow, unsigned=False)
        for logical in system.logical_ops:
            pauli = stim.PauliString(system.num_qubits)
            for qubit, factor in logical["pauli"].items():
                pauli[qubit] = factor
            assert block.has_flow(stim.Flow(input=pauli, output=pauli), unsigned=False)


@pytest.mark.parametrize("r", [True, 2, 5, 3.5, "3"])
def test_unsupported_simplex_dimension_is_rejected(r):
    with pytest.raises(ValueError, match="r"):
        SHYPSCode(r)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_public_shyps_memory_tracks_all_nine_logicals(basis):
    experiment = MemoryExperiment(qec_patch=SHYPSCode(), rounds=2, basis=basis)
    original_centers = frozenset(experiment.system.active_stabilizer_indices)
    original_gauges = frozenset(experiment.system.active_gauge_indices)
    circuit = experiment.build()
    assert experiment.system.active_stabilizer_indices == original_centers
    assert experiment.system.active_gauge_indices == original_gauges
    assert circuit.num_observables == 9
    assert circuit.num_detectors > 0
    assert circuit.detector_error_model().num_observables == 9
    detectors, observables = circuit.compile_detector_sampler(seed=31).sample(
        64, separate_observables=True
    )
    assert observables.shape == (64, 9)
    assert not detectors.any()
    assert not observables.any()


def test_shyps_circuit_noise_preserves_the_nine_observable_interface():
    circuit = MemoryExperiment(
        qec_patch=SHYPSCode(),
        rounds=2,
        basis="Z",
        noise_params=NoiseConfig(p_2q=0.001, p_meas=0.001),
    ).build()
    dem = circuit.detector_error_model()
    assert dem.num_errors > 0
    assert dem.num_observables == circuit.num_observables == 9
    detectors, observables = circuit.compile_detector_sampler(seed=32).sample(
        64, separate_observables=True
    )
    assert detectors.shape == (64, circuit.num_detectors)
    assert observables.shape == (64, 9)
