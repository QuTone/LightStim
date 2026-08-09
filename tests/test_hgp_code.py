"""Phase-1 tests for hypergraph-product code patches."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.HGP import (
    BinaryParityCheck,
    HGPCode,
    HGPProductColorationExtractionBlock,
    canonical_kernel_basis,
    hgp_13_1_3,
    hgp_13_1_3_seed,
    hgp_18_2_3,
    hgp_18_2_3_seed,
    hgp_225_9_4,
    hgp_225_9_4_seed,
)
from lightstim.qec_code.generic_css import (
    GenericCSSColorationExtractionBlock,
    color_bipartite_edges,
)
from lightstim.qec_code.surface_code.toric import ToricCode
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode
from lightstim.utils.linear_algebra import row_echelon


def _gf2_rank(matrix: np.ndarray) -> int:
    return int(row_echelon(matrix.astype(np.uint8))[1])


def _classical_distance(matrix: np.ndarray) -> int:
    for weight in range(1, matrix.shape[1] + 1):
        for support in combinations(range(matrix.shape[1]), weight):
            if not np.any(np.bitwise_xor.reduce(matrix[:, support], axis=1)):
                return weight
    raise ValueError("The classical code has no nonzero codewords.")


def _stabilizer_signatures(
    patch,
    coords_by_index: dict[int, tuple[float, float]] | None = None,
    transform: Callable[[tuple[float, float]], tuple[float, float]] = lambda coord: coord,
) -> set[tuple[str, tuple[float, float], frozenset[tuple[float, float]]]]:
    coords = patch.qubit_coords if coords_by_index is None else coords_by_index
    return {
        (
            stabilizer["type"],
            transform(coords[stabilizer["syn_idx"]]),
            frozenset(
                transform(coords[index])
                for index in stabilizer["data_indices"]
            ),
        )
        for stabilizer in patch.stabilizers
    }


def _checkerboard_tanner_embedding(code: HGPCode) -> dict[int, tuple[float, float]]:
    """Embed semantic product indices in the conventional surface-code grid."""
    coords: dict[int, tuple[float, float]] = {}
    coords.update(
        {
            index: (2 * bit_2, 2 * bit_1)
            for (bit_1, bit_2), index in code.vv_qubits.items()
        }
    )
    coords.update(
        {
            index: (2 * check_2 + 1, 2 * check_1 + 1)
            for (check_1, check_2), index in code.cc_qubits.items()
        }
    )
    coords.update(
        {
            index: (2 * check_2 + 1, 2 * bit_1)
            for (bit_1, check_2), index in code.x_check_qubits.items()
        }
    )
    coords.update(
        {
            index: (2 * bit_2, 2 * check_1 + 1)
            for (check_1, bit_2), index in code.z_check_qubits.items()
        }
    )
    return coords


def _assert_hgp_parity_check_formula(code: HGPCode) -> None:
    h1 = code.h1.to_dense()
    h2 = code.h2.to_dense()
    n1, n2 = code.h1.num_bits, code.h2.num_bits
    m1, m2 = code.h1.num_checks, code.h2.num_checks
    expected_hx = np.concatenate(
        [np.kron(np.eye(n1, dtype=np.uint8), h2), np.kron(h1.T, np.eye(m2, dtype=np.uint8))],
        axis=1,
    )
    expected_hz = np.concatenate(
        [np.kron(h1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(m1, dtype=np.uint8), h2.T)],
        axis=1,
    )
    hx, hz = code.get_data_parity_check_matrices()
    assert np.array_equal(hx, expected_hx)
    assert np.array_equal(hz, expected_hz)


def _assert_product_block_layout(code: HGPCode) -> None:
    """Check all four sectors, including the single-coordinate gutter."""
    right_start = code.h1.num_bits + 1
    lower_start = code.h2.num_bits + 1

    assert {
        key: code.qubit_coords[index] for key, index in code.vv_qubits.items()
    } == {
        (bit_1, bit_2): (bit_1, bit_2)
        for bit_1 in range(code.h1.num_bits)
        for bit_2 in range(code.h2.num_bits)
    }
    assert {
        key: code.qubit_coords[index] for key, index in code.z_check_qubits.items()
    } == {
        (check_1, bit_2): (right_start + check_1, bit_2)
        for check_1 in range(code.h1.num_checks)
        for bit_2 in range(code.h2.num_bits)
    }
    assert {
        key: code.qubit_coords[index] for key, index in code.x_check_qubits.items()
    } == {
        (bit_1, check_2): (bit_1, lower_start + check_2)
        for bit_1 in range(code.h1.num_bits)
        for check_2 in range(code.h2.num_checks)
    }
    assert {
        key: code.qubit_coords[index] for key, index in code.cc_qubits.items()
    } == {
        (check_1, check_2): (right_start + check_1, lower_start + check_2)
        for check_1 in range(code.h1.num_checks)
        for check_2 in range(code.h2.num_checks)
    }

    all_coords = list(code.qubit_coords.values())
    assert len(all_coords) == len(set(all_coords)) == code.num_qubits

    # Coordinates changed, but stable semantic registration order did not.
    offset = 0
    for mapping in (
        code.vv_qubits,
        code.cc_qubits,
        code.x_check_qubits,
        code.z_check_qubits,
    ):
        indices = [mapping[key] for key in sorted(mapping)]
        assert indices == list(range(offset, offset + len(mapping)))
        offset += len(mapping)


def _assert_product_layer_semantics(
    block: HGPProductColorationExtractionBlock,
) -> None:
    """Verify that direction relabeling does not alter any scheduled target."""
    code = block.patch
    global_index = block.local_to_global

    for layer in block.layers:
        expected_seed = "H1" if layer.direction == "horizontal" else "H2"
        assert layer.seed == expected_seed
        expected_pairs: list[tuple[int, int]] = []

        if layer.basis == "X" and layer.direction == "vertical":
            for check_2, bit_2 in layer.seed_edges:
                for bit_1 in range(code.h1.num_bits):
                    expected_pairs.append(
                        (
                            global_index[code.x_check_qubits[(bit_1, check_2)]],
                            global_index[code.vv_qubits[(bit_1, bit_2)]],
                        )
                    )
        elif layer.basis == "X" and layer.direction == "horizontal":
            for check_1, bit_1 in layer.seed_edges:
                for check_2 in range(code.h2.num_checks):
                    expected_pairs.append(
                        (
                            global_index[code.x_check_qubits[(bit_1, check_2)]],
                            global_index[code.cc_qubits[(check_1, check_2)]],
                        )
                    )
        elif layer.basis == "Z" and layer.direction == "vertical":
            for check_2, bit_2 in layer.seed_edges:
                for check_1 in range(code.h1.num_checks):
                    expected_pairs.append(
                        (
                            global_index[code.cc_qubits[(check_1, check_2)]],
                            global_index[code.z_check_qubits[(check_1, bit_2)]],
                        )
                    )
        else:
            for check_1, bit_1 in layer.seed_edges:
                for bit_2 in range(code.h2.num_bits):
                    expected_pairs.append(
                        (
                            global_index[code.vv_qubits[(bit_1, bit_2)]],
                            global_index[code.z_check_qubits[(check_1, bit_2)]],
                        )
                    )

        assert layer.cnot_pairs == tuple(sorted(expected_pairs))


def _logical_vectors(code: HGPCode) -> tuple[np.ndarray, np.ndarray]:
    data_order = code.data_index_order
    data_to_column = {index: column for column, index in enumerate(data_order)}
    logical_x = np.zeros((code.num_logicals, len(data_order)), dtype=np.uint8)
    logical_z = np.zeros_like(logical_x)

    for pair in code.logical_pairs:
        logical_id = pair["logical_id"]
        for index in pair["x"]["data_indices"]:
            logical_x[logical_id, data_to_column[index]] = 1
        for index in pair["z"]["data_indices"]:
            logical_z[logical_id, data_to_column[index]] = 1
    return logical_x, logical_z


def _assert_css_and_logical_invariants(code: HGPCode) -> None:
    hx, hz = code.get_data_parity_check_matrices()
    logical_x, logical_z = _logical_vectors(code)

    assert not np.any((hx @ hz.T) % 2)
    assert not np.any((hz @ logical_x.T) % 2)
    assert not np.any((hx @ logical_z.T) % 2)
    assert np.array_equal(
        (logical_x @ logical_z.T) % 2,
        np.eye(code.num_logicals, dtype=np.uint8),
    )
    assert code.num_logicals == code.num_data_qubits - _gf2_rank(hx) - _gf2_rank(hz)


def test_binary_parity_check_constructors_share_one_representation():
    open_repetition = hgp_13_1_3_seed().to_dense()
    dense = BinaryParityCheck.from_dense(open_repetition)
    supports = BinaryParityCheck.from_row_supports(
        (2, 3),
        [(0, 1), (1, 2)],
    )
    cycle = BinaryParityCheck.from_cyclic_polynomial([0, 1], size=3)

    assert dense.shape == (2, 3)
    assert dense.row_supports == supports.row_supports
    assert np.array_equal(dense.to_dense(), open_repetition)
    assert np.array_equal(
        cycle.to_dense(),
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ],
            dtype=np.uint8,
        ),
    )


def test_sparse_and_bivariate_parity_check_constructors():
    class SparseLike:
        shape = (2, 3)
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 2, 1, 1])
        data = np.array([1, 3, 1, 1])

        def tocoo(self, *, copy):
            assert copy
            return self

    sparse = BinaryParityCheck.from_scipy_sparse(SparseLike())
    bivariate = BinaryParityCheck.from_bivariate_polynomial(
        [(0, 0), (1, 0)],
        shape=(2, 3),
    )

    assert sparse.row_supports == ((0, 2), ())
    assert bivariate.shape == (6, 6)
    assert bivariate.row_supports == (
        (0, 3),
        (1, 4),
        (2, 5),
        (0, 3),
        (1, 4),
        (2, 5),
    )


def test_canonical_kernel_basis_returns_paired_pivot_vectors():
    canonical = canonical_kernel_basis(hgp_13_1_3_seed())

    assert canonical.rank == 2
    assert canonical.nullity == 1
    assert canonical.pivots == (2,)
    assert np.array_equal(canonical.basis, np.array([[1, 1, 1]], dtype=np.uint8))
    assert np.array_equal(
        (canonical.unit_complement @ canonical.basis.T) % 2,
        np.eye(1, dtype=np.uint8),
    )


def test_repetition_product_is_the_13_1_3_unrotated_surface_patch():
    seed = hgp_13_1_3_seed()
    hgp = hgp_13_1_3()
    surface = UnrotatedSurfaceCode(distance=3)

    assert seed.shape == (2, 3)
    assert seed.row_supports == ((0, 1), (1, 2))
    assert seed.source_metadata["code_family"] == "open_repetition"
    assert hgp.code_distance == 3
    assert hgp.num_data_qubits == 13
    assert hgp.num_qubits == 25
    assert len(hgp.syndrome_indices_x) == 6
    assert len(hgp.syndrome_indices_z) == 6
    assert hgp.num_logicals == 1
    assert [pair["sector"] for pair in hgp.logical_pairs] == ["bit_bit"]

    # The default HGP display separates the four product sectors.  Its Tanner
    # graph is nevertheless exactly the standard unrotated surface-code graph
    # under this independent semantic-index embedding.
    assert _stabilizer_signatures(
        hgp,
        _checkerboard_tanner_embedding(hgp),
    ) == _stabilizer_signatures(surface)
    _assert_hgp_parity_check_formula(hgp)
    _assert_product_block_layout(hgp)
    assert {len(op["data_indices"]) for op in hgp.logical_ops} == {3}
    _assert_css_and_logical_invariants(hgp)


def test_cycle_product_is_the_18_2_3_toric_patch():
    seed = hgp_18_2_3_seed()
    hgp = hgp_18_2_3()
    toric = ToricCode(distance=3)
    transpose_coords = lambda coord: (coord[1], coord[0])

    assert seed.shape == (3, 3)
    assert seed.row_supports == ((0, 1), (1, 2), (0, 2))
    assert seed.source_metadata["code_family"] == "cyclic_repetition"
    assert hgp.code_distance == 3
    assert hgp.num_data_qubits == 18
    assert hgp.num_qubits == 36
    assert len(hgp.syndrome_indices_x) == 9
    assert len(hgp.syndrome_indices_z) == 9
    assert hgp.num_logicals == 2
    assert [pair["sector"] for pair in hgp.logical_pairs] == [
        "bit_bit",
        "check_check",
    ]

    # ToricCode uses the transposed checkerboard orientation.  Compare Tanner
    # graphs through semantic indices instead of constraining the HGP display.
    assert _stabilizer_signatures(
        hgp,
        _checkerboard_tanner_embedding(hgp),
    ) == _stabilizer_signatures(toric, transform=transpose_coords)
    _assert_hgp_parity_check_formula(hgp)
    _assert_product_block_layout(hgp)
    assert {len(op["data_indices"]) for op in hgp.logical_ops} == {3}
    _assert_css_and_logical_invariants(hgp)


def test_distinct_rectangular_rank_deficient_seeds_are_preserved():
    h1 = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    h2 = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    original_h1 = h1.copy()
    original_h2 = h2.copy()

    code = HGPCode(h1, h2)

    # Canonical kernel analysis is separate from the physical Tanner graph:
    # neither caller-owned matrix nor the stored seed support is row-reduced.
    assert np.array_equal(h1, original_h1)
    assert np.array_equal(h2, original_h2)
    assert np.array_equal(code.h1.to_dense(), original_h1)
    assert np.array_equal(code.h2.to_dense(), original_h2)

    assert code.h1.shape == (3, 4)
    assert code.h2.shape == (2, 3)
    assert code.kernel_h1.rank == 2
    assert code.kernel_h2.rank == 1
    assert code.num_data_qubits == 18
    assert code.num_logicals == 5
    assert [pair["sector"] for pair in code.logical_pairs].count("bit_bit") == 4
    assert [pair["sector"] for pair in code.logical_pairs].count("check_check") == 1
    _assert_hgp_parity_check_formula(code)
    _assert_product_block_layout(code)
    _assert_css_and_logical_invariants(code)

    system = QECSystem()
    system.add_patch(code, name="asymmetric_hgp")
    block = HGPProductColorationExtractionBlock(system)
    assert block.depth_x == 6
    assert block.depth_z == 6


def test_fixed_3_4_biregular_seed_builds_225_9_4_hgp_instance():
    seed = hgp_225_9_4_seed()
    matrix = seed.to_dense()
    code = hgp_225_9_4()

    assert seed.shape == (9, 12)
    assert set(matrix.sum(axis=0)) == {3}
    assert set(matrix.sum(axis=1)) == {4}
    assert _gf2_rank(matrix) == 9
    assert _classical_distance(matrix) == 4

    assert code.num_data_qubits == 225
    assert code.num_qubits == 441
    assert code.num_logicals == 9
    assert len(code.syndrome_indices_x) == 108
    assert len(code.syndrome_indices_z) == 108
    _assert_css_and_logical_invariants(code)


def test_public_bipartite_coloring_is_optimal_on_biregular_seed():
    seed = hgp_225_9_4_seed()
    edges = [
        (check, bit)
        for check, row in enumerate(seed.row_supports)
        for bit in row
    ]
    colors = color_bipartite_edges(edges)

    assert len(colors) == 4
    assert sorted(edge for color in colors for edge in color) == sorted(edges)
    for color in colors:
        assert len({check for check, _ in color}) == len(color)
        assert len({bit for _, bit in color}) == len(color)


def test_public_bipartite_coloring_accepts_non_orderable_hashable_labels():
    left_a = object()
    left_b = object()
    right_a = object()
    right_b = object()
    edges = [(left_a, right_b), (left_b, right_a), (left_a, right_a)]

    colors = color_bipartite_edges(edges)

    assert len(colors) == 2
    assert {edge for color in colors for edge in color} == set(edges)


def test_hgp_product_coloration_has_stable_four_phase_16_layer_schedule():
    system = QECSystem()
    system.add_patch(hgp_225_9_4(), name="hgp225")
    block = HGPProductColorationExtractionBlock(system)

    assert block.depth_x == 8
    assert block.depth_z == 8
    assert block.cnot_depth == 16
    assert len(block.measurement_blocks) == 2
    assert [(layer.basis, layer.direction, layer.color) for layer in block.layers] == [
        *(('X', 'vertical', color) for color in range(4)),
        *(('X', 'horizontal', color) for color in range(4)),
        *(('Z', 'vertical', color) for color in range(4)),
        *(('Z', 'horizontal', color) for color in range(4)),
    ]
    assert [layer.seed for layer in block.layers] == [
        *("H2" for _ in range(4)),
        *("H1" for _ in range(4)),
        *("H2" for _ in range(4)),
        *("H1" for _ in range(4)),
    ]

    for layer in block.layers:
        qubits = [qubit for pair in layer.cnot_pairs for qubit in pair]
        assert len(qubits) == len(set(qubits))

    _assert_product_layer_semantics(block)
    assert sum(len(layer.cnot_pairs) for layer in block.x_layers) == 108 * 7
    assert sum(len(layer.cnot_pairs) for layer in block.z_layers) == 108 * 7
    assert block.measurement_blocks[0].num_measurements == 108
    assert block.measurement_blocks[1].num_measurements == 108


def test_product_coloration_ignores_registered_inactive_hgp_patches():
    system = QECSystem()
    system.add_patch(hgp_13_1_3(), name="active")
    system.add_patch(
        hgp_13_1_3(),
        offset=(20, 0),
        name="inactive",
        is_active=False,
    )

    block = HGPProductColorationExtractionBlock(system)

    assert block.patch_name == "active"
    assert block.cnot_depth == 8


def test_generic_css_coloration_remains_an_optional_hgp_fallback():
    system = QECSystem()
    system.add_patch(hgp_13_1_3(), name="surface_hgp")

    generic = GenericCSSColorationExtractionBlock(system)
    product = HGPProductColorationExtractionBlock(system)

    assert generic.cnot_depth == 8
    assert product.cnot_depth == 8
    generic_edges = {
        ("X", edge)
        for layer in generic.x_layers
        for edge in layer
    } | {
        ("Z", (data, syndrome))
        for layer in generic.z_layers
        for syndrome, data in layer
    }
    product_edges = {
        (layer.basis, pair)
        for layer in product.layers
        for pair in layer.cnot_pairs
    }
    assert generic_edges == product_edges


def test_hgp_225_product_coloration_memory_is_noiseless_in_both_bases():
    for basis in ("X", "Z"):
        code = hgp_225_9_4()
        circuit = MemoryExperiment(
            qec_patch=code,
            patch_name="hgp225",
            extraction_block_class=HGPProductColorationExtractionBlock,
            rounds=code.code_distance,
            noise_params=None,
            basis=basis,
        ).build()
        detectors, observables = circuit.compile_detector_sampler(seed=42).sample(
            shots=20,
            separate_observables=True,
        )

        assert circuit.num_qubits == 441
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 9
        assert not np.any(detectors)
        assert not np.any(observables)
