"""
Patch-level tests for the rotated XZZX surface code.

Covers the code-construction invariants that the protocol smoke tests in
test_protocols.py do not touch:
  1. build() is idempotent (QECPatch.__init__ already builds once).
  2. The checkerboard formula has a single source: the patch-local
     data_basis_map() and the system-global xzzx_memory_basis() must agree
     per coordinate.

Run:  pytest tests/test_xzzx_code.py -q
"""
import pytest
import numpy as np

from lightstim.qec_code.surface_code.xzzx import XZZXSurfaceCode, xzzx_memory_basis


def _gf2_rank(matrix) -> int:
    mat = (matrix.copy() % 2).astype(np.uint8)
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = next((row for row in range(rank, rows) if mat[row, col]), None)
        if pivot is None:
            continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _pauli_vec(code, pauli):
    data = sorted(code.data_indices)
    position = {idx: i for i, idx in enumerate(data)}
    vec = np.zeros(2 * len(data), dtype=np.uint8)
    for idx, p in pauli.items():
        i = position[idx]
        if p in ("X", "Y"):
            vec[i] = 1
        if p in ("Z", "Y"):
            vec[len(data) + i] = 1
    return vec


def _symplectic_product(a, b) -> int:
    n = len(a) // 2
    return int((np.dot(a[:n], b[n:]) + np.dot(a[n:], b[:n])) % 2)


def _in_rowspan(rows, vec) -> bool:
    return _gf2_rank(np.vstack([rows, vec])) == _gf2_rank(rows)


@pytest.mark.smoke
class TestXZZXPatch:

    def test_build_idempotent(self):
        """QECPatch.__init__ already calls build(); a second call must not duplicate state."""
        code = XZZXSurfaceCode(distance=3)
        n_stabs = len(code.stabilizers)
        n_logicals = len(code.logical_ops)
        code.build()
        assert len(code.stabilizers) == n_stabs, \
            f"stabilizers duplicated on rebuild: {n_stabs} -> {len(code.stabilizers)}"
        assert len(code.logical_ops) == n_logicals

    @pytest.mark.parametrize("distance_z,distance_x", [(3, 3), (5, 5), (5, 3), (3, 5)])
    def test_logicals_are_handwritten_alternating_boundary_strings(
        self, distance_z, distance_x
    ):
        """XZZX logicals are explicit X/Z-alternating boundary strings, not
        arbitrary nullspace representatives."""
        code = XZZXSurfaceCode(distance_z=distance_z, distance_x=distance_x)
        by_type = {op["type"]: op for op in code.logical_ops}

        expected_x = {
            (2 * k + 1, 1): ("X" if k % 2 == 0 else "Z")
            for k in range(distance_z)
        }
        expected_z = {
            (1, 2 * k + 1): ("Z" if k % 2 == 0 else "X")
            for k in range(distance_x)
        }

        actual_x = {code.qubit_coords[q]: p for q, p in by_type["X"]["pauli"].items()}
        actual_z = {code.qubit_coords[q]: p for q, p in by_type["Z"]["pauli"].items()}

        assert actual_x == expected_x
        assert actual_z == expected_z

        stabs = np.array(
            [_pauli_vec(code, st["pauli"]) for st in code.stabilizers],
            dtype=np.uint8,
        )
        logical_x = _pauli_vec(code, by_type["X"]["pauli"])
        logical_z = _pauli_vec(code, by_type["Z"]["pauli"])

        assert all(_symplectic_product(logical_x, st) == 0 for st in stabs)
        assert all(_symplectic_product(logical_z, st) == 0 for st in stabs)
        assert not _in_rowspan(stabs, logical_x)
        assert not _in_rowspan(stabs, logical_z)
        assert _symplectic_product(logical_x, logical_z) == 1

    @pytest.mark.parametrize("basis", ["Y", "W"])
    def test_memory_basis_helpers_reject_non_xz(self, basis):
        """The XZZX checkerboard is only defined for X/Z memory bases — both helpers
        must fail fast with ValueError, not crash with KeyError inside flip[]."""
        from lightstim.ir.qec_system import QECSystem
        system = QECSystem()
        system.add_patch(XZZXSurfaceCode(distance=3), name="xzzx_sc")
        patch, _ = system.patches["xzzx_sc"]
        with pytest.raises(ValueError, match="'X' or 'Z'"):
            xzzx_memory_basis(system, basis)
        with pytest.raises(ValueError, match="'X' or 'Z'"):
            patch.data_basis_map(basis)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_checkerboard_local_and_global_maps_agree(self, basis):
        """data_basis_map (local keys) and xzzx_memory_basis (global keys) must give
        the same basis for the same coordinate — i.e. one checkerboard formula."""
        from lightstim.ir.qec_system import QECSystem
        system = QECSystem()
        system.add_patch(XZZXSurfaceCode(distance=3), name="xzzx_sc")
        patch, _ = system.patches["xzzx_sc"]

        global_map = xzzx_memory_basis(system, basis)
        local_map = patch.data_basis_map(basis)

        assert len(global_map) == len(local_map)
        for coord in system.data_coords:
            g_idx = system.index_map[coord]
            l_idx = patch.index_map[coord]
            assert global_map[g_idx] == local_map[l_idx], \
                f"checkerboard mismatch at {coord}: global={global_map[g_idx]} local={local_map[l_idx]}"
