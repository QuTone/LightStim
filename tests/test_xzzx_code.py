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

from lightstim.qec_code.surface_code.xzzx import XZZXSurfaceCode, xzzx_memory_basis


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
