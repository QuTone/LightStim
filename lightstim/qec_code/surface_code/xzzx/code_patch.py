from typing import Dict

from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode


class XZZXSurfaceCode(RotatedSurfaceCode):
    """Rotated XZZX surface code (Clifford-deformed CSS rotated surface code).

    Geometry, syndrome roles, and boundary structure are identical to
    :class:`RotatedSurfaceCode`.  The difference is purely the per-data-qubit
    Pauli assigned to each stabilizer: instead of an all-``X`` or all-``Z``
    check, every check becomes ``XZZX`` via the *diagonal rule* relative to its
    syndrome coordinate::

        SW / NE neighbour  (dx * dy > 0)  ->  'X'
        NW / SE neighbour  (dx * dy < 0)  ->  'Z'

    This reproduces the canonical XZZX surface code (Bonilla-Ataides et al.) and
    matches the reference circuit of Etxezarreta Martinez et al.,
    *Phys. Rev. Applied* **25**, 014021 (2026).

    The ``type`` field of each stabilizer is preserved — it labels the syndrome
    *sublattice* (used by :class:`XZZXSurfaceCodeExtractionBlock` to pick the
    per-tick interaction offsets), not the Pauli content.

    Memory experiments on this code require a checkerboard of initialization /
    measurement bases (see :meth:`data_basis_map` and :func:`xzzx_memory_basis`)
    because every stabilizer mixes ``X`` and ``Z``; a uniform product state
    makes no stabilizer deterministic.
    """

    def build(self):
        # Build the CSS rotated surface code (geometry, uniform stabilizers,
        # logicals, optional shift), then deform it in place.
        # QECPatch.__init__ already runs build() once; super().build() is additive,
        # so a second call must be a no-op or the stabilizer set duplicates.
        if self.stabilizers:
            return
        super().build()
        self._deform_to_xzzx()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _data_color(coord) -> int:
        """2-colouring of the data sublattice (checkerboard).

        The two diagonals of every plaquette carry opposite colours, so this is
        exactly the X/Z split needed for a deterministic XZZX memory state.
        """
        x, y = coord
        return (round((x - 1) / 2) + round((y - 1) / 2)) % 2

    def _diagonal_pauli(self, syn_coord, data_idx) -> str:
        dx = self.qubit_coords[data_idx][0] - syn_coord[0]
        dy = self.qubit_coords[data_idx][1] - syn_coord[1]
        return "X" if (dx * dy) > 0 else "Z"

    def _deform_to_xzzx(self):
        # Stabilizers: reassign Pauli per neighbour by the diagonal rule.
        for st in self.stabilizers:
            syn = st["syn_coord"]
            st["pauli"] = {idx: self._diagonal_pauli(syn, idx) for idx in st["pauli"]}

        self._define_xzzx_logicals()

    def _define_xzzx_logicals(self):
        """Hand-written XZZX logical operators.

        Use boundary strings matching the XZZX convention:

        * logical X: top horizontal X/Z-alternating string
        * logical Z: left vertical Z/X-alternating string

        These commute with every XZZX stabilizer and anticommute with each other
        at the top-left data qubit.  The logical label is the encoded operator
        label; the physical Pauli string itself is mixed.
        """
        self.logical_ops = []
        min_x = min(x for x, _ in self.data_coords)
        min_y = min(y for _, y in self.data_coords)

        lx_targets = {
            self.snap_coord((min_x + 2 * k, min_y)): ("X" if k % 2 == 0 else "Z")
            for k in range(self.distance_z)
        }
        self.create_stim_logical(lx_targets, "X")

        lz_targets = {
            self.snap_coord((min_x, min_y + 2 * k)): ("Z" if k % 2 == 0 else "X")
            for k in range(self.distance_x)
        }
        self.create_stim_logical(lz_targets, "Z")

    def data_basis_map(self, memory_basis: str) -> Dict[int, str]:
        """Local data-index -> init/measure basis for an XZZX memory experiment.

        colour-0 data qubits use ``memory_basis``; colour-1 use the opposite
        basis.  With this checkerboard, exactly one syndrome sublattice has
        deterministic first-round outcomes (handled automatically by the
        tableau tracker).
        """
        flip = {"Z": "X", "X": "Z"}
        mb = memory_basis.upper()
        if mb not in flip:
            raise ValueError(f"XZZX memory basis must be 'X' or 'Z'; got {memory_basis!r}")
        return {
            idx: (mb if self._data_color(self.qubit_coords[idx]) == 0 else flip[mb])
            for idx in self.data_indices
        }


def xzzx_memory_basis(system, memory_basis: str) -> Dict[int, str]:
    """Global data-index -> basis map for a system whose patch(es) are XZZX.

    Mirrors :meth:`XZZXSurfaceCode.data_basis_map` but keyed by *global* indices
    (after ``system.add_patch`` remapping), computed straight from coordinates so
    it is robust to index relabeling.
    """
    flip = {"Z": "X", "X": "Z"}
    mb = memory_basis.upper()
    if mb not in flip:
        raise ValueError(f"XZZX memory basis must be 'X' or 'Z'; got {memory_basis!r}")
    out: Dict[int, str] = {}
    for coord in system.data_coords:
        gidx = system.index_map[coord]
        c = XZZXSurfaceCode._data_color(coord)
        out[gidx] = mb if c == 0 else flip[mb]
    return out
