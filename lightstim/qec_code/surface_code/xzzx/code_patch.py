from typing import Dict, List

import numpy as np

from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode


def _gf2_rref(mat: np.ndarray):
    """Reduced row-echelon form over GF(2). Returns (rref, pivot_cols)."""
    M = (mat.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    pivots: List[int] = []
    r = 0
    for c in range(cols):
        piv = None
        for rr in range(r, rows):
            if M[rr, c]:
                piv = rr
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for rr in range(rows):
            if rr != r and M[rr, c]:
                M[rr] ^= M[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return M, pivots


def _gf2_nullspace(mat: np.ndarray) -> np.ndarray:
    """Basis (as rows) of the null space {v : mat @ v = 0} over GF(2)."""
    M = (mat.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    rref, pivots = _gf2_rref(M)
    pivot_set = set(pivots)
    free = [c for c in range(cols) if c not in pivot_set]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for ri, pc in enumerate(pivots):
            if rref[ri, f]:
                v[pc] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), dtype=np.uint8)


def _in_rowspan(rowspan: np.ndarray, vec: np.ndarray) -> bool:
    """True if vec is in the GF(2) row span of `rowspan`."""
    if rowspan.shape[0] == 0:
        return not vec.any()
    stacked = np.vstack([rowspan, vec.reshape(1, -1)])
    r_before = len(_gf2_rref(rowspan)[1])
    r_after = len(_gf2_rref(stacked)[1])
    return r_after == r_before


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

        # Logicals: derive a valid anticommuting pair from the deformed
        # stabilizers (the centralizer modulo the stabilizer group).  The
        # CSS-style line operators are NOT valid for the diagonal-rule XZZX code,
        # so we recompute them numerically.  These are used only for distance /
        # tooling — the memory experiment's observable is auto-discovered by the
        # tableau tracker.
        self.logical_ops = self._compute_xzzx_logicals()

    def _compute_xzzx_logicals(self) -> list:
        data = sorted(self.data_indices)
        nd = len(data)
        pos = {idx: i for i, idx in enumerate(data)}

        # Stabilizer symplectic rows over data qubits: [X-part | Z-part].
        S = np.zeros((len(self.stabilizers), 2 * nd), dtype=np.uint8)
        for r, st in enumerate(self.stabilizers):
            for idx, p in st["pauli"].items():
                j = pos[idx]
                if p in ("X", "Y"):
                    S[r, j] = 1
                if p in ("Z", "Y"):
                    S[r, nd + j] = 1

        # Centralizer: v=[ax|az] commutes with stab (sx|sz) iff sx·az + sz·ax = 0.
        # That is M @ v = 0 with M row = [sz | sx] (X/Z halves swapped).
        M = np.hstack([S[:, nd:], S[:, :nd]])
        null = _gf2_nullspace(M)

        # Logical reps = null-space vectors independent of the stabilizer span.
        logicals = []
        span = S.copy()
        for v in null:
            if not _in_rowspan(span, v):
                logicals.append(v)
                span = np.vstack([span, v.reshape(1, -1)])
            if len(logicals) == 2:
                break

        def _to_pauli(vec):
            ps = {}
            for i, idx in enumerate(data):
                x, z = int(vec[i]), int(vec[nd + i])
                if x and z:
                    ps[idx] = "Y"
                elif x:
                    ps[idx] = "X"
                elif z:
                    ps[idx] = "Z"
            return ps

        records = []
        for li, v in enumerate(logicals):
            ps = _to_pauli(v)
            # Label by dominant Pauli content (purely cosmetic).
            n_x = sum(1 for p in ps.values() if p == "X")
            n_z = sum(1 for p in ps.values() if p == "Z")
            op_type = "X" if n_x >= n_z else "Z"
            records.append({
                "pauli": ps,
                "type": op_type,
                "data_indices": list(ps.keys()),
            })
        # Ensure the two carry distinct type labels for downstream lookups.
        if len(records) == 2 and records[0]["type"] == records[1]["type"]:
            records[1]["type"] = "Z" if records[0]["type"] == "X" else "X"
        return records

    def data_basis_map(self, memory_basis: str) -> Dict[int, str]:
        """Local data-index -> init/measure basis for an XZZX memory experiment.

        colour-0 data qubits use ``memory_basis``; colour-1 use the opposite
        basis.  With this checkerboard, exactly one syndrome sublattice has
        deterministic first-round outcomes (handled automatically by the
        tableau tracker).
        """
        flip = {"Z": "X", "X": "Z"}
        mb = memory_basis.upper()
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
    out: Dict[int, str] = {}
    for coord in system.data_coords:
        gidx = system.index_map[coord]
        x, y = coord
        c = (round((x - 1) / 2) + round((y - 1) / 2)) % 2
        out[gidx] = mb if c == 0 else flip[mb]
    return out
