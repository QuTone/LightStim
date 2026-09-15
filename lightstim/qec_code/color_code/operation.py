"""Transversal Clifford gates for ColorCode.

Adapted from Maggie Bao's PR #98. Phase gates extend the original d=3-only
uniform implementation to the registered color-code checks at larger distances.
The S/S_DAG pattern satisfies the signed-support conditions of Kubica and
Beverland, arXiv:1410.0069, Sec. II.3 / Eqs. (8)-(12), solved algebraically
instead of depending on a particular geometric embedding or vertex numbering.
"""

import numpy as np
import stim

from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.utils.linear_algebra import row_echelon
from .code_patch import ColorCode


class ColorCodeLogicalOpSet(CSSLogicalOpSet):
    """H, S, S_DAG, plus inherited logical X/Z and inter-patch CNOT.

    Methods act on global patch views returned by system.add_patch(). They
    preserve the registered logical convention, including the sign of logical
    Y. They operate at a full code boundary; inserting them inside a middle-out
    extraction cycle requires a separately validated spacetime construction.
    """

    def __init__(self):
        super().__init__()
        self.name = "ColorCode"

    def _supports(self, builder, patch):
        self._require_global_patch(builder, patch)
        if not isinstance(patch, ColorCode) or patch.num_logicals != 1:
            raise ValueError("ColorCodeLogicalOpSet requires a one-logical-qubit ColorCode patch.")
        by_basis = {}
        for basis in ("X", "Z"):
            supports = []
            for check in patch.stabilizers:
                if check.get("type") != basis:
                    continue
                support = frozenset(check["pauli"])
                if (not support or not support <= patch.data_indices
                        or set(check["pauli"].values()) != {basis} or len(support) % 2):
                    raise ValueError("Expected even-weight, pure X/Z color-code checks.")
                supports.append(support)
            by_basis[basis] = set(supports)
        if not by_basis["X"] or by_basis["X"] != by_basis["Z"]:
            raise ValueError("Color-code transversal gates require matching X/Z check supports.")
        xl = self._logical_support(patch, "X", 0)
        zl = self._logical_support(patch, "Z", 0)
        if xl != zl or len(xl) % 2 != 1:
            raise ValueError("Color-code transversal gates require matching odd-weight X/Z logicals.")
        return sorted(by_basis["X"], key=lambda s: tuple(sorted(s))), xl

    def transversal_hadamard(self, builder, patch, noiseless: bool = False):
        """Logical H via physical H on every data qubit."""
        self._supports(builder, patch)
        circuit = stim.Circuit()
        circuit.append("H", sorted(patch.data_indices))
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)

    def _s_pattern(self, builder, patch):
        checks, logical = self._supports(builder, patch)
        data = sorted(patch.data_indices)
        # Preserve the original uniform implementation when all checks are
        # doubly even (e.g. d=3). Odd logical weight fixes S versus S_DAG.
        if all(len(s) % 4 == 0 for s in checks):
            gate = "S" if len(logical) % 4 == 1 else "S_DAG"
            return {gate: data}

        # r_q=1 chooses S_DAG, r_q=0 chooses S. For each check c,
        # sum_q(1-2*r_q) over c must be 0 mod 4; on logical X it must be
        # 1 mod 4, so X_L -> i X_L Z_L = Y_L, not -Y_L.
        column = {q: i for i, q in enumerate(data)}
        supports = [*checks, logical]
        augmented = np.zeros((len(supports), len(data) + 1), dtype=np.uint8)
        for row, support in enumerate(supports):
            augmented[row, [column[q] for q in support]] = 1
            augmented[row, -1] = (len(support) // 2) % 2
        reduced, _, _, pivots = row_echelon(augmented, reduced=True)
        if len(data) in pivots:
            raise ValueError("Color-code checks/logicals admit no consistent transversal S pattern.")
        dagger = np.zeros(len(data), dtype=np.uint8)
        # Deterministic solution: all free variables zero.
        for row, pivot in enumerate(pivots):
            dagger[pivot] = reduced[row, -1]
        return {
            "S": [q for i, q in enumerate(data) if not dagger[i]],
            "S_DAG": [q for i, q in enumerate(data) if dagger[i]],
        }

    def transversal_s(self, builder, patch, noiseless: bool = False):
        """Logical S, using a validated physical S/S_DAG pattern."""
        self._apply_s(builder, patch, inverse=False, noiseless=noiseless)

    def transversal_s_dag(self, builder, patch, noiseless: bool = False):
        """Logical S_DAG: invert every gate in the logical-S pattern."""
        self._apply_s(builder, patch, inverse=True, noiseless=noiseless)

    def _apply_s(self, builder, patch, *, inverse, noiseless):
        circuit = stim.Circuit()
        for gate, data in self._s_pattern(builder, patch).items():
            if data:
                if inverse:
                    gate = "S_DAG" if gate == "S" else "S"
                circuit.append(gate, data)
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)


__all__ = ["ColorCodeLogicalOpSet"]
