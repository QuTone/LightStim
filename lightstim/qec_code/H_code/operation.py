"""Logical operations for Jones's H-code family.

Transversal H exchanges the matching X/Z checks and each registered logical
pair (Jones, arXiv:1210.3388, Section II). The physical-H implementation is
adapted from Maggie Bao's H6 operation set in PR #98.
"""

from typing import Sequence

import stim

from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.qec_code._operation_utils import require_global_patch
from .code_patch import HCode


class HCodeLogicalOpSet(CSSLogicalOpSet):
    """Transversal H on all n-4 logical slots, plus inherited block CNOT.

    Use a global patch view at a full H-code boundary. A physical H layer
    acts on every logical slot together; it does not select one logical qubit.
    """

    def __init__(self):
        super().__init__()
        self.name = "HCode"

    def transversal_hadamard(self, builder, patch, noiseless: bool = False):
        """Apply H to every data qubit, exchanging X_L and Z_L in every slot."""
        if not isinstance(patch, HCode):
            raise ValueError("HCodeLogicalOpSet requires an HCode or HSixCode patch.")
        require_global_patch(builder, patch)
        for records, count in ((patch.stabilizers, 2),
                               (patch.logical_ops, patch.n - 4)):
            supports = {"X": [], "Z": []}
            for record in records:
                basis = record.get("type")
                support = frozenset(record["pauli"])
                if (basis not in supports or not support
                        or support != set(record["data_indices"])
                        or not support <= patch.data_indices
                        or set(record["pauli"].values()) != {basis}):
                    raise ValueError("H-code H requires pure X/Z checks and logicals on data.")
                supports[basis].append(support)
            if (len(supports["X"]) != count
                    or supports["X"] != supports["Z"]):
                raise ValueError("H-code H requires matching X/Z checks and logical pairs.")
        circuit = stim.Circuit()
        circuit.append("H", sorted(patch.data_indices))
        builder.apply_unitary_block(unitary_block=circuit, noiseless=noiseless)


class HSixLogicalOpSet(HCodeLogicalOpSet):
    """H6 operations, including the unflagged encoder of Dasu et al., Fig. 1(d).

    The encoder follows the gate order ported by Maggie Bao in PR #98's
    ``prep_circuits.get_dist_circ``. Its two input states are independently
    selectable here. It is distinct from the flagged |00> preparation in
    Fig. 5 and does not by itself provide fault-tolerant state preparation.
    """

    def __init__(self):
        super().__init__()
        self.name = "HSix"

    def encode(
        self,
        builder,
        patch,
        logical_bases: Sequence[str] = ("Z", "Z"),
        noiseless: bool = False,
    ):
        """Initialize and encode two +1 Pauli eigenstates in logical-slot order.

        ``logical_bases=("X", "Y")`` prepares |+>_L0 |+i>_L1. Each basis
        must be X, Y, or Z. Both HSixCode and HCode(n=6) are supported, using
        the global patch returned by system.add_patch(). The six data qubits
        must not already be active; other patches and syndrome ancillas are
        untouched. As with PQRM encoding, canonicalize the tracked stabilizers
        before subsequent syndrome extraction.

        Reference: https://arxiv.org/html/2506.14688v1, Fig. 1(d).
        """
        if not isinstance(patch, HCode) or patch.n != 6:
            raise ValueError("The Fig. 1(d) encoder requires HSixCode or HCode(n=6).")
        require_global_patch(builder, patch)
        bases = tuple(logical_bases)
        if len(bases) != 2 or any(b not in ("X", "Y", "Z") for b in bases):
            raise ValueError("logical_bases must contain two bases, each X, Y, or Z.")
        if patch.data_indices & builder.system.active_qubit_indices:
            raise ValueError("H6 encode requires uninitialized data qubits.")

        data = sorted(patch.data_indices)
        # Fig. 1(d): the two input wires are 0 and 1; the remaining inputs
        # are |+>, |0>, |+>, |0>. No flag ancillas are involved.
        builder.initialize(
            dict(zip(data, (*bases, "X", "Z", "X", "Z"))),
            n=builder.system.num_qubits,
            noiseless=noiseless,
        )
        for labels in ((2, 3, 4, 5), (2, 0, 3, 1),
                       (0, 4, 1, 5), (4, 2, 5, 3)):
            layer = stim.Circuit()
            layer.append("CX", [data[q] for q in labels])
            # One block per disjoint layer keeps ticks/noise between CNOTs
            # that share qubits, instead of merging the whole encoder.
            builder.apply_unitary_block(layer, noiseless=noiseless)

    def prepare_logical_zz(self, builder, patch, noiseless: bool = False):
        """Prepare |00>_L using the unflagged Fig. 1(d) encoder."""
        self.encode(builder, patch, ("Z", "Z"), noiseless=noiseless)

    def prepare_logical_xx(self, builder, patch, noiseless: bool = False):
        """Prepare |++>_L using the unflagged Fig. 1(d) encoder."""
        self.encode(builder, patch, ("X", "X"), noiseless=noiseless)

    def prepare_logical_yy(self, builder, patch, noiseless: bool = False):
        """Prepare |+i,+i>_L using the unflagged Fig. 1(d) encoder."""
        self.encode(builder, patch, ("Y", "Y"), noiseless=noiseless)

    # Preserve the whole-patch preparation names used by the CSS operation API.
    prepare_logical_z = prepare_logical_zz
    prepare_logical_x = prepare_logical_xx
    prepare_logical_y = prepare_logical_yy


__all__ = ["HCodeLogicalOpSet", "HSixLogicalOpSet"]
