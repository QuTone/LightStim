"""Logical operations for Jones's H-code family.

Transversal H exchanges the matching X/Z checks and each registered logical
pair (Jones, arXiv:1210.3388, Section II). The physical-H implementation is
adapted from Maggie Bao's H6 operation set in PR #98.
"""

from typing import Optional, Sequence

import stim

from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code._operation_utils import require_global_patch
from .code_patch import HCode


class FlagAncillaPatch(QECPatch):
    """Two bare ancilla qubits for a flagged syndrome-extraction gadget.

    Not a code: no stabilizers, no logical operators. Registered as its own
    ``QECSystem`` patch purely so the flag ancillas get real coordinates like
    every other qubit in the circuit. Used by ``HSixLogicalOpSet.encode``'s
    ``flagged=True`` path (Fig. 5 of arXiv:2506.14688); reusable by any other
    flagged gadget that just needs a pair of bare ancillas.
    """

    def _process_params(self):
        if self.params:
            raise ValueError(
                f"FlagAncillaPatch takes no parameters; got {sorted(self.params)}."
            )

    def build(self):
        self.add_qubit(0.0, 0.0, role="syndrome")
        self.add_qubit(1.0, 0.0, role="syndrome")


# Fig. 5's data-CX core (flag-ancilla entangle/disentangle CX excluded):
# prepares |00>_L from |000000>. Data labels 0 and 2 are the two controls
# the flag ancillas hook onto ("the encoder spine").
_FIG5_ENCODER_CORE = ((0, 1), (2, 3), (0, 4), (2, 5), (0, 5), (2, 4))
_FIG5_FLAG_CONTROLS = (0, 2)


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
        flagged: bool = False,
        flag_qubits: Optional[Sequence[int]] = None,
    ):
        """Initialize and encode two +1 Pauli eigenstates in logical-slot order.

        ``logical_bases=("X", "Y")`` prepares |+>_L0 |+i>_L1. Each basis
        must be X, Y, or Z. Both HSixCode and HCode(n=6) are supported, using
        the global patch returned by system.add_patch(). The six data qubits
        must not already be active; other patches and syndrome ancillas are
        untouched. As with PQRM encoding, canonicalize the tracked stabilizers
        before subsequent syndrome extraction.

        ``flagged=True`` uses the fault-tolerant, flag-verified Fig. 5
        encoder instead of the unflagged Fig. 1(d) one above: it only
        prepares ``|00>_L`` (``logical_bases`` must be ``("Z", "Z")``), and
        requires ``flag_qubits`` -- two global qubit indices from a patch
        such as :class:`FlagAncillaPatch`, registered on the system before
        the tracker is constructed. Both flag qubits are reset, entangled
        onto the encoder's two control qubits (data labels 0 and 2),
        disentangled after the data-CX core, then measured; the caller must
        post-select on both reading 0 (each gets its own
        ``"post-select"``-tagged ``DETECTOR``). Unlike Fig. 1(d), a single
        fault during preparation is *detected*, not merely left for a later
        round to catch.

        Reference: https://arxiv.org/html/2506.14688v1, Fig. 1(d) (unflagged,
        arbitrary-basis) and Fig. 5 (flagged, |00>_L only).
        """
        if not isinstance(patch, HCode) or patch.n != 6:
            raise ValueError("The Fig. 1(d) encoder requires HSixCode or HCode(n=6).")
        require_global_patch(builder, patch)
        bases = tuple(logical_bases)
        if len(bases) != 2 or any(b not in ("X", "Y", "Z") for b in bases):
            raise ValueError("logical_bases must contain two bases, each X, Y, or Z.")
        if patch.data_indices & builder.system.active_qubit_indices:
            raise ValueError("H6 encode requires uninitialized data qubits.")
        if flagged:
            if bases != ("Z", "Z"):
                raise ValueError(
                    "The Fig. 5 flagged encoder only prepares |00>_L "
                    "(logical_bases must be ('Z', 'Z'))."
                )
            flag_qubits = tuple(flag_qubits) if flag_qubits is not None else ()
            if len(flag_qubits) != 2:
                raise ValueError(
                    "flagged=True requires flag_qubits to be exactly 2 global qubit indices."
                )

        data = sorted(patch.data_indices)

        if flagged:
            self._encode_fig5_flagged(builder, data, flag_qubits, noiseless=noiseless)
            return

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

    def _encode_fig5_flagged(self, builder, data, flag_qubits, noiseless: bool = False):
        """Fig. 5's flag-verified ``|00>_L`` encoder; see ``encode``'s
        ``flagged`` kwarg for the public contract."""
        system = builder.system
        builder.initialize(
            {q: "Z" for q in data}, n=system.num_qubits, noiseless=noiseless
        )

        flags = list(flag_qubits)
        tag = "noiseless" if noiseless else ""
        builder.circuit.append("R", flags, tag=tag)

        def _gate(name, qubits):
            # Each physical gate gets its own apply_unitary_block call (as
            # opposed to bundling several into one stim.Circuit) so stim
            # cannot auto-merge adjacent same-name instructions into one
            # multi-pair gate -- which would place a single noise channel
            # over several physical gates once noise is injected.
            block = stim.Circuit()
            block.append(name, qubits)
            builder.apply_unitary_block(block, noiseless=noiseless)

        c0, c1 = (data[i] for i in _FIG5_FLAG_CONTROLS)
        a0, a1 = flags
        _gate("H", [c0, c1])
        _gate("CX", [c0, a0])
        _gate("CX", [c1, a1])
        for i, j in _FIG5_ENCODER_CORE:
            _gate("CX", [data[i], data[j]])
        _gate("CX", [c0, a0])
        _gate("CX", [c1, a1])

        if builder.circuit[-1].name != "TICK":
            builder.circuit.append("TICK")
        flag_meas_base = builder.tracker.total_measurements
        builder.circuit.append("M", flags, tag=tag)
        builder.tracker.total_measurements += len(flags)
        # The entangle/disentangle CX brackets above give the tracked H6
        # stabilizer/logical rows a phantom dependency on the flag qubits
        # (they're conjugated through apply_unitary_block like any other
        # tracked qubit); fold that dependency into each row's own record
        # history before it's measured out, exactly like the Bell-pair and
        # Y2 flag gadgets in lightstim.protocols.h6_distillation do for the
        # identical reason.
        builder.tracker._replace_measured_ancillas_with_records(
            measurement_qubit_indices=flags,
            measurement_bases=["Z", "Z"],
            measurement_base_idx=flag_meas_base,
        )
        for i, a in enumerate(flags):
            coord = tuple(system.qubit_coords[a]) + (0.0,)
            builder.circuit.append(
                "DETECTOR", [stim.target_rec(-len(flags) + i)], list(coord),
                tag="post-select",
            )

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
