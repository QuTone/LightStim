"""Fault-tolerant logical operations on hypergraph-product (HGP) patches.

This module follows Xu et al., "Fast and Parallelizable Logical Computation
with Homological Product Codes" (arXiv:2407.18490), whose Table I lists the
HGP gadgets used for logical computation.  Each gadget is transcribed here
with its paper rule; deviations from the paper are stated explicitly.

What these gates are, and what they are not
-------------------------------------------
* ``fold_transversal_h_swap`` is the H-SWAP gate.  It is not a logical
  Hadamard: its logical action is H̄ on every logical qubit together with
  the transposition Q̄_{i,j} <-> Q̄_{j,i} of the logical grid.  Only the
  diagonal logical qubits (i = j) receive a plain H̄; every other logical
  qubit receives H̄ and is moved to its mirror position.

* ``fold_transversal_cz_s`` is the CZ-S gate.  It is not a logical S gate
  (and not a logical CZ either): its logical action is S̄ on the diagonal
  logical qubits Q̄_{i,i} together with CZ̄ on every mirror pair
  (Q̄_{i,j}, Q̄_{j,i}).  On a code whose seed has redundant checks, the
  C1xC2 diagonal logical qubits receive S̄† instead.

References
    [1] N. P. Breuckmann and S. Burton, Fold-transversal Clifford gates
        for quantum codes, Quantum 8, 1372 (2024).
    [2] A. O. Quintavalle, P. Webster, and M. Vasmer, Partitioning qubits
        in hypergraph product codes to implement logical gates, Quantum 7,
        1153 (2023).
    [3] Q. Xu, H. Zhou, G. Zheng, D. Bluvstein, J. P. Bonilla Ataides,
        M. D. Lukin, and L. Jiang, Fast and parallelizable logical
        computation with homological product codes, Phys. Rev. X 15,
        021065 (2025).

Fold-transversal H-SWAP (Table I; introduced in Ref. [27] of the paper,
Quintavalle-Webster-Vasmer, under the fold-transversal framework of Ref. [26],
Breuckmann-Burton)
--------------------------------------------------------------------------
Paper rule.  For a *symmetric* HGP code, i.e. one whose two base codes are
the same matrix (H1 = H2), with physical qubits ``Q_{i,j}`` at coordinates
``O = [n1] x [n2]  ∪  {n1+1..2n1-k1} x {n2+1..2n2-k2}`` and ``O+`` the upper
blocks ``{(i,j) ∈ O | j > i}``:

    physical:  (H^{⊗n}) · ⊗_{(i,j)∈O+} SWAP(Q_{i,j}, Q_{j,i})
    logical:   (H̄^{⊗k}) · ⊗_{(i,j)∈Ō+} SWAP(Q̄_{i,j}, Q̄_{j,i})

Time cost O(1); no ancilla.  The gate is a code automorphism composed with a
transversal single-qubit layer, so a single physical fault stays a single
physical error (up to the two qubits of one SWAP, see ``noisy_swap``).

Transcription into LightStim's HGPCode.  ``HGPCode`` stores the two data
sectors as ``vv_qubits[(bit_1, bit_2)]`` (V1 x V2) and
``cc_qubits[(check_1, check_2)]`` (C1 x C2).  The fold is the transposition
``(bit_1, bit_2) ↔ (bit_2, bit_1)`` on V1 x V2 and
``(check_1, check_2) ↔ (check_2, check_1)`` on C1 x C2; both sectors are
square exactly when H1 = H2.  Under H^{⊗n} followed by this transposition,

    X check (bit_1, check_2)  ↦  Z check (check_1 = check_2, bit_2 = bit_1)
    Z check (check_1, bit_2)  ↦  X check (bit_1 = bit_2, check_2 = check_1)

so the stabilizer generators are permuted among themselves (sign +1), and in
the canonical logical basis of ``HGPCode._build_logical_operators`` (X̄ =
kernel(H1) x pivot(H2), Z̄ = pivot(H1) x kernel(H2) on V1 x V2; the dual
pattern on C1 x C2)

    X̄ of pair (l1, l2)  ↦  +Z̄ of pair (l2, l1)
    Z̄ of pair (l1, l2)  ↦  +X̄ of pair (l2, l1)

within the same sector, i.e. exactly H̄ on every logical qubit composed with
the swap of the logical pairs ``(l1, l2) ↔ (l2, l1)``.  Pairs with
``l1 == l2`` (the logical diagonal) receive a plain H̄.  These identities hold
exactly, not only modulo stabilizers, because the supports map onto each
other qubit by qubit; the tests check this with stim tableaux.

Beyond the paper.  The paper assumes full-rank base codes, in which case all
logical qubits live on V1 x V2.  ``HGPCode`` also registers logical pairs on
C1 x C2 when the seed has redundant checks (e.g. the [[18,2,3]] toric
instance); the same derivation applies to that sector, and the tests cover
it.  This is a statement about the code, not a change to the gadget.

Deviations / choices.
* The SWAP layer is emitted as physical ``SWAP`` gates, the same way
  ``UnrotatedSurfaceCodeLogicalOpSet.fold_transversal_hadamard`` does.  With
  ``noisy_swap=True`` (default) they take the two-qubit gate noise of the
  noise model, so a single fault can produce a weight-two error on one mirror
  pair.  ``noisy_swap=False`` tags the SWAP layer noiseless, modelling the
  paper's picture of the fold as a relabelling of qubits (Sec. VII: qubit
  movement) with no gate error.
* The H layer is emitted before the SWAP layer.  The two commute (H^{⊗n} is
  invariant under any qubit permutation), so the operator equals the paper's
  product in either order.
* No Pauli frame is involved: the logical action is exactly H̄^{⊗k} · SWAP,
  with no residual logical Pauli.

Fold-transversal CZ-S (Table I; same origin as the H-SWAP)
--------------------------------------------------------------------------
Paper rule.  For a symmetric HGP code (H1 = H2), with ``O+`` as above and
``n1 + 1 → 2n1 − k1`` the C1 x C2 diagonal:

    physical:  (⊗_{i∈[n1]} S(Q_{i,i})) · (⊗_{i∈{n1+1→2n1−k1}} S†(Q_{i,i}))
               · ⊗_{(i,j)∈O+} CZ(Q_{i,j}, Q_{j,i})
    logical:   (⊗_{i∈[k1]} S̄(Q̄_{i,i})) · ⊗_{(i,j)∈Ō+} CZ̄(Q̄_{i,j}, Q̄_{j,i})

Time cost O(1); no ancilla.  Every gate acts on the diagonal or on one
mirror pair, so a single fault stays on at most one mirror pair.

Transcription.  S on the V1 x V2 diagonal ``(bit, bit)``, S† on the C1 x C2
diagonal ``(check, check)``, CZ on every mirror pair of both sectors (the
pairs of :func:`fold_mirror_pairs`).  Under this layer

    X on (r, c)  ↦  X on (r, c) · Z on (c, r)      (phase i on the V1xV2
                                                   diagonal, −i on the C1xC2
                                                   diagonal),
    Z            ↦  Z,

so an X check ``(bit_1, check_2)`` maps to itself times the Z check
``(check_2, bit_1)``; its support meets the two diagonals on the same
condition ``H[check_2, bit_1] = 1``, and the phases ``i · (−i) = 1`` cancel.
That cancellation is why the paper puts S† on the C1 x C2 diagonal.  Z
checks are fixed.  In the canonical basis, for V1 x V2 logical pairs:

    X̄ of pair (l1, l2), l1 ≠ l2  ↦  X̄(l1, l2) · Z̄(l2, l1)   exactly (CZ̄ with
                                                               the mirror pair)
    X̄ of pair (l, l)             ↦  +Ȳ(l, l)                exactly (S̄)
    Z̄                            ↦  Z̄

The kernel vector of ``l1`` has a 1 on its own pivot only, so a logical
support meets the diagonal exactly when ``l1 == l2``; the tests check
these identities with stim tableaux.

Beyond the paper.  C1 x C2 logical pairs (rank-deficient seeds) see the
S† diagonal instead, so their diagonal pairs receive S̄† (image −Ȳ) while
their mirror pairs still receive CZ̄.  ``fold_transversal_cz_s_dag`` is
the inverse layer (S†/S exchanged, CZ unchanged) and is the S̄† gate on
V1 x V2 diagonal pairs.

Deviations / choices.  The CZ layer is emitted as physical ``CZ`` gates,
which take the two-qubit gate noise of the noise model; unlike the SWAP
of the H-SWAP there is no relabelling reading of a CZ, so no noiseless
option is offered beyond ``noiseless``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code._operation_utils import require_global_patch

from .code_patch import HGPCode


IndexPair = Tuple[int, int]


# ---------------------------------------------------------------------------
# Geometry of the fold
# ---------------------------------------------------------------------------

def is_symmetric_hgp(patch: HGPCode) -> bool:
    """True if the two base codes of ``patch`` are the same matrix (H1 = H2).

    This is matrix equality, not code equivalence: a row-permuted copy of H1
    defines the same classical code but is rejected, because the fold pairs
    ``(check_1, check_2) ↔ (check_2, check_1)`` on C1 x C2 and the mapping of
    X checks onto Z checks only hold entry by entry.
    """
    dense_1 = patch.h1.to_dense()
    dense_2 = patch.h2.to_dense()
    return dense_1.shape == dense_2.shape and bool(np.array_equal(dense_1, dense_2))


def _require_symmetric(patch: HGPCode) -> None:
    if not is_symmetric_hgp(patch):
        raise ValueError(
            "The fold-transversal gates (H-SWAP, CZ-S) need a symmetric HGP code (H1 = H2, "
            "Xu et al. Table I); this patch has "
            f"H1 of shape {patch.h1.shape} and H2 of shape {patch.h2.shape}"
            + ("" if patch.h1.shape != patch.h2.shape else " with different entries")
            + "."
        )


def _global_index(builder: Optional[CircuitBuilder], patch: HGPCode, local_index: int) -> int:
    """Map a local data index of ``patch`` to its index in the builder's system.

    ``QECSystem.add_patch`` deep-copies the patch and shifts its coordinates,
    so the returned patch's ``qubit_coords`` are global coordinates while its
    ``vv_qubits``/``cc_qubits`` values stay local.  The system's ``index_map``
    turns a global coordinate into the global qubit index.  With
    ``builder=None`` the local index is returned unchanged.
    """
    if builder is None:
        return int(local_index)
    return int(builder.system.index_map[patch.qubit_coords[local_index]])


def _check_patch(patch: HGPCode, builder: Optional[CircuitBuilder]) -> None:
    """Shared validation of the public helpers and the op-set method."""
    if not isinstance(patch, HGPCode):
        raise TypeError(f"Expected an HGPCode patch, got {type(patch).__name__}.")
    if builder is not None:
        # A local (unregistered) patch carries local coordinates; looking them
        # up in the system's index_map would silently return whatever patch
        # happens to cover those coordinates.
        require_global_patch(builder, patch)
    _require_symmetric(patch)


def fold_mirror_pairs(patch: HGPCode, builder: Optional[CircuitBuilder] = None) -> List[IndexPair]:
    """Return the SWAP pairs of the fold, ``(Q_{i,j}, Q_{j,i})`` for ``j > i``.

    V1 x V2 pairs come first (ordered by ``(bit_1, bit_2)``), then C1 x C2
    pairs (ordered by ``(check_1, check_2)``).  Diagonal qubits are not
    listed.  Indices are global when ``builder`` is given (``patch`` must then
    be the global patch returned by ``QECSystem.add_patch``), local otherwise.
    """
    _check_patch(patch, builder)
    pairs: List[IndexPair] = []
    for sector in (patch.vv_qubits, patch.cc_qubits):
        for (a, b), local_index in sorted(sector.items()):
            if b > a:
                mirror = sector[(b, a)]
                pairs.append(
                    (_global_index(builder, patch, local_index), _global_index(builder, patch, mirror))
                )
    return pairs


def fold_diagonal_qubits(patch: HGPCode, builder: Optional[CircuitBuilder] = None) -> List[int]:
    """Return the data qubits fixed by the fold, ``Q_{i,i}`` of both sectors.

    Indices are global when ``builder`` is given (global patch required),
    local otherwise.
    """
    _check_patch(patch, builder)
    diagonal: List[int] = []
    for sector in (patch.vv_qubits, patch.cc_qubits):
        for (a, b), local_index in sorted(sector.items()):
            if a == b:
                diagonal.append(_global_index(builder, patch, local_index))
    return diagonal


def fold_diagonal_qubits_by_sector(
    patch: HGPCode, builder: Optional[CircuitBuilder] = None
) -> Tuple[List[int], List[int]]:
    """Return ``(V1xV2 diagonal, C1xC2 diagonal)`` data qubits (global with a builder)."""
    _check_patch(patch, builder)
    vv = [_global_index(builder, patch, q) for (a, b), q in sorted(patch.vv_qubits.items()) if a == b]
    cc = [_global_index(builder, patch, q) for (a, b), q in sorted(patch.cc_qubits.items()) if a == b]
    return vv, cc


def fold_logical_permutation(patch: HGPCode) -> Dict[int, int]:
    """Return ``{logical_id: mirror logical_id}`` induced by the fold.

    The canonical pair with ``seed_indices=(l1, l2)`` in a sector is mapped
    to the pair ``(l2, l1)`` of the same sector; diagonal pairs map to
    themselves.  Read from ``patch.logical_pairs`` (``HGPCode`` metadata).
    """
    _check_patch(patch, None)
    by_key = {
        (record["sector"], tuple(record["seed_indices"])): int(record["logical_id"])
        for record in patch.logical_pairs
    }
    permutation: Dict[int, int] = {}
    for record in patch.logical_pairs:
        l1, l2 = record["seed_indices"]
        key = (record["sector"], (l2, l1))
        if key not in by_key:
            raise RuntimeError(
                f"Logical pair {record['logical_id']} ({record['sector']}, seed {(l1, l2)}) "
                f"has no mirror pair {(l2, l1)}; the canonical bases of H1 and H2 differ."
            )
        permutation[int(record["logical_id"])] = by_key[key]
    return permutation


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------

def fold_cz_s_circuit(
    vv_diagonal: List[int], cc_diagonal: List[int], mirror_pairs: List[IndexPair], dagger: bool = False
) -> stim.Circuit:
    """Physical CZ-S: S on the V1xV2 diagonal, S† on the C1xC2 diagonal, TICK, CZ on mirror pairs.

    ``dagger=True`` exchanges S and S† (the inverse layer, CZ unchanged).
    """
    circuit = stim.Circuit()
    first, second = ("S_DAG", "S") if dagger else ("S", "S_DAG")
    if vv_diagonal:
        circuit.append(first, sorted(int(q) for q in vv_diagonal))
    if cc_diagonal:
        circuit.append(second, sorted(int(q) for q in cc_diagonal))
    if mirror_pairs:
        if len(circuit):
            circuit.append("TICK")
        circuit.append("CZ", [int(q) for pair in mirror_pairs for q in pair])
    return circuit


def fold_logical_cz_s_action(patch: HGPCode, dagger: bool = False) -> Dict[int, Tuple[str, Optional[int]]]:
    """Logical action of the CZ-S layer (or its inverse) per logical id.

    ``("S", None)`` for V1xV2 diagonal pairs, ``("S_DAG", None)`` for C1xC2
    diagonal pairs, ``("CZ", partner_id)`` for the others (partner = mirror
    pair of :func:`fold_logical_permutation`).  With ``dagger=True`` the
    diagonal entries are exchanged (V1xV2 diagonal pairs get S̄†, C1xC2
    diagonal pairs S̄); CZ̄ is its own inverse.
    """
    permutation = fold_logical_permutation(patch)
    sector = {int(r["logical_id"]): r["sector"] for r in patch.logical_pairs}
    action: Dict[int, Tuple[str, Optional[int]]] = {}
    for logical_id, mirror in permutation.items():
        if mirror == logical_id:
            s_here = (sector[logical_id] == "bit_bit") != dagger
            action[logical_id] = ("S" if s_here else "S_DAG", None)
        else:
            action[logical_id] = ("CZ", mirror)
    return action


def fold_h_layer_circuit(data_qubits: List[int]) -> stim.Circuit:
    """The transversal layer ``H^{⊗n}`` of the H-SWAP on ``data_qubits``."""
    circuit = stim.Circuit()
    if data_qubits:
        circuit.append("H", sorted(int(q) for q in data_qubits))
    return circuit


def fold_swap_layer_circuit(mirror_pairs: List[IndexPair]) -> stim.Circuit:
    """The fold layer ``⊗ SWAP(Q_{i,j}, Q_{j,i})`` of the H-SWAP."""
    circuit = stim.Circuit()
    if mirror_pairs:
        circuit.append("SWAP", [int(q) for pair in mirror_pairs for q in pair])
    return circuit


def fold_h_swap_circuit(data_qubits: List[int], mirror_pairs: List[IndexPair]) -> stim.Circuit:
    """Physical H-SWAP: ``H`` on every data qubit, TICK, SWAP of every mirror pair."""
    circuit = fold_h_layer_circuit(data_qubits)
    swaps = fold_swap_layer_circuit(mirror_pairs)
    if len(circuit) and len(swaps):
        circuit.append("TICK")
    circuit += swaps
    return circuit


# ---------------------------------------------------------------------------
# Logical operation set
# ---------------------------------------------------------------------------

class HGPCodeLogicalOpSet(CSSLogicalOpSet):
    """Logical operation set for :class:`HGPCode` patches.

    Inherits ``transversal_cnot`` (between two identical patches) from
    :class:`CSSLogicalOpSet` and adds the fold-transversal gates of a
    symmetric HGP code (module docstring)::

        executor.apply_logical_operation("fold_transversal_h_swap", [patch])
        executor.apply_logical_operation("fold_transversal_cz_s", [patch])
        executor.apply_logical_operation("fold_transversal_cz_s_dag", [patch])

    ``patch`` must be the global patch returned by ``QECSystem.add_patch``.

    ``extraction_block_class`` is stored for interface parity with the other
    op sets (e.g. ``RotatedSurfaceCodeLogicalOpSet``); the H-SWAP does not
    use it.
    """

    def __init__(self, extraction_block_class: Optional[Type] = None):
        super().__init__()
        self.name = "HGPCode"
        self.extraction_block_class = extraction_block_class

    def fold_transversal_h_swap(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        noiseless: bool = False,
        noisy_swap: bool = True,
    ) -> stim.Circuit:
        """Apply the fold-transversal H-SWAP to every logical qubit of ``patch``.

        Logical action (exact): H̄ on all k logical qubits composed with the
        permutation ``fold_logical_permutation(patch)`` of the logical pairs,
        ``(l1, l2) ↔ (l2, l1)``.  The builder's tracker follows this through
        the physical Clifford; nothing is relabelled by hand.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global HGPCode patch (returned by ``system.add_patch``);
                a local patch is rejected with ValueError.  Its two base
                codes must be the same matrix, else ValueError.
            noiseless: If True, tag both layers (H and SWAP) as noiseless.
            noisy_swap: Only used when ``noiseless`` is False.  True: the
                SWAP layer takes the noise model's two-qubit gate noise.
                False: the SWAP layer is tagged noiseless, modelling the fold
                as a relabelling of qubits without gate error; the H layer
                stays noisy.

        Returns:
            The appended circuit (H layer, TICK, SWAP layer), i.e.
            :func:`fold_h_swap_circuit` on this patch's global indices.  The
            returned copy carries no ``noiseless`` tags; the builder's
            circuit does when requested.
        """
        data_qubits = sorted(int(q) for q in patch.data_indices)
        mirror_pairs = fold_mirror_pairs(patch, builder)   # validates type, global patch, symmetry
        if not set(q for pair in mirror_pairs for q in pair) <= set(data_qubits):
            raise RuntimeError("Fold pairs are not data qubits of this patch.")

        h_layer = fold_h_layer_circuit(data_qubits)
        swap_layer = fold_swap_layer_circuit(mirror_pairs)
        builder.apply_unitary_block(h_layer, noiseless=noiseless)
        if len(swap_layer):
            builder.apply_unitary_block(swap_layer, noiseless=(noiseless or not noisy_swap))
        return fold_h_swap_circuit(data_qubits, mirror_pairs)

    def fold_transversal_cz_s(
        self, builder: CircuitBuilder, patch: QECPatch, noiseless: bool = False
    ) -> stim.Circuit:
        """Apply the fold-transversal CZ-S layer to ``patch`` (module docstring).

        Logical action (exact): S̄ on every V1xV2 diagonal logical pair,
        S̄† on every C1xC2 diagonal pair, CZ̄ on every mirror pair of logical
        qubits; see :func:`fold_logical_cz_s_action`.  The tracker follows
        the symplectic part of the physical Clifford; signs (the Pauli
        frame, e.g. S̄² = Z̄ sending X̄ to −X̄) are not tracked, so they show
        up in raw measurement parities but not in detector-sampler flips.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global symmetric HGPCode patch (``system.add_patch``).
            noiseless: If True, tag both layers (S/S† and CZ) as noiseless.

        Returns:
            The appended circuit, i.e. :func:`fold_cz_s_circuit` on this
            patch's global indices (without ``noiseless`` tags).
        """
        return self._apply_cz_s(builder, patch, noiseless=noiseless, dagger=False)

    def fold_transversal_cz_s_dag(
        self, builder: CircuitBuilder, patch: QECPatch, noiseless: bool = False
    ) -> stim.Circuit:
        """Inverse of :meth:`fold_transversal_cz_s` (S and S† exchanged, CZ unchanged).

        Logical action: S̄† on V1xV2 diagonal pairs, S̄ on C1xC2 diagonal
        pairs, CZ̄ on mirror pairs (``fold_logical_cz_s_action(patch, dagger=True)``).
        """
        return self._apply_cz_s(builder, patch, noiseless=noiseless, dagger=True)

    def _apply_cz_s(self, builder, patch, *, noiseless: bool, dagger: bool) -> stim.Circuit:
        vv_diagonal, cc_diagonal = fold_diagonal_qubits_by_sector(patch, builder)  # validates
        mirror_pairs = fold_mirror_pairs(patch, builder)
        data = set(int(q) for q in patch.data_indices)
        if not (set(vv_diagonal) | set(cc_diagonal) | {q for p in mirror_pairs for q in p}) <= data:
            raise RuntimeError("Fold qubits are not data qubits of this patch.")
        circuit = fold_cz_s_circuit(vv_diagonal, cc_diagonal, mirror_pairs, dagger=dagger)
        # Two blocks so the noise model sees the single-qubit layer and the
        # CZ layer as separate moments, like the H-SWAP.
        phase_layer = stim.Circuit()
        cz_layer = stim.Circuit()
        for inst in circuit:
            if inst.name == "TICK":
                continue
            (cz_layer if inst.name == "CZ" else phase_layer).append(inst)
        if len(phase_layer):
            builder.apply_unitary_block(phase_layer, noiseless=noiseless)
        if len(cz_layer):
            builder.apply_unitary_block(cz_layer, noiseless=noiseless)
        return circuit


__all__ = [
    "HGPCodeLogicalOpSet",
    "fold_cz_s_circuit",
    "fold_diagonal_qubits",
    "fold_diagonal_qubits_by_sector",
    "fold_h_layer_circuit",
    "fold_h_swap_circuit",
    "fold_logical_cz_s_action",
    "fold_logical_permutation",
    "fold_mirror_pairs",
    "fold_swap_layer_circuit",
    "is_symmetric_hgp",
]
