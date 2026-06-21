"""
Routed multi-patch lattice-surgery helpers.

The routed coupler is basis-aware: each selected interface has a native X/Z
basis.  Logical H is applied only when the requested target Pauli differs from
that native interface basis.  For paper-style full-width routes, the ancillary
region is interpreted as one bent surface-code patch; any leftover ancillary
boundary support is tracked as a known ancilla logical factor instead of as
individual data readouts.
"""
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
from dataclasses import dataclass, field

import numpy as np
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedRoutedMultiPatchCoupler,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.utils.tableau_utils import stabilizers_to_symplectic


@dataclass(frozen=True)
class SyndromeProductTerm:
    """One syndrome outcome used in a logical Pauli-product decomposition."""

    source: str
    patch_name: str
    stabilizer_uid: int
    local_stabilizer_index: int
    stype: str
    syn_idx: int
    syn_coord: Tuple[float, float]
    weight: int
    rec_offset: int


@dataclass(frozen=True)
class AncillaReadoutTerm:
    """
    One ancillary-patch data readout/known-boundary term in a product decomposition.

    A full-width ancillary patch can leave an ancillary logical boundary factor.
    These terms represent the extra ancilla data outcomes, or known prepared
    boundary eigenvalues, needed in addition to syndrome outcomes.
    """

    source: str
    coupler_name: str
    local_qubit_index: int
    global_qubit_index: int
    coord: Tuple[float, float]
    pauli: str
    rec_offset: Optional[int] = None


@dataclass(frozen=True)
class AncillaKnownTerm:
    """
    One known ancillary-patch initialization eigenvalue used by a decomposition.

    These terms are not final readout outcomes.  They are deterministic +1
    factors supplied by the chosen ancillary preparation basis, so the measured
    logical-product bit still comes only from the selected syndrome outcomes.
    """

    source: str
    coupler_name: str
    local_qubit_index: int
    global_qubit_index: int
    coord: Tuple[float, float]
    pauli: str


@dataclass(frozen=True)
class AncillaLogicalTerm:
    """
    One known logical boundary factor of a long ancillary patch.

    In the long-ancilla picture used by Litinski/von Oppen, the bent bus is a
    real surface-code patch.  Merge-check products can leave the bus logical
    operator, which is not a set of independent physical readouts; its outcome is
    supplied by the ancillary patch preparation/readout frame.
    """

    source: str
    coupler_name: str
    logical_name: str
    pauli: str
    support_terms: List[AncillaReadoutTerm]
    known_from: str = "long_ancilla_logical_frame"


@dataclass(frozen=True)
class SyndromeProductDecomposition:
    """
    Algebraic expression for a logical product as measured outcomes.

    The product of all listed syndrome outcomes, and any listed ancillary
    readout/known-boundary terms, is equal to the requested target logical
    product in the stabilizer algebra.  Terms from ``source='patch'`` are active
    patch stabilizers that supply the stabilizer-valued correction needed by the
    local routed geometry.
    """

    coupler_name: str
    patch_names: List[str]
    target_paulis: List[str]
    selected_coupler_terms: List[SyndromeProductTerm]
    selected_patch_terms: List[SyndromeProductTerm]
    verified: bool
    selected_ancilla_terms: List[AncillaReadoutTerm] = field(default_factory=list)
    selected_ancilla_known_terms: List[AncillaKnownTerm] = field(default_factory=list)

    @property
    def selected_terms(self) -> List[SyndromeProductTerm]:
        return [*self.selected_coupler_terms, *self.selected_patch_terms]

    @property
    def uses_ancilla_readout_terms(self) -> bool:
        return bool(self.selected_ancilla_terms)


@dataclass(frozen=True)
class LongAncillaProductDecomposition:
    """
    Routed Pauli-product expression using a paper-style long ancillary patch.

    ``syndrome_decomposition`` contains the measured syndrome outcomes.  Its
    ancillary support is reinterpreted as one or more known logical factors of
    the long ancillary patch and exposed through
    ``selected_ancilla_logical_terms``.
    """

    coupler_name: str
    patch_names: List[str]
    target_paulis: List[str]
    syndrome_decomposition: SyndromeProductDecomposition
    selected_ancilla_logical_terms: List[AncillaLogicalTerm]
    verified: bool

    @property
    def selected_terms(self) -> List[SyndromeProductTerm]:
        return self.syndrome_decomposition.selected_terms

    @property
    def selected_coupler_terms(self) -> List[SyndromeProductTerm]:
        return self.syndrome_decomposition.selected_coupler_terms

    @property
    def selected_patch_terms(self) -> List[SyndromeProductTerm]:
        return self.syndrome_decomposition.selected_patch_terms


@dataclass(frozen=True)
class MergeCheckProductTerm:
    """One local green/merge check used in a routed product measurement."""

    source: str
    patch_name: str
    stabilizer_uid: int
    local_stabilizer_index: int
    stype: str
    syn_idx: int
    syn_coord: Tuple[float, float]
    weight: int
    rec_offset: int
    pauli: Dict[int, str]
    base_weight: int


@dataclass(frozen=True)
class ResidualPauliTerm:
    """One uncancelled Pauli left by a strict merge-check product."""

    global_qubit_index: int
    coord: Tuple[float, float]
    pauli: str
    owner: Optional[str]


@dataclass(frozen=True)
class MergeCheckProductDecomposition:
    """
    Green-check expression for a routed logical product.

    ``selected_merge_terms`` are the measured local merge checks.  The
    ``patch_correction_terms`` are original code stabilizers used only for the
    algebraic equivalence ``product(green checks) = logical × stabilizers``;
    they are not part of the green-check outcome product.
    """

    coupler_name: str
    patch_names: List[str]
    target_paulis: List[str]
    selected_merge_terms: List[MergeCheckProductTerm]
    patch_correction_terms: List[SyndromeProductTerm]
    verified: bool
    trimmed_boundary_terms: List[AncillaKnownTerm] = field(default_factory=list)
    residual_terms: List[ResidualPauliTerm] = field(default_factory=list)
    selection_rule: str = "strict_geometry"

    @property
    def selected_terms(self) -> List[MergeCheckProductTerm]:
        return list(self.selected_merge_terms)


def x_to_z_basis_change_indices(paulis: Sequence[str]) -> List[int]:
    """
    Return the patch indices that need a logical H before a Z-product measurement.

    Supports X/Z strings only.  For example, ``ZZZX`` returns ``[3]`` because
    H on patch 3 maps the requested X measurement into a Z measurement.
    """
    result = []
    for i, pauli in enumerate(paulis):
        p = pauli.upper()
        if p == "X":
            result.append(i)
        elif p != "Z":
            raise ValueError(f"Only X/Z Pauli products are supported, got '{pauli}' at index {i}.")
    return result


def infer_interface_paulis(
    system: QECSystem,
    patch_names: Sequence[str],
    sides: Sequence[str],
) -> List[str]:
    """Infer the native X/Z basis of each selected patch side."""
    if len(patch_names) != len(sides):
        raise ValueError("patch_names and sides must have the same length.")
    result = []
    for patch_name, side in zip(patch_names, sides):
        patch = system.patches[patch_name][0]
        result.append(UnrotatedRoutedMultiPatchCoupler.infer_side_basis(patch, side))
    return result


def basis_change_indices_for_interfaces(
    target_paulis: Sequence[str],
    interface_paulis: Sequence[str],
) -> List[int]:
    """
    Return patch indices needing logical H because target and interface differ.

    Decision table:
        interface X, target X -> no H
        interface X, target Z -> H
        interface Z, target Z -> no H
        interface Z, target X -> H
    """
    if len(target_paulis) != len(interface_paulis):
        raise ValueError("target_paulis and interface_paulis must have the same length.")

    result = []
    for i, (target, interface) in enumerate(zip(target_paulis, interface_paulis)):
        t = target.upper()
        native = interface.upper()
        if t not in ("X", "Z") or native not in ("X", "Z"):
            raise ValueError("Only X/Z Pauli products are supported.")
        if t != native:
            result.append(i)
    return result


def apply_logical_h_on_patches(
    builder: CircuitBuilder,
    system: QECSystem,
    patch_names: Sequence[str],
    indices: Sequence[int],
    noiseless: bool = False,
):
    """Apply the unrotated surface-code fold-transversal logical H to selected patches."""
    op_set = UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock)
    for i in indices:
        patch_name = patch_names[i]
        patch = system.patches[patch_name][0]
        op_set.fold_transversal_hadamard(builder, patch, noiseless=noiseless)


def logical_pauli_product_vector(
    system: QECSystem,
    patch_names: Sequence[str],
    paulis: Sequence[str],
) -> np.ndarray:
    """Return the symplectic vector for a requested logical Pauli product."""
    if len(patch_names) != len(paulis):
        raise ValueError("patch_names and paulis must have the same length.")

    row = np.zeros(2 * system.num_qubits, dtype=np.uint8)
    for patch_name, pauli in zip(patch_names, paulis):
        p = pauli.upper()
        if p not in ("X", "Z"):
            raise ValueError(f"Only X/Z logical products are supported, got {pauli!r}.")
        matches = [
            op for op in system.logical_ops
            if op.get("patch_name") == patch_name and op.get("type") == p
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one logical {p} operator for patch {patch_name!r}, "
                f"found {len(matches)}."
            )
        for q, term in matches[0].get("pauli", {}).items():
            if term in ("X", "Y"):
                row[q] ^= 1
            if term in ("Z", "Y"):
                row[system.num_qubits + q] ^= 1
    return row.reshape(1, -1)


def routed_coupler_data_basis(
    system: QECSystem,
    coupler_name: str,
    mode: str = "opposite",
    default_route_basis: str = "Z",
) -> Dict[int, str]:
    """
    Return a global-qubit basis map for a routed coupler's data qubits.

    ``mode='same'`` returns the local route label itself: Z-labeled ancillary
    data are measured in Z and X-labeled ancillary data are measured in X.

    ``mode='opposite'`` returns the conjugate preparation basis: Z-labeled
    ancillary data are prepared in X and X-labeled ancillary data are prepared
    in Z.  This is the mixed-route analogue of preparing a ZZ-surgery ancilla
    in |+> and an XX-surgery ancilla in |0>.
    """
    if coupler_name not in system.coupler_patches:
        raise ValueError(f"Coupler {coupler_name!r} is not registered.")

    normalized_mode = mode.lower()
    default_route_basis = default_route_basis.upper()
    if default_route_basis not in ("X", "Z"):
        raise ValueError("default_route_basis must be 'X' or 'Z'.")

    cp = system.coupler_patches[coupler_name]
    local_to_global = system.local_to_global_map.get(coupler_name, {})
    basis_map: Dict[int, str] = {}
    for local_idx in sorted(cp.data_indices):
        global_idx = local_to_global.get(local_idx)
        if global_idx is None:
            continue
        coord = cp.qubit_coords[local_idx]
        route_basis = str(
            getattr(cp, "route_coord_basis", {}).get(coord, default_route_basis)
        ).upper()
        if route_basis not in ("X", "Z"):
            route_basis = default_route_basis

        if normalized_mode in ("same", "route", "readout"):
            basis = route_basis
        elif normalized_mode in ("opposite", "conjugate", "prep", "prepare", "init"):
            basis = _opposite_xz(route_basis)
        elif normalized_mode in ("x", "z"):
            basis = normalized_mode.upper()
        else:
            raise ValueError(
                "mode must be one of 'same', 'opposite', 'X', or 'Z'."
            )
        basis_map[global_idx] = basis
    return basis_map


def solve_routed_pauli_product_syndromes(
    system: QECSystem,
    patch_names: Sequence[str],
    paulis: Sequence[str],
    coupler_name: str,
    include_patch_stabilizers: bool = True,
    include_ancilla_readout_terms: bool = True,
    ancilla_readout_bases: Optional[Mapping[int, str]] = None,
    include_ancilla_known_terms: bool = False,
    ancilla_known_bases: Optional[Mapping[int, str]] = None,
    reduce_weight: bool = True,
) -> SyndromeProductDecomposition:
    """
    Solve which syndrome outcomes multiply to a routed logical Pauli product.

    This is an algebraic extractor for native mixed X/Z routed measurements.  It
    searches the measured stabilizers of the coupler-on system.  If
    ``include_patch_stabilizers`` is true, still-active patch stabilizers are
    allowed as stabilizer-valued correction terms.  This is necessary for local
    mixed geometries where the coupler checks alone leave boundary stabilizer
    factors.

    For full-width ancillary-patch geometries, a pure syndrome product can need
    deterministic ancillary initialization eigenvalues to close the algebra.
    When ``include_ancilla_known_terms`` is true, those +1 preparation factors
    are allowed and reported separately in ``selected_ancilla_known_terms``;
    they are not readout terms and do not change the measured syndrome product.

    If that is still insufficient and ``include_ancilla_readout_terms`` is true,
    the solver falls back to allowing ancillary-patch data readout terms and
    reports them separately in ``selected_ancilla_terms``.

    If ``ancilla_readout_bases`` is provided, only those actual readout bases are
    allowed.  Keys are global qubit indices and values are ``'X'`` or ``'Z'``.
    This prevents solving with an unphysical mixture where the same data qubit
    contributes both X and Z readout outcomes.

    ``ancilla_known_bases`` has the same shape, but represents deterministic
    preparation eigenvalues instead of final readout outcomes.
    """
    if coupler_name not in system.coupler_patches:
        raise ValueError(f"Coupler {coupler_name!r} is not registered.")

    target_paulis = [p.upper() for p in paulis]
    target = logical_pauli_product_vector(system, patch_names, target_paulis)
    active_uids = _coupler_on_active_stabilizer_uids(system, coupler_name)
    coupler_uids = [
        uid for uid in sorted(active_uids)
        if system.stabilizers[uid].get("patch_name") == coupler_name
    ]
    patch_uids = []
    if include_patch_stabilizers:
        patch_set = set(patch_names)
        patch_uids = [
            uid for uid in sorted(active_uids)
            if system.stabilizers[uid].get("patch_name") in patch_set
        ]

    basis_uids = coupler_uids + patch_uids
    if not basis_uids:
        raise ValueError("No candidate syndrome stabilizers are available.")

    basis_records = [system.stabilizers[uid] for uid in basis_uids]
    basis = stabilizers_to_symplectic(system, basis_records, system.num_qubits)
    coeffs, is_dependent, _ = solve_linear_decomposition(
        basis=basis,
        targets=target,
        reduce_weight=reduce_weight,
    )
    ancilla_known_labels: List[AncillaKnownTerm] = []
    ancilla_labels: List[AncillaReadoutTerm] = []
    combined_basis = basis
    known_start = None
    readout_start = None

    if not is_dependent[0] and include_ancilla_known_terms:
        known_basis, known_readout_labels = _coupler_data_readout_basis(
            system,
            coupler_name,
            readout_bases=ancilla_known_bases,
        )
        ancilla_known_labels = [
            AncillaKnownTerm(
                source="ancilla_initialization",
                coupler_name=label.coupler_name,
                local_qubit_index=label.local_qubit_index,
                global_qubit_index=label.global_qubit_index,
                coord=label.coord,
                pauli=label.pauli,
            )
            for label in known_readout_labels
        ]
        if known_basis.shape[0]:
            known_start = combined_basis.shape[0]
            combined_basis = np.vstack([combined_basis, known_basis])
            coeffs, is_dependent, _ = solve_linear_decomposition(
                basis=combined_basis,
                targets=target,
                reduce_weight=reduce_weight,
            )

    if not is_dependent[0] and include_ancilla_readout_terms:
        ancilla_basis, ancilla_labels = _coupler_data_readout_basis(
            system,
            coupler_name,
            readout_bases=ancilla_readout_bases,
        )
        if ancilla_basis.shape[0]:
            readout_start = combined_basis.shape[0]
            combined_basis = np.vstack([basis, ancilla_basis])
            if known_start is not None:
                combined_basis = np.vstack([
                    basis,
                    known_basis,
                    ancilla_basis,
                ])
            coeffs, is_dependent, _ = solve_linear_decomposition(
                basis=combined_basis,
                targets=target,
                reduce_weight=reduce_weight,
            )

    if not is_dependent[0]:
        mode = "coupler plus active patch stabilizers" if include_patch_stabilizers else "coupler stabilizers only"
        if include_ancilla_known_terms:
            mode += " plus known ancillary initialization terms"
        if include_ancilla_readout_terms:
            mode += " plus ancillary readout terms"
        raise ValueError(
            f"Target logical product {''.join(target_paulis)} is not generated by {mode} "
            f"for coupler {coupler_name!r}."
        )

    selected_positions = np.where(coeffs[0])[0].tolist()
    product = np.zeros_like(target[0])
    for pos in selected_positions:
        product ^= combined_basis[pos]
    verified = bool(np.array_equal(product.reshape(1, -1), target))

    rec_offsets = _single_round_record_offsets(system, active_uids)
    local_indices = _coupler_local_stabilizer_indices(system, coupler_name)
    selected_coupler_terms: List[SyndromeProductTerm] = []
    selected_patch_terms: List[SyndromeProductTerm] = []
    selected_ancilla_terms: List[AncillaReadoutTerm] = []
    selected_ancilla_known_terms: List[AncillaKnownTerm] = []

    for pos in selected_positions:
        if known_start is not None and known_start <= pos < known_start + len(ancilla_known_labels):
            known_pos = pos - known_start
            selected_ancilla_known_terms.append(ancilla_known_labels[known_pos])
            continue
        if readout_start is not None and pos >= readout_start:
            ancilla_pos = pos - readout_start
            selected_ancilla_terms.append(ancilla_labels[ancilla_pos])
            continue
        uid = basis_uids[pos]
        source = "coupler" if uid in coupler_uids else "patch"
        term = _syndrome_product_term_from_uid(
            system=system,
            uid=uid,
            source=source,
            rec_offsets=rec_offsets,
            coupler_local_indices=local_indices,
        )
        if source == "coupler":
            selected_coupler_terms.append(term)
        else:
            selected_patch_terms.append(term)

    return SyndromeProductDecomposition(
        coupler_name=coupler_name,
        patch_names=list(patch_names),
        target_paulis=target_paulis,
        selected_coupler_terms=selected_coupler_terms,
        selected_patch_terms=selected_patch_terms,
        verified=verified,
        selected_ancilla_terms=selected_ancilla_terms,
        selected_ancilla_known_terms=selected_ancilla_known_terms,
    )


def solve_routed_pauli_product_long_ancilla(
    system: QECSystem,
    patch_names: Sequence[str],
    paulis: Sequence[str],
    coupler_name: str,
    ancilla_frame_mode: str = "same",
    reduce_weight: bool = True,
) -> LongAncillaProductDecomposition:
    """
    Interpret a routed product measurement using one bent long ancillary patch.

    This follows the patch picture used in long-range lattice-surgery layouts:
    the routed bus is a valid ancillary surface-code patch, and the measured
    syndrome product uses the full connected ancillary region, not a minimum
    weight algebraic shortcut near the data-patch interfaces.  The product of
    those merge-check syndromes may leave the bus logical boundary operator.
    That boundary support is returned as one known ``AncillaLogicalTerm``
    instead of as individual ancillary data readouts.
    """
    del ancilla_frame_mode  # The long-ancilla path uses geometry, not readout rows.
    merge_decomp = solve_routed_pauli_product_merge_checks(
        system=system,
        patch_names=patch_names,
        paulis=paulis,
        coupler_name=coupler_name,
        reduce_weight=reduce_weight,
    )

    selected_coupler_terms = [
        SyndromeProductTerm(
            source="coupler",
            patch_name=term.patch_name,
            stabilizer_uid=term.stabilizer_uid,
            local_stabilizer_index=term.local_stabilizer_index,
            stype=term.stype,
            syn_idx=term.syn_idx,
            syn_coord=term.syn_coord,
            weight=term.weight,
            rec_offset=term.rec_offset,
        )
        for term in merge_decomp.selected_merge_terms
    ]

    residual_terms = list(merge_decomp.residual_terms)
    residual_is_long_ancilla = all(term.owner == coupler_name for term in residual_terms)
    verified = bool(residual_is_long_ancilla)

    syndrome_decomp = SyndromeProductDecomposition(
        coupler_name=coupler_name,
        patch_names=list(patch_names),
        target_paulis=[p.upper() for p in paulis],
        selected_coupler_terms=selected_coupler_terms,
        selected_patch_terms=list(merge_decomp.patch_correction_terms),
        verified=merge_decomp.verified,
        selected_ancilla_terms=[],
        selected_ancilla_known_terms=[],
    )

    logical_terms: List[AncillaLogicalTerm] = []
    local_to_global = system.local_to_global_map.get(coupler_name, {})
    global_to_local = {global_idx: local_idx for local_idx, global_idx in local_to_global.items()}
    support = [
        AncillaReadoutTerm(
            source="long_ancilla_logical_support",
            coupler_name=coupler_name,
            local_qubit_index=global_to_local.get(term.global_qubit_index, -1),
            global_qubit_index=term.global_qubit_index,
            coord=term.coord,
            pauli=term.pauli,
            rec_offset=None,
        )
        for term in residual_terms
        if term.owner == coupler_name
    ]
    if support:
        pauli_types = sorted({term.pauli for term in support})
        logical_pauli = pauli_types[0] if len(pauli_types) == 1 else "MIXED"
        logical_terms.append(AncillaLogicalTerm(
            source="ancilla_logical",
            coupler_name=coupler_name,
            logical_name=f"{coupler_name}_long_ancilla_boundary",
            pauli=logical_pauli,
            support_terms=support,
        ))

    return LongAncillaProductDecomposition(
        coupler_name=coupler_name,
        patch_names=list(patch_names),
        target_paulis=[p.upper() for p in paulis],
        syndrome_decomposition=syndrome_decomp,
        selected_ancilla_logical_terms=logical_terms,
        verified=verified,
    )


def solve_routed_pauli_product_merge_checks(
    system: QECSystem,
    patch_names: Sequence[str],
    paulis: Sequence[str],
    coupler_name: str,
    boundary_basis: Optional[Mapping[int, str]] = None,
    reduce_weight: bool = True,
) -> MergeCheckProductDecomposition:
    """
    Derive the local green/merge checks for a routed logical product.

    This is the basis-aware geometric check used for debugging the routed mixed
    templates.  It multiplies complete coupler stabilizers from the ancillary
    region without trimming any boundary Pauli factors, but only from the
    X/Z sheet that matches the local routed interface label.  Mixed seam checks
    are kept as local mixed templates.  Original data-patch stabilizers may be
    used only as code-space equivalence terms.  If the product still leaves
    ancillary/data residual Paulis, ``verified`` is false and ``residual_terms``
    lists the leftover support.
    """
    del boundary_basis  # Kept for API compatibility with older notebook cells.

    target_paulis = [p.upper() for p in paulis]
    target = logical_pauli_product_vector(system, patch_names, target_paulis)
    active_uids = _coupler_on_active_stabilizer_uids(system, coupler_name)
    rec_offsets = _single_round_record_offsets(system, active_uids)
    local_indices = _coupler_local_stabilizer_indices(system, coupler_name)

    selected_uids = _basis_matched_geometry_merge_check_uids(
        system=system,
        patch_names=patch_names,
        coupler_name=coupler_name,
    )
    selected_merge_terms: List[MergeCheckProductTerm] = []
    selected_records = []
    for uid in selected_uids:
        stab = system.stabilizers[uid]
        pauli = dict(stab.get("pauli", {}))
        if not pauli:
            continue
        selected_records.append(stab)
        key = (stab.get("type"), stab.get("syn_coord"))
        selected_merge_terms.append(MergeCheckProductTerm(
            source="coupler_merge",
            patch_name=stab.get("patch_name"),
            stabilizer_uid=uid,
            local_stabilizer_index=local_indices.get(key, -1),
            stype=stab.get("type"),
            syn_idx=stab.get("syn_idx"),
            syn_coord=stab.get("syn_coord"),
            weight=len(pauli),
            rec_offset=rec_offsets.get(stab.get("syn_idx"), 0),
            pauli=pauli,
            base_weight=len(stab.get("pauli", {})),
        ))

    if not selected_records:
        raise ValueError("No geometric merge-check terms were selected.")

    rows = stabilizers_to_symplectic(system, selected_records, system.num_qubits)
    product = np.bitwise_xor.reduce(rows, axis=0).reshape(1, -1)
    patch_target = np.bitwise_xor(product, target)

    patch_correction_terms: List[SyndromeProductTerm] = []
    residual = patch_target

    for patch_name in patch_names:
        patch_uids = [
            uid for uid, stab in enumerate(system.stabilizers)
            if stab.get("patch_name") == patch_name
        ]
        if not patch_uids:
            continue

        patch_qubits = {
            q for q, owner in system.index_to_owner_map.items()
            if owner == patch_name
        }
        patch_mask = np.zeros_like(residual[0])
        for q in patch_qubits:
            patch_mask[q] = 1
            patch_mask[system.num_qubits + q] = 1
        patch_residual = (residual[0] & patch_mask).reshape(1, -1)
        if not np.any(patch_residual):
            continue

        patch_records = [system.stabilizers[uid] for uid in patch_uids]
        patch_basis = stabilizers_to_symplectic(system, patch_records, system.num_qubits)
        coeffs, is_dependent, _ = solve_linear_decomposition(
            basis=patch_basis,
            targets=patch_residual,
            reduce_weight=reduce_weight,
        )
        if is_dependent[0]:
            patch_positions = np.where(coeffs[0])[0].tolist()
            patch_active = active_uids | set(patch_uids)
            patch_rec_offsets = _single_round_record_offsets(system, patch_active)
            patch_terms = [
                _syndrome_product_term_from_uid(
                    system=system,
                    uid=patch_uids[pos],
                    source="patch",
                    rec_offsets=patch_rec_offsets,
                    coupler_local_indices=local_indices,
                )
                for pos in patch_positions
            ]
            patch_correction_terms.extend(patch_terms)
            if not patch_terms:
                continue
            correction_rows = stabilizers_to_symplectic(
                system,
                [system.stabilizers[t.stabilizer_uid] for t in patch_terms],
                system.num_qubits,
            )
            correction = np.bitwise_xor.reduce(correction_rows, axis=0).reshape(1, -1)
            residual = np.bitwise_xor(residual, correction)

    verified = not np.any(residual)

    return MergeCheckProductDecomposition(
        coupler_name=coupler_name,
        patch_names=list(patch_names),
        target_paulis=target_paulis,
        selected_merge_terms=selected_merge_terms,
        patch_correction_terms=patch_correction_terms,
        verified=verified,
        trimmed_boundary_terms=[],
        residual_terms=_residual_pauli_terms(system, residual),
        selection_rule="basis_matched_geometry_no_trim",
    )


def _basis_matched_geometry_merge_check_uids(
    system: QECSystem,
    patch_names: Sequence[str],
    coupler_name: str,
) -> List[int]:
    """
    Select complete coupler stabilizers belonging to the active product sheet.

    The static routed map includes some coupler-owned boundary templates whose
    syndrome coordinate sits inside a data-patch box.  Those are interface
    boundary checks drawn for geometry/tracker context; the no-trim product
    sheet below only multiplies the ancillary-region checks whose syndrome
    coordinates live on the routed bus itself.

    Inside the ancillary region, a routed mixed product has multiple local
    sheets.  A pure X/Z check contributes only when its type matches the local
    route label.  For example, on the Z-labeled side of a ZZZX route we multiply
    Z checks and skip the interleaved X checks; on the X-labeled side we do the
    converse.  Mixed seam checks are retained because their Pauli support is the
    local X/Z transition template.
    """
    patch_bounds = []
    for patch_name in patch_names:
        patch = system.patches[patch_name][0]
        patch_bounds.append(patch._get_bounds())

    cp = system.coupler_patches[coupler_name]
    local_indices = _coupler_local_stabilizer_indices(system, coupler_name)
    keyed_uids = []
    for uid, stab in enumerate(system.stabilizers):
        if stab.get("patch_name") != coupler_name:
            continue
        syn_coord = stab.get("syn_coord")
        if _coord_in_any_box(syn_coord, patch_bounds):
            continue
        stype = stab.get("type")
        route_basis = getattr(cp, "route_coord_basis", {}).get(syn_coord)
        if stype != "MIXED" and stype != route_basis:
            continue
        key = (stab.get("type"), syn_coord)
        keyed_uids.append((local_indices.get(key, 10**9), uid))
    return [uid for _, uid in sorted(keyed_uids)]


def _coord_in_any_box(
    coord: Tuple[float, float],
    bounds: Sequence[Tuple[float, float, float, float]],
) -> bool:
    x, y = coord
    for x0, x1, y0, y1 in bounds:
        if x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9:
            return True
    return False


def _residual_pauli_terms(system: QECSystem, residual: np.ndarray) -> List[ResidualPauliTerm]:
    row = residual.reshape(-1)
    n = system.num_qubits
    terms: List[ResidualPauliTerm] = []
    for q in range(n):
        has_x = bool(row[q])
        has_z = bool(row[n + q])
        if not has_x and not has_z:
            continue
        pauli = "Y" if has_x and has_z else ("X" if has_x else "Z")
        terms.append(ResidualPauliTerm(
            global_qubit_index=q,
            coord=tuple(system.qubit_coords[q]),
            pauli=pauli,
            owner=system.index_to_owner_map.get(q),
        ))
    return sorted(terms, key=lambda t: (t.coord[1], t.coord[0], t.pauli))


def _coupler_on_active_stabilizer_uids(system: QECSystem, coupler_name: str) -> set:
    """Return stabilizer UIDs that would be active during one coupler-on round."""
    cp = system.coupler_patches[coupler_name]
    active = set(system.active_stabilizer_indices)
    if coupler_name in system.paused_stabilizer_indices:
        return active

    coupler_uids = set(getattr(cp, "_registered_stabilizer_uids", set()))
    if not coupler_uids:
        coupler_uids = {
            uid for uid, stab in enumerate(system.stabilizers)
            if stab.get("patch_name") == coupler_name
        }
    conflict_coords = set(getattr(cp, "conflicting_stabilizer_coords", set()))
    conflict_uids = {
        uid for uid in active
        if system.stabilizers[uid].get("syn_coord") in conflict_coords
    }
    return (active - conflict_uids) | coupler_uids


def _single_round_record_offsets(system: QECSystem, active_uids: set) -> Dict[int, int]:
    """Map syndrome qubit index to Stim rec offset after one SE round's final M."""
    syn_indices = sorted({
        system.stabilizers[uid]["syn_idx"]
        for uid in active_uids
        if system.stabilizers[uid].get("syn_idx") is not None
    })
    n = len(syn_indices)
    return {syn_idx: i - n for i, syn_idx in enumerate(syn_indices)}


def _coupler_local_stabilizer_indices(system: QECSystem, coupler_name: str) -> Dict[Tuple[str, Tuple[float, float]], int]:
    cp = system.coupler_patches[coupler_name]
    return {
        (stab.get("type"), stab.get("syn_coord")): i
        for i, stab in enumerate(cp.stabilizers)
    }


def _coupler_data_readout_basis(
    system: QECSystem,
    coupler_name: str,
    readout_bases: Optional[Mapping[int, str]] = None,
) -> Tuple[np.ndarray, List[AncillaReadoutTerm]]:
    """Return single-qubit rows for allowed ancillary-patch data readout terms."""
    cp = system.coupler_patches[coupler_name]
    rows = []
    labels: List[AncillaReadoutTerm] = []
    local_to_global = system.local_to_global_map.get(coupler_name, {})

    for local_idx in sorted(cp.data_indices):
        global_idx = local_to_global.get(local_idx)
        if global_idx is None:
            continue
        coord = system.qubit_coords[global_idx]
        if readout_bases is None:
            allowed_paulis = ("X", "Z")
        else:
            pauli = readout_bases.get(global_idx)
            if pauli is None:
                continue
            pauli = str(pauli).upper()
            if pauli not in ("X", "Z"):
                raise ValueError(
                    f"Ancilla readout basis for global qubit {global_idx} "
                    f"must be 'X' or 'Z', got {pauli!r}."
                )
            allowed_paulis = (pauli,)

        for pauli in allowed_paulis:
            row = np.zeros(2 * system.num_qubits, dtype=np.uint8)
            if pauli == "X":
                row[global_idx] = 1
            else:
                row[system.num_qubits + global_idx] = 1
            rows.append(row)
            labels.append(AncillaReadoutTerm(
                source="ancilla_readout",
                coupler_name=coupler_name,
                local_qubit_index=local_idx,
                global_qubit_index=global_idx,
                coord=coord,
                pauli=pauli,
            ))

    if not rows:
        return np.zeros((0, 2 * system.num_qubits), dtype=np.uint8), labels
    return np.array(rows, dtype=np.uint8), labels


def _opposite_xz(pauli: str) -> str:
    pauli = str(pauli).upper()
    if pauli == "X":
        return "Z"
    if pauli == "Z":
        return "X"
    raise ValueError(f"Expected 'X' or 'Z', got {pauli!r}.")


def _syndrome_product_term_from_uid(
    system: QECSystem,
    uid: int,
    source: str,
    rec_offsets: Dict[int, int],
    coupler_local_indices: Dict[Tuple[str, Tuple[float, float]], int],
) -> SyndromeProductTerm:
    stab = system.stabilizers[uid]
    syn_idx = stab["syn_idx"]
    key = (stab.get("type"), stab.get("syn_coord"))
    local_idx = coupler_local_indices.get(key, -1) if source == "coupler" else -1
    return SyndromeProductTerm(
        source=source,
        patch_name=stab.get("patch_name"),
        stabilizer_uid=uid,
        local_stabilizer_index=local_idx,
        stype=stab.get("type"),
        syn_idx=syn_idx,
        syn_coord=stab.get("syn_coord"),
        weight=len(stab.get("pauli", {})),
        rec_offset=rec_offsets[syn_idx],
    )


def build_routed_pauli_product_readout_circuit(
    system: QECSystem,
    patch_names: Sequence[str],
    paulis: Sequence[str],
    sides: Sequence[str],
    rounds: int,
    coupler_name: str = "routed_pauli_product",
    init_basis: str = "X",
    coupler_init_basis: str = "X",
    route_padding: int = 4,
    route_width: int = 1,
    interface_paulis: Sequence[str] = None,
    mixed_stabilizers: bool = False,
    if_detector: bool = True,
    h_noiseless: bool = False,
    restore_h_before_readout: bool = True,
) -> Tuple[stim.Circuit, Dict[str, object], QECSystem]:
    """
    Build a complete final-readout circuit for an X/Z routed Pauli product.

    The generated measurement is interpreted in the original frame.  Each target
    Pauli is compared against the native Pauli basis of the selected interface.
    Logical H is applied only on mismatches, and by default the same H gates are
    applied again before final destructive readout.

    For a mid-circuit reusable measurement, use
    ``basis_change_indices_for_interfaces`` and ``apply_logical_h_on_patches``
    directly around your own split/readout flow.
    """
    if len(patch_names) != len(paulis) or len(patch_names) != len(sides):
        raise ValueError("patch_names, paulis, and sides must have the same length.")

    target_paulis = [p.upper() for p in paulis]
    if interface_paulis is None:
        interface_paulis = infer_interface_paulis(system, patch_names, sides)
    interface_paulis = [p.upper() for p in interface_paulis]
    h_indices = basis_change_indices_for_interfaces(target_paulis, interface_paulis)

    system.register_coupler(
        UnrotatedRoutedMultiPatchCoupler(),
        list(patch_names),
        coupler_name,
        sides=list(sides),
        interface_paulis=interface_paulis,
        target_paulis=target_paulis,
        mixed_stabilizers=mixed_stabilizers,
        route_padding=route_padding,
        route_width=route_width,
    )

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=if_detector)
    builder.write_coordinates()

    non_coupler = {
        q: init_basis
        for q in system.data_indices
        if system.index_to_owner_map.get(q) != coupler_name
    }
    builder.initialize(init_dict=non_coupler, n=system.num_qubits)

    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    apply_logical_h_on_patches(
        builder=builder,
        system=system,
        patch_names=patch_names,
        indices=h_indices,
        noiseless=h_noiseless,
    )

    builder.activate_coupler(coupler_name)
    cp = system.coupler_patches[coupler_name]
    coupler_data = {
        system.local_to_global_map[coupler_name][q]: coupler_init_basis
        for q in cp.data_indices
    }
    builder.initialize(init_dict=coupler_data, n=system.num_qubits)

    se2 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds)

    if restore_h_before_readout:
        apply_logical_h_on_patches(
            builder=builder,
            system=system,
            patch_names=patch_names,
            indices=h_indices,
            noiseless=h_noiseless,
        )

    builder.apply_data_readout(final_measurements={**non_coupler, **coupler_data})

    info: Dict[str, object] = {
        "patch_names": list(patch_names),
        "paulis": target_paulis,
        "sides": [s.lower() for s in sides],
        "interface_paulis": interface_paulis,
        "mixed_stabilizers": mixed_stabilizers,
        "route_width": route_width,
        "h_basis_change_indices": h_indices,
        "restore_h_before_readout": restore_h_before_readout,
        "coupler_name": coupler_name,
        "num_qubits": builder.circuit.num_qubits,
        "num_detectors": builder.circuit.num_detectors,
        "num_observables": builder.circuit.num_observables,
    }
    return builder.circuit, info, system
