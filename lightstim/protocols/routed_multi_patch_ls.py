"""
Routed multi-patch lattice-surgery helpers.

The routed coupler is basis-aware: each selected interface has a native X/Z
basis.  Logical H is applied only when the requested target Pauli differs from
that native interface basis.  If X and Z interfaces both connect to the same
ancillary region, the coupler creates local mixed XZ stabilizers at the seams and
corners of the routed path.
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

    This implements the lattice-surgery identity used in the usual pictures:
    the product of the newly measured merge checks equals the target logical
    product up to original code stabilizers.  Operationally this starts from the
    full-routed decomposition, identifies ancillary boundary factors that would
    otherwise be treated as known preparation eigenvalues, and truncates the
    corresponding local merge checks at that boundary.  The resulting measured
    terms are local weight-2/3/4 checks and require no ancillary data readout or
    known ancillary initialization factors in the final outcome product.
    """
    if boundary_basis is None:
        boundary_basis = routed_coupler_data_basis(system, coupler_name, mode="same")

    seed = solve_routed_pauli_product_syndromes(
        system=system,
        patch_names=patch_names,
        paulis=paulis,
        coupler_name=coupler_name,
        include_patch_stabilizers=True,
        include_ancilla_readout_terms=False,
        include_ancilla_known_terms=True,
        ancilla_known_bases=boundary_basis,
        reduce_weight=reduce_weight,
    )
    if seed.selected_ancilla_terms:
        raise ValueError(
            "Merge-check decomposition must not use ancillary data readout terms."
        )

    target_paulis = [p.upper() for p in paulis]
    target = logical_pauli_product_vector(system, patch_names, target_paulis)
    active_uids = _coupler_on_active_stabilizer_uids(system, coupler_name)
    rec_offsets = _single_round_record_offsets(system, active_uids)

    to_trim: Set[Tuple[int, str]] = {
        (term.global_qubit_index, term.pauli)
        for term in seed.selected_ancilla_known_terms
    }
    remaining = set(to_trim)
    selected_merge_terms: List[MergeCheckProductTerm] = []
    trimmed_records = []

    for term in seed.selected_coupler_terms:
        stab = system.stabilizers[term.stabilizer_uid]
        pauli = dict(stab.get("pauli", {}))
        for key in list(remaining):
            q, p = key
            if pauli.get(q) == p:
                del pauli[q]
                remaining.remove(key)

        if not pauli:
            continue

        trimmed_record = dict(stab)
        trimmed_record["pauli"] = pauli
        trimmed_record["data_indices"] = sorted(pauli)
        trimmed_records.append(trimmed_record)

        selected_merge_terms.append(MergeCheckProductTerm(
            source="coupler_merge",
            patch_name=stab.get("patch_name"),
            stabilizer_uid=term.stabilizer_uid,
            local_stabilizer_index=term.local_stabilizer_index,
            stype=stab.get("type"),
            syn_idx=stab.get("syn_idx"),
            syn_coord=stab.get("syn_coord"),
            weight=len(pauli),
            rec_offset=rec_offsets[stab.get("syn_idx")],
            pauli=pauli,
            base_weight=len(stab.get("pauli", {})),
        ))

    if remaining:
        examples = sorted(remaining)[:5]
        raise ValueError(
            "Could not attach all routed boundary terms to local merge checks; "
            f"examples: {examples}."
        )

    patch_records = [
        system.stabilizers[term.stabilizer_uid]
        for term in seed.selected_patch_terms
    ]
    verify_records = [*trimmed_records, *patch_records]
    if not verify_records:
        raise ValueError("No merge-check terms were selected.")

    rows = stabilizers_to_symplectic(system, verify_records, system.num_qubits)
    product = np.bitwise_xor.reduce(rows, axis=0).reshape(1, -1)
    verified = bool(np.array_equal(product, target))
    if not verified:
        raise ValueError(
            f"Trimmed merge checks do not generate target logical product "
            f"{''.join(target_paulis)} for coupler {coupler_name!r}."
        )

    return MergeCheckProductDecomposition(
        coupler_name=coupler_name,
        patch_names=list(patch_names),
        target_paulis=target_paulis,
        selected_merge_terms=selected_merge_terms,
        patch_correction_terms=seed.selected_patch_terms,
        verified=verified,
        trimmed_boundary_terms=seed.selected_ancilla_known_terms,
    )


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
