"""Hypergraph-product quantum error-correcting codes."""

from .algebra import CanonicalKernelBasis, canonical_kernel_basis
from .binary_parity_check import BinaryParityCheck
from .code_patch import HGPCode
from .instances import (
    hgp_13_1_3,
    hgp_13_1_3_seed,
    hgp_18_2_3,
    hgp_18_2_3_seed,
    hgp_225_9_4,
    hgp_225_9_4_seed,
)
from .operation import (
    HGPCodeLogicalOpSet,
    fold_cz_s_circuit,
    fold_diagonal_qubits,
    fold_diagonal_qubits_by_sector,
    fold_h_layer_circuit,
    fold_h_swap_circuit,
    fold_logical_cz_s_action,
    fold_logical_permutation,
    fold_mirror_pairs,
    fold_swap_layer_circuit,
    is_symmetric_hgp,
)
from .SE_block import (
    HGPCodeExtractionBlock,
    HGPProductColorLayer,
    HGPProductColorationExtractionBlock,
)

__all__ = [
    "BinaryParityCheck",
    "CanonicalKernelBasis",
    "HGPCode",
    "HGPCodeExtractionBlock",
    "HGPCodeLogicalOpSet",
    "HGPProductColorLayer",
    "HGPProductColorationExtractionBlock",
    "canonical_kernel_basis",
    "fold_cz_s_circuit",
    "fold_diagonal_qubits",
    "fold_diagonal_qubits_by_sector",
    "fold_h_layer_circuit",
    "fold_h_swap_circuit",
    "fold_logical_cz_s_action",
    "fold_logical_permutation",
    "fold_mirror_pairs",
    "fold_swap_layer_circuit",
    "hgp_13_1_3",
    "hgp_13_1_3_seed",
    "hgp_18_2_3",
    "hgp_18_2_3_seed",
    "hgp_225_9_4",
    "hgp_225_9_4_seed",
    "is_symmetric_hgp",
]
