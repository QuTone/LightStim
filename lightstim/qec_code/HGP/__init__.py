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
    "HGPProductColorLayer",
    "HGPProductColorationExtractionBlock",
    "canonical_kernel_basis",
    "hgp_13_1_3",
    "hgp_13_1_3_seed",
    "hgp_18_2_3",
    "hgp_18_2_3_seed",
    "hgp_225_9_4",
    "hgp_225_9_4_seed",
]
