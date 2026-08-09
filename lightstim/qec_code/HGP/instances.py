"""Fixed, reproducible HGP instances used in examples and tests."""

from __future__ import annotations

from .binary_parity_check import BinaryParityCheck
from .code_patch import HGPCode


_HGP_13_1_3_ROWS = (
    (0, 1),
    (1, 2),
)

_HGP_225_9_4_ROWS = (
    (1, 4, 5, 10),
    (0, 5, 8, 9),
    (3, 4, 7, 11),
    (0, 4, 6, 11),
    (3, 7, 8, 10),
    (0, 1, 2, 3),
    (1, 8, 9, 11),
    (2, 6, 9, 10),
    (2, 5, 6, 7),
)


def hgp_13_1_3_seed() -> BinaryParityCheck:
    """Return the open-boundary length-3 repetition-code parity check."""
    return BinaryParityCheck.from_row_supports(
        (2, 3),
        _HGP_13_1_3_ROWS,
        source_metadata={
            "code_family": "open_repetition",
            "classical_parameters": (3, 1, 3),
        },
    )


def hgp_13_1_3() -> HGPCode:
    """Build the ``[[13, 1, 3]]`` unrotated-surface-code HGP patch."""
    return HGPCode(hgp_13_1_3_seed(), d=3)


def hgp_18_2_3_seed() -> BinaryParityCheck:
    """Return the cyclic length-3 repetition-code parity check."""
    return BinaryParityCheck.from_cyclic_polynomial(
        (0, 1),
        size=3,
        source_metadata={
            "code_family": "cyclic_repetition",
            "classical_parameters": (3, 1, 3),
        },
    )


def hgp_18_2_3() -> HGPCode:
    """Build the ``[[18, 2, 3]]`` toric-code HGP patch."""
    return HGPCode(hgp_18_2_3_seed(), d=3)


def hgp_225_9_4_seed() -> BinaryParityCheck:
    """Return a fixed ``[12, 3, 4]`` (3,4)-biregular classical seed.

    Xu et al. construct their HGP family by rejection-sampling (3,4)-biregular
    Tanner graphs but do not publish a canonical parity-check matrix for each
    random instance.  This independently generated, fixed matrix matches the
    degree profile and the smallest reported code parameters, making the
    resulting self-product a reproducible ``[[225, 9, 4]]`` representative.
    """
    return BinaryParityCheck.from_row_supports(
        (9, 12),
        _HGP_225_9_4_ROWS,
        source_metadata={
            "construction": "fixed_biregular_instance",
            "bit_degree": 3,
            "check_degree": 4,
            "classical_parameters": (12, 3, 4),
            "reference": "https://arxiv.org/abs/2308.08648",
            "relationship_to_reference": "independent_parameter_matching_instance",
        },
    )


def hgp_225_9_4() -> HGPCode:
    """Build the reproducible ``[[225, 9, 4]]`` self-product HGP patch."""
    return HGPCode(hgp_225_9_4_seed(), d=4)


__all__ = [
    "hgp_13_1_3",
    "hgp_13_1_3_seed",
    "hgp_18_2_3",
    "hgp_18_2_3_seed",
    "hgp_225_9_4",
    "hgp_225_9_4_seed",
]
