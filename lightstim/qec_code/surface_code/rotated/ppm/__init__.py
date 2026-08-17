"""Rotated-surface-code backend for explicit-route PPM lowering."""

from .coupler import RotatedSurfacePPMCoupler
from .layout import MultiPatchLayout, SubsetRoute, build_explicit_ppm_layout
from .lowering import (
    RotatedSurfacePPMCertificate,
    RotatedSurfacePPMPlan,
    RotatedSurfacePPMRequest,
    UnsupportedPauliError,
    apply_ppm_plan,
    is_cell_adjacent_pair,
    lower_ppm,
)
from .placement import (
    RotatedSurfacePatchPlacement,
    RotatedSurfacePPMLayoutError,
    cell,
    cell_index,
    conjugate_patch_records,
    origin_of,
    place_patch,
)
from .seam_rules import (
    MergeRule,
    PatchView,
    RotatedSeamWallCoupler,
    SeamRuleError,
    WallSpec,
    classify_seam,
    patch_view,
    wall_spec,
)

__all__ = [
    "RotatedSurfacePatchPlacement",
    "RotatedSurfacePPMLayoutError",
    "cell",
    "cell_index",
    "conjugate_patch_records",
    "origin_of",
    "place_patch",
    "MergeRule",
    "PatchView",
    "RotatedSeamWallCoupler",
    "SeamRuleError",
    "WallSpec",
    "classify_seam",
    "patch_view",
    "wall_spec",
    "MultiPatchLayout",
    "SubsetRoute",
    "build_explicit_ppm_layout",
    "RotatedSurfacePPMCoupler",
    "RotatedSurfacePPMCertificate",
    "RotatedSurfacePPMPlan",
    "RotatedSurfacePPMRequest",
    "UnsupportedPauliError",
    "apply_ppm_plan",
    "is_cell_adjacent_pair",
    "lower_ppm",
]
