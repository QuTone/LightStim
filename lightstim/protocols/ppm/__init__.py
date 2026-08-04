# protocols/ppm/__init__.py
#
# Sequential PPM stack — joint Pauli-product measurements on rotated routed
# surface-code patches with explicit corridor routes.  Copied from
# https://github.com/John-YuehanZhang/CircLS @ 8802a5b (the author's own
# repository); minimal explicit-route variant, intentionally independent of
# CircLS.

from .spec import (BentLayoutError, PatchSpec, PPMStep, cell, cell_index,
                   conjugate_patch_records, origin_of, place_patch)
from .seam_rules import (MergeRule, PatchView, RotatedSeamWallCoupler,
                         SeamRuleError, WallSpec, classify_seam, patch_view,
                         wall_spec)
from .coupler import (MultiPatchLayout, RotatedRoutedMultiPatchCoupler,
                      SubsetRoute, route_and_build)
from .sequential import SequentialPPMExperiment

__all__ = [
    "BentLayoutError", "PatchSpec", "PPMStep", "cell", "cell_index",
    "conjugate_patch_records", "origin_of", "place_patch",
    "MergeRule", "PatchView", "RotatedSeamWallCoupler", "SeamRuleError",
    "WallSpec", "classify_seam", "patch_view", "wall_spec",
    "MultiPatchLayout", "RotatedRoutedMultiPatchCoupler", "SubsetRoute",
    "route_and_build", "SequentialPPMExperiment",
]
