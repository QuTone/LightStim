from typing import Tuple, List, Optional, Dict, Set, Iterable
from dataclasses import dataclass, field
from collections import deque
import math
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.coupler import LogicalCouplerProtocol
from .two_patch_coupler import UnrotatedTwoPatchCoupler

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
@dataclass
class InterfaceInfo:
    """Describes how one patch connects to the corridor."""
    patch: QECPatch
    side: str                    # 'left', 'right', 'top', 'bottom'
    boundary_edge_coord: float   # the patch edge coordinate facing the corridor

@dataclass
class PathInfo:
    """Complete description of the corridor geometry."""
    path_axis: str                          # 'vertical' or 'horizontal'
    corridor_bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max) — outer bounds
    anchor_patch: QECPatch                  # parity reference for role inference
    interfaces: List[InterfaceInfo] = field(default_factory=list)


@dataclass
class RoutedPathInfo(PathInfo):
    """PathInfo for a routed, possibly-bent ancillary region."""
    route_coords: Set[Tuple[float, float]] = field(default_factory=set)
    coord_basis: Dict[Tuple[float, float], str] = field(default_factory=dict)
    interface_bases: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Multi-Patch Coupler
# -----------------------------------------------------------------------------
class UnrotatedMultiPatchCoupler(LogicalCouplerProtocol):
    """
    Multi-patch lattice surgery coupler for Unrotated Surface Codes.

    Creates a one-way ancilla path (corridor) connecting N>=2 patches for
    Z-product measurements (ZZ, ZZZ, ZZZZ, ...).

    Required params:
        path_axis: 'vertical' or 'horizontal'

    Mode A — center_axis specified (no start/end):
        center_axis: float — x-coord (vertical) or y-coord (horizontal)
        Splits patches into two groups. Corridor fills the gap between groups.

    Mode B — start_patch specified (end_patch optional):
        start_patch: int — index into patches list (defines one end of path)
        end_patch: int — index into patches list (defines other end)
        Uses two-patch containment conditions. Remaining patches are side patches.
    """

    EXPECTED_PATCH_COUNT = None  # Variable number of patches

    def __init__(self):
        super().__init__(name_prefix="unrotated_multi_coupler")

    def _build_coupler_geometry(self, coupler_patch: QECPatch, patches: List[QECPatch], **params):
        if len(patches) < 2:
            raise ValueError(f"Multi-patch coupler requires at least 2 patches, got {len(patches)}.")

        path_axis = params.get('path_axis')
        if path_axis not in ('vertical', 'horizontal'):
            raise ValueError(f"path_axis is required and must be 'vertical' or 'horizontal', got '{path_axis}'.")

        start_idx = params.get('start_patch')
        end_idx = params.get('end_patch')
        center_axis = params.get('center_axis')

        if start_idx is not None:
            path_info = self._analyze_with_endpoints(patches, path_axis, start_idx, end_idx)
        elif center_axis is not None:
            path_info = self._analyze_with_center_axis(patches, path_axis, center_axis)
        else:
            raise ValueError("Must provide either 'start_patch' (index) or 'center_axis' (float).")

        # Inherit transposition/rotation from anchor so SE block uses consistent directions
        coupler_patch.is_transposed = path_info.anchor_patch.is_transposed
        coupler_patch.rotation_angle = path_info.anchor_patch.rotation_angle

        self._construct_coupling_region(coupler_patch, patches, path_info)
        self._init_stabilizers(coupler_patch, patches, path_info)

    # =========================================================================
    # Geometry Analysis — Mode A: start/end patches
    # =========================================================================
    def _analyze_with_endpoints(self, patches, path_axis, start_idx, end_idx) -> PathInfo:
        """
        Analyze geometry when start (and optionally end) patch indices are given.
        Uses two-patch coupler containment conditions for start/end.
        """
        start_patch = patches[start_idx]
        sb = start_patch._get_bounds()  # (min_x, max_x, min_y, max_y)

        if path_axis == 'vertical':
            return self._endpoints_vertical(patches, start_idx, end_idx, start_patch, sb)
        else:
            return self._endpoints_horizontal(patches, start_idx, end_idx, start_patch, sb)

    def _endpoints_vertical(self, patches, start_idx, end_idx, start_patch, sb):
        """Vertical path with start/end patches (patches stacked vertically, y disjoint, x contained)."""
        interfaces = []
        anchor_patch = start_patch

        if end_idx is not None:
            end_patch = patches[end_idx]
            eb = end_patch._get_bounds()

            # Check x containment (like two-patch coupler)
            s_contains_e = (sb[0] <= eb[0] and sb[1] >= eb[1])
            e_contains_s = (eb[0] <= sb[0] and eb[1] >= sb[1])
            if not (s_contains_e or e_contains_s):
                raise ValueError("Start/end patches must have x-range containment for vertical path.")

            # Anchor = smaller x-range
            if e_contains_s:
                anchor_patch = start_patch
            else:
                anchor_patch = end_patch

            # Check y disjoint
            if sb[3] < eb[2]:
                gap_y_min, gap_y_max = sb[3], eb[2]
            elif eb[3] < sb[2]:
                gap_y_min, gap_y_max = eb[3], sb[2]
            else:
                raise ValueError("Start/end patches must have disjoint y-ranges for vertical path.")

            # Corridor x bounds = narrowest x overlap
            corr_x_min = max(sb[0], eb[0])
            corr_x_max = min(sb[1], eb[1])

            # Interfaces for start/end
            if sb[3] <= eb[2]:
                interfaces.append(InterfaceInfo(patch=start_patch, side='top', boundary_edge_coord=sb[3]))
                interfaces.append(InterfaceInfo(patch=end_patch, side='bottom', boundary_edge_coord=eb[2]))
            else:
                interfaces.append(InterfaceInfo(patch=start_patch, side='bottom', boundary_edge_coord=sb[2]))
                interfaces.append(InterfaceInfo(patch=end_patch, side='top', boundary_edge_coord=eb[3]))

            corr_y_min = gap_y_min
            corr_y_max = gap_y_max

        else:
            # Only start patch — corridor extends to cover side patches
            corr_x_min = sb[0]
            corr_x_max = sb[1]

            # First pass: find side patches to determine corridor direction
            side_centroids_y = []
            for i, p in enumerate(patches):
                if i == start_idx:
                    continue
                pb = p._get_bounds()
                side_centroids_y.append((pb[2] + pb[3]) / 2)

            start_cy = (sb[2] + sb[3]) / 2
            avg_side_cy = sum(side_centroids_y) / len(side_centroids_y) if side_centroids_y else start_cy

            if start_cy > avg_side_cy:
                # Start patch is BELOW side patches → corridor extends UPWARD
                # Interface is start patch's TOP edge (y_min)
                corr_y_max = sb[2]  # corridor ends at start patch's top edge
                corr_y_min = sb[2]  # will be reduced by side patches
                interfaces.append(InterfaceInfo(patch=start_patch, side='bottom', boundary_edge_coord=sb[2]))
            else:
                # Start patch is ABOVE side patches → corridor extends DOWNWARD
                # Interface is start patch's BOTTOM edge (y_max)
                corr_y_min = sb[3]  # corridor starts at start patch's bottom edge
                corr_y_max = sb[3]  # will be extended by side patches
                interfaces.append(InterfaceInfo(patch=start_patch, side='top', boundary_edge_coord=sb[3]))

        # Classify remaining patches as side patches
        endpoint_indices = {start_idx}
        if end_idx is not None:
            endpoint_indices.add(end_idx)

        for i, p in enumerate(patches):
            if i in endpoint_indices:
                continue
            pb = p._get_bounds()

            # Determine side (left or right of corridor) — must be strictly separated
            if pb[1] < corr_x_min - 1e-3:
                interfaces.append(InterfaceInfo(patch=p, side='left', boundary_edge_coord=pb[1]))
            elif pb[0] > corr_x_max + 1e-3:
                interfaces.append(InterfaceInfo(patch=p, side='right', boundary_edge_coord=pb[0]))
            else:
                raise ValueError(
                    f"Side patch at bounds {pb} overlaps or touches corridor x-range [{corr_x_min}, {corr_x_max}]. "
                    f"Side patches must be strictly separated (gap >= 1) from the corridor.")

            # Extend corridor y-range to cover this side patch
            corr_y_min = min(corr_y_min, pb[2])
            corr_y_max = max(corr_y_max, pb[3])

        # Validate side patches
        self._validate_side_patches_vertical(interfaces, corr_y_min, corr_y_max, anchor_patch)

        corridor_bounds = (corr_x_min, corr_x_max, corr_y_min, corr_y_max)
        return PathInfo(path_axis='vertical', corridor_bounds=corridor_bounds,
                        anchor_patch=anchor_patch, interfaces=interfaces)

    def _endpoints_horizontal(self, patches, start_idx, end_idx, start_patch, sb):
        """Horizontal path with start/end patches (x disjoint, y contained)."""
        interfaces = []
        anchor_patch = start_patch

        if end_idx is not None:
            end_patch = patches[end_idx]
            eb = end_patch._get_bounds()

            s_contains_e = (sb[2] <= eb[2] and sb[3] >= eb[3])
            e_contains_s = (eb[2] <= sb[2] and eb[3] >= sb[3])
            if not (s_contains_e or e_contains_s):
                raise ValueError("Start/end patches must have y-range containment for horizontal path.")

            if e_contains_s:
                anchor_patch = start_patch
            else:
                anchor_patch = end_patch

            if sb[1] < eb[0]:
                gap_x_min, gap_x_max = sb[1], eb[0]
            elif eb[1] < sb[0]:
                gap_x_min, gap_x_max = eb[1], sb[0]
            else:
                raise ValueError("Start/end patches must have disjoint x-ranges for horizontal path.")

            corr_y_min = max(sb[2], eb[2])
            corr_y_max = min(sb[3], eb[3])

            if sb[1] <= eb[0]:
                interfaces.append(InterfaceInfo(patch=start_patch, side='left', boundary_edge_coord=sb[1]))
                interfaces.append(InterfaceInfo(patch=end_patch, side='right', boundary_edge_coord=eb[0]))
            else:
                interfaces.append(InterfaceInfo(patch=start_patch, side='right', boundary_edge_coord=sb[0]))
                interfaces.append(InterfaceInfo(patch=end_patch, side='left', boundary_edge_coord=eb[1]))

            corr_x_min = gap_x_min
            corr_x_max = gap_x_max
        else:
            corr_y_min = sb[2]
            corr_y_max = sb[3]
            corr_x_min = sb[1]
            corr_x_max = sb[1]
            interfaces.append(InterfaceInfo(patch=start_patch, side='left', boundary_edge_coord=sb[1]))

        endpoint_indices = {start_idx}
        if end_idx is not None:
            endpoint_indices.add(end_idx)

        for i, p in enumerate(patches):
            if i in endpoint_indices:
                continue
            pb = p._get_bounds()
            if pb[3] < corr_y_min - 1e-3:
                interfaces.append(InterfaceInfo(patch=p, side='top', boundary_edge_coord=pb[3]))
            elif pb[2] > corr_y_max + 1e-3:
                interfaces.append(InterfaceInfo(patch=p, side='bottom', boundary_edge_coord=pb[2]))
            else:
                raise ValueError(
                    f"Side patch at bounds {pb} overlaps or touches corridor y-range [{corr_y_min}, {corr_y_max}]. "
                    f"Side patches must be strictly separated (gap >= 1) from the corridor.")
            corr_x_min = min(corr_x_min, pb[0])
            corr_x_max = max(corr_x_max, pb[1])

        self._validate_side_patches_horizontal(interfaces, corr_x_min, corr_x_max, anchor_patch)

        corridor_bounds = (corr_x_min, corr_x_max, corr_y_min, corr_y_max)
        return PathInfo(path_axis='horizontal', corridor_bounds=corridor_bounds,
                        anchor_patch=anchor_patch, interfaces=interfaces)

    # =========================================================================
    # Geometry Analysis — Mode B: center_axis
    # =========================================================================
    def _analyze_with_center_axis(self, patches, path_axis, center_axis) -> PathInfo:
        if path_axis == 'vertical':
            return self._center_axis_vertical(patches, center_axis)
        else:
            return self._center_axis_horizontal(patches, center_axis)

    def _center_axis_vertical(self, patches, center_x) -> PathInfo:
        """
        Vertical corridor defined by center_axis splitting patches left/right.

        Patches whose x_range straddles the center_axis are classified as
        top/bottom endpoints (they sit within the corridor, not on the side).
        """
        left_patches = []  # (index, patch, bounds)
        right_patches = []
        endpoint_patches = []  # patches that straddle the center axis

        for i, p in enumerate(patches):
            b = p._get_bounds()
            # Check if patch x_range contains the center_axis
            if b[0] < center_x - 1e-3 and b[1] > center_x + 1e-3:
                # Patch straddles the center axis → endpoint (top or bottom)
                endpoint_patches.append((i, p, b))
            else:
                cx = (b[0] + b[1]) / 2
                if cx < center_x - 1e-3:
                    left_patches.append((i, p, b))
                elif cx > center_x + 1e-3:
                    right_patches.append((i, p, b))
                else:
                    raise ValueError(f"Patch {i} centroid x={cx} is on the center_axis={center_x} "
                                     f"but does not straddle it. Cannot classify.")

        if not left_patches and not right_patches:
            raise ValueError("No patches on either side of center_axis.")

        # Corridor interior x bounds: one step inside from side patch edges
        if left_patches:
            corr_x_min = max(b[1] for _, _, b in left_patches) + 1.0
        else:
            corr_x_min = center_x

        if right_patches:
            corr_x_max = min(b[0] for _, _, b in right_patches) - 1.0
        else:
            corr_x_max = center_x

        if corr_x_min > corr_x_max + 1e-3:
            raise ValueError(f"No room for corridor interior: left_interior={corr_x_min}, right_interior={corr_x_max}. "
                             f"Need gap >= 3 between left/right patch edges.")

        # Validate endpoint patches sit within corridor x_range
        for i, p, b in endpoint_patches:
            if b[0] < corr_x_min - 1e-3 or b[1] > corr_x_max + 1e-3:
                raise ValueError(
                    f"Endpoint patch {i} x_range [{b[0]}, {b[1]}] exceeds corridor x_range "
                    f"[{corr_x_min}, {corr_x_max}]. Adjust side patch positions to widen corridor.")

        # Corridor y bounds from side patches
        side_bounds = [b for _, _, b in left_patches + right_patches]
        corr_y_min = min(b[2] for b in side_bounds)
        corr_y_max = max(b[3] for b in side_bounds)

        # Extend corridor y to reach endpoint patches (with gap=1)
        for i, p, b in endpoint_patches:
            cy = (b[2] + b[3]) / 2
            side_cy = (corr_y_min + corr_y_max) / 2
            if cy < side_cy:
                # Endpoint is above side patches
                corr_y_min = min(corr_y_min, b[3])  # corridor extends up to endpoint's bottom edge
            else:
                # Endpoint is below side patches
                corr_y_max = max(corr_y_max, b[2])  # corridor extends down to endpoint's top edge

        # Build interfaces
        interfaces = []
        for _, p, b in left_patches:
            interfaces.append(InterfaceInfo(patch=p, side='left', boundary_edge_coord=b[1]))
        for _, p, b in right_patches:
            interfaces.append(InterfaceInfo(patch=p, side='right', boundary_edge_coord=b[0]))
        for _, p, b in endpoint_patches:
            cy = (b[2] + b[3]) / 2
            side_cy = (corr_y_min + corr_y_max) / 2
            if cy < side_cy:
                interfaces.append(InterfaceInfo(patch=p, side='top', boundary_edge_coord=b[3]))
            else:
                interfaces.append(InterfaceInfo(patch=p, side='bottom', boundary_edge_coord=b[2]))

        # Select anchor (smallest patch)
        anchor_patch = min(patches, key=lambda p: (p._get_bounds()[1] - p._get_bounds()[0]) * (p._get_bounds()[3] - p._get_bounds()[2]))

        # Validate
        self._validate_side_patches_vertical(interfaces, corr_y_min, corr_y_max, anchor_patch)

        corridor_bounds = (corr_x_min, corr_x_max, corr_y_min, corr_y_max)
        return PathInfo(path_axis='vertical', corridor_bounds=corridor_bounds,
                        anchor_patch=anchor_patch, interfaces=interfaces)

    def _center_axis_horizontal(self, patches, center_y) -> PathInfo:
        """Horizontal corridor defined by center_axis splitting patches top/bottom.
        Patches straddling the center_axis become left/right endpoints."""
        top_patches = []
        bottom_patches = []
        endpoint_patches = []

        for i, p in enumerate(patches):
            b = p._get_bounds()
            if b[2] < center_y - 1e-3 and b[3] > center_y + 1e-3:
                endpoint_patches.append((i, p, b))
            else:
                cy = (b[2] + b[3]) / 2
                if cy < center_y - 1e-3:
                    top_patches.append((i, p, b))
                elif cy > center_y + 1e-3:
                    bottom_patches.append((i, p, b))
                else:
                    raise ValueError(f"Patch {i} centroid y={cy} is on the center_axis={center_y} "
                                     f"but does not straddle it. Cannot classify.")

        if not top_patches and not bottom_patches:
            raise ValueError("No patches on either side of center_axis.")

        if top_patches:
            corr_y_min = max(b[3] for _, _, b in top_patches) + 1.0
        else:
            corr_y_min = center_y
        if bottom_patches:
            corr_y_max = min(b[2] for _, _, b in bottom_patches) - 1.0
        else:
            corr_y_max = center_y

        if corr_y_min > corr_y_max + 1e-3:
            raise ValueError(f"No room for corridor interior: top_interior={corr_y_min}, bottom_interior={corr_y_max}. "
                             f"Need gap >= 3 between top/bottom patch edges.")

        for i, p, b in endpoint_patches:
            if b[2] < corr_y_min - 1e-3 or b[3] > corr_y_max + 1e-3:
                raise ValueError(
                    f"Endpoint patch {i} y_range [{b[2]}, {b[3]}] exceeds corridor y_range "
                    f"[{corr_y_min}, {corr_y_max}]. Adjust side patch positions.")

        side_bounds = [b for _, _, b in top_patches + bottom_patches]
        corr_x_min = min(b[0] for b in side_bounds)
        corr_x_max = max(b[1] for b in side_bounds)

        for i, p, b in endpoint_patches:
            cx = (b[0] + b[1]) / 2
            side_cx = (corr_x_min + corr_x_max) / 2
            if cx < side_cx:
                corr_x_min = min(corr_x_min, b[1])
            else:
                corr_x_max = max(corr_x_max, b[0])

        interfaces = []
        for _, p, b in top_patches:
            interfaces.append(InterfaceInfo(patch=p, side='top', boundary_edge_coord=b[3]))
        for _, p, b in bottom_patches:
            interfaces.append(InterfaceInfo(patch=p, side='bottom', boundary_edge_coord=b[2]))
        for _, p, b in endpoint_patches:
            cx = (b[0] + b[1]) / 2
            side_cx = (corr_x_min + corr_x_max) / 2
            if cx < side_cx:
                interfaces.append(InterfaceInfo(patch=p, side='left', boundary_edge_coord=b[1]))
            else:
                interfaces.append(InterfaceInfo(patch=p, side='right', boundary_edge_coord=b[0]))

        anchor_patch = min(patches, key=lambda p: (p._get_bounds()[1] - p._get_bounds()[0]) * (p._get_bounds()[3] - p._get_bounds()[2]))
        self._validate_side_patches_horizontal(interfaces, corr_x_min, corr_x_max, anchor_patch)

        corridor_bounds = (corr_x_min, corr_x_max, corr_y_min, corr_y_max)
        return PathInfo(path_axis='horizontal', corridor_bounds=corridor_bounds,
                        anchor_patch=anchor_patch, interfaces=interfaces)

    # =========================================================================
    # Validation
    # =========================================================================
    def _validate_side_patches_vertical(self, interfaces, corr_y_min, corr_y_max, anchor_patch):
        """Validate side patches for a vertical corridor."""
        left_y_ranges = []
        right_y_ranges = []

        for iface in interfaces:
            if iface.side in ('top', 'bottom'):
                continue  # endpoint patches, not side patches
            b = iface.patch._get_bounds()
            y_range = (b[2], b[3])

            # Check y_range contained in corridor
            if y_range[0] < corr_y_min - 1e-3 or y_range[1] > corr_y_max + 1e-3:
                raise ValueError(
                    f"Side patch y-range [{y_range[0]}, {y_range[1]}] exceeds corridor y-range [{corr_y_min}, {corr_y_max}].")

            if iface.side == 'left':
                left_y_ranges.append(y_range)
            elif iface.side == 'right':
                right_y_ranges.append(y_range)

            # Parity alignment check
            self._check_parity_alignment(iface.patch, anchor_patch)

        # Check pairwise disjoint within each side
        self._check_disjoint_ranges(left_y_ranges, "left")
        self._check_disjoint_ranges(right_y_ranges, "right")

    def _validate_side_patches_horizontal(self, interfaces, corr_x_min, corr_x_max, anchor_patch):
        """Validate side patches for a horizontal corridor."""
        top_x_ranges = []
        bottom_x_ranges = []

        for iface in interfaces:
            if iface.side in ('left', 'right'):
                continue
            b = iface.patch._get_bounds()
            x_range = (b[0], b[1])

            if x_range[0] < corr_x_min - 1e-3 or x_range[1] > corr_x_max + 1e-3:
                raise ValueError(
                    f"Side patch x-range [{x_range[0]}, {x_range[1]}] exceeds corridor x-range [{corr_x_min}, {corr_x_max}].")

            if iface.side == 'top':
                top_x_ranges.append(x_range)
            elif iface.side == 'bottom':
                bottom_x_ranges.append(x_range)

            self._check_parity_alignment(iface.patch, anchor_patch)

        self._check_disjoint_ranges(top_x_ranges, "top")
        self._check_disjoint_ranges(bottom_x_ranges, "bottom")

    @staticmethod
    def _check_disjoint_ranges(ranges, side_name):
        """Check that ranges are pairwise disjoint."""
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] > sorted_ranges[i + 1][0] + 1e-3:
                raise ValueError(
                    f"Overlapping y/x-ranges on {side_name} side: "
                    f"[{sorted_ranges[i][0]}, {sorted_ranges[i][1]}] and "
                    f"[{sorted_ranges[i+1][0]}, {sorted_ranges[i+1][1]}].")

    @staticmethod
    def _check_parity_alignment(side_patch, anchor_patch):
        """
        Check that a side patch is lattice-aligned with the anchor (corridor).

        Uses same-type syndrome qubits (Z or X) from both patches. If delta_x and
        delta_y between a pair of same-type syndromes are both even, the lattices
        are aligned. This is robust to patch rotation (unlike data-qubit-based checks).
        """
        # Try Z syndromes first, fall back to X
        side_z = getattr(side_patch, 'syndrome_coords_z', [])
        anchor_z = getattr(anchor_patch, 'syndrome_coords_z', [])
        side_x = getattr(side_patch, 'syndrome_coords_x', [])
        anchor_x = getattr(anchor_patch, 'syndrome_coords_x', [])

        if side_z and anchor_z:
            s_coord = side_z[0]
            a_coord = anchor_z[0]
            label = "Z-syndrome"
        elif side_x and anchor_x:
            s_coord = side_x[0]
            a_coord = anchor_x[0]
            label = "X-syndrome"
        else:
            return  # Can't check

        dx = s_coord[0] - a_coord[0]
        dy = s_coord[1] - a_coord[1]

        if not (math.isclose(dx % 2, 0, abs_tol=1e-3) and math.isclose(dy % 2, 0, abs_tol=1e-3)):
            raise ValueError(
                f"Parity mismatch: {label} at {s_coord} (side) vs {a_coord} (anchor). "
                f"Delta=({dx}, {dy}) — both must be even. Adjust patch offset or rotation.")

    # =========================================================================
    # Construction
    # =========================================================================
    def _construct_coupling_region(self, coupler_patch: QECPatch, patches: List[QECPatch], path_info: PathInfo):
        """Fill the corridor and extend to reach side patches."""
        anchor = path_info.anchor_patch
        grid_step = 1.0
        gx_min, gx_max, gy_min, gy_max = path_info.corridor_bounds

        # Interior range (the main corridor strip)
        if path_info.path_axis == 'vertical':
            # Fill main corridor — full x_range, _coord_in_any_patch skips code patch qubits
            current_y = gy_min
            while current_y <= gy_max + 1e-3:
                current_x = gx_min
                while current_x <= gx_max + 1e-3:
                    if not self._coord_in_any_patch(current_x, current_y, patches):
                        role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor, current_x, current_y)
                        if role:
                            coupler_patch.add_qubit(current_x, current_y, role=role)
                    current_x += grid_step
                current_y += grid_step

            # Extend to side patches that are not adjacent to corridor
            for iface in path_info.interfaces:
                if iface.side == 'left':
                    self._extend_to_side_patch(coupler_patch, patches, anchor, iface,
                                               from_x=gx_min, to_x=iface.boundary_edge_coord,
                                               direction='left', grid_step=grid_step)
                elif iface.side == 'right':
                    self._extend_to_side_patch(coupler_patch, patches, anchor, iface,
                                               from_x=gx_max, to_x=iface.boundary_edge_coord,
                                               direction='right', grid_step=grid_step)

        else:  # horizontal
            current_x = gx_min
            while current_x <= gx_max + 1e-3:
                current_y = gy_min
                while current_y <= gy_max + 1e-3:
                    if not self._coord_in_any_patch(current_x, current_y, patches):
                        role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor, current_x, current_y)
                        if role:
                            coupler_patch.add_qubit(current_x, current_y, role=role)
                    current_y += grid_step
                current_x += grid_step

            for iface in path_info.interfaces:
                if iface.side == 'top':
                    self._extend_to_side_patch(coupler_patch, patches, anchor, iface,
                                               from_x=gy_min, to_x=iface.boundary_edge_coord,
                                               direction='top', grid_step=grid_step)
                elif iface.side == 'bottom':
                    self._extend_to_side_patch(coupler_patch, patches, anchor, iface,
                                               from_x=gy_max, to_x=iface.boundary_edge_coord,
                                               direction='bottom', grid_step=grid_step)

    def _extend_to_side_patch(self, coupler_patch, patches, anchor, iface,
                               from_x, to_x, direction, grid_step):
        """
        Fill the gap between the main corridor edge and a side patch boundary.
        For 'left'/'right': fill columns between from_x and to_x at the patch's y_range.
        For 'top'/'bottom': fill rows between from_x and to_x at the patch's x_range.
        """
        pb = iface.patch._get_bounds()

        if direction in ('left', 'right'):
            # Fill between corridor edge (from_x) and patch edge (to_x)
            # Corridor edge is already filled; patch edge is not (it belongs to the patch)
            # Fill the gap: x from to_x+1 to from_x-1 (or from_x+1 to to_x-1)
            x_start = min(from_x, to_x) + 1.0
            x_end = max(from_x, to_x) - 1.0

            if x_start > x_end + 1e-3:
                return  # Adjacent, no gap to fill

            y_start = pb[2]
            y_end = pb[3]

            current_y = y_start
            while current_y <= y_end + 1e-3:
                current_x = x_start
                while current_x <= x_end + 1e-3:
                    if not self._coord_in_any_patch(current_x, current_y, patches):
                        role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor, current_x, current_y)
                        if role:
                            coupler_patch.add_qubit(current_x, current_y, role=role)
                    current_x += grid_step
                current_y += grid_step

        elif direction in ('top', 'bottom'):
            y_start = min(from_x, to_x) + 1.0
            y_end = max(from_x, to_x) - 1.0

            if y_start > y_end + 1e-3:
                return

            x_start = pb[0]
            x_end = pb[1]

            current_x = x_start
            while current_x <= x_end + 1e-3:
                current_y = y_start
                while current_y <= y_end + 1e-3:
                    if not self._coord_in_any_patch(current_x, current_y, patches):
                        role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor, current_x, current_y)
                        if role:
                            coupler_patch.add_qubit(current_x, current_y, role=role)
                    current_y += grid_step
                current_x += grid_step

    @staticmethod
    def _coord_in_any_patch(x: float, y: float, patches: List[QECPatch]) -> bool:
        """Check if a coordinate is already owned by any code patch."""
        for p in patches:
            if (x, y) in p.index_map:
                return True
        return False

    # =========================================================================
    # Stabilizer Initialization
    # =========================================================================
    def _init_stabilizers(self, coupler_patch: QECPatch, patches: List[QECPatch], path_info: PathInfo):
        """Two-phase stabilizer construction, generalized for N patches."""
        coupler_patch.conflicting_stabilizer_coords = set()

        # Phase 1: Gap-internal syndrome qubits (new qubits in coupler)
        for uid in coupler_patch.syndrome_indices:
            syn_coord = coupler_patch.qubit_coords[uid]
            if uid in coupler_patch.syndrome_indices_x:
                stype = 'X'
            elif uid in coupler_patch.syndrome_indices_z:
                stype = 'Z'
            else:
                raise ValueError(f"Syndrome qubit {uid} has undefined type.")
            self._probe_and_create_stabilizer(coupler_patch, patches, syn_coord, stype)

        # Phase 2: Boundary syndrome qubits (existing qubits in code patches)
        boundary_candidates = self._find_boundary_syndrome_candidates(patches, path_info)
        for syn_coord in boundary_candidates:
            stype = UnrotatedTwoPatchCoupler._resolve_existing_syndrome_type(patches, syn_coord)
            if not stype:
                continue
            success = self._probe_and_create_stabilizer(coupler_patch, patches, syn_coord, stype)
            if success:
                coupler_patch.conflicting_stabilizer_coords.add(syn_coord)

    def _probe_and_create_stabilizer(self, coupler_patch, patches, syn_coord, stype) -> bool:
        """Probes 4 directions from a syndrome coordinate, finds data neighbors, creates stabilizer."""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            tx, ty = syn_coord[0] + dx, syn_coord[1] + dy
            if UnrotatedTwoPatchCoupler._is_data_qubit_at(coupler_patch, patches, tx, ty):
                neighbors.append((tx, ty))
        if neighbors:
            coupler_patch.stabilizers.append({
                'pauli': {coord: stype for coord in neighbors},
                'type': stype,
                'syn_coord': syn_coord,
            })
            return True
        return False

    def _find_boundary_syndrome_candidates(self, patches, path_info) -> List[Tuple[float, float]]:
        """Find existing syndrome qubits on each patch's boundary edge facing the corridor."""
        candidates = []
        for iface in path_info.interfaces:
            edge = iface.boundary_edge_coord
            for coord in iface.patch.syndrome_coords:
                x, y = coord
                if iface.side in ('left', 'right'):
                    if math.isclose(x, edge, abs_tol=1e-3):
                        candidates.append(coord)
                elif iface.side in ('top', 'bottom'):
                    if math.isclose(y, edge, abs_tol=1e-3):
                        candidates.append(coord)
        return candidates


class UnrotatedRoutedMultiPatchCoupler(UnrotatedMultiPatchCoupler):
    """
    Routed multi-patch coupler for Unrotated Surface Codes.

    Unlike ``UnrotatedMultiPatchCoupler``, this protocol does not require a
    single vertical/horizontal corridor.  It builds a rectilinear Manhattan
    ancillary region that can bend while connecting explicitly selected patch
    boundary sides.

    Required params:
        sides: list[str]
            One boundary side per patch, each in {'left', 'right', 'top', 'bottom'}.

    Optional params:
        interface_paulis: list[str] = None
            Native Pauli basis of each selected interface.  If omitted, inferred
            from the selected side and the patch logical operator orientations.
        target_paulis: list[str] = None
            Requested logical Pauli product.  Stored for introspection; basis
            changes are handled by higher-level builders.
        mixed_stabilizers: bool = False
            If true, routed stabilizers infer Pauli labels locally from the
            X/Z-labeled route and may produce mixed XZ checks at seams/corners.
        route_width: int = 2 * min patch distance - 1
            Filled width of the ancillary patch in integer lattice-coordinate
            rows/columns.  Defaults to the full coordinate span of a
            distance-d unrotated patch, not the logical distance d itself.
        route_padding: int = 4
            Extra grid margin around terminals during routing.
        obstacle_patches: list[QECPatch] = []
            Additional patches that the route must not cross.  Participating
            patches are always treated as obstacles.
        route_order: list[int] = None
            Order in which terminals are connected.  Defaults to patch order.

    Notes:
        The routed region is a geometry generator.  Arbitrary turns/endpoints can
        still require a syndrome-extraction schedule that avoids X/Z CNOT
        conflicts.  Always validate generated circuits with a detector error
        model for the intended layout.
    """

    def __init__(self):
        super().__init__()
        self.name_prefix = "unrotated_routed_multi_coupler"

    def _build_coupler_geometry(self, coupler_patch: QECPatch, patches: List[QECPatch], **params):
        if len(patches) < 2:
            raise ValueError(f"Routed multi-patch coupler requires at least 2 patches, got {len(patches)}.")

        sides = params.get('sides') or params.get('boundary_sides')
        if sides is None:
            sides = self._auto_select_sides(patches)
        if len(sides) != len(patches):
            raise ValueError(f"Expected {len(patches)} boundary sides, got {len(sides)}.")

        sides = [self._normalize_side(s) for s in sides]
        interface_bases = params.get('interface_paulis') or params.get('interface_bases')
        if interface_bases is None:
            interface_bases = [
                self.infer_side_basis(patch, side)
                for patch, side in zip(patches, sides)
            ]
        interface_bases = [self._normalize_pauli(p) for p in interface_bases]
        if len(interface_bases) != len(patches):
            raise ValueError(f"Expected {len(patches)} interface paulis, got {len(interface_bases)}.")

        target_paulis = params.get('target_paulis')
        if target_paulis is not None:
            target_paulis = [self._normalize_pauli(p) for p in target_paulis]
            if len(target_paulis) != len(patches):
                raise ValueError(f"Expected {len(patches)} target paulis, got {len(target_paulis)}.")

        mixed_stabilizers = bool(params.get('mixed_stabilizers', False))
        route_width = int(params.get(
            'route_width',
            params.get('ancilla_distance', self._infer_route_width(patches)),
        ))
        if route_width < 1:
            raise ValueError(f"route_width must be positive, got {route_width}.")
        route_padding = int(params.get('route_padding', 4))
        route_order = params.get('route_order')
        if route_order is None:
            route_order = list(range(len(patches)))
        if sorted(route_order) != list(range(len(patches))):
            raise ValueError("route_order must be a permutation of patch indices.")

        obstacle_patches = list(params.get('obstacle_patches') or [])
        all_obstacles = self._dedupe_patches([*patches, *obstacle_patches])

        anchor_patch = params.get('anchor_patch')
        if anchor_patch is None:
            anchor_patch = min(
                patches,
                key=lambda p: (p._get_bounds()[1] - p._get_bounds()[0]) *
                              (p._get_bounds()[3] - p._get_bounds()[2]),
            )

        self._validate_integer_lattice(all_obstacles)

        route_coords, interfaces, strips, already_full_width = self._route_interfaces(
            patches=patches,
            sides=sides,
            route_order=route_order,
            obstacle_patches=all_obstacles,
            padding=route_padding,
            route_width=route_width,
        )
        if not already_full_width:
            route_coords = self._thicken_route_coords(
                route_coords=route_coords,
                strips=strips,
                sides=sides,
                obstacle_patches=all_obstacles,
                route_width=route_width,
            )
        coord_basis = self._label_route_coords(route_coords, strips, interface_bases)

        bounds = self._coords_bounds(route_coords)
        path_info = RoutedPathInfo(
            path_axis='routed',
            corridor_bounds=bounds,
            anchor_patch=anchor_patch,
            interfaces=interfaces,
            route_coords=route_coords,
            coord_basis=coord_basis,
            interface_bases=interface_bases,
        )

        coupler_patch.is_transposed = anchor_patch.is_transposed
        coupler_patch.rotation_angle = anchor_patch.rotation_angle
        coupler_patch.interface_bases = list(interface_bases)
        coupler_patch.target_paulis = list(target_paulis) if target_paulis is not None else None
        coupler_patch.boundary_sides = list(sides)
        coupler_patch.route_width = route_width
        coupler_patch.route_coord_basis = dict(coord_basis)

        self._construct_routed_region(coupler_patch, all_obstacles, anchor_patch, route_coords)
        if mixed_stabilizers:
            self._init_routed_stabilizers(coupler_patch, patches, path_info)
        else:
            self._init_stabilizers(coupler_patch, patches, path_info)
        if any(stab.get('type') == 'MIXED' for stab in coupler_patch.stabilizers):
            self._mark_anticommuting_patch_stabilizers(coupler_patch, patches)
        self._validate_routed_connectivity(coupler_patch, patches, path_info)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_interfaces(
        self,
        patches: List[QECPatch],
        sides: List[str],
        route_order: List[int],
        obstacle_patches: List[QECPatch],
        padding: int,
        route_width: int,
    ) -> Tuple[Set[Tuple[float, float]], List[InterfaceInfo], List[Set[Tuple[float, float]]], bool]:
        obstacle_coords = self._patch_coords(obstacle_patches)
        interfaces: List[InterfaceInfo] = []
        strips: List[Set[Tuple[float, float]]] = []

        for patch, side in zip(patches, sides):
            iface = self._make_interface(patch, side)
            strip = self._interface_strip(patch, side)
            blocked = sorted(strip & obstacle_coords)
            if blocked:
                raise ValueError(
                    f"Interface strip for patch '{getattr(patch, 'name', '<unnamed>')}' on side '{side}' "
                    f"collides with existing patch coordinates, e.g. {blocked[:4]}."
                )
            interfaces.append(iface)
            strips.append(strip)
            self._check_parity_alignment(patch, patches[route_order[0]])

        bounds = self._routing_bounds(strips, obstacle_patches, padding)
        ordered_strips = [strips[i] for i in route_order]
        if route_width <= 1:
            route_coords: Set[Tuple[float, float]] = set(ordered_strips[0])
            for target_strip in ordered_strips[1:]:
                path = self._bfs_path(
                    sources=route_coords,
                    goals=target_strip,
                    blocked=obstacle_coords,
                    bounds=bounds,
                )
                route_coords.update(path)
                route_coords.update(target_strip)
            return route_coords, interfaces, strips, False

        base_x, base_y, occupied_cells = self._validate_patch_block_grid(
            obstacle_patches,
            route_width,
        )
        terminal_cells = [
            self._terminal_cell_for_side(patch, side, base_x, base_y, route_width)
            for patch, side in zip(patches, sides)
        ]
        for patch, side, terminal in zip(patches, sides, terminal_cells):
            if terminal in occupied_cells:
                raise ValueError(
                    f"Interface terminal for patch '{getattr(patch, 'name', '<unnamed>')}' "
                    f"on side '{side}' overlaps an occupied data patch block {terminal}."
                )

        bounds = self._coarse_routing_bounds(
            terminal_cells=terminal_cells,
            occupied_cells=occupied_cells,
            padding_blocks=max(1, math.ceil(padding / route_width) + 1),
        )
        ordered_terminals = [terminal_cells[i] for i in route_order]
        route_cells: Set[Tuple[int, int]] = {ordered_terminals[0]}
        for target_cell in ordered_terminals[1:]:
            path = self._bfs_cell_path(
                sources=route_cells,
                goals={target_cell},
                blocked=occupied_cells,
                bounds=bounds,
            )
            route_cells.update(path)
            route_cells.add(target_cell)

        route_coords = self._expand_coarse_cells(route_cells, base_x, base_y, route_width)
        return route_coords, interfaces, strips, True

    @staticmethod
    def _normalize_side(side: str) -> str:
        normalized = str(side).lower()
        if normalized not in ('left', 'right', 'top', 'bottom'):
            raise ValueError(f"Invalid boundary side '{side}'. Expected left/right/top/bottom.")
        return normalized

    @staticmethod
    def _normalize_pauli(pauli: str) -> str:
        normalized = str(pauli).upper()
        if normalized not in ('X', 'Z'):
            raise ValueError(f"Invalid Pauli '{pauli}'. Only X/Z routed products are supported.")
        return normalized

    @classmethod
    def infer_side_basis(cls, patch: QECPatch, side: str) -> str:
        """
        Infer whether a selected global boundary side is an X or Z logical interface.

        The inference compares the geometric orientation of the boundary data
        line with the orientations of the patch's stored logical X and Z
        representatives.  This handles opposite boundaries of the same type
        (e.g. left and right are both X-oriented in the default unrotated patch).
        """
        side = cls._normalize_side(side)
        side_axis = cls._side_axis(patch, side)
        logical_x_axis = cls._logical_axis(patch, 'X')
        logical_z_axis = cls._logical_axis(patch, 'Z')

        if side_axis == logical_x_axis and side_axis != logical_z_axis:
            return 'X'
        if side_axis == logical_z_axis and side_axis != logical_x_axis:
            return 'Z'

        # Square symmetric or degenerate fallback: use overlap on the selected
        # boundary first, then the unrotated global convention.
        boundary = cls._boundary_data_coords(patch, side)
        x_coords = cls._logical_coords(patch, 'X')
        z_coords = cls._logical_coords(patch, 'Z')
        x_overlap = len(boundary & x_coords)
        z_overlap = len(boundary & z_coords)
        if x_overlap > z_overlap:
            return 'X'
        if z_overlap > x_overlap:
            return 'Z'
        return 'X' if side in ('left', 'right') else 'Z'

    @staticmethod
    def _logical_coords(patch: QECPatch, pauli: str) -> Set[Tuple[float, float]]:
        coords = set()
        for op in getattr(patch, 'logical_ops', []):
            if op.get('type') != pauli:
                continue
            for idx in op.get('pauli', {}):
                if idx in patch.qubit_coords:
                    coords.add(patch.qubit_coords[idx])
            break
        return coords

    @classmethod
    def _logical_axis(cls, patch: QECPatch, pauli: str) -> str:
        coords = cls._logical_coords(patch, pauli)
        if not coords:
            return 'unknown'
        xs = {c[0] for c in coords}
        ys = {c[1] for c in coords}
        if len(xs) == 1 and len(ys) > 1:
            return 'vertical'
        if len(ys) == 1 and len(xs) > 1:
            return 'horizontal'
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        return 'horizontal' if x_span >= y_span else 'vertical'

    @staticmethod
    def _side_axis(patch: QECPatch, side: str) -> str:
        # A left/right side is a vertical boundary; top/bottom is horizontal.
        return 'vertical' if side in ('left', 'right') else 'horizontal'

    @classmethod
    def _boundary_data_coords(cls, patch: QECPatch, side: str) -> Set[Tuple[float, float]]:
        x0, x1, y0, y1 = patch._get_bounds()
        if side == 'left':
            return {c for c in patch.data_coords if math.isclose(c[0], x0, abs_tol=1e-3)}
        if side == 'right':
            return {c for c in patch.data_coords if math.isclose(c[0], x1, abs_tol=1e-3)}
        if side == 'top':
            return {c for c in patch.data_coords if math.isclose(c[1], y0, abs_tol=1e-3)}
        return {c for c in patch.data_coords if math.isclose(c[1], y1, abs_tol=1e-3)}

    @staticmethod
    def _dedupe_patches(patches: Iterable[QECPatch]) -> List[QECPatch]:
        result = []
        seen = set()
        for patch in patches:
            marker = id(patch)
            if marker not in seen:
                result.append(patch)
                seen.add(marker)
        return result

    @staticmethod
    def _patch_coords(patches: Iterable[QECPatch]) -> Set[Tuple[float, float]]:
        coords = set()
        for patch in patches:
            coords.update(patch.index_map.keys())
        return coords

    @staticmethod
    def _infer_route_width(patches: List[QECPatch]) -> int:
        distances = []
        for patch in patches:
            dz = getattr(patch, 'distance_z', None)
            dx = getattr(patch, 'distance_x', None)
            if dz is not None:
                distances.append(int(dz))
            if dx is not None:
                distances.append(int(dx))
        d = min(distances) if distances else 1
        return 2 * d - 1

    @staticmethod
    def _validate_integer_lattice(patches: Iterable[QECPatch]):
        for patch in patches:
            for coord in patch.index_map:
                x, y = coord
                if not (math.isclose(x, round(x), abs_tol=1e-6) and
                        math.isclose(y, round(y), abs_tol=1e-6)):
                    raise ValueError(
                        "UnrotatedRoutedMultiPatchCoupler currently routes on integer lattice coordinates. "
                        f"Patch '{getattr(patch, 'name', '<unnamed>')}' contains non-integer coord {coord}."
                    )

    @staticmethod
    def _auto_select_sides(patches: List[QECPatch]) -> List[str]:
        centers = []
        for patch in patches:
            x0, x1, y0, y1 = patch._get_bounds()
            centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)

        sides = []
        for px, py in centers:
            dx = cx - px
            dy = cy - py
            if abs(dx) >= abs(dy):
                sides.append('right' if dx >= 0 else 'left')
            else:
                sides.append('bottom' if dy >= 0 else 'top')
        return sides

    @staticmethod
    def _make_interface(patch: QECPatch, side: str) -> InterfaceInfo:
        x0, x1, y0, y1 = patch._get_bounds()
        if side == 'left':
            return InterfaceInfo(patch=patch, side=side, boundary_edge_coord=x0)
        if side == 'right':
            return InterfaceInfo(patch=patch, side=side, boundary_edge_coord=x1)
        if side == 'top':
            return InterfaceInfo(patch=patch, side=side, boundary_edge_coord=y0)
        return InterfaceInfo(patch=patch, side=side, boundary_edge_coord=y1)

    @staticmethod
    def _interface_strip(patch: QECPatch, side: str) -> Set[Tuple[float, float]]:
        x0, x1, y0, y1 = patch._get_bounds()
        xi0, xi1 = int(round(x0)), int(round(x1))
        yi0, yi1 = int(round(y0)), int(round(y1))

        coords = set()
        if side == 'left':
            x = xi0 - 1
            for y in range(yi0, yi1 + 1):
                coords.add((float(x), float(y)))
        elif side == 'right':
            x = xi1 + 1
            for y in range(yi0, yi1 + 1):
                coords.add((float(x), float(y)))
        elif side == 'top':
            y = yi0 - 1
            for x in range(xi0, xi1 + 1):
                coords.add((float(x), float(y)))
        elif side == 'bottom':
            y = yi1 + 1
            for x in range(xi0, xi1 + 1):
                coords.add((float(x), float(y)))
        else:
            raise ValueError(f"Invalid boundary side '{side}'.")
        return coords

    @staticmethod
    def _strip_center(strip: Set[Tuple[float, float]]) -> Tuple[float, float]:
        """Return the integer-coordinate center of a selected boundary strip."""
        xs = sorted(int(round(x)) for x, _ in strip)
        ys = sorted(int(round(y)) for _, y in strip)
        mid = len(xs) // 2
        return (float(xs[mid]), float(ys[mid]))

    @staticmethod
    def _validate_patch_block_grid(
        patches: List[QECPatch],
        route_width: int,
    ) -> Tuple[int, int, Set[Tuple[int, int]]]:
        """
        Validate full-width routed surgery uses patch-sized coarse cells.

        For ``route_width > 1`` the ancillary bus is built from whole
        patch-sized blocks, so every data/obstacle patch must occupy exactly one
        ``route_width x route_width`` physical-coordinate block on a common
        coarse grid.  For a distance-d unrotated square patch this width is
        ``2*d - 1``.
        """
        if not patches:
            return 0, 0, set()

        first_bounds = patches[0]._get_bounds()
        base_x = int(round(first_bounds[0]))
        base_y = int(round(first_bounds[2]))
        occupied_cells: Set[Tuple[int, int]] = set()

        for patch in patches:
            x0f, x1f, y0f, y1f = patch._get_bounds()
            x0, x1 = int(round(x0f)), int(round(x1f))
            y0, y1 = int(round(y0f)), int(round(y1f))
            span_x = x1 - x0 + 1
            span_y = y1 - y0 + 1
            name = getattr(patch, 'name', '<unnamed>')
            if span_x != route_width or span_y != route_width:
                raise ValueError(
                    f"Full-width routed ancillary patch requires each data/obstacle patch "
                    f"to span exactly route_width={route_width} integer coordinates. "
                    f"Patch '{name}' spans {span_x}x{span_y}."
                )
            if (x0 - base_x) % route_width != 0 or (y0 - base_y) % route_width != 0:
                raise ValueError(
                    f"Patch '{name}' is not aligned to the common coarse grid. "
                    f"For route_width={route_width}, patch origins must differ by integer "
                    f"multiples of {route_width} from base origin ({base_x}, {base_y}); "
                    f"got origin ({x0}, {y0})."
                )

            cell = ((x0 - base_x) // route_width, (y0 - base_y) // route_width)
            if cell in occupied_cells:
                raise ValueError(
                    f"Multiple patches occupy coarse grid cell {cell}; move one patch."
                )
            occupied_cells.add(cell)

        return base_x, base_y, occupied_cells

    @classmethod
    def _patch_coarse_cell(
        cls,
        patch: QECPatch,
        base_x: int,
        base_y: int,
        route_width: int,
    ) -> Tuple[int, int]:
        x0, _, y0, _ = patch._get_bounds()
        return (
            (int(round(x0)) - base_x) // route_width,
            (int(round(y0)) - base_y) // route_width,
        )

    @classmethod
    def _terminal_cell_for_side(
        cls,
        patch: QECPatch,
        side: str,
        base_x: int,
        base_y: int,
        route_width: int,
    ) -> Tuple[int, int]:
        cx, cy = cls._patch_coarse_cell(patch, base_x, base_y, route_width)
        if side == 'left':
            return cx - 1, cy
        if side == 'right':
            return cx + 1, cy
        if side == 'top':
            return cx, cy - 1
        if side == 'bottom':
            return cx, cy + 1
        raise ValueError(f"Invalid boundary side '{side}'.")

    @staticmethod
    def _coarse_routing_bounds(
        terminal_cells: List[Tuple[int, int]],
        occupied_cells: Set[Tuple[int, int]],
        padding_blocks: int,
    ) -> Tuple[int, int, int, int]:
        xs = [x for x, _ in terminal_cells]
        ys = [y for _, y in terminal_cells]
        xs.extend(x for x, _ in occupied_cells)
        ys.extend(y for _, y in occupied_cells)
        return (
            min(xs) - padding_blocks,
            max(xs) + padding_blocks,
            min(ys) - padding_blocks,
            max(ys) + padding_blocks,
        )

    @staticmethod
    def _bfs_cell_path(
        sources: Set[Tuple[int, int]],
        goals: Set[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
        bounds: Tuple[int, int, int, int],
    ) -> List[Tuple[int, int]]:
        x_min, x_max, y_min, y_max = bounds
        allowed_blocked = set(sources) | set(goals)
        queue = deque()
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

        for node in sorted(sources):
            queue.append(node)
            prev[node] = None

        found = None
        while queue:
            node = queue.popleft()
            if node in goals:
                found = node
                break

            x, y = node
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = nxt
                if nx < x_min or nx > x_max or ny < y_min or ny > y_max:
                    continue
                if nxt in prev:
                    continue
                if nxt in blocked and nxt not in allowed_blocked:
                    continue
                prev[nxt] = node
                queue.append(nxt)

        if found is None:
            raise ValueError(
                "Unable to route a connected full-width ancillary patch between the selected "
                "coarse-grid interfaces. Try different sides, increase route_padding, or move "
                "blocking patches."
            )

        path = []
        node = found
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    @staticmethod
    def _expand_coarse_cells(
        cells: Set[Tuple[int, int]],
        base_x: int,
        base_y: int,
        route_width: int,
    ) -> Set[Tuple[float, float]]:
        coords: Set[Tuple[float, float]] = set()
        for cx, cy in cells:
            x_start = base_x + cx * route_width
            y_start = base_y + cy * route_width
            for x in range(x_start, x_start + route_width):
                for y in range(y_start, y_start + route_width):
                    coords.add((float(x), float(y)))
        return coords

    @staticmethod
    def _routing_bounds(
        strips: List[Set[Tuple[float, float]]],
        obstacle_patches: List[QECPatch],
        padding: int,
    ) -> Tuple[int, int, int, int]:
        xs = []
        ys = []
        for strip in strips:
            for x, y in strip:
                xs.append(int(round(x)))
                ys.append(int(round(y)))
        for patch in obstacle_patches:
            x0, x1, y0, y1 = patch._get_bounds()
            xs.extend([int(round(x0)), int(round(x1))])
            ys.extend([int(round(y0)), int(round(y1))])
        return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding

    @staticmethod
    def _bfs_path(
        sources: Set[Tuple[float, float]],
        goals: Set[Tuple[float, float]],
        blocked: Set[Tuple[float, float]],
        bounds: Tuple[int, int, int, int],
    ) -> List[Tuple[float, float]]:
        x_min, x_max, y_min, y_max = bounds
        source_nodes = {(int(round(x)), int(round(y))) for x, y in sources}
        goal_nodes = {(int(round(x)), int(round(y))) for x, y in goals}
        blocked_nodes = {(int(round(x)), int(round(y))) for x, y in blocked}

        allowed_blocked = source_nodes | goal_nodes
        queue = deque()
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

        for node in sorted(source_nodes):
            queue.append(node)
            prev[node] = None

        found = None
        while queue:
            node = queue.popleft()
            if node in goal_nodes:
                found = node
                break

            x, y = node
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = nxt
                if nx < x_min or nx > x_max or ny < y_min or ny > y_max:
                    continue
                if nxt in prev:
                    continue
                if nxt in blocked_nodes and nxt not in allowed_blocked:
                    continue
                prev[nxt] = node
                queue.append(nxt)

        if found is None:
            raise ValueError(
                "Unable to route a connected ancillary path between the selected patch interfaces. "
                "Try different sides, increase route_padding, or move blocking patches."
            )

        path = []
        node = found
        while node is not None:
            path.append((float(node[0]), float(node[1])))
            node = prev[node]
        path.reverse()
        return path

    @classmethod
    def _thicken_route_coords(
        cls,
        route_coords: Set[Tuple[float, float]],
        strips: List[Set[Tuple[float, float]]],
        sides: List[str],
        obstacle_patches: List[QECPatch],
        route_width: int,
    ) -> Set[Tuple[float, float]]:
        """
        Expand a centerline skeleton into a patch-width ancillary region.

        Each selected interface first becomes a terminal block with the same
        coordinate span as the neighboring distance-d patch.  Centerline edges
        then become rectangular corridors of the same width.  This avoids the
        older "union of many local squares" thickening, which made terminal
        regions longer than a distance-d patch.
        """
        if route_width <= 1:
            obstacle_coords = cls._patch_coords(obstacle_patches)
            return {c for c in route_coords if c not in obstacle_coords}

        before = (route_width - 1) // 2
        after = route_width - 1 - before
        thick: Set[Tuple[float, float]] = set()

        for strip, side in zip(strips, sides):
            thick.update(cls._terminal_block_coords(strip, side, route_width))

        route_nodes = {(int(round(x)), int(round(y))) for x, y in route_coords}
        incident_axes: Dict[Tuple[int, int], Set[str]] = {node: set() for node in route_nodes}
        for x, y in sorted(route_nodes):
            for nx, ny, axis in ((x + 1, y, 'horizontal'), (x, y + 1, 'vertical')):
                if (nx, ny) not in route_nodes:
                    continue
                if axis == 'horizontal':
                    for xx in range(x, nx + 1):
                        for yy in range(y - before, y + after + 1):
                            thick.add((float(xx), float(yy)))
                else:
                    for xx in range(x - before, x + after + 1):
                        for yy in range(y, ny + 1):
                            thick.add((float(xx), float(yy)))
                incident_axes[(x, y)].add(axis)
                incident_axes[(nx, ny)].add(axis)

        for (x, y), axes in incident_axes.items():
            if len(axes) > 1:
                for dx in range(-before, after + 1):
                    for dy in range(-before, after + 1):
                        thick.add((float(x + dx), float(y + dy)))

        obstacle_coords = cls._patch_coords(obstacle_patches)
        return {coord for coord in thick if coord not in obstacle_coords}

    @staticmethod
    def _terminal_block_coords(
        strip: Set[Tuple[float, float]],
        side: str,
        route_width: int,
    ) -> Set[Tuple[float, float]]:
        """Return the flush ancillary block attached to one data-patch side."""
        xs = [int(round(x)) for x, _ in strip]
        ys = [int(round(y)) for _, y in strip]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        block = set()

        if side == 'left':
            xr = range(x0 - route_width + 1, x0 + 1)
            yr = range(y0, y1 + 1)
        elif side == 'right':
            xr = range(x0, x0 + route_width)
            yr = range(y0, y1 + 1)
        elif side == 'top':
            xr = range(x0, x1 + 1)
            yr = range(y0 - route_width + 1, y0 + 1)
        elif side == 'bottom':
            xr = range(x0, x1 + 1)
            yr = range(y0, y0 + route_width)
        else:
            raise ValueError(f"Invalid boundary side '{side}'.")

        for x in xr:
            for y in yr:
                block.add((float(x), float(y)))
        return block

    @staticmethod
    def _label_route_coords(
        route_coords: Set[Tuple[float, float]],
        strips: List[Set[Tuple[float, float]]],
        interface_bases: List[str],
    ) -> Dict[Tuple[float, float], str]:
        """Assign each routed coordinate to the nearest X/Z interface region."""
        route_nodes = {(int(round(x)), int(round(y))) for x, y in route_coords}
        queue = deque()
        labels: Dict[Tuple[int, int], str] = {}
        distances: Dict[Tuple[int, int], int] = {}

        for strip, basis in zip(strips, interface_bases):
            for x, y in sorted(strip):
                node = (int(round(x)), int(round(y)))
                if node not in route_nodes:
                    continue
                if node not in labels:
                    labels[node] = basis
                    distances[node] = 0
                    queue.append(node)

        while queue:
            node = queue.popleft()
            x, y = node
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nxt not in route_nodes:
                    continue
                nd = distances[node] + 1
                if nxt not in labels:
                    labels[nxt] = labels[node]
                    distances[nxt] = nd
                    queue.append(nxt)
                elif nd == distances[nxt] and labels[nxt] != labels[node]:
                    # Deterministic tie break at seams.  The mixed stabilizer
                    # generation sees neighboring labels and creates XZ checks.
                    labels[nxt] = 'X' if 'X' in (labels[nxt], labels[node]) else 'Z'

        return {
            (float(x), float(y)): basis
            for (x, y), basis in labels.items()
        }

    # ------------------------------------------------------------------
    # Coupler construction and validation
    # ------------------------------------------------------------------

    def _construct_routed_region(
        self,
        coupler_patch: QECPatch,
        obstacle_patches: List[QECPatch],
        anchor_patch: QECPatch,
        route_coords: Set[Tuple[float, float]],
    ):
        obstacle_coords = self._patch_coords(obstacle_patches)
        for x, y in sorted(route_coords, key=lambda c: (c[1], c[0])):
            if (x, y) in obstacle_coords:
                continue
            role = UnrotatedTwoPatchCoupler._infer_role_from_anchor(anchor_patch, x, y)
            if role:
                coupler_patch.add_qubit(x, y, role=role)

    def _init_routed_stabilizers(self, coupler_patch: QECPatch, patches: List[QECPatch], path_info: RoutedPathInfo):
        """Create local pure/mixed stabilizers from routed basis labels."""
        coupler_patch.conflicting_stabilizer_coords = set()

        for uid in coupler_patch.syndrome_indices:
            syn_coord = coupler_patch.qubit_coords[uid]
            self._probe_and_create_routed_stabilizer(coupler_patch, patches, syn_coord, path_info)

        boundary_candidates = self._find_boundary_syndrome_candidates(patches, path_info)
        for syn_coord in boundary_candidates:
            success = self._probe_and_create_routed_stabilizer(coupler_patch, patches, syn_coord, path_info)
            if success:
                coupler_patch.conflicting_stabilizer_coords.add(syn_coord)

    def _probe_and_create_routed_stabilizer(self, coupler_patch, patches, syn_coord, path_info: RoutedPathInfo) -> bool:
        pauli = {}
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            tx, ty = QECPatch.snap_coord((syn_coord[0] + dx, syn_coord[1] + dy))
            if UnrotatedTwoPatchCoupler._is_data_qubit_at(coupler_patch, patches, tx, ty):
                basis = self._neighbor_basis((tx, ty), syn_coord, patches, path_info)
                pauli[(tx, ty)] = basis

        if len(pauli) < 2:
            return False

        bases = set(pauli.values())
        if bases == {'X'}:
            stype = 'X'
        elif bases == {'Z'}:
            stype = 'Z'
        else:
            stype = 'MIXED'

        coupler_patch.stabilizers.append({
            'pauli': pauli,
            'type': stype,
            'syn_coord': syn_coord,
        })
        return True

    def _neighbor_basis(
        self,
        neighbor_coord: Tuple[float, float],
        syn_coord: Tuple[float, float],
        patches: List[QECPatch],
        path_info: RoutedPathInfo,
    ) -> str:
        if neighbor_coord in path_info.coord_basis:
            return path_info.coord_basis[neighbor_coord]
        if syn_coord in path_info.coord_basis:
            return path_info.coord_basis[syn_coord]

        for iface, basis in zip(path_info.interfaces, path_info.interface_bases):
            uid = iface.patch.index_map.get(neighbor_coord)
            if uid is not None and uid in iface.patch.data_indices:
                return basis

        return 'Z'

    def _mark_anticommuting_patch_stabilizers(self, coupler_patch, patches: List[QECPatch]):
        """
        Pause every existing patch stabilizer that anticommutes with a routed check.

        Mixed-boundary and corner templates can replace more than just the
        literal boundary syndrome coordinate.  This algebraic conflict pass keeps
        the active stabilizer set commuting.
        """
        conflicts = set(getattr(coupler_patch, 'conflicting_stabilizer_coords', set()))
        for patch in patches:
            for stab in patch.stabilizers:
                if self._anticommutes_with_any_coupler_stabilizer(stab, coupler_patch.stabilizers, patch):
                    syn_coord = stab.get('syn_coord')
                    if syn_coord is not None:
                        conflicts.add(syn_coord)
        coupler_patch.conflicting_stabilizer_coords = conflicts

    @staticmethod
    def _anticommutes_with_any_coupler_stabilizer(stab, coupler_stabilizers, patch: QECPatch) -> bool:
        patch_pauli = {}
        for key, pauli in stab.get('pauli', {}).items():
            if isinstance(key, int):
                coord = patch.qubit_coords.get(key)
            else:
                coord = key
            if coord is not None:
                patch_pauli[coord] = pauli

        for coupler_stab in coupler_stabilizers:
            parity = 0
            for coord, pauli in coupler_stab.get('pauli', {}).items():
                other = patch_pauli.get(coord)
                if other is not None and other != pauli:
                    parity ^= 1
            if parity:
                return True
        return False

    @staticmethod
    def _coords_bounds(coords: Set[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), max(xs), min(ys), max(ys)

    def _validate_routed_connectivity(self, coupler_patch, patches, path_info: RoutedPathInfo):
        coupler_coords = set(coupler_patch.index_map.keys())
        if not coupler_coords:
            raise ValueError("Routed coupler produced no ancillary qubits.")

        start = next(iter(coupler_coords))
        queue = deque([start])
        visited = {start}
        while queue:
            x, y = queue.popleft()
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nxt = QECPatch.snap_coord(nxt)
                if nxt in coupler_coords and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        if visited != coupler_coords:
            missing = sorted(coupler_coords - visited)[:8]
            raise ValueError(f"Routed coupler is disconnected; example unreachable coords: {missing}.")

        conflicts = getattr(coupler_patch, 'conflicting_stabilizer_coords', set())
        for iface in path_info.interfaces:
            edge = iface.boundary_edge_coord
            has_boundary_stabilizer = False
            for coord in conflicts:
                x, y = coord
                if iface.side in ('left', 'right') and math.isclose(x, edge, abs_tol=1e-3):
                    has_boundary_stabilizer = True
                    break
                if iface.side in ('top', 'bottom') and math.isclose(y, edge, abs_tol=1e-3):
                    has_boundary_stabilizer = True
                    break
            if not has_boundary_stabilizer:
                raise ValueError(
                    f"Routed coupler did not create any boundary stabilizer for patch "
                    f"'{getattr(iface.patch, 'name', '<unnamed>')}' on side '{iface.side}'."
                )
