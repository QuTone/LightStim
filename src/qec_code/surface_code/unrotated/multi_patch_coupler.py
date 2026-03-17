from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import math
from src.ir.qec_patch import QECPatch
from src.ir.coupler import LogicalCouplerProtocol
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
