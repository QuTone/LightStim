"""Smoke tests for the subset-joint report front-end (``report_subset_joint``).

Every ``show_*`` toggle off: the call must still route, build and verify — touching neither
matplotlib nor IPython — and hand back the verified ``SubsetRoute`` for programmatic use.  The
passing geometry is the simplest obstacle case (two targets on a free straight corridor, two
obstacle patches below); the failing one puts an obstacle inside the keepout of a target.
"""

import pytest

from lightstim.qec_code.surface_code.rotated import PatchSpec, report_subset_joint
from lightstim.qec_code.surface_code.rotated.subset_routing import (
    origin_of, acceptance_of_layout, ACCEPTANCE_ITEMS)

pytestmark = pytest.mark.smoke

D = 3

NO_SHOW = dict(show_path=False, show_acceptance=False, show_layout=False, show_detslice=False)


def test_report_returns_verified_route_headless():
    patches = [PatchSpec("X1", origin_of(0,  0, D), D, "X", "X_horizontal"),
               PatchSpec("Z2", origin_of(3,  0, D), D, "Z", "X_horizontal"),
               PatchSpec("B1", origin_of(0, -2, D), D, "X", "X_horizontal"),
               PatchSpec("B2", origin_of(3, -2, D), D, "X", "X_horizontal")]
    target = [("X1", "X"), ("Z2", "Z")]
    r = report_subset_joint(patches, target, **NO_SHOW)
    assert r.ok and r.layout is not None and r.tree
    a = acceptance_of_layout(r.layout)
    assert tuple(a["items"]) == ACCEPTANCE_ITEMS   # the strict gate, item for item
    assert a["accept"], a["items"]
    assert a["N"] == 2 and a["k"] == 1             # dof == N-1


def test_report_raises_when_unroutable():
    # Z2 walled in on all four edge-adjacent cells -> no corridor can reach any of its
    # faces -> no_path.  (An obstacle merely adjacent to a target is legal now: the
    # collision criterion is actual ancilla-site sharing, and the rule constructor
    # interleaves with the idle neighbour's used sites.)
    patches = [PatchSpec("X1", origin_of(0, 0, D), D, "X", "X_horizontal"),
               PatchSpec("Z2", origin_of(4, 0, D), D, "Z", "X_horizontal"),
               PatchSpec("B1", origin_of(3, 0, D), D, "X", "X_horizontal"),
               PatchSpec("B2", origin_of(5, 0, D), D, "X", "X_horizontal"),
               PatchSpec("B3", origin_of(4, 1, D), D, "X", "X_horizontal"),
               PatchSpec("B4", origin_of(4, -1, D), D, "X", "X_horizontal")]
    target = [("X1", "X"), ("Z2", "Z")]
    with pytest.raises(RuntimeError, match="route_and_build failed"):
        report_subset_joint(patches, target, **NO_SHOW)
