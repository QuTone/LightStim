import numpy as np
import pytest

from lightstim.qec_code.surface_code.rotated.bent_layout import (
    _symplectic, _gf2_rank, _in_span, build_rotated_bent_xz_layout, PatchSpec)


def _lvec(sv, measured, support):
    return sv({c: measured for c in support})


def joint_acceptance(data, checks, logicals):
    """The five algebraic acceptance checks for an N-term joint ∏ P̄_i.

    ``logicals`` is a list of ``(measured_pauli, support)`` per patch.  Returns a dict:
    commute / joint (∏ P̄_i in span) / no_single (no individual P̄_i in span) / no_twist /
    logical_count (#data − rank == N − 1).
    """
    sv, n = _symplectic(data)
    S = [sv(ch["pauli"]) for ch in checks]
    comm = lambda a, b: int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0
    commute = all(comm(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S)))
    twist = any(P == "Y" for ch in checks for P in ch["pauli"].values())
    singles = [_lvec(sv, P, sup) for P, sup in logicals]
    joint = np.zeros(2 * n, np.uint8)
    for v in singles:
        joint ^= v
    N = len(logicals)
    return dict(commute=commute, joint=_in_span(S, joint),
                no_single=not any(_in_span(S, v) for v in singles),
                no_twist=not twist, logical_count=len(data) - _gf2_rank(S) == N - 1)


def test_joint_acceptance_on_two_patch_reference():
    # Sanity: the validated two-patch X̄₁Z̄₂ layout satisfies the N-logical helper with N=2.
    lay = build_rotated_bent_xz_layout([PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"),
                                        PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal")])
    a = joint_acceptance(lay.data, lay.checks, [("X", lay.x_logical), ("Z", lay.z_logical)])
    assert a == {"commute": True, "joint": True, "no_single": True,
                 "no_twist": True, "logical_count": True}
