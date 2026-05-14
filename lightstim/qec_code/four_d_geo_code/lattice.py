"""4D lattice arithmetic for loop-only toric codes on rotated lattices.

Given a 4x4 integer lattice matrix L in Hermite Normal Form (HNF),
this module handles:
- Point reduction modulo the lattice Λ
- Enumeration of cells (vertices, edges, faces, cubes, hypercubes)
- Boundary and coboundary operators for stabilizer construction
- Direction-to-face mapping for syndrome extraction circuits
"""

from typing import List, Tuple, Optional


class Lattice4D:
    """Arithmetic on the 4-torus T^4_Λ = Z^4 / Λ.

    The lattice Λ is specified by an upper-triangular HNF matrix L (4x4).
    Rows of L are the generating vectors of Λ.
    """

    # 6 face types: pairs of free directions
    FACE_TYPES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # Unit vectors in Z^4
    E = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

    def __init__(self, L: List[List[int]]):
        """
        Args:
            L: 4x4 HNF matrix (upper triangular, 0 <= a_ij < a_jj for i < j).
        """
        self.L = [list(row) for row in L]
        self.det = L[0][0] * L[1][1] * L[2][2] * L[3][3]
        self._points = None
        self._point_to_idx = None

    def reduce(self, p: tuple) -> tuple:
        """Reduce point p modulo Λ to canonical representative.

        For upper-triangular HNF, process rows top-to-bottom:
        subtracting row i only affects coordinates j >= i,
        so previously fixed coordinates (j < i) are preserved.
        """
        p = list(p)
        for i in range(4):
            q = p[i] // self.L[i][i]
            for j in range(i, 4):
                p[j] -= q * self.L[i][j]
        return tuple(p)

    def add(self, p: tuple, q: tuple) -> tuple:
        """Add two lattice points mod Λ."""
        return self.reduce((p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3]))

    def sub(self, p: tuple, q: tuple) -> tuple:
        """Subtract: p - q mod Λ."""
        return self.reduce((p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3] - q[3]))

    def enumerate_points(self) -> List[tuple]:
        """All Det(L) canonical lattice point representatives."""
        if self._points is None:
            self._points = []
            for a in range(self.L[0][0]):
                for b in range(self.L[1][1]):
                    for c in range(self.L[2][2]):
                        for d in range(self.L[3][3]):
                            self._points.append((a, b, c, d))
        return self._points

    def point_to_idx(self, p: tuple) -> int:
        """Map a canonical lattice point to its index in enumerate_points()."""
        if self._point_to_idx is None:
            self._point_to_idx = {pt: i for i, pt in enumerate(self.enumerate_points())}
        return self._point_to_idx[p]

    # ------------------------------------------------------------------
    # Boundary / Coboundary operators
    # ------------------------------------------------------------------

    def x_stabilizer_support(self, edge_dir: int, point: tuple) -> List[Tuple[int, tuple]]:
        """Coboundary of edge → 6 faces touched by this X-stabilizer.

        An edge in direction d at point p connects to:
          For each d' != d:
            - face {d, d'} at p
            - face {d, d'} at p - e_{d'}

        Returns:
            List of (face_type_idx, lattice_point) pairs.
        """
        faces = []
        for d_prime in range(4):
            if d_prime == edge_dir:
                continue
            ft = tuple(sorted((edge_dir, d_prime)))
            ft_idx = self.FACE_TYPES.index(ft)
            faces.append((ft_idx, point))
            faces.append((ft_idx, self.sub(point, self.E[d_prime])))
        return faces

    def z_stabilizer_support(self, cube_miss: int, point: tuple) -> List[Tuple[int, tuple]]:
        """Boundary of cube → 6 faces touched by this Z-stabilizer.

        A cube with missing direction m at point p:
          free_dirs = {0,1,2,3} \\ {m}
          For each d in free_dirs:
            - face type = free_dirs \\ {d}, at p
            - face type = free_dirs \\ {d}, at p + e_d

        Returns:
            List of (face_type_idx, lattice_point) pairs.
        """
        free_dirs = [d for d in range(4) if d != cube_miss]
        faces = []
        for d in free_dirs:
            remaining = tuple(sorted(dd for dd in free_dirs if dd != d))
            ft_idx = self.FACE_TYPES.index(remaining)
            faces.append((ft_idx, point))
            faces.append((ft_idx, self.add(point, self.E[d])))
        return faces

    # ------------------------------------------------------------------
    # Syndrome extraction circuit helpers
    # ------------------------------------------------------------------

    def se_edge_to_face(self, sign: int, axis: int,
                        edge_dir: int, point: tuple) -> Optional[Tuple[int, tuple]]:
        """Map a signed direction to the face connected to an edge.

        For edge in direction d at point p, signed direction (sign, axis):
          - If axis == d: no connection (idle)
          - sign == +1: face {d, axis} at p
          - sign == -1: face {d, axis} at p - e_{axis}

        Returns:
            (face_type_idx, point) or None if no connection.
        """
        if axis == edge_dir:
            return None
        ft = tuple(sorted((edge_dir, axis)))
        ft_idx = self.FACE_TYPES.index(ft)
        if sign == +1:
            return (ft_idx, point)
        else:
            return (ft_idx, self.sub(point, self.E[axis]))

    def se_cube_to_face(self, sign: int, axis: int,
                        cube_miss: int, point: tuple) -> Optional[Tuple[int, tuple]]:
        """Map a signed direction to the face connected to a cube.

        For cube with missing direction m at point p, signed direction (sign, axis):
          - If axis == m: no connection (idle)
          - sign == +1: face type {free_dirs \\ {axis}} at p + e_{axis}
          - sign == -1: face type {free_dirs \\ {axis}} at p

        Returns:
            (face_type_idx, point) or None if no connection.
        """
        if axis == cube_miss:
            return None
        free_dirs = [d for d in range(4) if d != cube_miss]
        remaining = tuple(sorted(d for d in free_dirs if d != axis))
        ft_idx = self.FACE_TYPES.index(remaining)
        if sign == +1:
            return (ft_idx, self.add(point, self.E[axis]))
        else:
            return (ft_idx, point)
