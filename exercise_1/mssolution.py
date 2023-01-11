from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.sparse.linalg as las
from numpy import ndarray, pi, sqrt
from scipy.sparse import spmatrix

from exercise_1.constants import GND, l_z, I, eps_s, mu_s, sig_cu, r1
from exercise_1.geometry import Geo
from exercise_1.knu_matrix import Knu
from exercise_1.load_vector import X
from exercise_1.mesh import Mesh
from exercise_1.solver_ms import inflate, deflate


@dataclass
class MSSolution:
    """An object for solving the magneto-statics problem Ka=j and calculating post-processing quantities.

    TODO: Add material parameters.
    """

    mesh: Mesh
    geo: Geo
    a: np.ndarray = field(init=False)

    def __post_init__(self):
        self.a = np.zeros(self.j.shape[0])

    @cached_property
    def knu(self) -> spmatrix:
        """The Knu matrix."""
        return Knu(self.mesh, self.geo)

    @cached_property
    def X(self) -> spmatrix:
        """The current distribution vector."""
        return X(self.mesh)

    @cached_property
    def j(self) -> spmatrix:
        """The grid current vector."""
        return self.X * I

    @cached_property
    def Q(self) -> ndarray:
        """The charge vector."""
        A, b = deflate(self.knu, self.X, self.idx_dof)
        x: np.ndarray = las.spsolve(A, b)
        Q = np.zeros(self.j.shape[0])
        return inflate(Q, x, self.idx_dof)

    @property
    def idx_dir(self):
        """The indices of dirichlet boundary nodes."""
        return self.mesh.nodes_in_group(GND)

    @property
    def idx_dof(self):
        """The indices for the degrees of freedom."""
        return np.setdiff1d(self.mesh.node_tags, self.idx_dir)

    def solve(self) -> np.ndarray:
        """Solves the magneto-static system Ka=j.

        :returns: The solution for the magnetic vector potential in z-direction on the nodes. Vector of size (N).
        """

        A, b = deflate(self.knu, self.j, self.idx_dof)
        x: np.ndarray = las.spsolve(A, b)
        return inflate(self.a, x, self.idx_dof)

    @cached_property
    def b(self) -> np.ndarray:
        """The values for the magnetic flux density in x- and y- direction. Matrix of size (E,2)."""
        a_z = self.a[self.mesh.elems]
        S = self.mesh.elem_areas[:, None]
        _, b, c = self.mesh.coeffs

        bx = np.sum(c * a_z / S, axis=1) / (2 * l_z)
        by = -np.sum(b * a_z / S, axis=1) / (2 * l_z)
        return np.vstack([bx, by]).T

    @cached_property
    def L(self) -> float:
        """The inductance L."""
        A, b = deflate(self.knu, self.X, self.idx_dof)
        y: np.ndarray = las.spsolve(A, b)
        return (self.X.T * inflate(self.a, y, self.idx_dof))[0]

    @cached_property
    def C(self) -> float:
        """The capacitance C."""
        return eps_s * mu_s / self.L

    @cached_property
    def R(self) -> float:
        """The per-unit-length resistance R'."""
        return 1 / (sig_cu * pi * r1**2)

    def Z(self, f: float) -> float:
        """The impedance Z."""
        w = 2*pi*f
        r = self.R
        l = self.L / l_z
        return r + w*l*1j

    def Y(self, f: float):
        """The admittance Y."""
        w = 2 * pi * f
        c = self.C / l_z
        return w*c*1j

    def Z_char(self, f: float) -> float:
        """The characteristic impedance for the given frequency."""
        return sqrt(self.Z(f) / self.Y(f))

    def beta(self, f: float) -> float:
        """The propagation constant beta."""
        return sqrt(self.Z(f) * self.Y(f))

    def A(self, f: float) -> np.ndarray:
        """The propagation matrix A."""
        bl = self.beta(f) * l_z
        Z = self.Z(f)
        Y = self.Y(f)
        return np.array([
            [np.cosh(bl), -Z * np.sinh(bl)],
            [-Y * np.sinh(bl), np.cosh(bl)]
        ])

    def B(self, f: float) -> np.ndarray:
        """The admittance matrix B."""
        bl = self.beta(f) * l_z
        Y = self.Y(f)
        return np.array([
            [np.cosh(bl), -1],
            [1, -np.cosh(bl)]
        ]) * Y / np.sinh(bl)
