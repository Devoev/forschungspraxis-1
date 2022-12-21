from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.sparse import spmatrix
import scipy.sparse.linalg as las

from exercise_1.constants import GND, l_z
from exercise_1.geometry import Geo
from exercise_1.knu_matrix import Knu
from exercise_1.load_vector import j_grid
from exercise_1.mesh import Mesh
from exercise_1.solver_ms import inflate, deflate


@dataclass
class MSSolution:
    """An object for solving the magneto-statics problem Ka=j and calculating post-processing quantities."""

    mesh: Mesh
    geo: Geo
    knu: spmatrix = field(init=False)
    j: spmatrix = field(init=False)
    a: np.ndarray = field(init=False)

    def __post_init__(self):
        self.knu = Knu(self.mesh, self.geo)
        self.j = j_grid(self.mesh)
        self.a = np.zeros(self.j.shape[0])

    def solve(self) -> np.ndarray:
        """Solves the magneto-static system Ka=j.

        :returns: The solution for the magnetic vector potential in z-direction on the nodes. Vector of size (N).
        """

        idx_dir = self.mesh.nodes_in_group(GND)  # Dirichlet (boundary) indices
        idx_dof = np.setdiff1d(self.mesh.node_tags, idx_dir)  # DOF indices

        A, b = deflate(self.knu, self.j, idx_dof)
        x = las.spsolve(A, b)
        return inflate(self.a, x, idx_dof)

    @property
    def b(self) -> np.ndarray:
        """The values for the magnetic flux density in x- and y- direction. Matrix of size (E,2)."""
        a_z = self.a[self.mesh.elems]
        S = self.mesh.elem_areas[:, None]
        _, b, c = self.mesh.coeffs

        bx = np.sum(c * a_z / S, axis=1) / (2*l_z)
        by = -np.sum(b * a_z / S, axis=1) / (2*l_z)
        return np.vstack([bx, by]).T

    @property
    def L(self) -> spmatrix:
        """The inductance matrix L of size (N,1)."""
        pass
