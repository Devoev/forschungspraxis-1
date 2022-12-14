from typing import Tuple

import numpy as np
import scipy.sparse.linalg as las
from scipy.sparse import spmatrix

from exercise_1.constants import GND
from exercise_1.geometry import Geo
from exercise_1.knu_matrix import Knu
from exercise_1.load_vector import j_grid
from exercise_1.mesh import Mesh


def solve_ms(mesh: Mesh, geo: Geo) -> np.ndarray:
    """Solves the magneto-static system Ka=j.

    :returns: The solution for the magnetic vector potential on the nodes. Vector of size (N).
    """

    knu = Knu(mesh, geo)
    j = j_grid(mesh)
    a = np.zeros(j.shape[0])

    idx_dir = mesh.nodes_in_group(GND)  # Dirichlet (boundary) indices
    idx_dof = np.setdiff1d(mesh.node_tags, idx_dir)  # DOF indices

    A, b = deflate(knu, j, idx_dof)
    x = las.spsolve(A, b)
    return inflate(a, x, idx_dof)


def deflate(A: spmatrix, b: spmatrix, idx: np.ndarray) -> Tuple[spmatrix, spmatrix]:
    """Deflates the system Ax=b by only using the rows and columns specified by idx."""
    return A[idx, :][:, idx], b[idx]


def inflate(v: np.ndarray, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Inflates the vector x by setting the values of v at the indices specified by idx to x."""
    v[idx] = x
    return v
