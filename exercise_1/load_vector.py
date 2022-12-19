import numpy as np
import scipy.sparse.linalg as las
from scipy.sparse import csr_matrix, csr_array, spmatrix

from exercise_1.constants import I, r1
from exercise_1.mesh import Mesh


def j_grid_elems(mesh: Mesh) -> np.ndarray:
    """A vector of size (E) of current densities inside the wire for each element."""
    idx = mesh.elem_in_group(1)
    J0 = I / (np.sum(mesh.elem_areas * idx))  # J = I/A
    return idx * J0


def j_grid(mesh: Mesh) -> spmatrix:
    """A vector of size (N) of current densities inside the wire for each node."""
    currents_e = j_grid_elems(mesh) * mesh.elem_areas  # I = J*A
    idx = np.zeros(mesh.num_elems * 3)
    currents_n = np.zeros(mesh.num_elems * 3)

    for j, nodes in enumerate(mesh.elems):
        idx[j * 3:(j + 1) * 3] = nodes
        currents_n[j * 3:(j + 1) * 3] = currents_e[j] / 3  # Divide with 3, because there are 3 elements per node

    return csr_matrix((currents_n, (idx, np.zeros(mesh.num_elems * 3))), shape=(mesh.num_node, 1))
