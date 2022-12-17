import numpy as np
from scipy.sparse import csr_matrix, csr_array, spmatrix

from exercise_1.constants import I, r1
from exercise_1.mesh import Mesh


def j_grid(mesh: Mesh) -> np.ndarray:
    """A vector of current densities inside the wire."""
    J0 = I / (np.pi * r1 ** 2)
    idx = mesh.elem_in_group(1)
    return idx * J0


def grid_current(mesh: Mesh) -> spmatrix:
    """A vector of grid currents inside the wire."""
    currents_e = j_grid(mesh) * mesh.elem_areas
    idx = np.zeros(mesh.num_elems * 3)
    currents_n = np.zeros(mesh.num_elems * 3)

    for j, nodes in enumerate(mesh.elems):
        idx[j * 3:(j + 1) * 3] = nodes
        currents_n[j * 3:(j + 1) * 3] = currents_e[j]

    return csr_matrix((currents_n, (idx, np.zeros(mesh.num_elems * 3))), shape=(mesh.num_node, 1))
