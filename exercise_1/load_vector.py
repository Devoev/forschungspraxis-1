import numpy as np
import scipy.sparse.linalg as las
from scipy.sparse import csr_matrix, csr_array, spmatrix

from exercise_1.constants import I, r1, WIRE
from exercise_1.mesh import Mesh


def X(mesh: Mesh) -> spmatrix:
    """The current distribution matrix X of size (N,1)."""
    x = np.zeros(mesh.num_elems * 3)
    rows = np.zeros(mesh.num_elems * 3)
    cols = np.zeros(mesh.num_elems * 3)  # TODO: Update cols if 2nd conductor is present.

    idx = mesh.elem_in_group(WIRE)  # The indices of wire elements with current I
    S: float = np.sum(mesh.elem_areas * idx) * 1  # The surface area of the wire

    for i, nodes in enumerate(mesh.elems):
        rows[i * 3:(i + 1) * 3] = nodes
        x[i * 3:(i + 1) * 3] = X_e(i, S, mesh) * idx[i]

    return csr_matrix((x, (rows, cols)), shape=(mesh.num_node, 1))


def X_e(elem: int, S: float, mesh: Mesh) -> np.ndarray:
    """The local 3x1 current distribution matrix X.

    :param elem: The element for the local matrix.
    :param S: The surface area of the entire region.
    :param mesh: The mesh object.
    """
    return mesh.elem_areas[elem] / (3*S) * np.ones(3)


def j_grid(mesh: Mesh) -> spmatrix:
    """The grid-current right hand side vector of size (N,1)."""
    return X(mesh) * I
