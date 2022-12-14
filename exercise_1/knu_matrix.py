import numpy as np
from scipy.sparse import coo_matrix, spmatrix, bmat, csr_matrix

from exercise_1.constants import l_z
from exercise_1.geometry import Geo
from exercise_1.mesh import Mesh


def Knu(mesh: Mesh, geo: Geo) -> spmatrix:
    """The stiffness matrix K.

    :param mesh: The mesh object.
    :param geo: The geometry object.
    """

    n = mesh.num_elems * 9  # Amount of matrix entries
    m = mesh.num_node  # Dimension of Knu matrix
    knu = np.zeros(n)  # Nonzero entries of the Knu matrix
    rows = np.zeros(n, dtype='int')  # Row indices for the entries
    cols = np.zeros(n, dtype='int')  # Column indices for the entries

    for elem in range(mesh.num_elems):
        idx = np.sort(mesh.elems[elem])
        j = elem*9
        rows[j:j+9] = np.repeat(idx, 3)
        cols[j:j+9] = np.reshape([idx, idx, idx], 9)
        knu[j:j+9] = Knu_e(elem, mesh, geo).flatten()

    return csr_matrix((knu, (rows, cols)), shape=(m, m))


def Knu_e(elem: int, mesh: Mesh, geo: Geo) -> np.ndarray:
    """Computes the 3x3 matrix of entries for the stiffness matrix K.

    :param elem: The element tag for the shape functions.
    :param mesh: The mesh object.
    :param geo: The geometry object.
    """

    _, b, c = mesh.coeffs
    b = b[elem]
    c = c[elem]
    S = mesh.elem_areas[elem]
    r = geo.reluctivity[elem]
    knu = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            if i != j:
                knu[i, j] = r * (b[i] * b[j] + c[i] * c[j]) / (4 * S * l_z)

    return knu
