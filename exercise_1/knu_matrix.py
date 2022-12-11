import numpy as np

from exercise_1.constants import l_z
from exercise_1.geometry import Geo
from exercise_1.mesh import Mesh


def Knu_e(elem: int) -> np.ndarray:
    """Computes the 3x3 matrix of entries for the stiffness matrix K.

    :param elem: The element for the shape functions.
    """

    mesh = Mesh.create()
    geo = Geo(mesh)

    _, b, c = mesh.coeffs
    b = b[elem]
    c = c[elem]
    S = mesh.elem_areas[elem]
    r = geo.reluctivity[elem]
    knu = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            knu[i, j] = r * (b[i]*b[j] + c[i]*c[j]) / (4 * S * l_z)

    return knu
