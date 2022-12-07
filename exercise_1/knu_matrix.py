import numpy as np

from exercise_1.constants import l_z
from exercise_1.geometry import element_areas, reluctivity
from exercise_1.mesh import Mesh


def Knu_e(elem: int) -> np.ndarray:
    """Computes the 3x3 matrix of entries for the stiffness matrix K.

    :param elem: The element for the shape functions.
    """

    mesh = Mesh.create()
    nodes = mesh.elem_to_node[elem]
    S = element_areas()[elem]  # TODO: Use function in mesh.
    r = reluctivity(1, 2)[elem]
    knu = np.zeros(3, 3)

    for i in nodes:
        for j in nodes:
            # TODO: Use a,b,c vectors.
            bi, bj = 0, 0
            ci, cj = 0, 0
            knu[i, j] = r * (bi*bj + ci*cj) / (4 * S * l_z)

    return knu
