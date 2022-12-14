import numpy as np

from exercise_1.constants import I, r1
from exercise_1.mesh import Mesh


def j_grid(mesh: Mesh) -> np.ndarray:
    """A vector of current densities inside the wire."""
    J0 = I / (np.pi * r1 ** 2)
    idx = mesh.elem_in_group(1)
    return idx * J0


def grid_current(mesh: Mesh) -> np.ndarray:
    """A vector of grid currents inside the wire."""
    return j_grid(mesh) * mesh.elem_areas
