from dataclasses import dataclass

import numpy as np
import numpy.linalg as la

from exercise_1.constants import mu_s, mu_w, WIRE, SHELL
from exercise_1.mesh import Mesh

@dataclass
class Geo:
    """An object for handling the geometry data."""

    mesh: Mesh

    @property
    def reluctivity(self) -> np.ndarray:
        """A vector with reluctivity values."""
        return self.mesh.elem_in_group(SHELL) / mu_s + self.mesh.elem_in_group(WIRE) / mu_w

    @property
    def r(self) -> np.ndarray:
        """The radius values for all nodes of the mesh."""
        return la.norm(self.mesh.node_coords, axis=1)
