from dataclasses import dataclass
from typing import Tuple

from exercise_1.mesh import Mesh
from util.model import Point2D


@dataclass
class ShapeFunction:
    """A 2 dimensional shape function."""

    a: float
    b: float
    c: float
    S: float

    def __call__(self, x: float, y: float) -> float:
        """Evaluates this shape function at the point (x,y)."""
        return (self.a + self.b * x + self.c * y) / (2 * self.S)

    @staticmethod
    def of_points(p_i: Point2D, p_j: Point2D, p_k: Point2D):
        """Creates the shape function at the triangle with corner points p_i, p_j and p_k"""

        if len(p_i) != 2 or len(p_j) != 2 or len(p_k) != 2:
            raise Exception("Points must be 2 dimensional!")
        return ShapeFunction.of_coords(p_i[0], p_i[1], p_j[0], p_j[1], p_k[0], p_k[1])

    @staticmethod
    def of_coords(x_i: float, y_i: float, x_j: float, y_j: float, x_k: float, y_k: float):
        """Creates the shape function at the triangle (x_i, y_i), (x_j, y_j), (x_k, y_k)."""

        return ShapeFunction(
            a=x_j * y_k - x_k * y_j,
            b=y_j - y_k,
            c=x_k - x_j,
            S=Mesh.elem_area((x_i, y_i), (x_j, y_j), (x_k, y_k))
        )
