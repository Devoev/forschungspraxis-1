import gmsh
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange

from exercise_1.geometry import plot_geometry, cable
from exercise_1.shape_function import ShapeFunction

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = ShapeFunction.of_points((0, 0), (0, 1), (1, 0))

    ax = plt.axes(projection='3d')
    x = arange(0, 1, 0.1)
    y = arange(0, 1, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = n(X, Y)

    ax.plot_surface(X, Y, Z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
