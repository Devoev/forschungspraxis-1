import gmsh
import numpy as np
from matplotlib import pyplot as plt

from exercise_1.h_field import plot_h_field
from exercise_1.geometry import cable, node_coords

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    xw, yw, zw = node_coords(2, wire)
    xs, ys, zs = node_coords(2, shell)
    xg, yg, _ = node_coords(1, gnd)

    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection='3d')

    # Creating plot
    ax.plot_trisurf(xw, yw, zw, color="blue")
    ax.plot_trisurf(xs, ys, zs, color="green", alpha=0.5)
    ax.plot(xg, yg, color="red")

    # show plot
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
