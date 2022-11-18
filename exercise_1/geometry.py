from typing import Tuple

import gmsh
import numpy as np
import numpy.typing
from matplotlib import pyplot as plt

from exercise_1.constants import l_z, r1, r2
from util.gmsh import model

gm = gmsh.model.occ
msh = gmsh.model.mesh


@model(name="coaxial_cable", dim=2, show_gui=False)
def cable() -> Tuple[int, int, int]:
    """
    Creates a 2D cross-section of the coaxial_cable.
    :return: The group tags for the wire, shell and ground.
    """

    # Inner and outer cable cross-section
    circ1 = gm.add_circle(0, 0, 0, r1)
    circ2 = gm.add_circle(0, 0, 0, r2)
    loop1 = gm.add_curve_loop([circ1])
    loop2 = gm.add_curve_loop([circ2])

    # Extrude to create volume
    # gm.extrude([(1, c1)], 0, 0, -l_z / 20)
    # gm.extrude([(1, c2)], 0, 0, -l_z / 20)

    # Create plane surfaces to connect loops
    surf1 = gm.add_plane_surface([loop1])
    surf2 = gm.add_plane_surface([loop2, loop1])

    # Create physical groups
    gmsh.model.occ.synchronize()
    wire: int = gmsh.model.add_physical_group(2, [surf1], name="WIRE")
    shell: int = gmsh.model.add_physical_group(2, [surf2], name="SHELL")
    gnd: int = gmsh.model.add_physical_group(1, [loop2], name="GND")
    return wire, shell, gnd


def plot_geometry():
    """
    Plots the cable geometry and visualizes the physical groups.
    """
    wire, shell, gnd = cable()
    xw, yw, zw = node_coords(2, wire)
    xs, ys, zs = node_coords(2, shell)
    xg, yg, _ = node_coords(1, gnd)

    # Creating plot
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(xw, yw, zw, color="blue")
    ax.plot_trisurf(xs, ys, zs, color="green", alpha=0.5)
    ax.plot(xg, yg, color="red")

    # show plot
    plt.show()


def node_coords(dim=-1, tag=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the coordinates of the nodes of the mesh.
    :return: The x,y,z coordinates.
    """

    node_tags, nodes = msh.get_nodes_for_physical_group(dim, tag)
    num_nodes = len(node_tags)

    i = np.arange(0, num_nodes)
    x = nodes[3 * i]
    y = nodes[3 * i + 1]
    z = nodes[3 * i + 2]
    return x, y, z
