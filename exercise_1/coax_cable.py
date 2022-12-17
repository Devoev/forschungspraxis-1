from typing import Tuple, List

import gmsh
import numpy as np
from matplotlib import pyplot as plt

from exercise_1.analytic import H_phi
from exercise_1.constants import r1, r2, WIRE, GND, SHELL
from util.gmsh import model

gm = gmsh.model.occ
msh = gmsh.model.mesh


@model(name="coaxial_cable", dim=2, show_gui=False)
def cable(tags: Tuple[int, int, int] = (WIRE, SHELL, GND)) -> Tuple[int, int, int]:
    """
    Creates a 2D cross-section of the coaxial_cable.

    :param tags: The group tags for the wire, shell and ground.
    :return: The group tags.
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
    wire: int = gmsh.model.add_physical_group(dim=2, tags=[surf1], tag=tags[0], name="WIRE")
    shell: int = gmsh.model.add_physical_group(dim=2, tags=[surf2], tag=tags[1], name="SHELL")
    gnd: int = gmsh.model.add_physical_group(dim=1, tags=[loop2], tag=tags[2], name="GND")
    return wire, shell, gnd


def plot_h_field():
    """Plots the h field of the coaxial cable in the range 0 to r2."""

    r = np.linspace(0, r2, 50)
    plt.plot(r, H_phi(r), 'r--')
    plt.show()


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
