from typing import Tuple, Dict, List

import gmsh
import numpy as np
import numpy.typing
from matplotlib import pyplot as plt

from exercise_1.constants import l_z, r1, r2
from exercise_1.shape_function import ShapeFunction
from util.gmsh import model
from util.model import Point2D

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


def element_node_tags(element_type: int, tag=-1) -> Dict[int, np.ndarray]:
    """An element-node tag dict.

    :param element_type: The type of elements to store in the dict.
    :param tag: The tag of elements to get.
    """

    tags, node_tags = msh.get_elements_by_type(element_type, tag)
    node_tags = np.array_split(node_tags, len(node_tags) / 3)
    return dict(zip(tags, node_tags))


def element_node_coords(element_type: int, tag=-1) -> Dict[int, List[np.ndarray]]:
    """An element-tag to node coords dict.

    :param element_type: The type of elements to store in the dict.
    :param tag: The tag of elements to get.
    """

    element_tags, node_tags = msh.get_elements_by_type(element_type, tag)
    node_tags = np.array_split(node_tags, len(node_tags) / 3)
    coords: List[List[np.ndarray]] = list()

    for tags in node_tags:
        coords.append([msh.get_node(t)[0] for t in tags])

    return dict(zip(element_tags, coords))


def triangle_node_coords(tag=-1) -> Dict[int, Tuple[Point2D, Point2D, Point2D]]:
    """A triangle-tag to node coords dict. Uses the element_node_coords function with element_type=2.

    :param tag: The tag of elements to get.
    """
    res = element_node_coords(2, tag)
    return {k: [(x[0], x[1]) for x in v] for (k, v) in res.items()}


def element_areas():
    """A list with the areas of the triangle elements."""
    return [ShapeFunction.area(x[0], x[1], x[2]) for x in triangle_node_coords().values()]
