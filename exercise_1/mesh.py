from dataclasses import dataclass
from typing import List, Dict, Tuple

import gmsh
import numpy as np

from util.model import Point2D

msh = gmsh.model.mesh


@dataclass
class Mesh:
    node_tag_data: np.ndarray
    node_data: np.ndarray
    elementTypes: np.ndarray
    element_tags: np.ndarray
    node_tags_elements: np.ndarray
    node_tags_groups: np.ndarray

    @property
    def num_node(self) -> int:
        """The number of nodes."""
        return int(len(self.node_data) / 3)

    @property
    def node_coords(self) -> List[Point2D]:
        """The list of node coordinates."""

        node = np.reshape(self.node_data, (self.num_node, 3))
        # coordinates of nodes. x-coordinate in first column
        # and y-coordinate in second column
        node = node[:, 0:2]
        return node

    @property
    def node_tags(self) -> List[int]:
        """A list of node tags. Node tags of gmsh -1"""
        # ID of nodes
        node_tag = self.node_tag_data - np.ones(len(self.node_tag_data))
        node_tag = node_tag.astype('int')
        np.put_along_axis(self.node_coords, np.c_[node_tag, node_tag], self.node_coords, axis=0)
        return node_tag

    @property
    def nodes(self) -> Dict[int, Point2D]:
        """A node tag-coord dict."""
        return dict(zip(self.node_tags, self.node_coords))

    @property
    def ind_elements(self) -> np.ndarray:
        """The indices of triangle elements."""
        return np.where(self.elementTypes == 2)[0]

    @property
    def elem_tags(self) -> np.ndarray:
        """The tags of all triangle elements."""
        return self.element_tags[self.ind_elements[0]]

    @property
    def elem_nodes(self) -> np.ndarray:
        """A vector of all nodes forming a triangle element.
        Shape: (e11,e12,e13,e21,e22,e23,...)
        """
        return np.array(self.node_tags_elements[self.ind_elements[0]])

    @property
    def num_elements(self) -> int:
        """The number of elements."""
        return int(len(self.elem_nodes) / 3)

    @property
    def elems(self) -> Dict[int, np.ndarray]:
        """A element tag-node tag dict."""
        node_tags = np.array_split(self.elem_nodes, self.num_elements)
        return dict(zip(self.elem_tags, node_tags))

    @property
    def elem_to_node(self) -> np.ndarray:
        """A matrix of elements. Each row contains the node tags of the element vertices."""

        # Associate elements (triangles) and their respective nodes.
        # Connection between elements and nodes.
        # Each line contains the indices of the contained nodes
        elem_to_node = np.reshape(self.elem_nodes, (self.num_elements, 3)) - np.ones(
            (self.num_elements, 1))
        elem_to_node = elem_to_node.astype('int')
        return elem_to_node

    @property
    def edges(self) -> np.ndarray:
        """A matrix of edges. Each row contains the start and end node tags."""
        return np.r_[self.elem_to_node[:, [0, 1]],
                     self.elem_to_node[:, [1, 2]],
                     self.elem_to_node[:, [0, 2]]]

    @property
    def edge_to_node(self) -> np.ndarray:
        """A matrix of edges. Each row contains the start and end node tags.
        Duplicate elements are removed and edges are sorted."""
        return np.unique(np.sort(self.edges), axis=0)

    def elem_in_group(self, tag: int) -> List[bool]:
        """A list of booleans to indicate, whether the element is in the group or not.

        :param tag: The tag of the physical group.
        """

        # TODO: Remove gmsh call
        nodes = msh.get_nodes_for_physical_group(2, tag)[0] - 1
        return [set(e) <= set(nodes) for e in self.elem_to_node]

    @staticmethod
    def create(dim: int = 2):
        """Creates an instance of a Mesh object."""
        node_tag, node, _ = msh.get_nodes()
        element_types, element_tags, node_tags_elements = msh.get_elements()
        groups = gmsh.model.get_physical_groups(dim)
        node_tags_groups = np.zeros(len(node_tag), 2)
        for i, _ in enumerate(groups):
            node_tags_groups[i] = msh.get_nodes_for_physical_group(dim, i)
        return Mesh(node_tag, node, element_types, element_tags, node_tags_elements, node_tags_groups)


def mat_nodes() -> np.ndarray:
    """ Function to create a matrix mat_nodes containing all coordinates of the nodes in the mesh msh """

    # Extract the data of all nodes in the mesh
    nodes_mesh = gmsh.model.mesh.get_nodes(dim=-1, tag=-1, includeBoundary=False, returnParametricCoord=True)

    # Create a matrix mat_nodes containing all coordinates of the nodes in the mesh
    num_nodes = int(len(nodes_mesh[0]))
    mat_nodes = np.zeros((num_nodes, 3))

    for i in range(0, num_nodes, 1):
        mat_nodes[i, 0] = nodes_mesh[1][3 * i]
        mat_nodes[i, 1] = nodes_mesh[1][3 * i + 1]
        mat_nodes[i, 2] = nodes_mesh[1][3 * i + 2]

    return mat_nodes


def mat_tri() -> np.ndarray:
    """ Function to create the matrix mat_tri containing the indices of all nodes of a triangle from mat_nodes in msh"""

    # Extract the data of all triangles in the mesh
    tri_mesh = gmsh.model.mesh.getElementsByType(2, tag=-1, task=0, numTasks=1)

    # Creating a matrix connecting node indices in mat_nodes to triangles
    num_tri = int(len(tri_mesh[0]))
    mat_tri = np.zeros((num_tri, 3))

    # Assign the indices of the nodes to the triangles
    for i in range(0, num_tri, 1):
        mat_tri[i, 0] = tri_mesh[1][3 * i] - 1
        mat_tri[i, 1] = tri_mesh[1][3 * i + 1] - 1
        mat_tri[i, 2] = tri_mesh[1][3 * i + 2] - 1

    return mat_tri


def tri_to_ph2d() -> np.ndarray:
    """ Function for creating a vector mapping which triangle belongs to which physical group in 2D """

    # Create mat_tri
    _mat_tri = mat_tri()

    # Create the "empty" vector
    tri_to_ph_2d = np.zeros(_mat_tri.shape[0])

    # Extract all 2D physical groups and store their tags in l_ph_groups_2d
    l_ph_groups_2d = []
    for phgroup in gmsh.model.getPhysicalGroups(dim=2):
        l_ph_groups_2d.append(phgroup[1])

    # Assign the indices of all triangles in mat_triangles to a physical group
    for g in l_ph_groups_2d:

        # Create a set containing the tags of all nodes from physical group g (Decrease the value of every tag by one
        # because gmash starts counting from 1 and not zero)
        nodes_g = gmsh.model.mesh.getNodesForPhysicalGroup(2, g)[0]
        for i in range(0, len(nodes_g), 1):
            nodes_g[i] -= 1
        nodes_g = set(nodes_g)

        # Assign all correct triangles to the ph group g by comparing the nodes of the triangle to the nodes in g
        for i in range(0, _mat_tri.shape[0], 1):

            setnodes = {_mat_tri[i][0], _mat_tri[i][1], _mat_tri[i][2]}
            if setnodes.intersection(nodes_g) == setnodes:
                tri_to_ph_2d[i] = g

    return tri_to_ph_2d
