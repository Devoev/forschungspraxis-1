from dataclasses import dataclass
from functools import cached_property
from typing import List, Dict, Tuple

import gmsh
import numpy as np

from util.model import Point2D

msh = gmsh.model.mesh


@dataclass
class Mesh:
    """An object for handling the mesh elements and nodes.

    node_tags_group: A matrix of size (N,G). Indicates, whether the node is included in the physical group or not.
    """
    node_tag_data: np.ndarray
    node_data: np.ndarray
    elementTypes: np.ndarray
    element_tags: np.ndarray
    node_tags_elements: np.ndarray
    node_tags_groups: np.ndarray

    @cached_property
    def num_node(self) -> int:
        """The number of nodes."""
        return int(len(self.node_data) / 3)

    @cached_property
    def node_coords(self) -> np.ndarray:
        """The list of node coordinates."""

        node = np.reshape(self.node_data, (self.num_node, 3))
        # coordinates of nodes. x-coordinate in first column
        # and y-coordinate in second column
        node = node[:, 0:2]
        return node

    @cached_property
    def node_tags(self) -> np.ndarray:
        """A list of node tags. Node tags of gmsh -1"""
        # ID of nodes
        node_tag = self.node_tag_data - np.ones(len(self.node_tag_data))
        node_tag = node_tag.astype('int')
        np.put_along_axis(self.node_coords, np.c_[node_tag, node_tag], self.node_coords, axis=0)
        return node_tag

    @cached_property
    def nodes(self) -> Dict[int, Point2D]:
        """A node tag-coord dict."""
        return dict(zip(self.node_tags, self.node_coords))

    def nodes_in_group(self, tag: int) -> np.ndarray:
        """The nodes in the given physical group.

        :param tag: The tag of the physical group.
        """
        return np.where(self.node_tags_groups[:, tag])[0]

    @cached_property
    def ind_elements(self) -> np.ndarray:
        """The indices of triangle elements."""
        return np.where(self.elementTypes == 2)[0]

    @cached_property
    def elem_tags(self) -> np.ndarray:
        """The tags of all triangle elements. **Not** necessarily starting from zero."""
        return self.element_tags[self.ind_elements[0]]

    @cached_property
    def elem_nodes(self) -> np.ndarray:
        """A vector of all nodes forming a triangle element.
        Shape: (e11,e12,e13,e21,e22,e23,...)
        """
        return np.array(self.node_tags_elements[self.ind_elements[0]])

    @cached_property
    def num_elems(self) -> int:
        """The number of elements."""
        return int(len(self.elem_nodes) / 3)

    @cached_property
    def elems_dict(self) -> Dict[int, np.ndarray]:
        """**DEPRECATED** A element tag-node tag dict."""
        node_tags = np.array_split(self.elem_nodes, self.num_elems)
        return dict(zip(self.elem_tags, node_tags))

    @cached_property
    def elems(self) -> np.ndarray:
        """A matrix of elements. Each row contains the node tags of the element vertices."""

        # Associate elements (triangles) and their respective nodes.
        # Connection between elements and nodes.
        # Each line contains the indices of the contained nodes
        elem_to_node = np.reshape(self.elem_nodes, (self.num_elems, 3)) - np.ones(
            (self.num_elems, 1))
        elem_to_node = elem_to_node.astype('int')
        return elem_to_node

    @cached_property
    def edges(self) -> np.ndarray:
        """A matrix of edges. Each row contains the start and end node tags."""
        return np.r_[self.elems[:, [0, 1]],
                     self.elems[:, [1, 2]],
                     self.elems[:, [0, 2]]]

    @cached_property
    def edge_to_node(self) -> np.ndarray:
        """A matrix of edges. Each row contains the start and end node tags.
        Duplicate elements are removed and edges are sorted."""
        return np.unique(np.sort(self.edges), axis=0)

    def elem_in_group(self, tag: int) -> np.ndarray:
        """A list of booleans to indicate, whether the element is in the group or not.

        :param tag: The tag of the physical group minus 1.
        """

        nodes = self.nodes_in_group(tag)
        return np.asarray([set(e) <= set(nodes) for e in self.elems])

    @staticmethod
    def elem_area(p_i: Point2D, p_j: Point2D, p_k: Point2D):
        """Computes the area of a triangle with the corner points p_i, p_j and p_k."""
        ax = p_j[0] - p_i[0]
        ay = p_j[1] - p_i[1]
        bx = p_k[0] - p_i[0]
        by = p_k[1] - p_i[1]
        return 0.5 * abs(ax * by - ay * bx)

    @cached_property
    def elem_areas(self):
        """A vector of areas for the triangle elements."""
        areas = np.zeros(self.num_elems)
        for i, nodes in enumerate(self.elems):
            x, y, z = self.node_coords[nodes]
            areas[i] = self.elem_area(x, y, z)
        return areas

    @staticmethod
    def coeffs_of(p_j: Point2D, p_k: Point2D) -> Tuple[float, float, float]:
        """The coefficients for a shape function."""
        x_j, y_j = p_j
        x_k, y_k = p_k
        a: float = x_j * y_k - x_k * y_j
        b: float = y_j - y_k
        c: float = x_k - x_j
        return a, b, c

    @cached_property
    def coeffs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A tuple of (E,3) matrices with the coefficients a,b,c for the shape functions."""
        a = np.zeros([self.num_elems, 3])
        b = np.zeros([self.num_elems, 3])
        c = np.zeros([self.num_elems, 3])
        for i, nodes in enumerate(self.elems):
            # Node indices
            n1, n2, n3 = self.node_coords[nodes]

            # a,b,c coefficients
            a1, b1, c1 = Mesh.coeffs_of(n2, n3)  # n1
            a2, b2, c2 = Mesh.coeffs_of(n3, n1)  # n2
            a3, b3, c3 = Mesh.coeffs_of(n1, n2)  # n3

            # Set values of a,b,c vectors
            a[i, :] = np.array([a1, a2, a3])
            b[i, :] = np.array([b1, b2, b3])
            c[i, :] = np.array([c1, c2, c3])
        return a, b, c

    @staticmethod
    def create():
        """Creates an instance of a Mesh object."""
        node_tag, node, _ = msh.get_nodes()
        element_types, element_tags, node_tags_elements = msh.get_elements()
        groups = gmsh.model.get_physical_groups()
        node_tags_groups = np.zeros((len(node_tag), len(groups)))
        for i, group in enumerate(groups):
            dim, tag = group
            nodes, _ = msh.get_nodes_for_physical_group(dim, tag)
            node_tags_groups[nodes - 1, i] = 1
        return Mesh(node_tag, node, element_types, element_tags, node_tags_elements, node_tags_groups)
