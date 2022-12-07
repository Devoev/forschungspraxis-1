from dataclasses import dataclass
from typing import List, Dict, Tuple

import gmsh
import numpy as np

from exercise_1.constants import mu_w, mu_s
from exercise_1.shape_function import ShapeFunction
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

    def nodes_in_group(self, tag: int) -> np.ndarray:
        """The nodes in the given physical group.

        :param tag: The tag of the physical group minus 1.
        """
        return np.where(self.node_tags_groups[:, tag - 1])[0]

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

    def elem_in_group(self, tag: int) -> np.ndarray:
        """A list of booleans to indicate, whether the element is in the group or not.

        :param tag: The tag of the physical group minus 1.
        """

        nodes = self.nodes_in_group(tag)
        return np.asarray([set(e) <= set(nodes) for e in self.elem_to_node])

    @property
    def elem_areas(self):
        """A vector of areas for the triangle elements."""
        areas = np.zeros(self.num_elements)
        for i, nodes in enumerate(self.elem_to_node):
            x, y, z = self.node_coords[nodes]
            areas[i] = ShapeFunction.area(x, y, z)
        return areas

    @property
    def reluctivity(self) -> np.ndarray:
        """A vector with reluctivity values."""
        return self.elem_in_group(0) * mu_s + self.elem_in_group(1) * mu_w

    @staticmethod
    def create(dim: int = 2):
        """Creates an instance of a Mesh object."""
        node_tag, node, _ = msh.get_nodes()
        element_types, element_tags, node_tags_elements = msh.get_elements()
        groups = gmsh.model.get_physical_groups(dim)
        node_tags_groups = np.zeros((len(node_tag), 2))
        for i, group in enumerate(groups):
            _, tag = group
            nodes, _ = msh.get_nodes_for_physical_group(dim, tag)
            node_tags_groups[nodes - 1, i] = 1
        return Mesh(node_tag, node, element_types, element_tags, node_tags_elements, node_tags_groups)
