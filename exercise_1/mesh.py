from dataclasses import dataclass

import gmsh
import numpy as np

msh = gmsh.model.mesh


@dataclass
class Mesh:
    node_tag_data: np.ndarray
    node_data: np.ndarray
    elementTypes: np.ndarray
    element_tags: np.ndarray
    nodeTags_elements: np.ndarray

    # treat nodes
    @property
    def num_node(self):
        # number of nodes
        return int(len(self.node_data) / 3)

    @property
    def node(self):
        # nodes
        node = np.reshape(self.node_data, (self.num_node, 3))
        # coordinates of nodes. x-coordinate in first column
        # and y-coordinate in second column
        node = node[:, 0:2]
        return node

    @property
    def node_tag(self):
        # ID of nodes
        node_tag = self.node_tag_data - np.ones(len(self.node_tag_data))
        node_tag = node_tag.astype('int')
        np.put_along_axis(self.node, np.c_[node_tag, node_tag], self.node, axis=0)
        return node_tag

        # treat elements

    @property
    def ind_elements(self):
        # index of elements
        return np.where(self.elementTypes == 2)[0]

    @property
    def elements(self):
        # elements
        return np.array(self.nodeTags_elements[self.ind_elements[0]])

    @property
    def num_elements(self):
        # number of elements
        return int(len(self.elements) / 3)

    @property
    def elem_to_node(self):
        # Associate elements (triangles) and their respective nodes.
        # Connection between elements and nodes.
        # Each line contains the indices of the contained nodes
        elem_to_node = np.reshape(self.elements, (self.num_elements, 3)) - np.ones(
            (self.num_elements, 1))
        elem_to_node = elem_to_node.astype('int')
        return elem_to_node

    @property
    def all_edges(self):
        # Contains all edges
        return np.r_[self.elem_to_node[:, [0, 1]],
                     self.elem_to_node[:, [1, 2]],
                     self.elem_to_node[:, [0, 2]]]

    @property
    def edge_to_node(self):
        # Associate edges and their respective nodes.
        return np.unique(np.sort(self.all_edges), axis=0)

    @staticmethod
    def create():
        """Creates an instance of a Mesh object."""
        node_tag, node, _ = msh.get_nodes()
        element_types, element_tags, node_tags_elements = msh.get_elements()
        return Mesh(node_tag, node, element_types, element_tags, node_tags_elements)
