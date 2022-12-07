import gmsh
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange

from exercise_1.geometry import plot_geometry, cable, element_node_tags, element_node_coords, triangle_node_coords, \
    element_areas, reluctivity
from exercise_1.mesh import Mesh, tri_to_ph2d, mat_tri
from exercise_1.shape_function import ShapeFunction

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    res = gmsh.model.mesh.get_nodes_for_physical_group(2, wire)
    mesh = Mesh.create()
    print(mesh.node_tags_groups)
    # rel = reluctivity(wire, shell)
    # mesh = Mesh.create()
    # print(rel)
