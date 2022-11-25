import gmsh
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange

from exercise_1.geometry import plot_geometry, cable, element_node_tags, element_node_coords, triangle_node_coords
from exercise_1.mesh import Mesh
from exercise_1.shape_function import ShapeFunction

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cable()
    print(element_node_coords(2))
    print(triangle_node_coords())
