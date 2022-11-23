import gmsh
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange

from exercise_1.geometry import plot_geometry, cable, element_node_tags, element_node_coords
from exercise_1.mesh import Mesh
from exercise_1.shape_function import ShapeFunction

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cable()
    mesh = Mesh.create()
    res = element_node_coords(2)
    element_areas = [ShapeFunction.area(x[0], x[1], x[2]) for x in res.values()]
    print(element_areas)
