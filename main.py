import gmsh
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import spy
import scipy.sparse.linalg as la

from exercise_1.coax_cable import cable
from exercise_1.geometry import Geo
from exercise_1.h_field import A_z
from exercise_1.knu_matrix import Knu_e, Knu
from exercise_1.load_vector import j_grid, grid_current
from exercise_1.mesh import Mesh

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    mesh = Mesh.create()
    geo = Geo(mesh)

    knu = Knu(mesh, geo)
    b = grid_current(mesh)

    a = la.spsolve(knu, b)

    r = np.zeros(mesh.num_node)
    for i, coord in enumerate(mesh.node_coords):
        x, y = coord
        r[i] = np.sqrt(x**2 + y**2)

    a_ana = A_z(r)
