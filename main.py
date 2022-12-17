import gmsh
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import spy
import scipy.sparse.linalg as las
import numpy.linalg as la

from exercise_1.coax_cable import cable, plot_geometry
from exercise_1.constants import GND, SHELL, WIRE
from exercise_1.geometry import Geo
from exercise_1.analytic import A_z, H_phi, W_mag
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

    # TODO: Boundary conditions!
    # a = las.spsolve(knu, b)

    r = np.zeros(mesh.num_node)
    for i, coord in enumerate(mesh.node_coords):
        x, y = coord
        r[i] = np.sqrt(x**2 + y**2)

    a_ana = A_z(r)
    h_ana = H_phi(r)
    w_ana = W_mag()

    plot_geometry()

    # spy(knu)
    # plt.show()

    # error = la.norm(a_ana - a) / la.norm(a_ana)
    # print(f"Relative error between analytic und numerical solution: {error}")
