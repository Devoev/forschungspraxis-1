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
from exercise_1.solver_ms import solve_ms

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    mesh = Mesh.create()
    geo = Geo(mesh)

    r = np.sort(geo.r)
    a_ana = A_z(r)
    h_ana = H_phi(r)
    w_ana = W_mag()

    a = solve_ms(mesh, geo)
    plt.plot(r, -np.sort(-a), 'r--')
    plt.plot(r, a_ana, 'b--')
    plt.show()

    error = la.norm(a_ana - a) / la.norm(a_ana)
    print(f"Relative error between analytic und numerical solution: {error}")
