import gmsh
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from matplotlib.pyplot import loglog

from exercise_1.analytic import A_z, H_phi, W_mag
from exercise_1.coax_cable import cable
from exercise_1.constants import l_z
from exercise_1.geometry import Geo
from exercise_1.knu_matrix import Knu
from exercise_1.mesh import Mesh
from exercise_1.mssolution import MSSolution
from exercise_1.solver_ms import solve_ms

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    mesh = Mesh.create()
    geo = Geo(mesh)

    knu = Knu(mesh, geo)

    r = geo.r
    a_ana = A_z(r)
    h_ana = H_phi(r)
    w_ana = W_mag()

    w_test = l_z ** 2 / 2 * a_ana @ knu @ a_ana  # Test if knu matrix gives the correct energy.

    solution = MSSolution(mesh, geo)
    a = solution.solve()
    w = 0.5 * np.dot(a, knu * a)

    # plt.plot(r, a/l_z, 'r--')
    # plt.plot(r, a_ana, 'b--')
    # # plt.scatter(r, a_ana)
    # # plt.scatter(r, a)
    # plt.show()

    # spy(knu, markersize=1)
    # plt.show()

    err = abs(w_ana - w) / w_ana
    print(f"Analytic magnetic energy {w_ana} and numerical magnetic energy {w}. Relative error of {err}.")

    freq = np.logspace(0, 5, 100)
    z = [abs(solution.Z_char(f)) for f in freq]
    loglog(freq, z)
    plt.show()

    print(solution.A(1e3))
    print(solution.B(1e3))

    print(np.sqrt(solution.L / solution.C))
