import gmsh
from matplotlib import pyplot as plt
from matplotlib.pyplot import spy

from exercise_1.coax_cable import cable
from exercise_1.geometry import Geo
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

    spy(knu)
    plt.show()
    print(j_grid(mesh))
    print(grid_current(mesh))
