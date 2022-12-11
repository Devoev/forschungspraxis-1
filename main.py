import gmsh

from exercise_1.coax_cable import cable
from exercise_1.knu_matrix import Knu_e
from exercise_1.mesh import Mesh

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    mesh = Mesh.create()
    print(Knu_e(1))
