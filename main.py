import gmsh

from exercise_1.geometry import cable, reluctivity
from exercise_1.mesh import Mesh

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    mesh = Mesh.create()
    rel = reluctivity(wire, shell)
