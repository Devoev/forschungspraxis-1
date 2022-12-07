import gmsh

from exercise_1.geometry import cable
from exercise_1.mesh import Mesh

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    mesh = Mesh.create()
    print(mesh.nodes_in_group(wire))
    print(mesh.elem_in_group(wire))
    # rel = reluctivity(wire, shell)
    # mesh = Mesh.create()
    # print(rel)
