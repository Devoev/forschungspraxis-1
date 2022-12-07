import gmsh

from exercise_1.geometry import cable
from exercise_1.mesh import Mesh

msh = gmsh.model.mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wire, shell, gnd = cable()
    res = gmsh.model.mesh.get_nodes_for_physical_group(2, wire)
    mesh = Mesh.create()
    print(mesh.node_tags_groups)
    print(mesh.node_tags_elements)
    # rel = reluctivity(wire, shell)
    # mesh = Mesh.create()
    # print(rel)
