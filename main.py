import gmsh

from exercise_1.h_field import plot_h_field
from exercise_1.geometry import cable


msh = gmsh.model.mesh


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cable()
    node_tag, node, _ = msh.get_nodes()
    element_types, element_tags, node_tags_elements = msh.get_elements()
    print(node_tag, node, element_types, element_tags, node_tags_elements)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
