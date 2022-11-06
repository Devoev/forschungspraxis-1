import gmsh

from exercise_1.constants import l_z, r1, r2
from util.gmsh import model

gm = gmsh.model.occ


@model("coaxial_cable", show_gui=True)
def cable():
    c1 = gm.add_circle(0, 0, 0, r1)
    c2 = gm.add_circle(0, 0, 0, r2)
    gm.extrude([(1, c1)], 0, 0, l_z / 20)
    gm.extrude([(1, c2)], 0, 0, l_z / 20)

