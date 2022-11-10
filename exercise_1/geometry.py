from typing import Tuple

import gmsh

from exercise_1.constants import l_z, r1, r2
from util.gmsh import model

gm = gmsh.model.occ


@model("coaxial_cable", show_gui=True)
def cable() -> Tuple[int, int, int]:
    """
    Creates a 2D cross-section of the coaxial_cable.
    :return: The group tags for the wire, shell and ground.
    """

    # Inner and outer cable cross-section
    circ1 = gm.add_circle(0, 0, 0, r1)
    circ2 = gm.add_circle(0, 0, 0, r2)
    loop1 = gm.add_curve_loop([circ1])
    loop2 = gm.add_curve_loop([circ2])

    # Extrude to create volume
    # gm.extrude([(1, c1)], 0, 0, -l_z / 20)
    # gm.extrude([(1, c2)], 0, 0, -l_z / 20)

    # Create plane surfaces to connect loops
    surf1 = gm.add_plane_surface([loop1])
    surf2 = gm.add_plane_surface([loop2, loop1])

    # Create physical groups
    wire: int = gmsh.model.add_physical_group(2, [surf1], name="WIRE")
    shell: int = gmsh.model.add_physical_group(2, [surf2], name="SHELL")
    gnd: int = gmsh.model.add_physical_group(1, [circ2], name="GND")
    return wire, shell, gnd
