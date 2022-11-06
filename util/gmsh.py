import sys

import gmsh


def model(name: str, dim: int = 3, show_gui: bool = False):
    """Indicates that the function generates a gmsh model.

    :param name The name of the gmsh model.
    :param dim: Dimension of the mesh.
    :param show_gui: Whether to show the gui.
    """

    def _model(func):
        def call(*args, **kwargs):
            gmsh.initialize()
            gmsh.model.add(name)

            func(*args, **kwargs)

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(dim)

            if '-nopopup' not in sys.argv and show_gui:
                gmsh.fltk.run()

            gmsh.finalize()
        return call
    return _model
