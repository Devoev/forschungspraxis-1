import math
from typing import Final

import matplotlib.pylab as plt
import numpy as np
import numpy.typing
from numpy.typing import ArrayLike
from scipy import constants as const

from exercise_1.constants import r1, I, r2, mu_s, mu_w, l_z
from util.array import arg_as_array


@arg_as_array()
def H_phi(r: ArrayLike):
    """
    Analytic solution for the phi-component of the magnetic field of the coaxial cable.

    :param r: Radius values.
    :return: Values of the magnetic field strength H.
    """

    @arg_as_array()
    def H_phi_i(r: ArrayLike):
        return I / (2 * np.pi * r1 ** 2) * r

    @arg_as_array()
    def H_phi_a(r: ArrayLike):
        return I / (2 * np.pi * r)

    condition = r < r1
    return condition * H_phi_i(r) + (~condition) * H_phi_a(r)


@arg_as_array()
def A_z(r: ArrayLike):
    """
    Analytic solution for the z-component of magnetic vector potential of the coaxial cable.

    :param r: Radius values.
    :return: Values of the magnetic vector potential A.
    """

    @arg_as_array()
    def A_z_i(r: ArrayLike):
        return -I / (2 * np.pi) * (mu_w / 2 * (r ** 2 - r1 ** 2) / r1 ** 2 + mu_s * np.log(r1 / r2))

    @arg_as_array()
    def A_z_a(r: ArrayLike):
        return -mu_s * I / (2 * np.pi) * np.log(r / r2)

    condition = r < r1
    return condition * A_z_i(r) + (~condition) * A_z_a(r)


def W_mag() -> float:
    """The magnetic energy of the coaxial cable."""
    return I ** 2 * l_z * mu_w * (1 + 20*math.log(r2/r1)) / (16*np.pi)
