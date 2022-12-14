import math
from typing import Final

import matplotlib.pylab as plt
import numpy as np
import numpy.typing
from numpy.typing import ArrayLike
from scipy import constants as const

from exercise_1.constants import r1, I, r2, mu_s, mu_w
from util.array import arg_as_array


@arg_as_array()
def h_field(r: ArrayLike) -> float:
    """
    The h field of the coaxial cable

    :param r: Radius values.
    :return: Values of the magnetic field strength H.
    """

    if r.any() < 0:
        raise ValueError("r must not be negative")

    in1 = r <= r1
    return in1 * _h_field_1(r) + ~in1 * _h_field_2(r)


@arg_as_array()
def A_z(r: ArrayLike):
    """
    Analytic solution of magnetic vector potential

    Parameters
    ----------
    r : np.ndarray
        radius in [m]

    Returns
    -------
    a_z : np.ndarray
        Magnetic vector potential in [Tm]
    """

    @arg_as_array()
    def A_z_i(r: ArrayLike):
        return -I / (2 * np.pi) * (mu_w / 2 * (r ** 2 - r1 ** 2) / r1 ** 2 + mu_s * np.log(r1 / r2))

    @arg_as_array()
    def A_z_a(r: ArrayLike):
        return -mu_s * I / (2 * np.pi) * np.log(r / r2)

    condition = r < r1
    return condition * A_z_i(r) + (~condition) * A_z_a(r)


def plot_h_field():
    """Plots the h field of the coaxial cable in the range 0 to r2."""

    r = np.linspace(0, r2, 50)
    plt.plot(r, h_field(r), 'r--')
    plt.show()


@arg_as_array()
def _h_field_1(r: ArrayLike) -> float:
    """The h field inside the conductor

    :param r: Radius values.
    :return: Values of the magnetic field strength H inside the conductor.
    """
    return I / (2 * np.pi * r1 ** 2) * r


@arg_as_array()
def _h_field_2(r: ArrayLike) -> float:
    """The h field inside the insulator. If r=0, the field strength will be nan.

    :param r: Radius values.
    :return: Values of the magnetic field strength H inside the insulator.
    """

    r = np.asarray(
        list(
            map(lambda x: x if x > 0 else math.nan, r)
        )
    )
    return I / (2 * np.pi * r)
