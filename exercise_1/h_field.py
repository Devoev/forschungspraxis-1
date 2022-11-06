import math
from typing import Final

import matplotlib.pylab as plt
import numpy as np
import numpy.typing
from numpy.typing import ArrayLike
from scipy import constants as const

from util.array import arg_as_array

l_z: Final[float] = 300e-3
I: Final[float] = 16
r1: Final[float] = 2e-3
r2: Final[float] = 3.5e-3
mu_s: Final[float] = 5 * const.mu_0
eps_s: Final[float] = const.epsilon_0
sig_cu: Final[float] = 57.7e6


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
