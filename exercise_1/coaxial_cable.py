from typing import Final

import matplotlib.pylab as plt
import numpy as np
import numpy.typing
from numpy.typing import ArrayLike
from scipy import constants as const

l_z: Final[float] = 300e-3
I: Final[float] = 16
r1: Final[float] = 2e-3
r2: Final[float] = 3.5e-3
mu_s: Final[float] = 5 * const.mu_0
eps_s: Final[float] = const.epsilon_0
sig_cu: Final[float] = 57.7e6


def h_field(r: ArrayLike) -> float:
    """The h field of the coaxial cable"""

    r = np.asarray(r)
    if r.any() < 0:
        raise ValueError("r must not be negative")

    in1 = r <= r1
    return in1 * _h_field_1(r) + ~in1 * _h_field_2(r)


def plot_h_field():
    """Plots the h field of the coaxial cable in the range 0 to r2."""

    r = np.linspace(0, r2, 50)
    plt.plot(r, h_field(r), 'r--')
    plt.show()


def _h_field_1(r) -> float:
    """The h field inside the conductor"""
    return I / (2 * np.pi * r1 ** 2) * r


def _h_field_2(r) -> float:
    """The h field inside the insulator"""
    return I / (2 * np.pi * r)
