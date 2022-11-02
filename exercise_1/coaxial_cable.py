from typing import Final

import matplotlib.pylab as plt
import numpy as np
from scipy import constants as const

l_z: Final[float] = 300e-3
I: Final[float] = 16
r1: Final[float] = 2e-3
r2: Final[float] = 3.5e-3
mu_s: Final[float] = 5 * const.mu_0
eps_s: Final[float] = const.epsilon_0
sig_cu: Final[float] = 57.7e6


def h_field(r) -> float:
    """The h field of the coaxial cable"""

    if r.any() < 0:
        raise ValueError("r must not be negative")

    r01 = r <= r1
    r12 = np.logical_and(r1 < r, r <= r2)
    return r01 * __h_field_1__(r) + r12 * __h_field_2__(r)


def plot_h_field():
    """Plots the h field of the coaxial cable in the range 0 to r2."""

    r = np.linspace(0, r2, 50)
    plt.plot(r, h_field(r), 'r--')
    plt.show()


def __h_field_1__(r) -> float:
    """The h field inside the conductor"""
    return I / (2 * np.pi * r1 ** 2) * r


def __h_field_2__(r) -> float:
    """The h field inside the insulator"""
    return I / (2 * np.pi * r)
