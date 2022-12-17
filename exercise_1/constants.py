from typing import Final
from scipy import constants as const

l_z: Final[float] = 300e-3
I: Final[float] = 16
r1: Final[float] = 2e-3
r2: Final[float] = 3.5e-3
mu_w: Final[float] = const.mu_0
mu_s: Final[float] = 5 * const.mu_0
eps_s: Final[float] = const.epsilon_0
sig_cu: Final[float] = 57.7e6

WIRE: Final[int] = 1
SHELL: Final[int] = 2
GND: Final[int] = 0
