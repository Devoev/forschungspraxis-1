from numpy import sqrt


def v_phase(eps: float, mu: float) -> float:
    """The phase velocity of the EM wave for the given material parameters."""
    return 1 / sqrt(eps * mu)


def wavelength(f: float, v: float) -> float:
    """The wavelength for the given frequency and phase velocity."""
    return v / f
