import numpy as np


def z(x, xl, theta):
    return (xl - x) * np.tan(theta)


def hyperbolic(r, zc, b, c, r0):
    return zc + np.sqrt(b**2 + c**2 * (r - r0) ** 2)
