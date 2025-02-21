import numpy as np


def z(x, xl, theta):
    return (xl - x) * np.tan(theta)


def hyperbolic(r, zc, b, c, r0):
    return zc + np.sqrt(b**2 + c**2 * (r - r0) ** 2)


def hyperbolic3D(x, y, zc, b, c, x0, y0):
    return zc + np.sqrt(b**2 + c**2 * (((x - x0) ** 2)+((y-y0)**2)))
