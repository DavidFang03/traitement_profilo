import numpy as np


def z(x, xl, theta):
    return (x-xl) * np.tan(theta)


def hyperbolic(r, zc, b, c, r0):
    return zc + np.sqrt(b**2 + c**2 * (r - r0) ** 2)


def hyperbolic3D(x, y, zc, b, c, x0, y0):
    return zc + np.sqrt(b**2 + c**2 * (((x - x0) ** 2)+((y-y0)**2)))


def zerohyperbolic(zc, b, c, x0, y0):
    '''
    '''
    r = np.sqrt((zc**2-b**2)/np.abs(c))
    x1 = x0 - r
    x2 = x0 + r
    y1 = y0 - r
    y2 = y0 + r
    return x1, x2, y1, y2


def zerohyperbolicv2(zc, b, c, x0, y0):
    '''
    '''
    r = np.sqrt((zc**2-b**2)/np.abs(c))
    x1 = x0 - 2*r
    x2 = x0 + 2*r
    y1 = y0 - 2*r
    y2 = y0 + 2*r
    return x1, x2, y1, y2
