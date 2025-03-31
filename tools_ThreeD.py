import numpy as np
from scipy.optimize import least_squares
import utilit


def residuals(params, x1, x2, y):
    '''
    Residu à minimiser
    '''
    return utilit.hyperbolic3D(x1, x2, *params) - y


def filter0(X1, X2, Y):
    '''
    Filtre les points où la valeur de Y est inférieure à 10% de la valeur minimale de Y
    '''
    mask = (Y < 0.1 * np.min(Y)
)
    x1 = X1[mask]
    x2 = X2[mask]

    return x1, x2, Y[mask]

def fit3D(x1, x2, Y):

    x1, x2, Y = filter0(x1, x2, Y)

    bottom = np.min(Y)
    scenter = x1[np.argmin(Y)]
    ycenter = x2[np.argmin(Y)]
    # Initialisation zc, b, c, x0, y0
    params0 = (bottom, 1, 1, ycenter, scenter)
    res = least_squares(residuals, params0, args=(x1, x2, Y))
    J = res.jac
    JTJ = np.matmul(J.T, J)
    mse = np.sum(res.fun**2) / (len(Y) - len(params0))
    cov = mse * np.linalg.inv(JTJ)

    # Incertitudes sur les paramètres
    uncertainties = np.sqrt(np.diag(cov))
    return res.x, uncertainties


def compute_fitv2(popt):
    size = 100
    x1, x2, y1, y2 = utilit.zerohyperbolicv2(*popt)

    # S = np.linspace(x1, x2, size)
    # X = np.linspace(y1, y2, size)
    S, X = np.meshgrid(np.linspace(x1, x2, size), np.linspace(y1, y2, size))
    ydata = utilit.hyperbolic3D(S, X, *popt)
    print(np.shape(ydata))
    return S, X, ydata


if __name__ == "__main__":
    pass
