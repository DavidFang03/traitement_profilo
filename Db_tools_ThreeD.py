import numpy as np
import scipy.optimize
import A_utilit as utilit


def residuals(params, x1, x2, y):
    '''
    Residu à minimiser
    '''
    return utilit.hyperbolic3D(x1, x2, *params) - y


def residuals_parabola(params, x1, x2, y):
    '''
    Residu à minimiser
    '''
    return utilit.parabolic3D(x1, x2, *params) - y


def filter0(X1, X2, Y):
    '''
    Filtre les points où la valeur de Y est inférieure à 10% de la valeur minimale de Y
    '''
    mask = (Y < 0.1 * np.min(Y))
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
    res = scipy.optimize.least_squares(residuals, params0, args=(x1, x2, Y))
    J = res.jac
    JTJ = np.matmul(J.T, J)
    mse = np.sum(res.fun**2) / (len(Y) - len(params0))
    cov = mse * np.linalg.inv(JTJ)

    # Incertitudes sur les paramètres
    uncertainties = np.sqrt(np.diag(cov))
    return res.x, uncertainties


def fit3Dparabola(x1, x2, Y):

    x1, x2, Y = filter0(x1, x2, Y)

    bottom = np.min(Y)
    scenter = x1[np.argmin(Y)]
    ycenter = x2[np.argmin(Y)]
    # Initialisation zc, b, c, x0, y0
    params0 = (bottom, 1, 1, ycenter, scenter)
    res = scipy.optimize.least_squares(
        residuals_parabola, params0, args=(x1, x2, Y))
    J = res.jac
    JTJ = np.matmul(J.T, J)
    mse = np.sum(res.fun**2) / (len(Y) - len(params0))
    cov = mse * np.linalg.inv(JTJ)

    # Incertitudes sur les paramètres
    uncertainties = np.sqrt(np.diag(cov))
    return res.x, uncertainties


def compute_fitv2(popt):
    size = 100
    x1, x2, y1, y2 = utilit.zerohyperbolic(*popt)

    # S = np.linspace(x1, x2, size)
    # X = np.linspace(y1, y2, size)
    X, Y = np.meshgrid(np.linspace(x1, x2, size), np.linspace(y1, y2, size))
    ydata = utilit.hyperbolic3D(X, Y, *popt)
    print(np.shape(ydata))
    return X, Y, ydata


def compute_fitv2_p(popt):
    size = 100
    x1, x2, y1, y2 = utilit.zerohyperbolic(*popt)

    # S = np.linspace(x1, x2, size)
    # X = np.linspace(y1, y2, size)
    X, Y = np.meshgrid(np.linspace(x1, x2, size), np.linspace(y1, y2, size))
    ydata = utilit.parabolic3D(X, Y, *popt)
    print(np.shape(ydata))
    return X, Y, ydata


def compute_fitv3(X, Y, popt):
    Z = []
    for x in X:
        for y in Y:
            ydata = utilit.hyperbolic3D(X, Y, *popt)
            Z.append(ydata)

    return Z


def fit_hyperbolic2D(arrX, arrY, arrZ, xplan, zplan):
    '''
    Fit hyperbolic 2D sur un plan
    '''
    bottom = np.min(arrZ)
    threshold = 0.1*bottom
    xcenter = xplan[np.argmin(zplan)]

    maskthresh = zplan < threshold
    xplan_masked = xplan[maskthresh]
    zplan_masked = zplan[maskthresh]
    if len(zplan[maskthresh]) > 20:
        try:
            popt, pcov = scipy.optimize.curve_fit(
                utilit.hyperbolic, xplan_masked, zplan_masked, p0=(bottom, 1, 1, xcenter))
            xfit = np.linspace(
                np.min(xplan_masked), np.max(xplan_masked), 100)
            zfit = utilit.hyperbolic(xfit, *popt)
            u_popt = np.sqrt(np.diag(pcov))
        except RuntimeError:
            print("error")
            popt = None
            u_popt = None
            xfit = np.linspace(np.min(arrY), np.max(arrY), 100)
            zfit = np.ones(100) * threshold

    else:
        popt = None
        u_popt = None
        xfit = np.linspace(np.min(arrY), np.max(arrY), 100)
        zfit = np.ones(100) * threshold

    return popt, u_popt, xfit, zfit


if __name__ == "__main__":
    pass
