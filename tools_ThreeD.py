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
    x1 = []
    x2 = []
    y = []
    for i in range(len(X1)):
        for j in range(len(X2)):
            if Y[i][j] < 0.1*np.min(Y):
                x1.append(X1[i])
                x2.append(X2[j])
                y.append(Y[i][j])
    # on un array x1 correspondant à arrS, un array x2 correspondant à rangeY
    return x1, x2, y


def extract_bottom_3D_Profile(X1, X2, Y):
    '''
    Extrait les points où la valeur de Y est inférieure à 10% de la valeur minimale de Y (90% les points les plus profonds)
    '''
    # Convertir les listes en tableaux NumPy si ce n'est pas déjà fait
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)

    # Calculer le seuil
    threshold = 0.1 * np.min(Y)

    # Créer un masque pour les valeurs de Y inférieures au seuil
    mask = Y < threshold

    # Extraire les valeurs correspondantes
    x1 = X1[mask.any(axis=1)]
    x2 = X2[mask.any(axis=0)]
    y = Y[mask]

    return x1, x2, y


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


def compute_fit(popt):
    size = 100
    x1, x2, y1, y2 = utilit.zerohyperbolic(*popt)
    ydata = np.zeros((size, size))
    S = np.linspace(x1, x2, size)
    X = np.linspace(y1, y2, size)
    for i in range(size):
        for j in range(size):
            ydata[i, j] = utilit.hyperbolic3D(S[i], X[j], *popt)

    return S, X, ydata


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
