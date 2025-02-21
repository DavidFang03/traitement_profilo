import numpy as np
from scipy.optimize import least_squares
from utilit import hyperbolic3D


def model(params, x1, x2):
    # a, b, c = params
    return hyperbolic3D(x1, x2, *params)
    return a * x1 + b * x2 + c

# Résidu à minimiser


def residuals(params, x1, x2, y):
    return model(params, x1, x2) - y


# def fit3D(x1, x2, Y):
#     y = Y.ravel()
#     X1, X2 = np.meshgrid(x1, x2)

#     # Fit
#     x1_flat, x2_flat = X1.ravel(), X2.ravel()
#     # print(x1_flat, x2_flat)
#     # print(np.shape(x1_flat), np.shape(x2_flat), np.shape(y), np.shape(Y))
#     params0 = [1, 1, 1, 1, 1]  # Initialisation
#     res = least_squares(residuals, params0, args=(x1_flat, x2_flat, y))
#     return res.x


# def fit3D(x1, x2, Y):
#     y = Y.ravel()

#     X1, X2 = np.meshgrid(x1, x2)

#     mask = Y != 0
#     Y = Y[mask]
#     xdata = np.array((X1.ravel(), X2.ravel()))[mask]

#     # Fit
#     x1_flat, x2_flat = xdata
#     # print(x1_flat, x2_flat)
#     # print(np.shape(x1_flat), np.shape(x2_flat), np.shape(y), np.shape(Y))
#     params0 = [1, 1, 1, 1, 1]  # Initialisation
#     res = least_squares(residuals, params0, args=(x1_flat, x2_flat, y))
#     return res.x

def filter0(X1, X2, Y):
    x1 = []
    x2 = []
    y = []
    for i in range(len(X1)):
        for j in range(len(X2)):
            if Y[i][j] < 0.1*np.min(Y):
                x1.append(X1[i])
                x2.append(X2[j])
                y.append(Y[i][j])
    # ! on un array x1 correspondant à la liste des s, un array x2 correspondant à l'ensemble des x
    return x1, x2, y


def fit3D(x1, x2, Y):

    x1, x2, Y = filter0(x1, x2, Y)

    # X1, X2 = np.meshgrid(x1, x2)

    # print(x1_flat, x2_flat)
    # print(np.shape(x1_flat), np.shape(x2_flat), np.shape(y), np.shape(Y))
    params0 = [-50, 1, 1, 100, 100]  # Initialisation zc, b, c, x0, y0
    res = least_squares(residuals, params0, args=(x1, x2, Y))
    return res.x


def compute_fit(S, X, popt):
    # size = np.max((len(X), len(S)))
    size = 2000
    x1 = np.linspace(np.min(S), np.max(S), size)
    x2 = np.linspace(np.min(X), np.max(X), size)

    y = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            y[i, j] = hyperbolic3D(x1[i], x2[j], *popt)
    return x1, x2, y


if __name__ == "__main__":
    x1 = np.arange(1)
    x2 = np.arange(2)
    Y = np.random.rand(1, 2)  # Exemple de données aplaties
    print(fit3D(x1, x2, Y))
