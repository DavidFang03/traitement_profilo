import numpy as np
import scipy.optimize
import A_utilit as utilit


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
    res = scipy.optimize.least_squares(residuals, params0, args=(x1, x2, Y))
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


def fit_hyperbolic2D(img):
    '''
    Fit hyperbolic 2D sur le profil
    '''
    bottom = np.min(img.profile)
    center = img.rangeY[np.argmin(img.profile)]
    p0 = [bottom, 1, 1, center]

    if np.max(img.profile) > np.abs(np.min(img.profile)):
        img.popt = (0, 0, 0, 0)
        return

    try:
        img.popt, img.pcov = scipy.optimize.curve_fit(
            # Ne pas oublier de préciser p0 sinon galère
            utilit.hyperbolic, img.rangeY[img.i1:img.i2], img.profile[img.i1:img.i2], p0=p0)
    except RuntimeError as e:
        if "Optimal parameters not found" in str(e):
            print(
                f"Frame {img.frame_nb} - Echec fit : setting to (0,0,0,0). Check THETA ({img.theta}) or ROTATE")
            img.popt = (img.hyperbolic_threshold*np.min(img.profile), 0, 0, 0)
            np.savetxt("test.txt", np.column_stack(
                (img.rangeY[img.i1:img.i2], img.profile[img.i1:img.i2])))
        else:
            raise  # Relève l'erreur si ce n'est pas celle attendue

    return


if __name__ == "__main__":
    pass
