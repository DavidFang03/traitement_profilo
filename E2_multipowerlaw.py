import numpy as np
import matplotlib.pyplot as plt
import json
import A_utilit as utilit
from matplotlib.ticker import LogLocator
import scipy.optimize

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20  # Par exemple, 14 points

json_path = "DATA2.json"
datas = []
with open(json_path, "r") as infile:
    try:
        datas = json.loads(infile.read())
    except json.JSONDecodeError:
        raise Exception(f"{json_path} is empty or not a valid JSON file")


our_mass = np.array([0, 0, 89.46, 0, 63.7, 0, 49.59,
                    0, 32.63, 0, 23.81, 0, 16.69, 0, 14.8])

ball_diameters = np.array(
    [0, 0, 2.8, 0, 2.5, 0, 2.3, 0, 2.1, 0, 1.9, 0, 1.7, 0, 1.5])

Zc = []
U_zc = []
Bottom = []
M = []
H = []
E = []

D = []
D_b = []
Rho_b = []
g = 9.81


def residuals(params, E, Db, Rhob, y):
    '''
    Residu à minimiser
    '''
    A, alpha, beta, gamma = params
    return A*E**alpha*Db**beta*Rhob**gamma - y


def fit4D(E, Db, Rhob, Y):

    # Initialisation A, alpha, beta, gamma
    params0 = (0.1, 0.2, 0.2, -0.2)
    res = scipy.optimize.least_squares(
        residuals, params0, args=(E, Db, Rhob, Y))
    J = res.jac
    JTJ = np.matmul(J.T, J)
    mse = np.sum(res.fun**2) / (len(Y) - len(params0))
    cov = mse * np.linalg.inv(JTJ)

    # Incertitudes sur les paramètres
    uncertainties = np.sqrt(np.diag(cov))
    return res.x, uncertainties


for data in datas:
    zc = np.abs(data["zc"])
    u_zc = np.abs(data["u_zc"])
    b = np.abs(data["b"])
    u_b = np.abs(data["u_b"])
    c = np.abs(data["c"])
    u_c = np.abs(data["u_c"])

    bottom = np.abs(data["bottom"])
    m_nb = int(data["mass"])
    m = our_mass[m_nb]
    h = data["height"]
    ball_d = ball_diameters[m_nb]
    e = (m*1e-3)*g*h
    rho_b = m / ((4/3) * np.pi * (ball_d/2)**3)

    d = utilit.diameter_hyperbola(zc, b, c)

    Zc.append(zc)
    U_zc.append(u_zc)
    Bottom.append(bottom)
    M.append(m)
    H.append(h)
    E.append(e)
    D.append(d)
    D_b.append(ball_d)
    Rho_b.append(rho_b)

    # plt.text(e, zc, f"{m_nb:.1f} : {m:.1f}g", fontsize=9, ha='right')

E = np.array(M) * 1e-3 * g * np.array(H)
print(np.average(E))


popt, u_popt = fit4D(E, D_b, Rho_b, D)
print(popt, u_popt)

plt.show()
