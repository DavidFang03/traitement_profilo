import numpy as np
import matplotlib.pyplot as plt
import json
import A_utilit as utilit
from matplotlib.ticker import LogLocator


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20  # Par exemple, 14 points

json_path = "DATA3.json"
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
Rho = []

D_hyper = []
D_detect = []
U_D_detect = []
Ball_D = []
g = 9.81


def get_polyfit(xdata, ydata):
    p, cov = np.polyfit(xdata, ydata, 1, cov=True)
    (a, b) = p
    u_a, u_b = np.sqrt(np.diag(cov))
    return a, b, u_a, u_b


def get_exp_fit_points(xdata, a, b):
    return np.exp(b) * xdata**a


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

    rho = m*1e-3 / (np.pi * (ball_d*1e-2 / 2)**3 * 4 / 3)

    d_hyper = data["diameter_hyperbola"]
    d_detect = data["diameter_detected"]
    u_d_detect = data["u_diameter_detected"]

    Zc.append(zc)
    U_zc.append(u_zc)
    Bottom.append(bottom)
    M.append(m)
    H.append(h)
    E.append(e)
    D_hyper.append(d_hyper)
    D_detect.append(d_detect)
    U_D_detect.append(u_d_detect)
    Ball_D.append(ball_d)
    Rho.append(rho)

    # plt.text(e, zc, f"{m_nb:.1f} : {m:.1f}g", fontsize=9, ha='right')

E = np.array(M) * 1e-3 * g * np.array(H)

print("RHO", [f"{rho:.2f}" for rho in Rho])


class CraterPlot:
    def __init__(self, name, xdata, ydata, ydata2, yerr, yerr2, xticks=None, yticks=None):
        self.name = name
        self.fig, self.ax = plt.subplots(2, num=name)
        self.xdata = xdata
        self.ydata = ydata
        self.ydata2 = ydata2
        self.yerr = yerr
        self.yerr2 = yerr2

        self.xticks = xticks
        self.yticks = yticks

    def fit_log(self):
        self.a, self.b, self.u_a, self.u_b = get_polyfit(
            np.log(self.xdata), np.log(self.ydata))
        self.a2, self.b2, self.u_a2, self.u_b2 = get_polyfit(
            np.log(self.xdata), np.log(self.ydata2))

        self.fity = get_exp_fit_points(self.xdata, self.a, self.b)
        self.fity2 = get_exp_fit_points(self.xdata, self.a2, self.b2)

    def plot(self, labelx, labely, labely2, exponent_symbol=r"\alpha"):
        self.fig.suptitle(self.name)
        self.ax[0].errorbar(self.xdata, self.ydata, fmt="x",
                            yerr=U_zc, label="Points expérimentaux")
        self.ax[0].plot(self.xdata, self.fity,
                        label=rf"${exponent_symbol}={self.a:.2f}\pm {self.u_a2:.2f}$")
        self.ax[1].plot(self.xdata, self.ydata2, "x",
                        label="Issus du fit hyperbolique")
        self.ax[1].plot(self.xdata, self.fity2,
                        label=rf"${exponent_symbol}={self.a2:.2f}\pm {self.u_a2:.2f}$")

        self.ax[0].set_ylabel(rf'{labely}', rotation=0, ha='right')
        self.ax[1].set_ylabel(rf'{labely2}', rotation=0, ha='right')

        for ax in self.ax:
            ax.set_xscale("log")
            ax.set_yscale("log")
            # xlim_min = ax.get_xlim()[0]
            # ax.set_xlim(xlim_min, 10)
            ax.set_xlabel(rf'{labelx}')

            ax.legend(loc="lower right")

        self.fig.subplots_adjust(left=0.2, bottom=0.15,
                                 right=0.85, top=0.85, wspace=0.4, hspace=0.4)


titledepth = "\\textbf{\\Huge Profondeur en fonction de l'énergie}\n$H_c\\propto E^{\\alpha}$\n(Échelle $\\log$--$\\log$)"
titlediameter = "\\textbf{\\Huge Diamètre en fonction de l'énergie}\n$D_c\\propto E^{\\alpha}$\n(Échelle $\\log$--$\\log$)"

DepthPlot = CraterPlot(
    titledepth, E, Bottom, Zc, U_zc, None, yticks=[1, 50])
print(len(D_detect), len(E))
DiameterPlot = CraterPlot(titlediameter, E, D_detect,
                          D_hyper, U_D_detect, U_zc, yticks=[1, 200])


DepthPlot.fit_log()
DiameterPlot.fit_log()

DepthPlot.plot(r"$E$", r"$H_c$ (mm)", r"$z_c$ (mm)", exponent_symbol=r"\alpha")
DiameterPlot.plot(r"$E$", r"$D_c$",
                  r"$D^{hyperbole}_c = \frac{\sqrt{z_c^2-b^2}}{c}$", exponent_symbol=r"\alpha'")


print(np.array(D_detect)/np.array(D_hyper))
plt.show()
