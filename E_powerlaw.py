import numpy as np
import matplotlib.pyplot as plt
import json
import A_utilit as utilit


json_path = "DATA2.json"
datas = []
with open(json_path, "r") as infile:
    try:
        datas = json.loads(infile.read())
    except json.JSONDecodeError:
        raise Exception(f"{json_path} is empty or not a valid JSON file")


our_mass = np.array([0, 0, 89.46, 0, 63.7, 0, 49.59,
                    0, 32.63, 0, 23.81, 0, 16.69, 0, 14.8])

Zc = []
U_zc = []
Bottom = []
M = []
H = []
E = []

D = []
g = 9.81


def get_polyfit(xdata, ydata):
    p, cov = np.polyfit(xdata, ydata, 1, cov=True)
    (a, b) = p
    u_a, u_b = np.sqrt(np.diag(cov))
    return a, b, u_a, u_b


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
    e = (m*1e-3)*g*h

    d = utilit.diameter_hyperbola(zc, b, c)

    Zc.append(zc)
    U_zc.append(u_zc)
    Bottom.append(bottom)
    M.append(m)
    H.append(h)
    E.append(e)
    D.append(d)

    # plt.text(e, zc, f"{m_nb:.1f} : {m:.1f}g", fontsize=9, ha='right')

E = np.array(M) * 1e-3 * g * np.array(H)


class CraterPlot:
    def __init__(self, name, xdata, ydata, ydata2, yerr, yerr2):
        self.name = name
        self.fig, self.ax = plt.subplots(2, num=name)
        self.xdata = xdata
        self.ydata = ydata
        self.ydata = ydata2
        self.logxdata = np.log(xdata)
        self.logydata = np.log(ydata)
        self.logydata2 = np.log(ydata2)
        self.yerr = yerr
        self.yerr2 = yerr2

    def fit_log(self):
        self.a, self.b, self.u_a, self.u_b = get_polyfit(
            self.logxdata, self.logydata)
        self.a2, self.b2, self.u_a2, self.u_b2 = get_polyfit(
            self.logxdata, self.logydata2)

        self.logfity = self.a * self.logxdata + self.b
        self.logfity2 = self.a2 * self.logxdata + self.b2

    def plot(self, labelx, labely, labely2):
        self.ax[0].errorbar(self.logxdata, self.logydata, fmt="x", yerr=U_zc)
        self.ax[0].plot(self.logxdata, self.logfity,
                        label=rf"${self.a:.2f}\pm {self.u_a2:.2f}$")
        self.ax[1].plot(self.logxdata, self.logydata2, "x")
        self.ax[1].plot(self.logxdata, self.logfity2,
                        label=rf"${self.a2:.2f}\pm {self.u_a2:.2f}$")

        self.ax[0].set_xlabel(labelx)
        self.ax[0].set_ylabel(labely)
        self.ax[0].set_title(self.name)

        self.ax[1].set_xlabel(labelx)
        self.ax[1].set_ylabel(labely2)
        self.ax[1].set_title(self.name)

        self.ax[0].legend(loc="upper left")
        self.ax[1].legend(loc="upper left")


DepthPlot = CraterPlot(
    "Profondeur", E, Zc, Bottom, U_zc, None)
DiameterPlot = CraterPlot(
    "Diamètre", E, D, D, U_zc, None)

DepthPlot.fit_log()
DiameterPlot.fit_log()

DepthPlot.plot("log(E)", "log(Zc)", "log(Bottom)")
DiameterPlot.plot("log(E)", "log(D)", "log(Bottom)")

plt.show()
