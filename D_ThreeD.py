import matplotlib.pyplot as plt
import numpy as np
import Db_tools_ThreeD as tools_ThreeD
import A_utilit as utilit
from matplotlib import cm
import matplotlib.colors
import json

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 16  # Par exemple, 14 points

index_plan = 0

scales = None
with open("SCALES.json", "r") as scalesfile:
    try:
        scales = json.loads(scalesfile.read())
    except json.JSONDecodeError:
        raise Exception(f"SCALES.json is empty or not a valid JSON file")

vmin = -60*scales["2803"]
print(vmin)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=5)

DATAPATH = "./DATA2.json"


def return_update_func(fig, ax):
    def update_view(event):
        if event.key == '4':  # left
            ax.view_init(elev=ax.elev, azim=ax.azim - 5)
        elif event.key == '6':  # right
            ax.view_init(elev=ax.elev, azim=ax.azim + 5)
        elif event.key == '8':  # up
            ax.view_init(elev=ax.elev + 5, azim=ax.azim)
        elif event.key == '5':  # down
            ax.view_init(elev=ax.elev - 5, azim=ax.azim)
        fig.canvas.draw()
        # print(ax.elev, ax.azim)
    return update_view


def format_popt(popt, u_popt):
    if popt is not None:
        formatted_popt = rf"$z_c={popt[0]:.2f} \pm {u_popt[0]:.2f} \, b={popt[1]:.2f} \pm {u_popt[1]:.2f} \, c={popt[2]:.2f} \pm {u_popt[2]:.2f}$"
    else:
        formatted_popt = "No fit"
    return formatted_popt


def return_update_plan(Data, sorted_unique_arrS):
    ax = Data.axplan
    fig = Data.figplan
    arrS = Data.arrS
    arrY = Data.arrY
    Z = Data.points
    tdpopt = Data.popt
    u_tdpopt = Data.uncertainties

    def update_plan(event):
        global index_plan

        if event == 'right' or event.key == 'right':
            index_plan = (index_plan + 1) % len(sorted_unique_arrS)
            print(event, event == 'right' or event.key ==
                  'right', sorted_unique_arrS[index_plan], index_plan)
        elif event == 'left' or event.key == 'left':
            index_plan = (index_plan - 1) % len(sorted_unique_arrS)
            print(event, event == 'left' or event.key ==
                  'left', sorted_unique_arrS[index_plan], index_plan)

        mask = (arrS == sorted_unique_arrS[index_plan])

        popt, u_popt, xfit, zfit = tools_ThreeD.fit_hyperbolic2D(
            arrS, arrY, Z, arrY[mask], Z[mask])

        points_fit_exact = []
        s = sorted_unique_arrS[index_plan]
        for y in arrY[mask]:
            points_fit_exact.append(utilit.hyperbolic3D(s, y, *tdpopt))

        formatted_tdpopt = format_popt(popt, u_popt)
        formatted_popt = format_popt(tdpopt, u_tdpopt)

        ax.cla()
        ax.plot(arrY[mask], Z[mask], 'o', color='blue', label="Données")
        ax.plot(xfit, zfit, "--", color='red',
                label=f"Fit 2D {formatted_popt}")
        ax.plot(arrY[mask], points_fit_exact, "x",
                color='green', alpha=0.4, label=f"Fit 3D {formatted_tdpopt}")

        ax.legend(loc="upper right")
        ax.set_title(r"Profil dans un plan $x$ fixé")
        # ax.set_title(
        #     f"s={sorted_unique_arrS[index_plan]} → {sorted_unique_arrS[-1]} ({len(sorted_unique_arrS)})")
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_xlim(np.min(arrY), np.max(arrY))
        ax.set_ylim(np.min(Z), np.max(Z))
        ax.set_aspect('equal')
        fig.canvas.draw()

        return popt, u_popt

    return update_plan


class ThreeD_Data:
    def __init__(self, npz_path, **kwargs):
        '''
        Import data from npz file.
        Attributes :
        - arrS
        - Y
        - height
        - scale
        - theta
        - profiles
        '''
        # ! IMPORT

        self.npz_path = npz_path
        data = np.load(npz_path, allow_pickle=True)

        params_npz = {}

        params_npz["vidpath"] = str(data["vidpath"])
        params_npz["timestamp"] = str(data["timestamp"])
        params_npz["npz_timestamp"] = utilit.get_timestamp()

        # params_npz["Y"] = data["Y"]
        params_npz["height"] = float(data["height"])
        if None in data["scale"]:
            params_npz["scale"] = 0
        else:
            params_npz["scale"] = float(data["scale"])
        params_npz["theta_deg"] = float(data["theta_deg"])
        params_npz["mass"] = int(data["mass"])
        params_npz["date"] = str(data["date"])

        for key, value in params_npz.items():
            setattr(self, key, value)

        scale = scales[self.date]
        self.params_npz = params_npz
        self.arrS = data["arrS"]*scale
        self.arrY = data["arrY"]*scale
        self.points = data["points"]*scale

        self.theta = np.radians(self.theta_deg)
        self.bottom = np.min(self.points)

        # ! Sous-échantillonnage

        self.step = 100  # Tracer une frame toutes les step_s

        self.reduced_arrS = self.arrS[::self.step]
        self.reduced_arrY = self.arrY[::self.step]
        self.reduced_points = self.points[::self.step]
        # return
        # ! INIT PLOT
        self.fig_profile = plt.figure(num='3D Profil')  # affiche data
        self.fig_fit = plt.figure(num='3D Fit')  # affiche fit
        self.fig_both = plt.figure(num='3D Profil & Fit')

        # return

        self.ax_profile = self.fig_profile.add_subplot(111, projection='3d')
        self.ax_fit = self.fig_fit.add_subplot(111, projection='3d')
        self.ax_both = self.fig_both.add_subplot(111, projection='3d')

        self.figplan, self.axplan = plt.subplots(num='Plan')

        self.list_of_popts = []
        self.list_of_u = []

        return

    def plot_profile3D(self):
        # self.Y_grid, self.S_grid = np.meshgrid(
        #     self.reduced_arrY, self.reduced_arrS)

        self.trisurf_profile = self.ax_profile.plot_trisurf(
            self.reduced_arrS, self.reduced_arrY, self.reduced_points, cmap='viridis', alpha=0.9, norm=norm)

        return

    def fit3D(self):

        self.popt, self.uncertainties = tools_ThreeD.fit3D(
            self.arrS, self.arrY, self.points)

        self.zc = self.popt[0]

        self.S_fit, self.X_fit, self.points_fit_reduced = tools_ThreeD.compute_fitv2(
            self.popt)

        # self.points_fit_exact = tools_ThreeD.compute_fitv3(
        #     self.arrS, self.arrY, self.popt)

        return

    def plot_fit3D(self):
        # self.X_fit_grid, self.S_fit_grid = np.meshgrid(self.X_fit, self.S_fit)
        # self.ax_fit.plot_surface(self.S_fit_grid, self.X_fit_grid,
        #                          self.profile_fit_reduced, cmap='viridis')
        self.trisurf_fit = self.ax_fit.plot_surface(self.S_fit, self.X_fit,
                                                    self.points_fit_reduced, cmap='viridis', alpha=0.9, norm=norm)
        print(self.popt, self.uncertainties)

    def view2D(self):
        pass

    def plot_both(self):
        self.trisurf_both = self.ax_both.plot_trisurf(
            self.reduced_arrS, self.reduced_arrY, self.reduced_points, cmap='viridis', alpha=0.7, norm=norm)
        self.ax_both.plot_surface(self.S_fit, self.X_fit,
                                  self.points_fit_reduced, cmap=cm.inferno, alpha=0.7)
        # Ca me semble inutile de se faire chier a tracer le fit seulement dans la zone. Autant tout tracer + accelere avec numpy. Jouer avec transparence, cmap, et mettre un bouton toggle puor masquer ou non.

    def end_plot(self):

        all_trifigs = [self.fig_profile, self.fig_fit,
                       self.fig_both]
        all_triaxes = [self.ax_profile, self.ax_fit, self.ax_both]
        all_trisurfs = [self.trisurf_profile,
                        self.trisurf_fit, self.trisurf_both]

        all_cbs = []

        for trifig, triax, trisurf in zip(all_trifigs, all_triaxes, all_trisurfs):
            trifig.set_edgecolor('none')
            trisurf.set_edgecolor('none')
            triax.set_title(f"{self.vidpath}, {self.height}")
            triax.axis('equal')
            triax.set_xlabel("$x$ (mm)")
            triax.set_ylabel('$y$ (mm)')
            triax.set_zlabel('$z$ (mm)', rotation=90)
            cb = trifig.colorbar(trisurf,
                                 ax=triax, label='$z$ (mm)')
            all_cbs.append(cb)

            cb.ax.axline(
                (0, self.bottom), (1, self.bottom), color='red', linestyle='--')

            cb.ax.annotate(
                "Fond du profil (Données)",
                xy=(0, self.bottom),
                xycoords='data',
                # Ajustez cette valeur pour déplacer le texte vers la gauche
                xytext=(-20, 0),
                textcoords='offset points',
                ha='right',
                va='center',
                fontsize=14,
                color='red',
                arrowprops=dict(arrowstyle="->", color='red')
            )

            trifig.tight_layout()

            trifig.canvas.mpl_connect(
                'key_press_event', return_update_func(trifig, triax))

        all_cbs[2].ax.annotate(
            "Fond du profil (Fit)",
            xy=(1, self.zc),
            xycoords='data',
            # Ajustez cette valeur pour déplacer le texte vers la gauche
            xytext=(20, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=14,
            color='orange',
            arrowprops=dict(arrowstyle="->", color='orange')
        )
        all_cbs[2].ax.axline(
            (0, self.zc), (1, self.zc), color='orange', linestyle='--')

        # plt.show()

        # self.fig_profile.canvas.mpl_connect(
        #     'key_press_event', return_update_func(self.fig_profile, self.ax_profile))
        # self.fig_fit.canvas.mpl_connect(
        #     'key_press_event', return_update_func(self.fig_fit, self.ax_fit))
        # self.fig_both.canvas.mpl_connect(
        #     'key_press_event', return_update_func(self.fig_both, self.ax_both))

    def from_everything_to_formatted_data(self):
        '''
        On a besoin de :
        - masse
        - hauteur
        - scale
        - popt (zc, b, c) #
        - uncertainties #
        - theta
        - date
        - timestamp
        - npz_timestamp
        '''
        big_data = self.params_npz.copy()

        big_data["zc"] = float(self.zc)  # zc,
        big_data["b"] = float(self.popt[1])  # b,
        big_data["c"] = float(self.popt[2])  # c,

        big_data["u_zc"] = float(self.uncertainties[0])  # u_zc,
        big_data["u_b"] = float(self.uncertainties[1])  # u_b,
        big_data["u_c"] = float(self.uncertainties[2])  # u_c,

        big_data["bottom"] = np.min(self.points)
        return big_data

    def export_data(self):
        '''
        self.popt, self.uncertainties
        '''

        big_data = self.from_everything_to_formatted_data()
        print(big_data)
        utilit.add_to_history(big_data, DATAPATH)

        self.params_npz["popt"] = list(self.popt)
        self.params_npz["uncertainties"] = list(self.uncertainties)

    def update_history_npz(self):
        utilit.add_to_history(self.params_npz, "history_npz.json")

    def plot_plan(self):
        sorted_unique_arrS = np.sort(np.unique(self.arrS))
        return_update_plan(self, sorted_unique_arrS)('right')
        self.figplan.canvas.mpl_connect(
            'key_press_event', return_update_plan(self, sorted_unique_arrS))

    def auto_plan(self):
        """
        Automatically browses through the plans.
        """
        sorted_unique_arrS = np.sort(np.unique(self.arrS))
        f_update_plan = return_update_plan(self, sorted_unique_arrS)

        for i in range(len(sorted_unique_arrS)):
            popt, u_popt = f_update_plan('right')
            if popt is not None:
                if u_popt[0] < abs(popt[0]):
                    self.list_of_popts.append(popt)
                    self.list_of_u.append(u_popt)
        self.fig_popt, self.ax_popt = plt.subplots(4, num='Fits des plans')
        self.ax_popt[0].set_title('zc')
        self.ax_popt[1].set_title('b')
        self.ax_popt[2].set_title('c')
        self.ax_popt[3].set_title('y0')

        arrpopt = np.array(self.list_of_popts)
        arru = np.array(self.list_of_u)
        self.ax_popt[0].errorbar(range(len(self.list_of_popts)), arrpopt[:, 0], yerr=arru[:, 0],
                                 fmt='o', label=f'zc')
        self.ax_popt[0].axline((0, np.min(self.points)), (len(
            self.list_of_popts), np.min(self.points)), color='red', linestyle='--', label="minimum")
        self.ax_popt[0].axline((0, self.zc), (len(
            self.list_of_popts), self.zc), color='green', linestyle='-', alpha=0.5, label="zc 3D")

        self.ax_popt[0].legend()


def RUN_TRID(path, name):
    three_d_data = ThreeD_Data(path)
    three_d_data.plot_profile3D()
    three_d_data.fit3D()
    three_d_data.plot_fit3D()
    three_d_data.plot_both()
    # three_d_data.plot_plan()
    # three_d_data.auto_plan()
    three_d_data.end_plot()
    three_d_data.fig_both.savefig(f"final_3d_imgs/{name}.png")
    plt.close("all")
    three_d_data.export_data()
    three_d_data.update_history_npz()


if __name__ == "__main__":
    # Example usage
    npz_path = "datanpz/m2_2803_h2-008_m2.npz"
    three_d_data = ThreeD_Data(npz_path)
    three_d_data.plot_profile3D()
    three_d_data.fit3D()
    three_d_data.plot_fit3D()
    three_d_data.plot_both()
    # three_d_data.plot_plan()
    # three_d_data.auto_plan()
    three_d_data.end_plot()
    three_d_data.export_data()
    three_d_data.update_history_npz()
    plt.show()
