import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import Db_tools_ThreeD as tools_ThreeD
import A_utilit as utilit
from matplotlib import cm
import matplotlib.colors
import json
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20  # Par exemple, 14 points

index_plan = 0

scales = None
with open("SCALES.json", "r") as scalesfile:
    try:
        scales = json.loads(scalesfile.read())
    except json.JSONDecodeError:
        raise Exception(f"SCALES.json is empty or not a valid JSON file")

mass = np.array([0, 0, 89.46, 0, 63.7, 0, 49.59,
                 0, 32.63, 0, 23.81, 0, 16.69, 0, 14.8])

vmin = -25
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=2)

DATAPATH = "./DATA3.json"


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


def return_update_plan(Data, sorted_unique_arrY):
    ax = Data.axplan
    fig = Data.figplan
    arrX = Data.arrX
    arrY = Data.arrY
    Z = Data.points
    tdpopt = Data.popt
    u_tdpopt = Data.uncertainties
    bottom = Data.bottom

    def update_plan(event):
        '''
        y fixé
        '''
        global index_plan

        if event == 'right' or event.key == 'right':
            index_plan = (index_plan + 1) % len(sorted_unique_arrY)
            print(event, event == 'right' or event.key ==
                  'right', sorted_unique_arrY[index_plan], index_plan)
        elif event == 'left' or event.key == 'left':
            index_plan = (index_plan - 1) % len(sorted_unique_arrY)
            print(event, event == 'left' or event.key ==
                  'left', sorted_unique_arrY[index_plan], index_plan)

        mask = (arrY == sorted_unique_arrY[index_plan])

        popt, u_popt, xfit, zfit = tools_ThreeD.fit_hyperbolic2D(
            arrX, arrY, Z, arrX[mask], Z[mask])

        points_fit_exact = []
        y = sorted_unique_arrY[index_plan]
        for x in arrX[mask]:
            points_fit_exact.append(utilit.hyperbolic3D(x, y, *tdpopt))

        formatted_tdpopt = format_popt(popt, u_popt)
        formatted_popt = format_popt(tdpopt, u_tdpopt)
        ax.clear()
        # ax.cla()
        ax.plot(arrX[mask], Z[mask], 'o', color='blue', label="Données")
        ax.plot(xfit, zfit, "--", color='red',
                label=f"Fit 2D")  # {formatted_popt}
        ax.plot(arrX[mask], points_fit_exact, "x",
                # {formatted_tdpopt}
                color='green', alpha=0.4, label=f"Fit 3D")

        ax.legend(bbox_to_anchor=(1, 0))
        ax.set_title(r"Profil dans un plan $x$ fixé")
        # ax.set_title(
        #     f"s={sorted_unique_arrY[index_plan]} → {sorted_unique_arrY[-1]} ({len(sorted_unique_arrY)})")
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_xlim(np.min(arrY), np.max(arrY))
        ax.set_ylim(bottom, 30)
        ax.set_aspect('equal')

        fig.canvas.draw()

        return popt, u_popt

    return update_plan


def draw_colorbar_datatext(cb, z, text="$H_c$ : Fond du profil (Données)"):
    cb.ax.annotate(
        text,
        xy=(0, z),
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


def draw_colorbar_fittext(cb, z, text="$z_c$ : Fond du profil (Fit)"):
    cb.ax.annotate(
        text,
        xy=(1, z),
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


def draw_colorbar_line(cb, z, color, alpha=1):
    cb.ax.axline(
        (0, z), (1, z), color=color, linestyle='--', alpha=alpha)


def draw_infos_on_colorsbars(all_cbs, zc, bottom, u_zc=0):
    ecartrelatif = np.abs((zc-bottom)/np.max((zc, bottom)))
    print(f"Ecart relatif {ecartrelatif*100:0f}%")
    draw_colorbar_line(all_cbs[0], bottom, color="red")
    draw_colorbar_line(all_cbs[1], zc, color="orange")

    draw_colorbar_line(all_cbs[2], bottom, color="red")
    draw_colorbar_line(all_cbs[2], zc, color="orange")

    draw_colorbar_datatext(all_cbs[0], bottom)
    draw_colorbar_fittext(all_cbs[1], zc)

    draw_colorbar_fittext(all_cbs[2], zc)
    draw_colorbar_datatext(all_cbs[2], bottom)

    if u_zc > 0:
        return
        draw_colorbar_line(all_cbs[2], zc+u_zc, color="green", alpha=1)
        draw_colorbar_line(all_cbs[2], zc-u_zc, color="green", alpha=1)


class ThreeD_Data:
    def __init__(self, npz_path, **kwargs):
        '''
        Import data from npz file.
        Attributes :
        - arrX
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

        params_npz["theta_deg"] = float(data["theta_deg"])
        params_npz["mass"] = int(data["mass"])
        params_npz["date"] = str(data["date"])
        params_npz["vidname"] = str(data["vidname"])

        for key, value in params_npz.items():
            setattr(self, key, value)

        scale = scales[self.date]
        self.params_npz = params_npz
        self.arrX = data["arrX"]*scale
        self.arrY = data["arrY"]*scale
        self.points = data["points"]*scale

        self.theta = np.radians(self.theta_deg)
        self.bottom = np.min(self.points)

        # ! Sous-échantillonnage

        self.step = 100  # Tracer une frame toutes les step_s

        self.reduced_arrX = self.arrX[::self.step]
        self.reduced_arrY = self.arrY[::self.step]
        self.reduced_points = self.points[::self.step]
        # return
        # ! INIT PLOT
        self.fig_profile = plt.figure(
            num='3D Profil', figsize=(16, 16*9/16))  # affiche data
        self.fig_fit = plt.figure(
            num='3D Fit', figsize=(16, 16*9/16))  # affiche fit
        self.fig_fit_p = plt.figure(
            num='3D Fit PARABOLA', figsize=(16, 16*9/16))  # affiche fit
        self.fig_both = plt.figure(
            num='3D Profil & Fit', figsize=(16, 16*9/16))

        # return

        self.ax_profile = self.fig_profile.add_subplot(111, projection='3d')
        self.ax_fit = self.fig_fit.add_subplot(111, projection='3d')
        self.ax_fit_p = self.fig_fit_p.add_subplot(111, projection='3d')
        self.ax_both = self.fig_both.add_subplot(111, projection='3d')

        self.figplan, self.axplan = plt.subplots(num='Plan')

        self.list_of_popts = []
        self.list_of_u = []

        return

    def plot_profile3D(self):
        # self.Y_grid, self.S_grid = np.meshgrid(
        #     self.reduced_arrY, self.reduced_arrX)

        self.trisurf_profile = self.ax_profile.plot_trisurf(
            self.reduced_arrX, self.reduced_arrY, self.reduced_points, cmap='viridis', alpha=0.9, norm=norm)

        return

    def fit3D(self):

        self.popt, self.uncertainties = tools_ThreeD.fit3D(
            self.arrX, self.arrY, self.points)

        self.zc = self.popt[0]
        self.b = self.popt[1]
        self.c = self.popt[2]
        self.u_zc = self.uncertainties[0]

        self.X_fit, self.Y_fit, self.points_fit_reduced = tools_ThreeD.compute_fitv2(
            self.popt)

        # self.points_fit_exact = tools_ThreeD.compute_fitv3(
        #     self.arrX, self.arrY, self.popt)

        return

    def fit3Dparabola(self):

        self.popt_p, self.uncertainties_p = tools_ThreeD.fit3Dparabola(
            self.arrX, self.arrY, self.points)

        self.zc_p = self.popt_p[0]
        self.b_p = self.popt_p[1]
        self.c_p = self.popt_p[2]
        self.u_zc_p = self.uncertainties_p[0]

        self.X_fit_p, self.Y_fit_p, self.points_fit_reduced_p = tools_ThreeD.compute_fitv2_p(
            self.popt_p)

    def plot_fit3D(self):
        # self.Y_fit_grid, self.X_fit_grid = np.meshgrid(self.Y_fit, self.X_fit)
        # self.ax_fit.plot_surface(self.X_fit_grid, self.Y_fit_grid,
        #                          self.profile_fit_reduced, cmap='viridis')
        self.trisurf_fit = self.ax_fit.plot_surface(self.X_fit, self.Y_fit,
                                                    self.points_fit_reduced, cmap='inferno', alpha=0.9, norm=norm)
        print(self.popt, self.uncertainties)

    def plot_fit3Dparabola(self):
        # self.Y_fit_grid, self.X_fit_grid = np.meshgrid(self.Y_fit, self.X_fit)
        # self.ax_fit.plot_surface(self.X_fit_grid, self.Y_fit_grid,
        #                          self.profile_fit_reduced, cmap='viridis')
        self.trisurf_fit_p = self.ax_fit_p.plot_surface(self.X_fit_p, self.Y_fit_p,
                                                        self.points_fit_reduced_p, cmap='inferno', alpha=0.9, norm=norm)

    def view2D(self):
        pass

    def plot_both(self):
        self.trisurf_both = self.ax_both.plot_trisurf(
            self.reduced_arrX, self.reduced_arrY, self.reduced_points, cmap='viridis', alpha=0.7, norm=norm)
        self.ax_both.plot_surface(self.X_fit, self.Y_fit,
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
            triax.set_title(
                f"$m_b={mass[self.mass]:.1f}$ g\n $H={self.height:.2f}$ m")
            triax.axis('equal')
            triax.set_xlabel("$x$ (mm)", labelpad=15)
            triax.set_ylabel('$y$ (mm)', labelpad=15)
            triax.zaxis.set_rotate_label(False)
            triax.set_zlabel('$z$ (mm)', rotation=0, labelpad=15)

            # z_label = triax.zaxis.get_label()
            # z_label.set_rotation(90)  # Set the desired rotation angle

            # divider = make_axes_locatable(triax)

            # # Ajout d'un axe pour la colorbar
            # cax = divider.append_axes("right", size="5%", pad=0.2)

            cb = trifig.colorbar(trisurf,
                                 ax=triax, pad=0.15)
            cb.ax.set_ylabel('$z$ (mm)', rotation=0, labelpad=35, loc="top")
            all_cbs.append(cb)

            trifig.tight_layout()

            trifig.canvas.mpl_connect(
                'key_press_event', return_update_func(trifig, triax))

        draw_infos_on_colorsbars(all_cbs, self.zc, self.bottom, self.u_zc)

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
        - diametre hyperbole
        - diametre detecte
        '''
        big_data = self.params_npz.copy()

        big_data["zc"] = float(self.zc)  # zc,
        big_data["b"] = float(self.popt[1])  # b,
        big_data["c"] = float(self.popt[2])  # c,

        big_data["u_zc"] = float(self.uncertainties[0])  # u_zc,
        big_data["u_b"] = float(self.uncertainties[1])  # u_b,
        big_data["u_c"] = float(self.uncertainties[2])  # u_c,

        big_data["bottom"] = np.min(self.points)

        big_data["diameter_hyperbola"] = self.diameter_hyperbola
        big_data["diameter_detected"] = self.diameter_detected
        big_data["u_diameter_detected"] = self.u_diameter_detected
        return big_data

    def export_data(self):
        '''
        self.popt, self.uncertainties, diameters
        '''

        big_data = self.from_everything_to_formatted_data()
        print(big_data)
        utilit.add_to_history(big_data, DATAPATH)

        self.params_npz["popt"] = list(self.popt)
        self.params_npz["uncertainties"] = list(self.uncertainties)
        self.params_npz["diameter_hyperbola"] = self.diameter_hyperbola
        self.params_npz["diameter_detected"] = self.diameter_detected

    def update_history_npz(self):
        utilit.add_to_history(self.params_npz, "history_npz.json")

    def plot_plan(self):
        sorted_unique_arrY = np.sort(np.unique(self.arrY))
        return_update_plan(self, sorted_unique_arrY)('right')
        self.figplan.canvas.mpl_connect(
            'key_press_event', return_update_plan(self, sorted_unique_arrY))

    def auto_plan(self):
        """
        Automatically browses through the plans.
        """
        sorted_unique_arrY = np.sort(np.unique(self.arrX))
        f_update_plan = return_update_plan(self, sorted_unique_arrY)

        for i in range(len(sorted_unique_arrY)):
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

    def anim_profile(self, name, nb_azim_angles=360, pbars={}):
        if name == "profile":
            fig, ax = self.fig_profile, self.ax_profile
        elif name == "fit":
            fig, ax = self.fig_fit, self.ax_fit
        elif name == "both":
            fig, ax = self.fig_both, self.ax_both
        else:
            raise ValueError(
                "Invalid name. Choose 'profile', 'fit', or 'both'.")

        def update(frame):
            ax.view_init(elev=30, azim=70+frame)
            if name in pbars:
                pbars[name].update()
            return fig,
        ani = FuncAnimation(fig, update, frames=range(0, nb_azim_angles))
        ani.save(f'./anim_profile/anim_{name}.gif',
                 fps=30, savefig_kwargs={"transparent": True})

    def get_diameters(self):
        '''
        Get the diameter of the hyperbola and the detected diameter.
        '''
        X, Y, Z = self.arrX, self.arrY, self.points
        fig, ax = plt.subplots(figsize=(16, 16*9/16), num="Diamètre")
        bottom = np.min(Z)
        mask1 = Z > 0.01*bottom
        mask2 = Z < np.abs(0.01*bottom)
        mask = np.logical_and(mask1, mask2)
        Xcircle = X[mask]
        Ycircle = Y[mask]
        ax.plot(Xcircle, Ycircle, "x",
                color="gray", alpha=0.1)
        hyperbola_radius = utilit.diameter_hyperbola(self.zc, self.b, self.c)/2
        xcenter = X[np.argmin(Z)]
        ycenter = Y[np.argmin(Z)]

        coeff_radius = 1.2
        Radiuses_squared = (X-xcenter)**2 + (Y-ycenter)**2
        mask_radius = Radiuses_squared < (coeff_radius*hyperbola_radius)**2

        circle = matplotlib.patches.Circle(
            (xcenter, ycenter), hyperbola_radius, edgecolor='green', facecolor='none', label=r"$D^{hyperbole}_c = \frac{\sqrt{z_c^2-b^2}}{c}$", zorder=5)
        ax.add_patch(circle)
        circle2 = matplotlib.patches.Circle(
            (xcenter, ycenter), coeff_radius*hyperbola_radius, edgecolor='green', facecolor='none', label=f"${coeff_radius}D^{{hyperbole}}_c$", zorder=5, linestyle='--')
        ax.add_patch(circle2)

        Xcircle = X[mask_radius]
        Ycircle = Y[mask_radius]
        Zcircle = Z[mask_radius]

        final_mask = np.logical_and(
            Zcircle > 0.01*bottom, Zcircle < np.abs(0.01*bottom))

        Xcircle = Xcircle[final_mask]
        Ycircle = Ycircle[final_mask]

        ax.plot(Xcircle, Ycircle, "x", color="blue",
                label="$z\\in [0.99z_c,0.01|z_c|] $")
        ax.set_xlabel('$X$ (mm)')
        ax.set_ylabel('$Y$ (mm)')

        # new_x_center, new_y_center = np.mean(
        #     Xcircle, dtype='float64'), np.mean(Ycircle, dtype='float64')
        new_x_center, new_y_center = xcenter, ycenter
        distances_from_center = (
            Xcircle-new_x_center)**2 + (Ycircle-new_y_center)**2
        detected_radius = np.mean(
            np.sqrt(distances_from_center, dtype="float64"))
        u_detected_radius = np.std(
            np.sqrt(distances_from_center, dtype="float64"))

        ax.plot((new_x_center,), (new_y_center,), "o")
        circle3 = matplotlib.patches.Circle(
            (new_x_center, new_y_center), detected_radius, edgecolor='red', facecolor='none', label=f"$r_{{detecte}}$", zorder=5, linewidth=4)
        ax.add_patch(circle3)

        ax.plot((1.*xcenter, 1.*xcenter), (ycenter-detected_radius, ycenter +
                                           detected_radius), color='red')
        ax.plot((xcenter-detected_radius, xcenter +
                 detected_radius), (1.*ycenter, 1.*ycenter),  color='red')
        # ax.plot((0.95*xcenter, 0.95*xcenter), (ycenter-hyperbola_radius, ycenter +
        #                                        hyperbola_radius), color='black')

        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1, 1))

        self.diameter_hyperbola = hyperbola_radius*2
        self.diameter_detected = detected_radius*2
        self.u_diameter_detected = u_detected_radius*2
        fig.savefig(f"diameters_img/{self.date}-{self.vidname}.png")
        return self.diameter_hyperbola, self.diameter_detected


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
    three_d_data.get_diameters()

    three_d_data.export_data()
    # three_d_data.update_history_npz()


if __name__ == "__main__":
    # # Example usage
    npz_path = "final_datanpz/m2-riv_centre_2803_h5-907_m2.npz"
    three_d_data = ThreeD_Data(npz_path)
    three_d_data.plot_profile3D()

    three_d_data.fit3D()
    three_d_data.plot_fit3D()

    three_d_data.fit3Dparabola()
    three_d_data.plot_fit3Dparabola()
    three_d_data.plot_both()
    three_d_data.plot_plan()

    three_d_data.end_plot()
    # three_d_data.anim_profile("profile")
    # # three_d_data.anim_profile("fit")
    # # three_d_data.anim_profile("both")
    # # three_d_data.auto_plan()
    # plt.close("all")
    three_d_data.get_diameters()

    # # three_d_data.export_data()
    # # three_d_data.update_history_npz()

    plt.show()
    # npz_name = "m2_2803_h2-008_m2.npz"
    # npz_path = f"./final_datanpz/{npz_name}"
    # print("running", npz_path)
    # three_d_data = RUN_TRID(npz_path, npz_name[:-4])
