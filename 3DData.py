import numpy as np
import matplotlib.pyplot as plt
import tools_ThreeD
from matplotlib import cm


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
    return update_view


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
        data = np.load(npz_path)

        self.arrS = data["arrS"]
        self.Y = data["Y"]
        self.height = data["height"]
        self.scale = data["scale"]
        self.theta = data["theta"]
        self.profiles = data["profiles"]

        # ! INIT
        self.rangeY = np.arange(self.Y)

        # ! Sous-échantillonnage

        self.step_y = 1  # Par exemple, tracer un point tous step_x
        self.step_s = 1  # Tracer une frame toutes les step_s

        self.reduced_arrS = self.arrS[::self.step_y]
        self.reduced_rangeY = self.rangeY[::self.step_s]
        self.profiles_reduced = self.profiles[::self.step_s, ::self.step_y]

        # ! INIT PLOT
        self.fig_profile = plt.figure()  # affiche data
        self.fig_fit = plt.figure()  # affiche fit
        self.fig_both = plt.figure()

        self.ax_profile = self.fig_profile.add_subplot(111, projection='3d')
        self.ax_fit = self.fig_fit.add_subplot(111, projection='3d')
        self.ax_both = self.fig_both.add_subplot(111, projection='3d')

        self.figplan, self.axplan = plt.subplots()

    def plot_profile3D(self):
        self.Y_grid, self.S_grid = np.meshgrid(
            self.reduced_rangeY, self.reduced_arrS)

        self.ax_profile.plot_surface(
            self.S_grid, self.Y_grid, self.profiles_reduced, cmap='viridis', alpha=0.9)

        return

    def fit3D(self):

        self.popt, self.uncertainties = tools_ThreeD.fit3D(
            self.arrS, self.rangeY, self.profiles_reduced)

        self.S_fit, self.X_fit, self.profile_fit_reduced = tools_ThreeD.compute_fitv2(
            self.popt)

        return

    def plot_fit3D(self):
        # self.X_fit_grid, self.S_fit_grid = np.meshgrid(self.X_fit, self.S_fit)
        # self.ax_fit.plot_surface(self.S_fit_grid, self.X_fit_grid,
        #                          self.profile_fit_reduced, cmap='viridis')
        self.ax_fit.plot_surface(self.S_fit, self.X_fit,
                                 self.profile_fit_reduced, cmap='viridis', alpha=0.9)
        print(self.popt, self.uncertainties)

    def view2D(self):
        pass

    def plot_both(self):
        self.ax_both.plot_surface(
            self.S_grid, self.Y_grid, self.profiles_reduced, cmap='viridis', alpha=0.7)
        self.ax_both.plot_surface(self.S_fit, self.X_fit,
                                  self.profile_fit_reduced, cmap=cm.inferno, alpha=0.7)
        # Ca me semble inutile de se faire chier a tracer le fit seulement dans la zone. Autant tout tracer + accelere avec numpy. Jouer avec transparence, cmap, et mettre un bouton toggle puor masquer ou non.

    def end(self):
        self.ax_profile.axis('equal')
        self.ax_profile.set_xlabel('S')
        self.ax_profile.set_ylabel('Y')
        self.fig_profile.tight_layout()

        self.ax_fit.axis('equal')
        self.ax_fit.set_xlabel('S')
        self.ax_fit.set_ylabel('Y')
        self.fig_fit.tight_layout()

        self.ax_both.axis('equal')
        self.ax_both.set_xlabel('S')
        self.ax_both.set_ylabel('Y')
        self.fig_both.tight_layout()

        self.fig_profile.canvas.mpl_connect(
            'key_press_event', return_update_func(self.fig_profile, self.ax_profile))
        self.fig_fit.canvas.mpl_connect(
            'key_press_event', return_update_func(self.fig_fit, self.ax_fit))
        self.fig_both.canvas.mpl_connect(
            'key_press_event', return_update_func(self.fig_both, self.ax_both))

        # plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example usage
    npz_path = "./datanpz/13-03_h1-985_m10.npz"
    three_d_data = ThreeD_Data(npz_path)
    three_d_data.plot_profile3D()
    three_d_data.fit3D()
    three_d_data.plot_fit3D()
    three_d_data.plot_both()
    three_d_data.end()
