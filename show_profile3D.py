from test_fit3D import fit3D, compute_fit

from mpl_toolkits.mplot3d import Axes3D
"""
Module: show_profile3D
Fonction plot_profile3D : trace profil 3D à partir d'un .npz.
Au passage, fit3D.
Dépendances:
    - matplotlib
    - numpy
    - utilit (hyperbolic)
    - profilo (fit)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from utilit import hyperbolic3D, hyperbolic
from profilo import fit


# def fit_3D(S, X, profiles_array):
#     profiles_flatten = profiles_array.flatten("F")
#     print(np.shape(S), np.shape(X), np.shape(
#         profiles_array), np.shape(profiles_flatten))

#     popt = curve_fit(hyperbolic3D, (S, X), profiles_flatten)[0]
#     return popt


def plot_profile3D(path_data="test.npz", dofit=False):
    # Créer la figure et l'axe 3D
    fig = plt.figure()  # affiche data
    figfit = plt.figure()  # affiche fit

    ax = fig.add_subplot(111, projection='3d')
    axfit = figfit.add_subplot(111, projection='3d')

    # ! Import
    data = np.load(path_data)
    profiles_array = data['profiles']
    S_array = data['S']
    X_array = np.arange(len(profiles_array[0]))

    # ! retourner si à l'envers.
    if np.mean(profiles_array) > 0:
        profiles_array = -profiles_array

    # ! Sous-échantillonnage
    step_x = 1  # Par exemple, tracer un point tous step_x
    step_s = 1  # Tracer une frame toutes les step_s

    X_reduced = X_array[::step_x]
    S_reduced = S_array[::step_s]
    profiles_reduced = profiles_array[::step_s, ::step_x]

    # ! Fit 3D
    # on fait le fit avec les arrays non réduits
    if dofit:
        # popt = [1.73621624e+00,  1.00335380e+00,2.78173453e-06, -5.81180250e-01,  1.96938539e+00]
        popt = [-2.70903727e+01,  1.71220450e-03, -
                1.31629496e-01,  2.41346151e+02,  4.69942968e+02]
        popt = fit3D(S_array, X_array, profiles_reduced)

        print(f'POPT {popt}')
        S_fit, X_fit, profile_fit_reduced = compute_fit(
            S_reduced, X_reduced, popt)

        # profile_fit_reduced = hyperbolic3D((S_reduced, X_reduced), *popt)

    # ! meshgrid
    # Créer une grille pour les axes X et S
    X_grid, S_grid = np.meshgrid(X_reduced, S_reduced)
    X_fit_grid, S_fit_grid = np.meshgrid(X_fit, S_fit)

    #! debug2D
    figtest, axtest = plt.subplots()
    imid = len(S_array)//2
    axtest.plot(X_array, profiles_array[imid], label="profil")
    zc, b, c, x0, y0 = popt
    r0 = np.sqrt(x0**2 + y0**2)
    axtest.plot(X_array, hyperbolic(X_array, zc, b, c, r0), "x", label="fit3D")

    popt2 = fit(profiles_array[imid])
    axtest.plot(X_array, hyperbolic(X_array, *popt2), "x",
                label=f"fit2 {np.around(popt2, 2)}")
    axtest.set_title(f"Profil et fit")
    #! fin debug 2D

    # Tracer le profil en 3D
    ax.plot_surface(S_grid, X_grid, profiles_reduced, cmap='viridis')
    if dofit:
        axfit.plot_surface(S_fit_grid, X_fit_grid,
                           profile_fit_reduced, cmap='viridis')
    # Tracer les points en 3D
    # ax.scatter(S_grid, X_grid, profiles_reduced, c=profiles_reduced, cmap='viridis')
    ax.axis('equal')
    axfit.axis('equal')
    # Ajouter des labels aux axes
    ax.set_xlabel('S')
    ax.set_ylabel('X')
    ax.set_zlabel('Profil')
    axfit.set_xlabel('S')
    axfit.set_ylabel('X')
    axfit.set_zlabel('Profil')

    # Connecter l'événement clavier à la fonction de mise à jour
    fig.canvas.mpl_connect('key_press_event', return_update_func(fig, ax))
    figfit.canvas.mpl_connect(
        'key_press_event', return_update_func(figfit, axfit))


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


def get_list_of_popt(S_array, profiles_array):
    list_of_fits = []
    X_array = np.arange(len(profiles_array[0]))
    for i in range(len(S_array)):
        popt = fit(profiles_array[i])
        hyperbolic_profile = hyperbolic(X_array, *popt)
        list_of_fits.append(np.copy(hyperbolic_profile))
    return np.array(list_of_fits)

    # axfit.plot_surface(S_grid, X_grid, list_of_fits, cmap='viridis')


if __name__ == "__main__":
    plot_profile3D(path_data="07-02_18-46-38_test.npz", dofit=True)
    plt.legend()
    plt.show()
