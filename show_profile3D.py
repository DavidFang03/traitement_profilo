from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from utilit import hyperbolic
from profilo import fit

def plot_profile3D(path_data="test.npz"):
        # Créer la figure et l'axe 3D
    fig = plt.figure()
    fig2 = plt.figure()
    figtest,axtest = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    axfit = fig2.add_subplot(111, projection='3d')

    # Charger le fichier .npz
    data = np.load(path_data)

    # Accéder aux tableaux sauvegardés
    profiles_array = data['profiles']
    popts_array = data['popts']
    S_array = data['S']
    X_array = np.arange(len(profiles_array[0]))

    # ! retourner si à l'envers.
    if np.mean(profiles_array)>0:
        profiles_array = -profiles_array


    # Sous-échantillonnage des données pour améliorer la lisibilité
    print(np.shape(profiles_array))
    print(np.shape(X_array))
    print(np.shape(S_array))
    step_x = 1 # Par exemple, tracer un point tous step_x
    step_s = 1  # Tracer une frame toutes les step_s
    

    X_reduced = X_array[::step_x]
    S_reduced = S_array[::step_s]
    profiles_reduced = profiles_array[::step_s, ::step_x]

    # #!  debug
    # X_reduced = X_array[::step_x]
    # S_reduced = S_array[15:17]
    # profiles_reduced = profiles_array[15:17, ::step_x]
    # # ! fin debug

    # Créer une grille pour les axes X et S
    X_grid, S_grid = np.meshgrid(X_reduced, S_reduced)
    list_of_fits=[]
    for i in range(len(S_reduced)):
        popt2 = fit(profiles_array[i])
        hyperbolic_profile = hyperbolic(X_reduced, *popt2)
        list_of_fits.append(np.copy(hyperbolic_profile))
    
    list_of_fits = np.array(list_of_fits)
    axfit.plot_surface(S_grid, X_grid, list_of_fits, cmap='viridis')

    imid = len(S_array)//2
    axtest.plot(X_array,hyperbolic(X_array,*popts_array[imid]),label=f"fit {np.around(popts_array[imid],2)}")
    axtest.plot(X_array,profiles_array[imid],label="profil")
    popt2 = fit(profiles_array[imid])
    axtest.plot(X_array,hyperbolic(X_array,*popt2),label=f"fit2 {np.around(popt2,2)}")
    axtest.set_title(f"Profil et fit")


    # Tracer le profil en 3D
    ax.plot_surface(S_grid, X_grid, profiles_reduced, cmap='viridis')
    # Tracer les points en 3D
    # ax.scatter(S_grid, X_grid, profiles_reduced, c=profiles_reduced, cmap='viridis')
    ax.axis('equal')
    axfit.axis('equal')
    # Ajouter des labels aux axes
    ax.set_xlabel('X')
    ax.set_ylabel('S')
    ax.set_zlabel('Profil')
    axfit.set_xlabel('X')
    axfit.set_ylabel('S')
    axfit.set_zlabel('Profil')

    # Afficher le graphique
    # Fonction pour mettre à jour l'angle de vue


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

    def update_viewfit(event):
        if event.key == '4':  # left
            axfit.view_init(elev=axfit.elev, azim=axfit.azim - 5)
        elif event.key == '6':  # right
            axfit.view_init(elev=axfit.elev, azim=axfit.azim + 5)
        elif event.key == '8':  # up
            axfit.view_init(elev=axfit.elev + 5, azim=axfit.azim)
        elif event.key == '5':  # down
            axfit.view_init(elev=axfit.elev - 5, azim=axfit.azim)
        fig.canvas.draw()


    # Connecter l'événement clavier à la fonction de mise à jour
    fig.canvas.mpl_connect('key_press_event', update_view)
    fig2.canvas.mpl_connect('key_press_event', update_viewfit)

if __name__ == "__main__":
    plot_profile3D()
    plt.show()
