import cv2
import numpy as np
import matplotlib.pyplot as plt
from VideoData import frame_to_profile, plot_profile_and_fit
import time
from datetime import datetime


film_path = "./test.mp4"

# !


def profilo_film(film_path, theta, **kwargs):
    '''
    Fonction principale
    Analyse d'un film. Renvoie data.
    Paramètres:
    film_path (str): chemin du film
    '''
    t1 = kwargs.get('t1', 0)
    t2 = kwargs.get('t2', None)
    rotate = kwargs.get('rotate', False)

    list_of_profiles, A, B, (nb_frames, res) = get_list_of_profiles_and_fit(
        film_path, t1, t2, theta, rotate)

    if len(list_of_profiles) == 0:
        raise Exception("No data - check timestamps")
    # Maintenant, il faut traiter L pour obtenir l'absisse de la translation à chaque instant.
    # Géométriquement, on a besoin du coefficient directeur de la baseline.

    # Mais les baselines ne sont pas forcément parallèles. Chaque $a$ est différent -> On prend la moyenne (d'autres meilleures méthodes ?).
    a_avg = np.mean(A)
    # On applique la formule (origine en haut a gauche de de l'image)
    # S = B*np.cos(np.arctan(a_avg))
    S = B

    X1 = S
    X2 = np.arange(len(list_of_profiles[0]))
    # list_of_fit = [hyperbolic(X2, *popt) for popt in list_of_popts]
    return X1, X2, list_of_profiles, (nb_frames, res)


def get_list_of_profiles_and_fit(videoData):

    # ! read frames
    while cap.isOpened() and frame_nb < max_frame:
        # important de meettre en premiere ligne de la boucle car sinon on ne passe pas à la frame suivante.
        ret, frame = cap.read()

        # ! Check if max frame>frame_nb>min frame
        frame_nb += 1
        # print(frame_nb, cap.get(cv2.CAP_PROP_POS_FRAMES), (min_frame, max_frame),
        #       (frame_nb > min_frame, frame_nb < max_frame))

        if not ret:
            raise Exception("Error reading frame - max_frame too big?")

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # ! get profile
        profile, (a, b), newframe = frame_to_profile(frame, theta)

        # ! gather data
        list_of_profiles.append(np.copy(profile))
        A.append(a)
        B.append(b)

        # ! show first frame
        if frame_nb == min_frame:
            ax.imshow(frame)
            ax2.plot(np.arange(len(profile)), profile)
        if frame_nb == max_frame-2:
            ax2.plot(np.arange(len(profile)), profile)
        # rangeX = get_xdata_profile(profile)
        # hyperbolic_profile = hyperbolic(rangeX, *popt)

    cap.release()
    cv2.destroyAllWindows()

    nb_frames = frame_nb - min_frame

    if len(list_of_profiles) > 2:
        if np.array_equal(list_of_profiles[1], list_of_profiles[2]):
            raise Exception(
                "The second element of list_of_profiles is identical to the third element.")
        else:
            print(
                "The second element of list_of_profiles is not identical to the third element.")

    return list_of_profiles, np.array(A), np.array(B), (nb_frames, res)


def test_film_path(path_film="./test.mp4", theta=np.radians(30), t1=5, t2=6):
    nb_frames, res, npz_name = get_profile3D(
        path_film, theta, t1=t1, t2=t2, info="test")
    return nb_frames, res, npz_name


def get_profile3D(path_film, theta, t1=0, t2=None, **kwargs):
    h = kwargs.get("h", None)
    scale = kwargs.get("scale", None)
    info = kwargs.get("info", "")
    rotate = kwargs.get("rotate", False)
    print(t2)

    npz_name = generate_file_name(info=info)

    X1, X2, list_of_profiles, (nb_frames, res) = profilo_film(
        path_film, theta, t1=t1, t2=t2, rotate=rotate)
    plot_profile_and_fit(list_of_profiles[0])

    np.savez(npz_name, profiles=np.array(list_of_profiles),
             S=X1, h=h, scale=scale, theta=theta)

    return nb_frames, res, npz_name


if __name__ == "__main__":
    start_time = time.time()

    # nb_frames, res, _ = test_film_path()
    nb_frames, res, npz_name = get_profile3D(
        "./vids/2102/3.mp4", np.radians(30), t1=2, t2=10, rotate=True)

    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"Temps d'exécution : {execution_time:.2f} secondes pour un film de {nb_frames} frames {res} soit {execution_time/nb_frames:.2f} par frame.")
    plt.show()
