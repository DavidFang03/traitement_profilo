import numpy as np
import scipy.optimize
import utilit


def boundaries(gray_image):
    """
    Détermine les limites de l'IMAGE
    Détermine deux délimitations :
        - y1 et y2 : limites de la nappe laser (seule région où on peut obtenir des données)
        - y1b et y2a : limites de la région centrale du cratère (partie à ignorer pour la baseline)
    Besoin d'une image en niveaux de gris pour cela car utilise un seuil.
    Paramètres:
    gray_image (numpy.ndarray): Un tableau 2D représentant l'image en niveaux de gris.
    Renvoie:
    tuple: Un tuple contenant quatre entiers:
        -y1,y2 : limites la nappe laser
        -y1b,y2a : limites de la région centrale du cratère
    """
    lenY, lenX = np.shape(gray_image)
    threshold = 0.4 * np.max(gray_image)

    # Ignorer les zones noires
    y1 = np.argmax(np.max(gray_image > threshold, axis=1))
    y2 = lenY - np.argmax(
        np.max((gray_image > threshold)[::-1, :], axis=1)
    )  # Ignorer les zones noires
    lencrat = int(y2 - y1)
    y1b = int(
        y1 + 0.05 * lencrat
        # pour la baseline, ignorer le cratère et prendre en compte uniquement les bords.
    )
    y2a = int(
        y1 + 0.95 * lencrat
        # pour la baseline, ignorer le cratère et prendre en compte uniquement les bords.
    )

    return y1, y2, y1b, y2a


def boundaries_Hyperbolic(profile, threshold):
    '''
    Détermine les limites du PROFIL pour faire le fit hyperbolique
    Paramètres:
    profile (numpy.ndarray): tableau 1D
    '''
    nonzero_indices = np.where(profile < threshold*np.min(profile))[0]

    # Vérifier si l'array contient des éléments non nuls
    if len(nonzero_indices) > 0:
        # Indice du premier élément non nul
        i1 = nonzero_indices[0]

        # Indice du dernier élément non nul
        i2 = nonzero_indices[-1]

    else:
        raise Exception("Le profil ne contient que des zéros.")

    return i1, i2


def frame_to_profile(image):
    """
    Convertit une image donnée en un profil en traitant l'image pour détecter et analyser la ligne de profil.
    Renvoie le profil et la position du profil.
    Paramètres:
    frame (numpy.ndarray): L'image d'entrée au format RGB.
    Renvoie:
    tuple: Un tuple contenant:
        - profile (numpy.ndarray): Le profil du cratère
        - l (float): variable intermédiaire reprénsentant la position de la baseline au cours de la translation (=position de la 
        baseline en haut de l'image)
    Remarques:
    - La fonction convertit l'image d'entrée en niveaux de gris.
    - Elle recadre l'image pour ignorer les régions noires et se concentrer sur la zone d'intérêt.
    - Elle calcule la ligne de base en ajustant une ligne aux données du profil.
    - La fonction renvoie l'image traitée ainsi que les données du profil et de la ligne de base.
    """

    # ! Detection de la nappe
    # Centre de masse plutôt ?
    image.Xline = np.argmax(image.gray_frame, axis=1)
    # sum = np.sum(image.gray_frame, axis=1)
    # image.Xline = np.sum(image.rangeX * image.gray_frame, axis=1) / sum

    if np.shape(image.Xline)[0] != image.Y:
        raise Exception("np.shape(Xline)[0] != lenY")

    # ! FIT linéaire de la baseline

    # Pour calculer la baseline ignorer la baseline
    baseline_abs = np.concatenate(
        (image.rangeY[image.y1:image.ya], image.rangeY[image.yb:image.y2]))
    # Pour calculer la baseline ignorer la baseline
    baseline_data = np.concatenate(
        (image.Xline[image.y1:image.ya], image.Xline[image.yb:image.y2]))
    image.baselinea, image.baselineb = np.polyfit(
        baseline_abs, baseline_data, 1)

    Xbaseline = image.baselinea * image.rangeY + image.baselineb
    # # Si on veut tracer sur l'image, on ne veut que des valeurs entières
    # Xbaseline_int = np.astype(Xbaseline, np.int16)

    # ! PROFIL
    profile_between_y1_and_y2 = utilit.z(
        image.Xline[image.y1:image.y2], Xbaseline[image.y1:image.y2], image.theta)
    # if np.max(profile_between_y1_and_y2) > np.abs(np.min(profile_between_y1_and_y2)):
    #     image.profile[image.y1:image.y2] = -profile_between_y1_and_y2

    # else:
    #     image.profile[image.y1:image.y2] = profile_between_y1_and_y2  # copy ?

    image.profile[image.y1:image.y2] = profile_between_y1_and_y2  # copy ?

    # newframe = np.copy(frame)
    # # for i in range(np.shape(newframe)[0]):
    # #     for j in range(np.shape(newframe)[1]):
    # #         newframe[i, int(a * i + b), :] = [255, 255, 0]
    # #         if y1b > i > y1 or y2a < i < y2:
    # #             newframe[i, j, :] = [255, 255, 255]

    return


def fit_hyperbolic(img):
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
            print("Echec fit - setting to (0,0,0,0) - check theta or rotate")
            img.popt = (img.hyperbolic_threshold*np.min(img.profile), 0, 0, 0)
            np.savetxt("test.txt", np.column_stack(
                (img.rangeY[img.i1:img.i2], img.profile[img.i1:img.i2])))
        else:
            raise  # Relève l'erreur si ce n'est pas celle attendue

    return
