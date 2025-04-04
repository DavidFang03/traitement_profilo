import numpy as np
import A_utilit as utilit


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
    Remarques:
    - La fonction convertit l'image d'entrée en niveaux de gris.
    - Elle recadre l'image pour ignorer les régions noires et se concentrer sur la zone d'intérêt.
    - Elle calcule la ligne de base en ajustant une ligne aux données du profil.
    - La fonction renvoie l'image traitée ainsi que les données du profil et de la ligne de base.
    """
    image.Xline = XLINE(image, method="weight center")

    image.Xbaseline = XBASELINE(image)

    profile_between_y1_and_y2 = utilit.z(
        image.Xline[image.y1:image.y2], image.Xbaseline[image.y1:image.y2], image.theta)

    image.profile[image.y1:image.y2] = profile_between_y1_and_y2  # copy ?

    image.points_list = image.profile
    image.new_S_list = image.Xline
    image.new_Y_list = image.rangeY

    return


def XLINE(image, method="weight center"):
    '''
    Detection de la nappe
    '''
    if method == "weight center":
        sum = np.sum(image.gray_frame, axis=1)
        Xline = np.sum(image.rangeX * image.gray_frame, axis=1) / sum
    elif method == "max":
        Xline = np.argmax(image.gray_frame, axis=1)
    return Xline


def XBASELINE(image):
    '''
    FIT linéaire de la baseline à partir de la nappe
    '''
    # Pour calculer la baseline ignorer la baseline
    baseline_abs = np.concatenate(
        (image.rangeY[image.y1:image.ya], image.rangeY[image.yb:image.y2]))
    # Pour calculer la baseline ignorer la baseline
    baseline_data = np.concatenate(
        (image.Xline[image.y1:image.ya], image.Xline[image.yb:image.y2]))
    image.baselinea, image.baselineb = np.polyfit(
        baseline_abs, baseline_data, 1)

    Xbaseline = image.baselinea * image.rangeY + image.baselineb
    return Xbaseline
