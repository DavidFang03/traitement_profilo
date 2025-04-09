import numpy as np
import A_utilit as utilit
import cv2

threshold_bin = 0.5  # seuil pour le binarisation de l'image


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


def XLINE(image, method="weight center"):
    '''
    Detection de la nappe
    '''

    if not image.filterred:
        method = "max"

    if method == "weight center":
        bin_mask = image.gray_frame > threshold_bin * np.max(image.gray_frame)
        image.filtered_frame = image.gray_frame * bin_mask
        sum = np.sum(image.filtered_frame, axis=1)
        if 0 in sum:
            raise Exception(image.vidpath, "0 dans la somme")
        Xline = np.sum(image.rangeX * image.filtered_frame, axis=1) / sum
    elif method == "max":
        image.filtered_frame = image.gray_frame
        Xline = np.argmax(image.filtered_frame, axis=1)
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


def redfilter(frame):
    '''
    Takes an RGB image
    '''
    # ! filter red
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Définir les plages de couleurs pour le rouge
    # Note : Le rouge peut être présent dans deux plages dans l'espace HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([165, 45, 50])
    upper_red2 = np.array([180, 255, 255])

    # Créer un masque pour chaque plage de rouge
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combiner les deux masques
    mask = cv2.bitwise_or(mask1, mask2)
    # Appliquer le masque à l'image originale pour extraire les pixels rouges
    return cv2.bitwise_and(frame, frame, mask=mask)
