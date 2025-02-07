import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import curve_fit
from utilit import z, hyperbolic

import time


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
        y1 + 0.1 * lencrat
        # pour la baseline, ignorer le cratère et prendre en compte uniquement les bords.
    )
    y2a = int(
        y1 + 0.9 * lencrat
        # pour la baseline, ignorer le cratère et prendre en compte uniquement les bords.
    )

    return y1, y2, y1b, y2a


def boundaries_profile(profile):
    '''
    Détermine les limites du PROFIL pour faire le fit hyperbolique
    Paramètres:
    profile (numpy.ndarray): tableau 1D
    '''
    lenX = np.shape(profile)[0]
    nonzero_indices = np.nonzero(profile)[0]

    # Vérifier si l'array contient des éléments non nuls
    if len(nonzero_indices) > 0:
        # Indice du premier élément non nul
        i1 = nonzero_indices[0]

        # Indice du dernier élément non nul
        i2 = nonzero_indices[-1]

    else:
        raise Exception("Le profil ne contient que des zéros.")

    len_crat = i2-i1

    x1 = i1 + int(0.15 * len_crat)
    x2 = i1 + int(0.66 * len_crat)

    return x1, x2
# %%


def frame_to_profile(frame,theta):
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
    lenY, lenX, _ = np.shape(frame)

    # ! Crop
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    y1, y2, y1b, y2a = boundaries(gray_image)

    # ! Detection de la nappe
    Xmax = np.argmax(gray_image, axis=1)
    if np.shape(Xmax)[0] != lenY:
        raise Exception("np.shape(Xmax)[0] != lenY")

    # ! FIT linéaire de la baseline
    rangeY = np.arange(0, lenY)

    # Pour calculer la baseline ignorer la baseline
    baseline_abs = np.concatenate((rangeY[y1:y1b], rangeY[y2a:y2]))
    # Pour calculer la baseline ignorer la baseline
    baseline_data = np.concatenate((Xmax[y1:y1b], Xmax[y2a:y2]))
    a, b = np.polyfit(baseline_abs, baseline_data, 1)

    Xbaseline = a * rangeY + b
    # Si on veut tracer sur l'image, on ne veut que des valeurs entières
    Xbaseline_int = np.astype(Xbaseline, np.int16)
    # print(np.shape(Xbaseline), np.shape(rangeY), lenY)

    # ! PROFIL
    profile = np.zeros(lenY)
    profile_between_y1_and_y2 = z(Xmax[y1:y2], Xbaseline[y1:y2], theta)

    # print(np.shape(profile[y1:y2]), np.shape(profile_between_y1_and_y2))
    profile[y1:y2] = np.copy(profile_between_y1_and_y2)

    return profile, (a, b)


def fit(profile):
    '''
    Fit hyperbolique du profil
    Paramètres:
    profile (numpy.ndarray): tableau 1D

    Renvoie:
    tuple: Les paramètres du fit hyperbolique : zc, b, c, r0
    '''
    x1, x2 = boundaries_profile(profile)
    rangeX = np.arange(len(profile))
    try:
        popt, pcov = curve_fit(hyperbolic, rangeX[x1:x2], profile[x1:x2])
    except RuntimeError as e:
        if "Optimal parameters not found" in str(e):
            print("Echec fit - check theta")
            popt = (0,0,0,0)
        else:
            raise  # Relève l'erreur si ce n'est pas celle attendue


    return popt


def get_xdata_profile(profile):
    return np.arange(len(profile))


def plot_profile_and_fit(profile, fit_profile=None):
    fig2, ax2 = plt.subplots()
    rangeX = np.arange(len(profile))
    ax2.plot(rangeX, profile)
    if fit_profile is not None:
        ax2.plot(rangeX, fit_profile)


# def test_frame_path(frame_path="./test.png"):
#     frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
#     test_frame(frame)
#     return np.shape(frame)

def test_frame_path(frame_path,rotate=False):
    fig,ax=plt.subplots()
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    test_frame(frame)
    ax.imshow(frame)
    return np.shape(frame)

def test_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ! get PROFIL and FIT
    profile, _ = frame_to_profile(frame,theta=np.radians(10))
    rangeX = get_xdata_profile(profile)
    popt = fit(profile)
    hyperbolic_profile = hyperbolic(rangeX, *popt)
    plot_profile_and_fit(profile, hyperbolic_profile)
    return np.shape(frame)


# %%Méthode 1
if __name__ == "__main__":
    start_time = time.time()

    dims = test_frame_path(frame_path="test.png")

    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"Temps d'exécution : {execution_time:.2f} secondes pour une image de taille {dims}")
    plt.show()
