import numpy as np
import json
from datetime import datetime
import cv2


def z(x, xl, theta):
    return (x-xl) * np.tan(theta)


def hyperbolic(r, zc, b, c, r0):
    return zc + np.sqrt(b**2 + c**2 * (r - r0) ** 2)


def hyperbolic3D(x, y, zc, b, c, x0, y0):
    return zc + np.sqrt(b**2 + c**2 * (((x - x0) ** 2)+((y-y0)**2)))


def zerohyperbolic(zc, b, c, x0, y0):
    '''
    '''
    r = np.sqrt((zc**2-b**2)/np.abs(c))
    x1 = x0 - r
    x2 = x0 + r
    y1 = y0 - r
    y2 = y0 + r
    return x1, x2, y1, y2


def zerohyperbolicv2(zc, b, c, x0, y0):
    '''
    '''
    r = np.sqrt((zc**2-b**2)/np.abs(c))
    x1 = x0 - 2*r
    x2 = x0 + 2*r
    y1 = y0 - 2*r
    y2 = y0 + 2*r
    return x1, x2, y1, y2


def add_to_history(dic, json_path):
    with open(json_path, "r") as infile:
        try:
            history = json.loads(infile.read())
        except json.JSONDecodeError:
            raise Exception(f"{json_path} is empty or not a valid JSON file")

    # Ajouter le nouveau dictionnaire à la liste
    if dic not in history:
        history.append(dic)

    # Écrire la liste mise à jour dans le fichier JSON
    with open(json_path, "w") as outfile:
        json.dump(history, outfile, indent=4)

    print(f"{json_path} updated")


def get_timestamp():
    now = datetime.now()
    date_str = now.strftime("%d%m")
    time_str = now.strftime("%H-%M-%S")
    return f"{date_str}_{time_str}"


def get_day():
    now = datetime.now()
    date_str = now.strftime("%d%m")
    return date_str


def get_time():
    now = datetime.now()
    time_str = now.strftime("%H%M%S")
    return time_str


def R0(zc, b, c):
    """
    Rayon d'un hyperbole
    """

    return np.abs((zc**2-b**2)/c)


def redfilter(frame):
    '''
    Takes an RGB image
    '''
    # ! filter red
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Définir les plages de couleurs pour le rouge
    # Note : Le rouge peut être présent dans deux plages dans l'espace HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Créer un masque pour chaque plage de rouge
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combiner les deux masques
    mask = cv2.bitwise_or(mask1, mask2)
    # Appliquer le masque à l'image originale pour extraire les pixels rouges
    return cv2.bitwise_and(frame, frame, mask=mask)


if __name__ == "__main__":
    print(get_timestamp())
    print(get_day())
    print(get_time())
