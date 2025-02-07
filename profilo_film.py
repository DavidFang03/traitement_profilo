import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utilit import z, hyperbolic
from profilo import boundaries, boundaries_profile, frame_to_profile, fit, get_xdata_profile, plot_profile_and_fit
import time
from datetime import datetime


film_path = "./test.mp4"

def profilo_film(film_path,theta,**kwargs):
    '''
    Fonction principale
    Analyse d'un film. Renvoie data.
    Paramètres:
    film_path (str): chemin du film
    '''
    t1 = kwargs.get('t1',0)
    t2 = kwargs.get('t2',None)
    rotate = kwargs.get('rotate',False)

    list_of_profiles, list_of_popts, A, B, (nb_frames,res) = get_list_of_profiles_and_fit(
        film_path,t1,t2,theta,rotate)
    
    if len(list_of_profiles)==0:
        raise Exception("No data - check timestamps")
    # Maintenant, il faut traiter L pour obtenir l'absisse de la translation à chaque instant.
    # Géométriquement, on a besoin du coefficient directeur de la baseline.

    # Mais les baselines ne sont pas forcément parallèles. Chaque $a$ est différent -> On prend la  (d'autres meilleures méthodes ?).
    a_avg = np.mean(A)
    # On applique la formule (origine en haut a gauche de de l'image)
    S = B*np.cos(np.arctan(a_avg))

    X1 = S
    X2 = np.arange(len(list_of_profiles[0]))
    # list_of_fit = [hyperbolic(X2, *popt) for popt in list_of_popts]
    return X1, X2, list_of_profiles, list_of_popts, (nb_frames,res)

def get_min_frame_and_max_frame(t1,t2,cap):

    fps = cap.get(cv2.CAP_PROP_FPS)
    if t2 is not None:
        max_frame = int(t2 * fps)
    else:
        max_frame=np.inf
    min_frame = int(t1 * fps)
    print(t2,fps,max_frame)
    return min_frame,max_frame


def get_res(cap):
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    res = np.array([width,height],np.int16)
    return res

def get_list_of_profiles_and_fit(film_path,t1,t2,theta,rotate):
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    frame_nb = 0
    list_of_profiles = []
    list_of_popts = []
    # position x de la baseline à chaque instant sur le haut de l'image.
    A = []
    B = []
    cap = cv2.VideoCapture(film_path)
    if not cap.isOpened():
        raise Exception("cap is not opened - check path")

    # ! get min frame and max frame
    min_frame,max_frame=get_min_frame_and_max_frame(t1,t2,cap)
    print(t2,max_frame)
    #!get res
    res = get_res(cap)

    # ! read frames
    while cap.isOpened() and frame_nb < max_frame:
        # important de meettre en premiere ligne de la boucle car sinon on ne passe pas à la frame suivante.
        ret, frame = cap.read()

        # ! Check if max frame>frame_nb>min frame 
        frame_nb += 1
        # print(frame_nb, cap.get(cv2.CAP_PROP_POS_FRAMES), (min_frame, max_frame),
        #       (frame_nb > min_frame, frame_nb < max_frame))
        if frame_nb < min_frame:
            continue

        if not ret:
            break


        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # ! get profile and fit
        profile, (a, b) = frame_to_profile(frame,theta)
        popt = fit(profile)
        print(popt)

        # ! gather data
        list_of_profiles.append(np.copy(profile))
        list_of_popts.append(np.copy(popt))
        A.append(a)
        B.append(b)

        # ! show first frame
        if frame_nb==min_frame:
            ax.imshow(frame)
            ax2.plot(np.arange(len(profile)),profile)
        if frame_nb==max_frame-2:
            ax2.plot(np.arange(len(profile)),profile)
        # rangeX = get_xdata_profile(profile)
        # hyperbolic_profile = hyperbolic(rangeX, *popt)

    cap.release()
    cv2.destroyAllWindows()


    nb_frames = frame_nb - min_frame

    if len(list_of_profiles) > 2:
        if np.array_equal(list_of_profiles[1], list_of_profiles[2]):
            raise Exception("The second element of list_of_profiles is identical to the third element.")
        else:
            print("The second element of list_of_profiles is not identical to the third element.")

    return list_of_profiles, list_of_popts, np.array(A), np.array(B), (nb_frames,res)


def test_film_path(path_film="./test.mp4",theta=np.radians(30),t1=5,t2=6):
    nb_frames,res,npz_name = get_profile3D(path_film,theta,t1=t1,t2=t2,info="test")
    return nb_frames,res,npz_name


def generate_file_name(info,ext="npz"):
    now = datetime.now()
    date_str = now.strftime("%d-%m")
    time_str = now.strftime("%H-%M-%S")
    return f"{date_str}_{time_str}_{info}.{ext}"

def get_profile3D(path_film,theta,t1=0,t2=None,**kwargs):
    h=kwargs.get("h",None)
    scale=kwargs.get("scale",None)
    info=kwargs.get("info","")
    rotate=kwargs.get("rotate",False)
    print(t2)

    npz_name = generate_file_name(info=info)

    X1, X2, list_of_profiles, list_of_popts, (nb_frames,res) = profilo_film(path_film,theta,t1=t1,t2=t2,rotate=rotate)
    plot_profile_and_fit(list_of_profiles[0])

    np.savez(npz_name, profiles=np.array(list_of_profiles),
             popts=np.array(list_of_popts), S=X1,h=h,scale=scale,theta=theta)
    
    return nb_frames,res,npz_name

if __name__ == "__main__":
    start_time = time.time()

    nb_frames,res,_ = test_film_path()

    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"Temps d'exécution : {execution_time:.2f} secondes pour un film de {nb_frames} frames {res} soit {execution_time/nb_frames:.2f} par frame.")
    plt.show()
