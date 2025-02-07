from profilo_film import get_profile3D
from show_profile3D import plot_profile3D
import time
import matplotlib.pyplot as plt

import numpy as np
# test

t1, t2 = 7, 14
t1, t2 = 0.5, 11
if __name__ == '__main__':
    start_time = time.time()
    # test_film = "K:/Users/fangd/AppData/Local/Programs/Python/Python313/Lib/mypackages/test.mp4"
    # path_film = test_film
    path_film = "./vids/0702/newnappe.mp4"
    theta = np.radians(30)

    nb_frames, res, npz_name = get_profile3D(
        path_film, theta, t1=t1, t2=t2, info="test", h=1.245, rotate=True)

    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"Temps d'exécution : {execution_time:.2f} secondes pour un film de {nb_frames} frames {res} soit {execution_time/nb_frames:.2f} par frame.")

    plot_profile3D(npz_name)
    plt.legend()
    plt.show()
